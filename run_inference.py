#!/usr/bin/env python3
"""
run_inference.py
----------------
Faster inference runner with explicit progress output, no TTA,
smaller tiles for CPU speed, and correct fine-tuned checkpoint loading.

Outputs per image (under outputs/<stem>/):
  mask.tif            georeferenced binary raster
  overlay.png         RGB + green turf overlay
  prediction.geojson  vectorised polygons (EPSG:4326)
  prediction.shp      same as Shapefile
  stats.json          polygon count, acres, timing
"""
import sys
import json
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn.functional as F
import rasterio
import cv2
from PIL import Image as PILImage

from geo_utils import open_raster_geo
from model import build_model
from utils import get_device, load_checkpoint
from polygonize import clean_mask, mask_to_geodataframe, export_gdf, summarize, meters_per_degree

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TILE_SIZE   = 512
OVERLAP     = 64        # smaller overlap -> much faster on CPU
THRESHOLD   = 0.5
OVERLAY_ALPHA = 0.45
OPEN_KERNEL = 3
CLOSE_KERNEL = 5
MIN_AREA_SQM = 5.0
SIMPLIFY_DEG = 1e-6

CHECKPOINT  = "models/best_model.pth"
BACKBONE    = "b0"
OUT_ROOT    = Path("outputs")


def make_blend(tile_size):
    ramp = np.bartlett(tile_size)
    ramp = np.clip(ramp, 1e-3, None)
    return np.outer(ramp, ramp).astype(np.float32)


@torch.no_grad()
def predict_image(rgb, model, device):
    h, w = rgb.shape[:2]
    pad  = TILE_SIZE
    stride = TILE_SIZE - OVERLAP

    rgb_p = np.pad(rgb, ((0, pad), (0, pad), (0, 0)), mode="reflect")
    accum  = np.zeros((h + pad, w + pad, 2), dtype=np.float32)
    wsum   = np.zeros((h + pad, w + pad),    dtype=np.float32)
    blend  = make_blend(TILE_SIZE)

    rows = list(range(0, h, stride))
    cols = list(range(0, w, stride))
    total = len(rows) * len(cols)
    done  = 0

    model.eval()
    for row in rows:
        for col in cols:
            tile = rgb_p[row:row+TILE_SIZE, col:col+TILE_SIZE]  # (T,T,3) uint8

            x = tile.astype(np.float32) / 255.0
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,T,T)

            logits = model(x)                     # (1,2,T,T)
            probs  = F.softmax(logits, dim=1)[0]  # (2,T,T)
            probs  = probs.float().cpu().numpy().transpose(1, 2, 0)  # (T,T,2)

            accum[row:row+TILE_SIZE, col:col+TILE_SIZE] += probs * blend[..., None]
            wsum [row:row+TILE_SIZE, col:col+TILE_SIZE] += blend

            done += 1
            if done % 20 == 0 or done == total:
                print(f"    tiles {done}/{total} ({100*done//total}%)", flush=True)

    wsum = np.clip(wsum, 1e-6, None)
    full = (accum / wsum[..., None])[:h, :w]   # (H,W,2)
    return full[..., 1]   # turf probability map (H,W)


def run_one(image_path: Path, model, device):
    stem   = image_path.stem
    out_dir = OUT_ROOT / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"\n[{stem}] Opening raster ...", flush=True)
    geo = open_raster_geo(str(image_path))
    with rasterio.open(image_path) as src:
        rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0))   # (H,W,3)

    print(f"[{stem}] {rgb.shape[1]}x{rgb.shape[0]} px — running inference ...", flush=True)
    turf_prob   = predict_image(rgb, model, device)
    binary_mask = (turf_prob >= THRESHOLD).astype(np.uint8)
    cleaned     = clean_mask(binary_mask, OPEN_KERNEL, CLOSE_KERNEL)

    # ── GIS vector output ────────────────────────────────────────────────────
    _, center_lat = geo.transform * (geo.width / 2, geo.height / 2)
    gdf = mask_to_geodataframe(cleaned, geo.transform, geo.crs, center_lat,
                               min_area_sqm=MIN_AREA_SQM,
                               simplify_tolerance_deg=SIMPLIFY_DEG)
    export_gdf(gdf,
               out_geojson=str(out_dir / "prediction.geojson"),
               out_shapefile=str(out_dir / "prediction.shp"))

    m_lon, m_lat = meters_per_degree(center_lat)
    img_area_sqm = (geo.width * geo.height
                    * abs(geo.transform.a) * abs(geo.transform.e)
                    * m_lon * m_lat)
    stats = summarize(gdf, full_image_area_sqm=img_area_sqm)
    stats["mean_confidence"] = float(turf_prob[cleaned == 1].mean()) if cleaned.sum() else None
    stats["inference_time_s"] = time.time() - t0
    stats["image"] = str(image_path)
    stats["image_size_px"] = [geo.width, geo.height]
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ── Raster mask output ───────────────────────────────────────────────────
    mask_profile = dict(driver="GTiff", height=geo.height, width=geo.width,
                        count=1, dtype="uint8", crs=geo.crs, transform=geo.transform)
    with rasterio.open(out_dir / "mask.tif", "w", **mask_profile) as dst:
        dst.write(cleaned, 1)

    # ── Overlay PNG ──────────────────────────────────────────────────────────
    overlay = rgb.copy()
    green   = np.zeros_like(overlay)
    green[..., 1] = 255
    alpha   = (cleaned[..., None] * OVERLAY_ALPHA).astype(np.float32)
    overlay = (overlay * (1 - alpha) + green * alpha).astype(np.uint8)
    cv2.imwrite(str(out_dir / "overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"[{stem}] DONE — {stats['num_polygons']} polygons | "
          f"{stats['total_turf_area_acres']:.2f} acres | "
          f"{stats.get('turf_area_fraction', 0):.1%} of image | "
          f"{stats['inference_time_s']:.0f}s", flush=True)
    return stats


def main():
    device = get_device()
    print(f"Device: {device}")
    print(f"Loading fine-tuned checkpoint from {CHECKPOINT} ...")
    model = build_model(BACKBONE, freeze_encoder=False).to(device)
    model, meta = load_checkpoint(model, CHECKPOINT, device=device)
    model.eval()
    print(f"Checkpoint meta: {meta}\n")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ── Training / validation images (images 1, 2, 3) ────────────────────────
    training_images = sorted(Path("dataset/raw/aerial_imagery_pack").glob("*.tiff"))
    all_stats = []
    for img_path in training_images:
        stats = run_one(img_path, model, device)
        all_stats.append(stats)

    # ── External test image (Austin TX) ──────────────────────────────────────
    external = Path("dataset/external/austin_zilker_park.tif")
    if external.exists():
        stats = run_one(external, model, device)
        all_stats.append(stats)
    else:
        print(f"\nExternal image not found at {external} — skipping.")

    with open(OUT_ROOT / "batch_summary.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    print("\n" + "=" * 60)
    print("ALL INFERENCE COMPLETE")
    print("=" * 60)
    for s in all_stats:
        print(f"  {Path(s['image']).stem:30s}  "
              f"{s['num_polygons']:4d} polygons  "
              f"{s['total_turf_area_acres']:7.2f} acres  "
              f"{s.get('turf_area_fraction', 0):5.1%}")


if __name__ == "__main__":
    main()
