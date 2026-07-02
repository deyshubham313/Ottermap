#!/usr/bin/env python3
"""
inference.py
------------
Single-command entrypoint for the Ottermap turf/grass segmentation
pipeline.

    python inference.py --image path/to/image.tif
    python inference.py --input ./images/                # batch mode

Produces, per input image (under outputs/<image_stem>/):
    mask.tif          - binary prediction, georeferenced, same grid as input
    overlay.png        - RGB image with turf prediction overlaid
    prediction.geojson  - vectorized turf polygons (EPSG:4326)
    prediction.shp (+ .dbf/.shx/.prj) - same, as Shapefile
    stats.json          - polygon count, area (sqm/acres), turf area fraction,
                           inference time, confidence summary

No manual editing required: the script resolves the model checkpoint from
models/best_model.pth by default and downloads nothing at runtime.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from geo_utils import open_raster_geo, RASTERIO_AVAILABLE
from tiling import predict_large_image, test_time_augment_predict
from polygonize import clean_mask, mask_to_geodataframe, export_gdf, summarize, meters_per_degree

try:
    import rasterio
except ImportError:
    rasterio = None

try:
    import torch
    import torch.nn.functional as F
    from model import build_model
    from utils import get_device, load_checkpoint
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
except ImportError:
    cv2 = None


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_predict_fn(model, device, amp: bool, use_tta: bool):
    @torch.no_grad()
    def _raw_predict(tile_batch: np.ndarray) -> np.ndarray:
        # tile_batch: (N, H, W, 3) uint8 -> normalized tensor
        x = tile_batch.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
        with torch.autocast(device_type=device.type if hasattr(device, "type") else "cpu",
                             enabled=amp):
            logits = model(x)
            probs = F.softmax(logits, dim=1)
        return probs.float().cpu().numpy()

    if use_tta:
        return lambda tile_batch: test_time_augment_predict(tile_batch, _raw_predict)
    return _raw_predict


def run_on_image(image_path: Path, model, device, args, out_root: Path):
    stem = image_path.stem
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    geo = open_raster_geo(str(image_path))
    with rasterio.open(image_path) as src:
        rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0))

    predict_fn = build_predict_fn(model, device, amp=args.amp, use_tta=args.tta)
    probs = predict_large_image(rgb, predict_fn, tile_size=args.tile_size,
                                 overlap=args.overlap, num_classes=2)
    turf_prob = probs[..., 1]
    binary_mask = (turf_prob >= args.threshold).astype(np.uint8)

    cleaned = clean_mask(binary_mask, open_kernel=args.open_kernel, close_kernel=args.close_kernel)

    # --- GIS vector output ---
    _, center_lat = geo.transform * (geo.width / 2, geo.height / 2)

    gdf = mask_to_geodataframe(cleaned, geo.transform, geo.crs, center_lat,
                                min_area_sqm=args.min_area_sqm,
                                simplify_tolerance_deg=args.simplify_tolerance)
    export_gdf(gdf, out_geojson=str(out_dir / "prediction.geojson"),
               out_shapefile=str(out_dir / "prediction.shp"))

    m_per_deg_lon, m_per_deg_lat = meters_per_degree(center_lat)
    full_image_area_sqm = geo.width * geo.height * \
        abs(geo.transform.a) * abs(geo.transform.e) * m_per_deg_lon * m_per_deg_lat
    stats = summarize(gdf, full_image_area_sqm=full_image_area_sqm)
    stats["mean_confidence"] = float(turf_prob[cleaned == 1].mean()) if cleaned.sum() else None
    stats["inference_time_s"] = time.time() - t0
    stats["image"] = str(image_path)
    stats["image_size"] = [geo.width, geo.height]
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # --- raster mask output (georeferenced) ---
    mask_profile = {
        "driver": "GTiff", "height": geo.height, "width": geo.width,
        "count": 1, "dtype": "uint8", "crs": geo.crs, "transform": geo.transform,
    }
    with rasterio.open(out_dir / "mask.tif", "w", **mask_profile) as dst:
        dst.write(cleaned, 1)

    # --- overlay PNG ---
    overlay = rgb.copy()
    green = np.zeros_like(overlay)
    green[..., 1] = 255
    alpha = (cleaned[..., None] * args.overlay_alpha).astype(np.float32)
    overlay = (overlay * (1 - alpha) + green * alpha).astype(np.uint8)
    if cv2 is not None:
        cv2.imwrite(str(out_dir / "overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    else:
        from PIL import Image as PILImage
        PILImage.fromarray(overlay).save(out_dir / "overlay.png")

    print(f"[{stem}] {stats['num_polygons']} polygons | "
          f"{stats['total_turf_area_acres']:.2f} acres "
          f"({stats.get('turf_area_fraction', 0):.1%} of image) | "
          f"{stats['inference_time_s']:.1f}s -> {out_dir}")
    return stats


def main():
    ap = argparse.ArgumentParser(description="Ottermap turf/grass segmentation inference.")
    ap.add_argument("--image", type=str, help="Path to a single GeoTIFF.")
    ap.add_argument("--input", type=str, help="Path to a directory of GeoTIFFs (batch mode).")
    ap.add_argument("--checkpoint", type=str, default="models/best_model.pth")
    ap.add_argument("--backbone", choices=["b0", "b2", "b5"], default="b2")
    ap.add_argument("--out-dir", type=str, default="outputs")
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--open-kernel", type=int, default=3)
    ap.add_argument("--close-kernel", type=int, default=5)
    ap.add_argument("--min-area-sqm", type=float, default=5.0)
    ap.add_argument("--simplify-tolerance", type=float, default=1e-6)
    ap.add_argument("--overlay-alpha", type=float, default=0.45)
    ap.add_argument("--tta", action="store_true", default=True, help="test-time augmentation")
    ap.add_argument("--no-tta", dest="tta", action="store_false")
    ap.add_argument("--amp", action="store_true", default=True, help="mixed precision inference")
    args = ap.parse_args()

    if not args.image and not args.input:
        ap.error("Provide --image <file.tif> or --input <directory>")

    if not RASTERIO_AVAILABLE or rasterio is None:
        raise SystemExit(
            "rasterio is not installed. Run:\n"
            "    pip install -r requirements.txt\n"
            "in an environment with internet access (see README.md)."
        )
    if not TORCH_AVAILABLE:
        raise SystemExit(
            "torch/transformers not installed. Run:\n"
            "    pip install -r requirements.txt\n"
            "in a GPU-enabled environment (see README.md)."
        )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(
            f"Model checkpoint not found at {checkpoint_path}.\n"
            f"Train one first:\n    python src/train.py --mode final\n"
            f"or place a provided checkpoint at that path."
        )

    device = get_device()
    model = build_model(args.backbone, freeze_encoder=False).to(device)
    model, meta = load_checkpoint(model, checkpoint_path, device=device)
    model.eval()
    print(f"Loaded checkpoint {checkpoint_path} ({meta}) on {device}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.image:
        images = [Path(args.image)]
    else:
        in_dir = Path(args.input)
        images = sorted(list(in_dir.glob("*.tif")) + list(in_dir.glob("*.tiff")))
        if not images:
            raise SystemExit(f"No .tif/.tiff files found in {in_dir}")

    all_stats = []
    for img_path in images:
        stats = run_on_image(img_path, model, device, args, out_root)
        all_stats.append(stats)

    if len(all_stats) > 1:
        with open(out_root / "batch_summary.json", "w") as f:
            json.dump(all_stats, f, indent=2)


if __name__ == "__main__":
    main()
