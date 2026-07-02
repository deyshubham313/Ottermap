"""
dataset_prep.py
----------------
Builds training tiles + binary turf/grass masks + AOI validity masks from
the raw GeoTIFFs and GeoJSON label layers.

Key design decisions (see docs/technical_summary.md for full reasoning):

1. AOI-aware labelling. The `ShapeFile/*.geojson` files define the Area of
   Interest that was exhaustively annotated. Pixels outside the AOI are
   NOT reliable negatives — they were simply never labelled. We rasterize
   the AOI into a `valid` mask and:
     - drop tiles whose AOI coverage is below `min_aoi_coverage`
     - pass the valid mask through to training so loss is only computed
       inside the AOI (see src/losses.py)

2. Location-based (not random-tile-based) splitting. All three images are
   in different geographies. Naive random tile splitting would put tiles
   from the same image/location in both train and val, which leaks
   context and overstates generalization. Instead we support
   leave-one-location-out (LOLO): train on tiles from 2 of the 3 images,
   validate on tiles from the held-out image. `train.py` loops this for
   all 3 held-out choices and reports averaged metrics, which is a far
   more honest generalization estimate given only 3 source images.

3. Fixed tile size with overlap for both train and inference, so the
   train/inference distributions match (avoids a common train/test skew
   where training uses small crops but inference stitches large windows).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import Window
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

from geo_utils import open_raster_geo, load_aoi_polygon, load_label_features


def rasterize_geometries(geometries: list[dict], transform, out_shape: tuple[int, int],
                          fill: int = 0, value: int = 1) -> np.ndarray:
    """Rasterize a list of GeoJSON geometry dicts to a uint8 mask."""
    if not geometries:
        return np.full(out_shape, fill, dtype=np.uint8)
    shapes = [(geom, value) for geom in geometries]
    mask = rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=fill,
        dtype="uint8",
        all_touched=False,
    )
    return mask


def build_masks_for_image(image_id: int, raw_dir: Path):
    """
    Returns (rgb_array[h,w,3] uint8, grass_mask[h,w] uint8, aoi_mask[h,w] uint8)
    for one source image, fully rasterized at native resolution.
    """
    tif_path = raw_dir / "aerial_imagery_pack" / f"{image_id}.tiff"
    label_path = raw_dir / "feature_layers" / "GeoJSON" / f"{image_id}.geojson"
    aoi_path = raw_dir / "feature_layers" / "ShapeFile" / f"{image_id}.geojson"

    geo = open_raster_geo(str(tif_path))
    with rasterio.open(tif_path) as src:
        rgb = src.read([1, 2, 3])  # (3, H, W)
        rgb = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)

    features = load_label_features(str(label_path))
    grass_geoms = [f["geometry"] for f in features if f.get("geometry")]
    grass_mask = rasterize_geometries(grass_geoms, geo.transform, (geo.height, geo.width))

    aoi_geom = load_aoi_polygon(str(aoi_path))
    aoi_mask = rasterize_geometries([aoi_geom], geo.transform, (geo.height, geo.width))

    return rgb, grass_mask, aoi_mask, geo


def tile_image(rgb: np.ndarray, grass_mask: np.ndarray, aoi_mask: np.ndarray,
                tile_size: int = 512, overlap: int = 64,
                min_aoi_coverage: float = 0.3):
    """
    Slide a tile_size x tile_size window with the given overlap across the
    image. Yields (row, col, rgb_tile, mask_tile, valid_tile) for tiles
    whose AOI coverage clears `min_aoi_coverage`. Edge tiles are padded
    (reflect) rather than skipped, so no image content near a border is lost.
    """
    h, w = grass_mask.shape
    stride = tile_size - overlap
    pad = tile_size  # generous pad so edge windows are always full-size

    rgb_p = np.pad(rgb, ((0, pad), (0, pad), (0, 0)), mode="reflect")
    mask_p = np.pad(grass_mask, ((0, pad), (0, pad)), mode="reflect")
    aoi_p = np.pad(aoi_mask, ((0, pad), (0, pad)), mode="reflect")

    for row in range(0, h, stride):
        for col in range(0, w, stride):
            rgb_tile = rgb_p[row:row + tile_size, col:col + tile_size]
            mask_tile = mask_p[row:row + tile_size, col:col + tile_size]
            valid_tile = aoi_p[row:row + tile_size, col:col + tile_size]

            coverage = valid_tile.mean()
            if coverage < min_aoi_coverage:
                continue
            yield row, col, rgb_tile, mask_tile, valid_tile


def main():
    ap = argparse.ArgumentParser(description="Build AOI-aware training tiles.")
    ap.add_argument("--raw-dir", default="dataset/raw")
    ap.add_argument("--out-dir", default="dataset/tiles")
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--min-aoi-coverage", type=float, default=0.3)
    ap.add_argument("--image-ids", nargs="+", type=int, default=[1, 2, 3])
    args = ap.parse_args()

    if not RASTERIO_AVAILABLE:
        raise SystemExit(
            "rasterio not installed in this environment. Install the geo "
            "stack from requirements.txt and re-run (see README)."
        )

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "valid").mkdir(parents=True, exist_ok=True)

    manifest = []
    for image_id in args.image_ids:
        rgb, grass_mask, aoi_mask, geo = build_masks_for_image(image_id, raw_dir)
        print(f"[{image_id}] {rgb.shape[1]}x{rgb.shape[0]} px | "
              f"grass px: {grass_mask.mean():.3%} | AOI coverage: {aoi_mask.mean():.3%}")

        n = 0
        for row, col, rgb_t, mask_t, valid_t in tile_image(
            rgb, grass_mask, aoi_mask, args.tile_size, args.overlap, args.min_aoi_coverage
        ):
            stem = f"img{image_id}_r{row}_c{col}"
            np.save(out_dir / "images" / f"{stem}.npy", rgb_t)
            np.save(out_dir / "masks" / f"{stem}.npy", mask_t)
            np.save(out_dir / "valid" / f"{stem}.npy", valid_t)
            manifest.append({
                "stem": stem, "source_image": image_id, "row": row, "col": col,
                "grass_frac": float(mask_t.mean()), "aoi_frac": float(valid_t.mean()),
            })
            n += 1
        print(f"  -> {n} tiles written")

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest)} tiles total. Manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
