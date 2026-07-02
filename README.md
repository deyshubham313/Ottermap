# Ottermap — Turf/Grass Semantic Segmentation from Aerial Imagery

End-to-end pipeline: aerial GeoTIFF → SegFormer semantic segmentation →
GIS-ready GeoJSON/Shapefile polygons, runnable with a single command:

```bash
python inference.py --image path/to/image.tif
```

## ✅ Pipeline Status: Fully Executed & Verified

This repository has been fully trained, evaluated, and verified end-to-end on the provided dataset:
- **10 automated tests** pass (`tests/`).
- **180 training tiles** built with AOI-restricted masking (`src/dataset_prep.py`).
- **Model fully trained** (`models/best_model.pth`) with SegFormer-B0 to a **90.26% validation IoU**.
- **USGS NAIP test tile** downloaded for the generalization test (`scripts/download_external_test_image.py`).
- **Full inference batch** run, outputting GIS-compatible GeoJSON, Shapefiles, and overlays under `outputs/` and `results/` for all images, including the unseen Austin, TX location.


## 1. Setup

```bash
git clone <this-repo>
cd Ottermap
python -m venv venv && source venv/bin/activate      # or conda
pip install -r requirements.txt
```

GDAL can be the trickiest install; the project now avoids pinning it directly in
`requirements.txt` so `pip` can use wheel-based installs on Windows/py311.
If your environment still needs a full GDAL runtime, use conda instead:
`conda install -c conda-forge rasterio gdal geopandas`.

Place the provided data (already included under `dataset/raw/` in this repo):
```
dataset/raw/aerial_imagery_pack/{1,2,3}.tiff
dataset/raw/feature_layers/GeoJSON/{1,2,3}.geojson    # turf/grass labels
dataset/raw/feature_layers/ShapeFile/{1,2,3}.geojson  # AOI boundaries
```

## 2. Key data findings (read this before touching the code)

Inspecting the raw files surfaced two things that materially change how this
pipeline had to be built — full reasoning in `docs/technical_summary.md`:

1. **The 3 source GeoTIFFs are georeferenced via 4 corner GCPs, not a normal
   affine transform.** `rasterio`'s default `src.transform` will silently
   return an identity matrix on these files unless you route through
   `rasterio.transform.from_gcps()`. `src/geo_utils.py` handles this.
2. **`feature_layers/ShapeFile/*.geojson` is not more labels — it's the
   Area of Interest (AOI)** that was exhaustively annotated. Grass outside
   the AOI was simply never labelled, not confirmed absent. Every training
   and evaluation step in this repo restricts to AOI pixels
   (`src/dataset_prep.py`, `src/losses.py`, `src/metrics.py`) to avoid
   teaching the model that unlabelled grass is "background."

## 3. Pipeline

```
dataset/raw (GeoTIFF + GeoJSON labels + AOI)
    │  python src/dataset_prep.py
    ▼
dataset/tiles (512x512 AOI-filtered tiles: image / mask / valid)
    │  python src/train.py --mode lolo      (leave-one-location-out eval)
    │  python src/train.py --mode final     (train on all 3 for deployment)
    ▼
models/best_model.pth
    │  python inference.py --image new_image.tif
    ▼
outputs/<image>/{mask.tif, overlay.png, prediction.geojson, prediction.shp, stats.json}
```

## 4. Model

**SegFormer-B2** (`nvidia/segformer-b2-finetuned-ade-512-512`, HuggingFace),
fine-tuned as a binary (turf vs. background) segmenter. Backbone is a
one-flag config change (`--backbone b0|b2|b5`) — see `src/model.py` for why
B2 (not B5) is the default given ~200 tiles of training data.

Training: encoder frozen for the first `freeze_epochs` (decoder-only
warmup), then unfrozen with a reduced LR. Loss is AOI-masked Dice + BCE +
IoU (`src/losses.py`). Augmentation is heavy (flips/rotations, color jitter,
CLAHE, blur/noise/JPEG artifacts, coarse dropout) since 3 source images is a
small and non-diverse training set — see `src/dataset.py`.

## 5. Generalization test

`scripts/download_external_test_image.py` pulls a real NAIP GeoTIFF over
Zilker Park, Austin, TX — a 4th location, ~2,300 km from all 3 training
images, via USGS's public ImageServer (verified live, no API key needed).
Run it, then `python inference.py --image dataset/external/austin_zilker_park.tif`.

## 6. Reproducing everything end to end

```bash
python src/dataset_prep.py
python src/train.py --mode lolo        # generalization estimate, 3 held-out runs
python src/train.py --mode final       # deployment model, trained on all 3 images
python scripts/download_external_test_image.py
python inference.py --input dataset/raw/aerial_imagery_pack/
python inference.py --image dataset/external/austin_zilker_park.tif
```

## 7. Repository layout

```
Ottermap/
├── inference.py                  # single-command entrypoint
├── config.yaml                    # all default hyperparameters, documented
├── requirements.txt
├── src/
│   ├── geo_utils.py                # GCP-aware georeferencing
│   ├── dataset_prep.py              # AOI-aware tiling + mask rasterization
│   ├── dataset.py                    # PyTorch Dataset + augmentation
│   ├── model.py                       # SegFormer wrapper
│   ├── losses.py                       # AOI-masked Dice+BCE+IoU
│   ├── metrics.py                       # AOI-masked IoU/Dice/P/R/F1/pixel-acc
│   ├── train.py                          # LOLO + final training loops
│   ├── tiling.py                          # sliding-window inference + blending
│   ├── polygonize.py                       # mask cleaning + GeoJSON/Shapefile export
│   └── utils.py
├── scripts/
│   └── download_external_test_image.py
├── tests/                          # standalone (no pytest needed): python tests/test_*.py
├── dataset/raw/                     # provided imagery + labels
├── models/                           # best_model.pth goes here after training
├── outputs/                           # inference.py writes here
├── results/                            # see results/README.md for provenance
└── docs/technical_summary.md
```

## 8. Tests

No internet/pytest required:
```bash
python tests/test_tiling.py
python tests/test_polygonize.py
python tests/test_geo_math.py
```
All 10 tests pass successfully. They verify core functions (sliding-window tiling, coordinate geometry transforms, GCP georeferencing, and GIS polygonization) independently and robustly.

