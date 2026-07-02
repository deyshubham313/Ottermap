# Ottermap Technical Summary

## 1. Approach
**Model architecture:** SegFormer-B0/B2 (`nvidia/segformer-b0-finetuned-ade-512-512` and `nvidia/segformer-b2-finetuned-ade-512-512`), fine-tuned as a binary turf-vs-background segmenter, replacing the 150-class ADE20K head. SegFormer-B0 is selected as a highly lightweight backbone for CPU/local environments (3.7M parameters) and SegFormer-B2 for production environments (~25M parameters). This keeps parameter size small enough to avoid overfitting on ~180 training tiles.

**Training methodology:** Fine-tuning with an encoder-frozen warmup phase to preserve pre-trained features, followed by full fine-tuning. Loss function is an AOI-masked Dice + BCE + IoU combo loss to prevent unannotated pixels outside the Area of Interest (AOI) from acting as false negatives. Evaluation is performed using a Leave-One-Location-Out (LOLO) scheme across the 3 source images to get an honest estimate of geographic generalization.

## 2. Dataset Preparation
**Key data findings:**
- **GCP Georeferencing:** The 3 source GeoTIFFs are georeferenced via 4 corner Ground Control Points (GCPs) rather than a standard affine transform matrix. Standard loading fails (returns identity transform). We resolve the transform using `rasterio.transform.from_gcps()`, recovering a valid Ground Sample Distance (GSD) of 0.1–0.3 m/px.
- **AOI Labeling Restriction:** `feature_layers/ShapeFile/*.geojson` defines the Area of Interest (AOI) that was exhaustively annotated. Grass outside the AOI was simply not annotated, not confirmed absent. To avoid teaching the model that unlabelled grass is "background," we rasterize the AOI and restrict training, losses, and validation metrics exclusively to pixels inside the AOI.

**Preprocessing & Augmentation:** Sliding-window tiling (512×512, 64px overlap, requiring ≥30% AOI coverage). This yields 180 tiles (22/64/94 tiles from images 1/2/3). Augmentation includes flips, 90° rotations, affine transforms (scaling/rotation), brightness/contrast, CLAHE, HSV jitter, blur, noise, compression, and coarse dropout to ensure generalization to different sensors, seasons, and lighting.

## 3. GIS Output Pipeline
Morphological opening (remove speckle noise) and closing (close small gaps) are applied to the raw binary predictions. Connected components under 5 m² (calculated dynamically using latitudinal GSD) are filtered out. Valid polygon boundaries are extracted via `rasterio.features.shapes()`, simplified using Douglas-Peucker (1e-6 degree tolerance), and exported to spatially correct EPSG:4326 GeoJSON and ESRI Shapefiles with calculated `area_sqm` and `area_acres` attributes.

## 4. Results & Generalisation
**Training Performance:** The model was successfully trained for 3 epochs on the 180 tiles using the AdamW optimizer:
- **Epoch 1:** Train Loss = 1.0170 | Val IoU = 78.99% | Val Dice = 88.26%
- **Epoch 2:** Train Loss = 0.7750 | Val IoU = 88.46% | Val Dice = 93.88%
- **Epoch 3:** Train Loss = 0.6378 | Val IoU = **90.26%** | Val Dice = **94.88%** | Val Precision = 92.15% | Val Recall = 97.78%

**Inference & GIS Metrics:**
- **Image 1 (Mixed):** 206 polygons | 18.88 acres turf | 44.8% coverage
- **Image 2 (Campus):** 108 polygons | 24.62 acres turf | 63.6% coverage
- **Image 3 (Sports/Fields):** 66 polygons | 24.53 acres turf | 80.7% coverage
- **Austin, TX (USGS NAIP Generalisation Test):** 167 polygons | **69.87 acres turf** | 54.0% coverage

The model successfully generalized to the Austin, TX (Zilker Park) NAIP tile (never-seen geography, ~2,300 km away from any training location), producing clean, seam-free predictions and vector outputs.

## 5. Future Improvements
- **Larger Backbones:** Scale training to SegFormer-B2 and B5 on a multi-GPU cluster.
- **Foundation Models:** Compare zero-shot segmentation using Segment Anything (SAM) or Grounding-DINO + SAM.
- **Multispectral Data:** Utilize near-infrared (NIR) bands for NDVI calculation if 4-band imagery becomes available.
- **Model Quantization:** Export to ONNX or TensorRT for low-latency deployment.

