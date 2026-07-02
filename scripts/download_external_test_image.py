#!/usr/bin/env python3
"""
download_external_test_image.py
--------------------------------
Fetches a real, unseen-location NAIP GeoTIFF from the USGS National Map
ImageServer for the required "generalization test on an additional
aerial imagery dataset from a different geographic location" (brief
section 2). This endpoint was verified live from this project's dev
environment (see docs/technical_summary.md) — it is USGS/USDA public
domain orthoimagery, no API key required.

Default location: Zilker Park, Austin, TX (30.266N, -97.772W) — a large
turf/lawn area, ~2,300 km from all 3 training locations
(Chico CA / Olympia-area WA / Myrtle Beach SC), so it's a genuine
out-of-distribution test of a different city, climate, and NAIP capture.

Usage:
    python scripts/download_external_test_image.py
    python scripts/download_external_test_image.py --bbox "-97.7760,30.2635,-97.7690,30.2695" --out dataset/external/austin_zilker.tif
    python scripts/download_external_test_image.py --place "Grant Park, Chicago, IL" --bbox "..."
"""
import argparse
from pathlib import Path

import requests

ENDPOINT = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"

DEFAULT_BBOX = "-97.7760,30.2635,-97.7690,30.2695"  # Zilker Park, Austin, TX


def download(bbox: str, out_path: Path, size: str = "1536,1536"):
    params = {
        "bbox": bbox,
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": size,
        "format": "tiff",
        "pixelType": "U8",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }
    resp = requests.get(ENDPOINT, params=params, timeout=60)
    resp.raise_for_status()
    ctype = resp.headers.get("Content-Type", "")
    if "image" not in ctype and "tiff" not in ctype:
        raise RuntimeError(
            f"Unexpected response ({ctype}) — the service may have changed. "
            f"First 300 bytes: {resp.content[:300]!r}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    print(f"Saved {len(resp.content):,} bytes -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", default=DEFAULT_BBOX,
                     help="xmin,ymin,xmax,ymax in EPSG:4326 (lon,lat)")
    ap.add_argument("--out", default="dataset/external/austin_zilker_park.tif")
    ap.add_argument("--size", default="1536,1536")
    args = ap.parse_args()
    download(args.bbox, Path(args.out), args.size)


if __name__ == "__main__":
    main()
