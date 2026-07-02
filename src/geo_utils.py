"""
geo_utils.py
------------
Georeferencing helpers for the Ottermap turf/grass segmentation pipeline.

IMPORTANT DATA GOTCHA (discovered by inspecting the provided GeoTIFFs):
The three source images are geo-referenced using FOUR corner tiepoints
(ModelTiepointTag with 4 entries) rather than a single tiepoint + pixel
scale (ModelPixelScaleTag). GDAL/rasterio expose this as Ground Control
Points (GCPs), NOT as `src.transform`. If you naively call `src.transform`
on these files you will get an identity/default affine and every
downstream coordinate will be wrong.

Because the four corners form an axis-aligned rectangle (no rotation),
`rasterio.transform.from_gcps()` gives an exact affine — there is no need
for polynomial/TPS rectification here. We still route through
`from_gcps` (rather than hand-rolling the math) so the code degrades
gracefully if a future image *does* carry a normal transform or a
rotated GCP set.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import rasterio
    from rasterio.transform import from_gcps
    from rasterio.control import GroundControlPoint
    from rasterio.crs import CRS
    RASTERIO_AVAILABLE = True
except ImportError:  # pragma: no cover - sandbox without rasterio
    RASTERIO_AVAILABLE = False


@dataclass
class RasterGeo:
    """Georeferencing info for one raster, resolved to a usable affine."""
    width: int
    height: int
    transform: "object"          # rasterio.Affine
    crs: "object"                # rasterio.crs.CRS
    source: str                  # "transform" or "gcps"


def open_raster_geo(path: str) -> RasterGeo:
    """
    Open a GeoTIFF and return a resolved (width, height, transform, crs),
    correctly handling both normal-affine and GCP-georeferenced files.
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError(
            "rasterio is required for open_raster_geo(). Install the geo "
            "stack (see requirements.txt) — this sandbox environment does "
            "not have it, so this path is exercised only in the real "
            "training/inference environment."
        )

    with rasterio.open(path) as src:
        width, height = src.width, src.height
        crs = src.crs

        has_real_transform = (
            src.transform is not None
            and not src.transform.is_identity
            and src.transform != rasterio.Affine.identity()
        )

        if has_real_transform and not src.gcps[0]:
            return RasterGeo(width, height, src.transform, crs, source="transform")

        gcps, gcp_crs = src.gcps
        if not gcps:
            # Some GDAL builds report a "fake" identity transform even
            # without GCPs; nothing we can do but fail loudly.
            raise ValueError(
                f"{path}: no usable transform and no GCPs found — "
                f"cannot georeference this raster."
            )
        transform = from_gcps(gcps)
        return RasterGeo(width, height, transform, gcp_crs or CRS.from_epsg(4326),
                          source="gcps")


def lonlat_to_pixel(transform, lon: float, lat: float) -> tuple[float, float]:
    """World (lon, lat) -> pixel (col, row) using the raster's affine."""
    col, row = ~transform * (lon, lat)
    return col, row


def pixel_to_lonlat(transform, col: float, row: float) -> tuple[float, float]:
    """Pixel (col, row) -> world (lon, lat) using the raster's affine."""
    lon, lat = transform * (col, row)
    return lon, lat


def load_aoi_polygon(shapefile_geojson_path: str) -> dict:
    """
    The ShapeFile/*.geojson files are NOT label FeatureCollections —
    each is a single bare `{"geometry": {...}}` object describing the
    Area of Interest (the region that was exhaustively annotated).
    Return it as a GeoJSON-style geometry dict.
    """
    with open(shapefile_geojson_path) as f:
        data = json.load(f)
    if "geometry" in data and "type" in data and data["type"] == "Feature":
        return data["geometry"]
    if "geometry" in data:
        return data["geometry"]
    raise ValueError(f"Unrecognized AOI file structure: {shapefile_geojson_path}")


def load_label_features(geojson_path: str) -> list[dict]:
    """Load the turf/grass polygon annotations (a normal FeatureCollection)."""
    with open(geojson_path) as f:
        data = json.load(f)
    if data.get("type") != "FeatureCollection":
        raise ValueError(f"Expected FeatureCollection in {geojson_path}")
    return data["features"]
