"""
polygonize.py
-------------
Converts a binary prediction mask (in pixel space, at native raster
resolution) into GIS-ready vector output:

  binary mask
    -> morphological cleaning (remove speckle, close small gaps)
    -> connected-component small-object removal (area threshold in m^2,
       not pixels, since GSD differs per image)
    -> rasterio.features.shapes() -> shapely polygons (in the raster's CRS)
    -> polygon simplification (Douglas-Peucker, tolerance in CRS units)
    -> geopandas GeoDataFrame with area_sqm / area_acres columns
    -> GeoJSON + Shapefile export

Note: the source rasters are in EPSG:4326 (degrees), so "meters per pixel"
is computed locally via an equirectangular approximation at the image's
center latitude, purely for the min-area filter and reported areas. The
exported vector geometry itself stays in EPSG:4326 (matches the source
labels) unless --reproject is passed.
"""
from __future__ import annotations

import argparse
import math

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import rasterio
    from rasterio.features import shapes as rio_shapes
    import geopandas as gpd
    from shapely.geometry import shape as shapely_shape
    GEO_AVAILABLE = True
except ImportError:
    GEO_AVAILABLE = False

EARTH_RADIUS_M = 6371000.0


def meters_per_degree(lat_deg: float) -> tuple[float, float]:
    """Approximate (meters per degree lon, meters per degree lat) at a latitude."""
    lat_rad = math.radians(lat_deg)
    m_per_deg_lat = (math.pi / 180) * EARTH_RADIUS_M
    m_per_deg_lon = (math.pi / 180) * EARTH_RADIUS_M * math.cos(lat_rad)
    return m_per_deg_lon, m_per_deg_lat


def clean_mask(mask: np.ndarray, open_kernel: int = 3, close_kernel: int = 5) -> np.ndarray:
    """Morphological opening (remove speckle) then closing (fill small gaps)."""
    if cv2 is None:
        raise RuntimeError("opencv-python is required for clean_mask().")
    mask = mask.astype(np.uint8)
    if open_kernel > 0:
        k = np.ones((open_kernel, open_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_kernel > 0:
        k = np.ones((close_kernel, close_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def mask_to_geodataframe(mask: np.ndarray, transform, crs, center_lat: float,
                          min_area_sqm: float = 5.0, simplify_tolerance_deg: float = 1e-6):
    """
    mask: cleaned binary (H, W) uint8 array, value 1 = turf.
    transform, crs: from geo_utils.open_raster_geo().
    Returns a geopandas.GeoDataFrame with columns [geometry, area_sqm, area_acres].
    """
    if not GEO_AVAILABLE:
        raise RuntimeError("rasterio/geopandas/shapely required for mask_to_geodataframe().")

    m_per_deg_lon, m_per_deg_lat = meters_per_degree(center_lat)
    deg_area_to_sqm = m_per_deg_lon * m_per_deg_lat

    records = []
    for geom, value in rio_shapes(mask, mask=mask.astype(bool), transform=transform):
        if value != 1:
            continue
        poly = shapely_shape(geom)
        if simplify_tolerance_deg > 0:
            poly = poly.simplify(simplify_tolerance_deg, preserve_topology=True)
        area_sqm = poly.area * deg_area_to_sqm
        if area_sqm < min_area_sqm:
            continue
        records.append({"geometry": poly, "area_sqm": area_sqm,
                         "area_acres": area_sqm / 4046.8564224})

    gdf = gpd.GeoDataFrame(records, crs=crs)
    return gdf


def export_gdf(gdf, out_geojson: str = None, out_shapefile: str = None):
    if out_geojson:
        gdf.to_file(out_geojson, driver="GeoJSON")
    if out_shapefile:
        gdf.to_file(out_shapefile, driver="ESRI Shapefile")


def summarize(gdf, full_image_area_sqm: float | None = None) -> dict:
    n = len(gdf)
    total_area_sqm = float(gdf["area_sqm"].sum()) if n else 0.0
    summary = {
        "num_polygons": n,
        "total_turf_area_sqm": total_area_sqm,
        "total_turf_area_acres": total_area_sqm / 4046.8564224,
    }
    if full_image_area_sqm:
        summary["turf_area_fraction"] = total_area_sqm / full_image_area_sqm
    return summary
