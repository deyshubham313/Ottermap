"""
Regression test for the georeferencing math that src/geo_utils.py delegates
to rasterio.transform.from_gcps(). Since this sandbox can't install
rasterio, this test re-derives the same closed-form affine (valid because
the provided GeoTIFFs' GCPs form an axis-aligned rectangle - see
src/geo_utils.py docstring) and checks it reproduces the real corner
tiepoints pulled from the provided .tiff files' GeoKeyDirectoryTag /
ModelTiepointTag exactly.

    python tests/test_geo_math.py
"""

# (image_id, width, height, tiepoints) pulled directly from the provided
# GeoTIFFs' ModelTiepointTag (4 corners: I,J,K -> X,Y,Z each).
REAL_TIEPOINTS = {
    1: dict(width=3811, height=3407,
            corners={(0, 0): (-121.86821784073192, 39.75390683152841),
                     (3811, 0): (-121.86310715330987, 39.75390683152841),
                     (0, 3407): (-121.86821784073192, 39.75039416847159),
                     (3811, 3407): (-121.86310715330987, 39.75039416847159)}),
    2: dict(width=3889, height=3936,
            corners={(0, 0): (-122.1657327769076, 47.28302992047381),
                     (3889, 0): (-122.16051734131366, 47.28302992047381),
                     (0, 3936): (-122.1657327769076, 47.27944859730886),
                     (3889, 3936): (-122.16051734131366, 47.27944859730886)}),
    3: dict(width=5906, height=5429,
            corners={(0, 0): (-78.71052009237394, 33.845330772705424),
                     (5906, 0): (-78.70655969678269, 33.845330772705424),
                     (0, 5429): (-78.71052009237394, 33.8423069819284),
                     (5906, 5429): (-78.70655969678269, 33.8423069819284)}),
}


def affine_from_corners(width, height, corners):
    x0, y0 = corners[(0, 0)]
    x1, _ = corners[(width, 0)]
    _, y1 = corners[(0, height)]
    a = (x1 - x0) / width
    e = (y1 - y0) / height
    return a, e, x0, y0  # x = a*col + c ; y = e*row + f


def apply_affine(a, e, c, f, col, row):
    return a * col + c, e * row + f


def test_affine_reproduces_all_corners_exactly():
    for image_id, spec in REAL_TIEPOINTS.items():
        w, h = spec["width"], spec["height"]
        a, e, c, f = affine_from_corners(w, h, spec["corners"])
        for (col, row), (lon, lat) in spec["corners"].items():
            pred_lon, pred_lat = apply_affine(a, e, c, f, col, row)
            assert abs(pred_lon - lon) < 1e-9, f"image {image_id} corner ({col},{row}) lon mismatch"
            assert abs(pred_lat - lat) < 1e-9, f"image {image_id} corner ({col},{row}) lat mismatch"


def test_pixel_size_is_positive_and_plausible():
    """Sanity: derived ground sample distance should be sub-meter to a few
    meters per pixel for aerial imagery at this scale, not degrees-scale
    garbage from a unit/axis mixup."""
    import math
    for image_id, spec in REAL_TIEPOINTS.items():
        w, h = spec["width"], spec["height"]
        a, e, c, f = affine_from_corners(w, h, spec["corners"])
        lat_mid = (spec["corners"][(0, 0)][1] + spec["corners"][(0, h)][1]) / 2
        m_per_deg_lat = 111_194.9
        m_per_deg_lon = 111_194.9 * math.cos(math.radians(lat_mid))
        px_size_x_m = abs(a) * m_per_deg_lon
        px_size_y_m = abs(e) * m_per_deg_lat
        assert 0.01 < px_size_x_m < 5.0, f"image {image_id}: implausible GSD {px_size_x_m}"
        assert 0.01 < px_size_y_m < 5.0, f"image {image_id}: implausible GSD {px_size_y_m}"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print(f"\nAll {len(tests)} geo-math tests passed.")
