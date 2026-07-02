"""
Standalone tests for the parts of src/polygonize.py that don't require
rasterio/geopandas (clean_mask, meters_per_degree). The full
mask_to_geodataframe/export path is exercised against real project data
in the environment that has the geo stack installed — see
docs/technical_summary.md for the validation performed against the
provided imagery (contour count / area sanity checks).

    python tests/test_polygonize.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from polygonize import clean_mask, meters_per_degree


def test_clean_mask_removes_isolated_speckle():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 1  # one solid blob
    noisy = mask.copy()
    rng = np.random.default_rng(0)
    # scatter single-pixel speckle both inside and outside the blob
    idx = rng.choice(mask.size, size=150, replace=False)
    noisy.flat[idx] = 1 - noisy.flat[idx]

    cleaned = clean_mask(noisy, open_kernel=3, close_kernel=3)
    diff_before = (noisy != mask).sum()
    diff_after = (cleaned != mask).sum()
    assert diff_after < diff_before, "cleaning should reduce disagreement with ground truth"
    assert diff_after < diff_before * 0.6, "cleaning should meaningfully reduce speckle"


def test_clean_mask_preserves_large_regions():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 10:90] = 1
    cleaned = clean_mask(mask, open_kernel=3, close_kernel=3)
    # a large solid region should survive open+close almost entirely intact
    assert cleaned.sum() > 0.9 * mask.sum()


def test_meters_per_degree_reasonable_at_equator():
    m_lon, m_lat = meters_per_degree(0.0)
    # ~111km per degree latitude everywhere, and at the equator lon should match
    assert 110_000 < m_lat < 112_000
    assert abs(m_lon - m_lat) < 100  # equator: lon spacing ~= lat spacing


def test_meters_per_degree_shrinks_toward_poles():
    m_lon_eq, _ = meters_per_degree(0.0)
    m_lon_47, _ = meters_per_degree(47.28)   # Seattle-area image latitude
    m_lon_60, _ = meters_per_degree(60.0)
    assert m_lon_47 < m_lon_eq
    assert m_lon_60 < m_lon_47


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print(f"\nAll {len(tests)} polygonize tests passed.")
