"""
Standalone tests for src/tiling.py. No test framework required:
    python tests/test_tiling.py
(pytest will also collect and run these if available in your environment.)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tiling import predict_large_image, make_blend_weight, sliding_windows


def test_blend_weight_shape_and_peak():
    w = make_blend_weight(256)
    assert w.shape == (256, 256)
    # center should have higher weight than the edges
    assert w[128, 128] > w[0, 0]
    assert w[128, 128] > w[0, 128]


def test_sliding_windows_covers_full_image():
    h, w = 1000, 800
    windows = list(sliding_windows(h, w, tile_size=256, overlap=64))
    max_row = max(r for r, c in windows)
    max_col = max(c for r, c in windows)
    assert max_row + 256 >= h
    assert max_col + 256 >= w


def test_predict_large_image_shape_and_range():
    h, w = 700, 620
    rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    def fake_predict(tile_batch):
        n, th, tw, c = tile_batch.shape
        gray = tile_batch.mean(axis=-1) / 255.0
        return np.stack([1 - gray, gray], axis=1).astype(np.float32)

    out = predict_large_image(rgb, fake_predict, tile_size=256, overlap=64, num_classes=2)
    assert out.shape == (h, w, 2)
    assert np.all(out >= -1e-4) and np.all(out <= 1 + 1e-4)
    # probabilities should sum ~1 across the class axis everywhere
    assert np.allclose(out.sum(axis=-1), 1.0, atol=1e-3)


def test_no_seam_discontinuity():
    """Regression check: stitched output shouldn't show a hard edge at tile
    stride boundaries when the underlying signal is smooth."""
    h, w = 600, 600
    tile_size, overlap = 256, 64
    # smooth ramp image so any seam would show up as a visible discontinuity
    ramp = np.tile(np.linspace(0, 255, w), (h, 1)).astype(np.uint8)
    rgb = np.stack([ramp, ramp, ramp], axis=-1)

    def fake_predict(tile_batch):
        gray = tile_batch.mean(axis=-1) / 255.0
        return np.stack([1 - gray, gray], axis=1).astype(np.float32)

    out = predict_large_image(rgb, fake_predict, tile_size=tile_size, overlap=overlap, num_classes=2)
    turf = out[..., 1]
    stride = tile_size - overlap
    boundary_col = stride
    left = turf[:, boundary_col - 1]
    right = turf[:, boundary_col + 1]
    assert np.abs(left - right).mean() < 0.05, "visible seam at tile boundary"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print(f"\nAll {len(tests)} tiling tests passed.")
