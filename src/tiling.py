"""
tiling.py
---------
Sliding-window inference with overlap-blended stitching, so predictions
on large GeoTIFFs don't show tile-seam artifacts and so train/inference
windows match (same tile_size used in dataset_prep.py).

Blending uses a per-pixel triangular (Bartlett) weight that peaks at the
tile center and decays to the edges, then normalizes by the accumulated
weight at each output pixel. This is the standard approach used in
production remote-sensing inference (e.g. solaris, robosat) to avoid the
"grid" artifact you get from naive hard-edge tile stitching.
"""
from __future__ import annotations

import numpy as np


def make_blend_weight(tile_size: int) -> np.ndarray:
    ramp = np.bartlett(tile_size)
    ramp = np.clip(ramp, 1e-3, None)  # avoid exact-zero weight at the very edge
    weight = np.outer(ramp, ramp).astype(np.float32)
    return weight


def sliding_windows(height: int, width: int, tile_size: int, overlap: int):
    stride = tile_size - overlap
    for row in range(0, height, stride):
        for col in range(0, width, stride):
            yield row, col


def predict_large_image(rgb: np.ndarray, predict_fn, tile_size: int = 512,
                         overlap: int = 128, num_classes: int = 2) -> np.ndarray:
    """
    rgb: (H, W, 3) uint8 array, arbitrary size.
    predict_fn: callable(tile_batch: (N, tile_size, tile_size, 3) uint8) ->
                (N, num_classes, tile_size, tile_size) float32 probabilities.
    Returns: (H, W, num_classes) float32 accumulated/normalized probability map.
    """
    h, w = rgb.shape[:2]
    pad = tile_size
    rgb_p = np.pad(rgb, ((0, pad), (0, pad), (0, 0)), mode="reflect")

    accum = np.zeros((h + pad, w + pad, num_classes), dtype=np.float32)
    weight_sum = np.zeros((h + pad, w + pad), dtype=np.float32)
    blend = make_blend_weight(tile_size)

    windows = list(sliding_windows(h, w, tile_size, overlap))
    for row, col in windows:
        tile = rgb_p[row:row + tile_size, col:col + tile_size]
        probs = predict_fn(tile[None, ...])[0]  # (num_classes, tile_size, tile_size)
        probs = np.transpose(probs, (1, 2, 0))   # (tile_size, tile_size, num_classes)
        accum[row:row + tile_size, col:col + tile_size] += probs * blend[..., None]
        weight_sum[row:row + tile_size, col:col + tile_size] += blend

    weight_sum = np.clip(weight_sum, 1e-6, None)
    full_probs = accum / weight_sum[..., None]
    return full_probs[:h, :w]


def test_time_augment_predict(tile_batch: np.ndarray, base_predict_fn) -> np.ndarray:
    """
    Averages predictions over {identity, h-flip, v-flip, 180-rotate}. Keeps
    TTA cheap (4x) while covering the symmetric transforms that are valid
    for top-down aerial imagery (no canonical "up" direction for grass).
    """
    preds = []
    variants = {
        "identity": lambda x: x,
        "hflip": lambda x: x[:, :, ::-1, :],
        "vflip": lambda x: x[:, ::-1, :, :],
        "rot180": lambda x: x[:, ::-1, ::-1, :],
    }
    inverse = {
        "identity": lambda x: x,
        "hflip": lambda x: x[:, :, ::-1],
        "vflip": lambda x: x[:, ::-1, :],
        "rot180": lambda x: x[:, ::-1, ::-1],
    }
    for name, fwd in variants.items():
        aug_tile = np.ascontiguousarray(fwd(tile_batch))
        probs = base_predict_fn(aug_tile)  # (N, C, H, W)
        probs = inverse[name](probs.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
        preds.append(np.ascontiguousarray(probs))
    return np.mean(preds, axis=0)
