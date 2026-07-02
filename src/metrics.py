"""
metrics.py
----------
AOI-masked evaluation metrics. All metrics are computed only over pixels
inside the AOI (see dataset_prep.py docstring for why).
"""
from __future__ import annotations

import numpy as np


def confusion_counts(pred: np.ndarray, target: np.ndarray, valid: np.ndarray):
    pred = pred.astype(bool) & valid.astype(bool)
    target = target.astype(bool) & valid.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()
    tn = np.logical_and(~pred, ~target).sum()
    return int(tp), int(fp), int(fn), int(tn)


def compute_metrics(pred: np.ndarray, target: np.ndarray, valid: np.ndarray, eps=1e-6) -> dict:
    tp, fp, fn, tn = confusion_counts(pred, target, valid)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    pixel_acc = (tp + tn) / (tp + fp + fn + tn + eps)
    return {
        "iou": float(iou), "dice": float(dice), "precision": float(precision),
        "recall": float(recall), "f1": float(f1), "pixel_accuracy": float(pixel_acc),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def aggregate_metrics(metric_dicts: list[dict]) -> dict:
    """Micro-average across tiles/images by summing confusion counts first."""
    tp = sum(m["tp"] for m in metric_dicts)
    fp = sum(m["fp"] for m in metric_dicts)
    fn = sum(m["fn"] for m in metric_dicts)
    tn = sum(m["tn"] for m in metric_dicts)
    eps = 1e-6
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    pixel_acc = (tp + tn) / (tp + fp + fn + tn + eps)
    return {
        "iou": float(iou), "dice": float(dice), "precision": float(precision),
        "recall": float(recall), "f1": float(f1), "pixel_accuracy": float(pixel_acc),
    }
