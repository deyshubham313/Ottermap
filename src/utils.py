"""utils.py - small shared helpers used across the pipeline."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.device("cuda")
    if TORCH_AVAILABLE and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu") if TORCH_AVAILABLE else "cpu"


def save_checkpoint(model, path: Path, meta: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "meta": meta or {}}
    torch.save(payload, path)


def load_checkpoint(model, path: Path, device=None):
    payload = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(payload["state_dict"])
    return model, payload.get("meta", {})
