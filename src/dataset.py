"""
dataset.py
----------
PyTorch Dataset over the pre-built tiles (see dataset_prep.py). Each item
returns (image_tensor, mask_tensor, valid_tensor) where `valid` marks
which pixels lie inside the annotated AOI — losses.py uses this to zero
out gradient contribution from unlabeled (outside-AOI) pixels.

Data-scarcity handling: with only 3 source images / ~200 tiles, heavy
geometric + photometric augmentation is doing a lot of the generalization
work here. We deliberately include augmentations that vary illumination,
color balance and ground-sample-distance-like scale jitter, since those
are exactly the axes an unseen aerial image (different sensor, sun angle,
season, altitude) will vary along.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # type: ignore


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_augmentation(tile_size: int = 512):
    return A.Compose([
        A.RandomCrop(tile_size, tile_size, pad_if_needed=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.85, 1.15),
                 rotate=(-15, 15), border_mode=0, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),
            A.CLAHE(clip_limit=2.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=15),
        ], p=0.7),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.GaussNoise(std_range=(0.02, 0.08)),
            A.ImageCompression(quality_range=(60, 95)),
        ], p=0.3),
        A.CoarseDropout(num_holes_range=(1, 4),
                        hole_height_range=(8, 32),
                        hole_width_range=(8, 32), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], additional_targets={"valid": "mask"})


def get_eval_augmentation(tile_size: int = 512):
    return A.Compose([
        A.PadIfNeeded(tile_size, tile_size, border_mode=0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], additional_targets={"valid": "mask"})


class TurfTileDataset(Dataset):
    """
    Reads pre-tiled .npy files produced by dataset_prep.py, filtered by
    `source_images` (used for leave-one-location-out splits).
    """

    def __init__(self, tiles_dir: str, source_images: list[int], augment, tile_size: int = 512):
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch/albumentations not installed in this environment.")
        self.tiles_dir = Path(tiles_dir)
        manifest = json.load(open(self.tiles_dir / "manifest.json"))
        self.items = [m for m in manifest if m["source_image"] in source_images]
        if not self.items:
            raise ValueError(f"No tiles found for source_images={source_images}. "
                              f"Did you run dataset_prep.py?")
        self.augment = augment
        self.tile_size = tile_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        stem = item["stem"]
        rgb = np.load(self.tiles_dir / "images" / f"{stem}.npy")
        mask = np.load(self.tiles_dir / "masks" / f"{stem}.npy")
        valid = np.load(self.tiles_dir / "valid" / f"{stem}.npy")

        out = self.augment(image=rgb, mask=mask, valid=valid)
        image_t = out["image"]
        mask_t = out["mask"].long()
        valid_t = out["valid"].float()
        return image_t, mask_t, valid_t
