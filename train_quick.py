#!/usr/bin/env python3
"""
train_quick.py
--------------
CPU-friendly quick training that:
1. Downloads SegFormer-B0 pretrained weights (smallest backbone)
2. Trains for 3 epochs on ALL 3 images (final mode, no holdout)
3. Saves models/best_model.pth
4. Uses batch_size=2, num_workers=0 for CPU compatibility

Usage:
    python train_quick.py
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset import TurfTileDataset, get_train_augmentation, get_eval_augmentation
from model import build_model
from losses import ComboLoss
from metrics import compute_metrics, aggregate_metrics
from utils import set_seed, save_checkpoint


def evaluate(model, loader, device):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for images, masks, valid in loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            masks_np = masks.numpy()
            valid_np = valid.numpy()
            for p, m, v in zip(preds, masks_np, valid_np):
                all_metrics.append(compute_metrics(p, m, v))
    return aggregate_metrics(all_metrics)


def main():
    TILES_DIR = "dataset/tiles"
    OUT_DIR = "models"
    BACKBONE = "b0"        # smallest backbone for CPU demo
    EPOCHS = 3
    BATCH_SIZE = 2
    LR = 3e-4
    SEED = 42

    set_seed(SEED)
    device = torch.device("cpu")
    print(f"Training on: {device} | backbone: segformer-{BACKBONE} | epochs: {EPOCHS}")

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_images = [1, 2, 3]
    train_ds = TurfTileDataset(TILES_DIR, all_images, get_train_augmentation(512), 512)
    val_ds   = TurfTileDataset(TILES_DIR, all_images, get_eval_augmentation(512), 512)
    print(f"Dataset: {len(train_ds)} tiles")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\nLoading SegFormer-{BACKBONE} pretrained weights ...")
    model = build_model(BACKBONE, freeze_encoder=True).to(device)
    criterion = ComboLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total\n")

    best_iou = -1.0
    history  = []

    for epoch in range(EPOCHS):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0

        for batch_idx, (images, masks, valid) in enumerate(train_loader):
            images, masks, valid = images.to(device), masks.to(device), valid.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss, _ = criterion(logits, masks, valid)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if batch_idx % 15 == 0:
                print(f"  epoch {epoch+1}/{EPOCHS} | batch {batch_idx+1}/{len(train_loader)} "
                      f"| loss={loss.item():.4f} | {time.time()-t0:.0f}s")

        epoch_loss /= max(len(train_loader), 1)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1} done | avg_loss={epoch_loss:.4f} | {elapsed:.0f}s")

        val_m = evaluate(model, val_loader, device)
        print(f"  IoU={val_m['iou']:.4f}  Dice={val_m['dice']:.4f}  "
              f"P={val_m['precision']:.4f}  R={val_m['recall']:.4f}")

        if val_m["iou"] > best_iou:
            best_iou = val_m["iou"]
            save_checkpoint(model, out_dir / "best_model.pth",
                            meta={"backbone": BACKBONE, "epoch": epoch, **val_m})
            print(f"  [SAVED] best_model.pth  (IoU={best_iou:.4f})\n")

        history.append({"epoch": epoch, "train_loss": epoch_loss,
                        "time_s": elapsed, **{f"val_{k}": v for k, v in val_m.items()}})

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*55}")
    print(f"DONE  |  Best IoU = {best_iou:.4f}")
    print(f"Checkpoint : models/best_model.pth")
    print(f"History    : models/training_history.json")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
