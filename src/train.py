"""
train.py
--------
Training entrypoint. Given only 3 labelled source images, the default
mode is Leave-One-Location-Out (LOLO): train 3 separate runs, each
holding out one full source image as validation, and report averaged
metrics. This is a far more honest estimate of "does this generalize to
a new place" than a random tile split, since random splitting would leak
near-duplicate context (same rooftops, same lawns, same lighting) between
train and val.

The FINAL model shipped in models/best_model.pth is then re-trained on
ALL 3 images (no holdout) using the hyperparameters selected during LOLO,
since at inference time we want to use every labelled pixel we have.

Usage:
    python src/train.py --mode lolo                 # 3 runs, report avg metrics
    python src/train.py --mode final                 # train on all 3 images
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from dataset import TurfTileDataset, get_train_augmentation, get_eval_augmentation
from model import build_model, unfreeze_schedule
from losses import ComboLoss
from metrics import compute_metrics, aggregate_metrics
from utils import set_seed, get_device, save_checkpoint


def run_one_split(train_images, val_images, args, run_name: str):
    device = get_device()
    train_ds = TurfTileDataset(args.tiles_dir, train_images,
                                get_train_augmentation(args.tile_size), args.tile_size)
    val_ds = TurfTileDataset(args.tiles_dir, val_images,
                              get_eval_augmentation(args.tile_size), args.tile_size) if val_images else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers) if val_ds else None

    model = build_model(args.backbone, freeze_encoder=True).to(device)
    criterion = ComboLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_iou = -1.0
    history = []

    for epoch in range(args.epochs):
        if unfreeze_schedule(model, epoch, args.freeze_epochs):
            print(f"[{run_name}] epoch {epoch}: unfreezing encoder, dropping LR x{args.unfreeze_lr_mult}")
            for g in optimizer.param_groups:
                g["lr"] *= args.unfreeze_lr_mult
            backbone = getattr(model.net.segformer, "encoder", None) or model.net.segformer
            optimizer.add_param_group({"params": backbone.parameters(),
                                        "lr": args.lr * args.unfreeze_lr_mult})

        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        for images, masks, valid in train_loader:
            images, masks, valid = images.to(device), masks.to(device), valid.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(images)
                loss, parts = criterion(logits, masks, valid)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        epoch_loss /= max(len(train_loader), 1)

        log = {"epoch": epoch, "train_loss": epoch_loss, "time_s": time.time() - t0}

        if val_loader:
            val_metrics = evaluate(model, val_loader, device)
            log.update({f"val_{k}": v for k, v in val_metrics.items()})
            if val_metrics["iou"] > best_iou:
                best_iou = val_metrics["iou"]
                save_checkpoint(model, args.out_dir / f"{run_name}_best.pth",
                                 meta={"backbone": args.backbone, "epoch": epoch, **val_metrics})
        else:
            # final run (no holdout) — just checkpoint every epoch's last state
            save_checkpoint(model, args.out_dir / f"{run_name}_last.pth",
                             meta={"backbone": args.backbone, "epoch": epoch})

        print(f"[{run_name}] epoch {epoch}: loss={epoch_loss:.4f} "
              + (f"val_iou={log.get('val_iou', float('nan')):.4f}" if val_loader else "(no holdout)"))
        history.append(log)

    return {"run_name": run_name, "best_iou": best_iou, "history": history}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_metrics = []
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-dir", default="dataset/tiles")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--mode", choices=["lolo", "final"], default="lolo")
    ap.add_argument("--backbone", choices=["b0", "b2", "b5"], default="b2")
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--freeze-epochs", type=int, default=10,
                     help="epochs to keep the encoder frozen before fine-tuning it")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--unfreeze-lr-mult", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not TORCH_AVAILABLE:
        raise SystemExit("torch not installed. Run in a GPU environment (see README) — "
                          "e.g. Google Colab with `pip install -r requirements.txt`.")

    set_seed(args.seed)
    all_images = [1, 2, 3]
    results = []

    if args.mode == "lolo":
        for held_out in all_images:
            train_images = [i for i in all_images if i != held_out]
            r = run_one_split(train_images, [held_out], args, run_name=f"lolo_holdout{held_out}")
            results.append(r)
        ious = [r["best_iou"] for r in results]
        print("\n=== Leave-One-Location-Out summary ===")
        for r in results:
            print(f"  {r['run_name']}: best val IoU = {r['best_iou']:.4f}")
        print(f"  mean IoU = {np.mean(ious):.4f}  (std {np.std(ious):.4f})")
        with open(args.out_dir / "lolo_results.json", "w") as f:
            json.dump(results, f, indent=2)

    else:  # final
        r = run_one_split(all_images, [], args, run_name="final")
        # rename last checkpoint of final epoch to the canonical name the
        # inference script expects
        last_ckpt = args.out_dir / "final_last.pth"
        best_ckpt = args.out_dir / "best_model.pth"
        if last_ckpt.exists():
            last_ckpt.replace(best_ckpt)
        print(f"Final model saved to {best_ckpt}")


if __name__ == "__main__":
    main()
