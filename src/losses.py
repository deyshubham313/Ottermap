"""
losses.py
---------
Combo loss (Dice + BCE + IoU) restricted to pixels inside the annotated
AOI. This is the single most important correctness detail in this
project: naively computing loss over the full tile would punish the
model for "false positives" on grass that was simply never labelled
(outside the AOI), teaching it to under-predict turf everywhere.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _masked_mean(loss_map: "torch.Tensor", valid: "torch.Tensor", eps: float = 1e-6):
    return (loss_map * valid).sum() / (valid.sum() + eps)


def dice_loss(logits, targets, valid, eps: float = 1e-6):
    probs = torch.softmax(logits, dim=1)[:, 1]  # P(turf)
    targets = targets.float()
    probs = probs * valid
    targets = targets * valid
    intersection = (probs * targets).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def iou_loss(logits, targets, valid, eps: float = 1e-6):
    probs = torch.softmax(logits, dim=1)[:, 1]
    targets = targets.float()
    probs = probs * valid
    targets = targets * valid
    intersection = (probs * targets).sum(dim=(1, 2))
    union = (probs + targets - probs * targets).sum(dim=(1, 2))
    iou = (intersection + eps) / (union + eps)
    return 1 - iou.mean()


def masked_bce_loss(logits, targets, valid):
    ce = F.cross_entropy(logits, targets, reduction="none")  # (B, H, W)
    return _masked_mean(ce, valid)


class ComboLoss(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, w_dice=1.0, w_bce=1.0, w_iou=0.5):
        super().__init__()
        self.w_dice, self.w_bce, self.w_iou = w_dice, w_bce, w_iou

    def forward(self, logits, targets, valid):
        d = dice_loss(logits, targets, valid)
        b = masked_bce_loss(logits, targets, valid)
        i = iou_loss(logits, targets, valid)
        total = self.w_dice * d + self.w_bce * b + self.w_iou * i
        return total, {"dice": d.item(), "bce": b.item(), "iou": i.item()}
