"""
model.py
--------
Thin wrapper around HuggingFace's SegFormer for binary (turf/grass vs.
background) semantic segmentation.

Backbone size is configurable via `--backbone`. The default is
`nvidia/segformer-b2-finetuned-ade-512-512` rather than B5.

Why B2 instead of B5 (documented reasoning, see technical_summary.md):
SegFormer-B5 has ~84M parameters. We have ~200 tiles from 3 source images
after AOI-aware tiling. Fine-tuning an 84M-parameter transformer on that
little data, even with a frozen-encoder warmup, carries real overfitting
risk and a much higher compute/time cost within a 72-hour, single-GPU
budget. SegFormer-B2 (~25M params) keeps the same MiT hierarchical
transformer architecture and pretrained ImageNet/ADE20K features, trains
faster, and is empirically more stable in low-data fine-tuning regimes.
B5 remains a one-flag config change (`--backbone b5`) for anyone with more
labelled data or compute — the architecture code is identical.
"""
from __future__ import annotations

BACKBONES = {
    "b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "b2": "nvidia/segformer-b2-finetuned-ade-512-512",
    "b5": "nvidia/segformer-b5-finetuned-ade-640-640",
}

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import SegformerForSemanticSegmentation
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object  # type: ignore


class TurfSegformer(nn.Module if TORCH_AVAILABLE else object):
    """
    Binary segmentation head over a pretrained SegFormer encoder-decoder.
    We replace the classifier to output 2 logits (background, turf) and
    keep the pretrained MiT encoder + all-MLP decoder weights.
    """

    def __init__(self, backbone: str = "b2", freeze_encoder: bool = True):
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch/transformers not installed in this environment.")
        super().__init__()
        model_name = BACKBONES[backbone]
        self.net = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=2,
            ignore_mismatched_sizes=True,  # replaces the ADE20K 150-class head
        )
        if freeze_encoder:
            self.set_encoder_trainable(False)

    def set_encoder_trainable(self, trainable: bool):
        # In transformers >=5.x SegformerModel no longer exposes .encoder
        # directly. Freeze the whole segformer backbone (everything except
        # the decode_head) which achieves the same decoder-warmup effect.
        backbone = getattr(self.net.segformer, "encoder", None) or self.net.segformer
        for param in backbone.parameters():
            param.requires_grad = trainable

    def forward(self, pixel_values):
        out = self.net(pixel_values=pixel_values)
        logits = out.logits  # (B, 2, H/4, W/4) — SegFormer decodes at 1/4 res
        logits = F.interpolate(logits, size=pixel_values.shape[-2:],
                                mode="bilinear", align_corners=False)
        return logits


def build_model(backbone: str = "b2", freeze_encoder: bool = True) -> "TurfSegformer":
    return TurfSegformer(backbone=backbone, freeze_encoder=freeze_encoder)


def unfreeze_schedule(model: "TurfSegformer", epoch: int, freeze_epochs: int):
    """Call once per epoch from train.py: unfreeze the encoder after warmup."""
    if epoch == freeze_epochs:
        model.set_encoder_trainable(True)
        return True
    return False
