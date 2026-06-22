"""Optimizer param groups for UPC.

Stage 1: only head/pooling/proj/prototypes/critic train (backbone frozen).
Stage 2: unfreeze the top-k transformer layers at a small LR (5e-6, validated).
"""

from __future__ import annotations

import torch


def build_optimizer(model, head_lr=1e-4, weight_decay=1e-4):
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("ssl.backbone")]
    return torch.optim.AdamW(head_params, lr=head_lr, weight_decay=weight_decay)


def unfreeze_top_k(model, k: int = 4):
    """Unfreeze the top-k WavLM encoder layers for UPC stage 2."""
    enc = model.ssl.backbone.encoder.layers
    n = len(enc)
    for i in range(n - k, n):
        for p in enc[i].parameters():
            p.requires_grad = True
    model.ssl.frozen = False  # backbone no longer fully frozen
    return [p for p in (model.ssl.backbone.encoder.layers[n-k:]).parameters()]


def build_optimizer_stage2(model, head_lr=5e-5, backbone_lr=5e-6, weight_decay=1e-4):
    head, backbone = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if n.startswith("ssl.backbone") else head).append(p)
    return torch.optim.AdamW(
        [{"params": head, "lr": head_lr},
         {"params": backbone, "lr": backbone_lr}],
        weight_decay=weight_decay)
