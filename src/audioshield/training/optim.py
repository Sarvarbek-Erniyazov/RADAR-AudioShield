"""Optimizer param groups for UPC.

Stage 1: only head/pooling/proj/prototypes/critic train (backbone frozen).
Stage 2: selectively unfreeze SSL encoder layers at a small LR.
"""

from __future__ import annotations

import torch


def build_optimizer(model, head_lr=1e-4, weight_decay=1e-4):
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("ssl.backbone")]
    return torch.optim.AdamW(head_params, lr=head_lr, weight_decay=weight_decay)


def _encoder_layers(model):
    try:
        return model.ssl.backbone.encoder.layers
    except AttributeError as exc:
        raise AttributeError("Could not locate model.ssl.backbone.encoder.layers") from exc


def _freeze_backbone(model) -> None:
    for p in model.ssl.backbone.parameters():
        p.requires_grad = False


def _set_encoder_layers_trainable(model, indices: list[int]) -> list[int]:
    enc = _encoder_layers(model)
    n = len(enc)
    chosen = sorted({i for i in indices if 0 <= i < n})
    if not chosen:
        raise ValueError(f"No valid encoder layers selected from {indices}; encoder has {n} layers")
    _freeze_backbone(model)
    for i in chosen:
        for p in enc[i].parameters():
            p.requires_grad = True
    model.ssl.frozen = False
    return chosen


def unfreeze_top_k(model, k: int = 4):
    """Unfreeze the top-k WavLM encoder layers for UPC stage 2."""
    enc = _encoder_layers(model)
    n = len(enc)
    if k <= 0:
        return []
    chosen = _set_encoder_layers_trainable(model, list(range(max(0, n - k), n)))
    return [p for i in chosen for p in enc[i].parameters()]


def _expanded_hidden_state_window(
    band: tuple[int, int],
    count: int,
    max_hidden_state: int,
    center: int | None = None,
) -> tuple[int, int]:
    """Choose a contiguous hidden-state window.

    Hidden state 0 is the feature-extractor output; hidden state i maps to
    encoder layer i - 1. The configured layer-weight band is therefore the
    natural default for partial fine-tuning.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    min_h, max_h = 1, max_hidden_state
    lo, hi = sorted((int(band[0]), int(band[1])))
    lo, hi = max(min_h, lo), min(max_h, hi)
    if lo > hi:
        raise ValueError(f"Invalid hidden-state band {band} for max hidden state {max_hidden_state}")

    base_count = hi - lo + 1
    count = min(int(count), max_h - min_h + 1)
    if count == base_count:
        return lo, hi
    if count > base_count:
        extra = count - base_count
        left = extra // 2
        right = extra - left
        lo -= left
        hi += right
        if lo < min_h:
            hi += min_h - lo
            lo = min_h
        if hi > max_h:
            lo -= hi - max_h
            hi = max_h
        return max(min_h, lo), min(max_h, hi)

    if center is None:
        center = (lo + hi) // 2
    center = min(max(int(center), min_h), max_h)
    half_left = (count - 1) // 2
    new_lo = center - half_left
    new_hi = new_lo + count - 1
    if new_lo < min_h:
        new_hi += min_h - new_lo
        new_lo = min_h
    if new_hi > max_h:
        new_lo -= new_hi - max_h
        new_hi = max_h
    return max(min_h, new_lo), min(max_h, new_hi)


def unfreeze_weighted_band(model, hidden_state_band, k: int | None = None, center: int | None = None):
    """Unfreeze encoder layers aligned with the SSL layer-weight hidden states.

    If the SSL readout is initialized on hidden states [8, 11], this unfreezes
    encoder layers [7, 10], because encoder layer j produces hidden state j+1.
    """
    enc = _encoder_layers(model)
    n = len(enc)
    band = (int(hidden_state_band[0]), int(hidden_state_band[1]))
    count = int(k) if k is not None and int(k) > 0 else abs(band[1] - band[0]) + 1
    hs_lo, hs_hi = _expanded_hidden_state_window(band, count, max_hidden_state=n, center=center)
    encoder_indices = [h - 1 for h in range(hs_lo, hs_hi + 1)]
    chosen = _set_encoder_layers_trainable(model, encoder_indices)
    return {
        "hidden_state_window": (hs_lo, hs_hi),
        "encoder_indices": chosen,
        "params": [p for i in chosen for p in enc[i].parameters()],
    }


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
