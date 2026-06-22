"""Binary spoof loss: weighted BCE with optional focal term (from old losses.py)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def spoof_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    spoof_pos_weight: float = 1.0,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    if spoof_pos_weight != 1.0:
        w = torch.where(target > 0.5,
                        target.new_tensor(spoof_pos_weight),
                        target.new_tensor(1.0))
        loss = loss * w
    if focal_gamma > 0.0:
        p = torch.sigmoid(logits)
        pt = torch.where(target > 0.5, p, 1.0 - p)
        loss = loss * (1.0 - pt).clamp_min(1e-6).pow(focal_gamma)
    return loss.mean()
