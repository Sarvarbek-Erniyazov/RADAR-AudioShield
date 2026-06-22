"""Attentive statistics pooling (P3 / ACS).

Pools a [B, T, H] sequence into [B, 2H] by attention-weighted mean and std,
which captures *where* in time artifacts live -- unlike the old time-mean.
"""

from __future__ import annotations

import torch
from torch import nn


class AttentiveStatsPooling(nn.Module):
    def __init__(self, hidden: int, attn_hidden: int = 128) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        w = torch.softmax(self.attn(x).squeeze(-1), dim=1).unsqueeze(-1)  # [B, T, 1]
        mean = (w * x).sum(dim=1)                                          # [B, H]
        var = (w * (x - mean.unsqueeze(1)) ** 2).sum(dim=1)
        std = var.clamp_min(1e-8).sqrt()
        return torch.cat([mean, std], dim=1)                              # [B, 2H]
