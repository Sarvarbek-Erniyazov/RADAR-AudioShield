"""Cosine prototype head with additive angular margin (P3 + P6).

1 shared bona-fide prototype + K spoof prototypes in a unit-normalized space.
Returns cosine similarities to all prototypes; the loss module applies AAM and
the intra/inter spoof-prototype regularizers (Huang eqs. 2-4).
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CosinePrototypes(nn.Module):
    def __init__(self, dim: int, num_bona: int = 1, num_spoof: int = 8) -> None:
        super().__init__()
        self.num_bona = num_bona
        self.num_spoof = num_spoof
        self.bona = nn.Parameter(torch.randn(num_bona, dim))
        self.spoof = nn.Parameter(torch.randn(num_spoof, dim))
        nn.init.normal_(self.bona, std=0.02)
        nn.init.normal_(self.spoof, std=0.02)

    def all_prototypes(self) -> torch.Tensor:
        return F.normalize(torch.cat([self.bona, self.spoof], dim=0), dim=1)

    def forward(self, z: torch.Tensor) -> dict:
        zer = F.normalize(z, dim=1)
        protos = self.all_prototypes()                # [1+K, D]
        cos = zer @ protos.t()                        # [B, 1+K]
        bona_cos = cos[:, : self.num_bona]            # [B, num_bona]
        spoof_cos = cos[:, self.num_bona :]           # [B, K]
        # spoof score proxy: closeness to nearest spoof vs bona prototype
        spoof_score = spoof_cos.max(dim=1).values - bona_cos.max(dim=1).values
        return {
            "cos": cos,
            "bona_cos": bona_cos,
            "spoof_cos": spoof_cos,
            "spoof_score": spoof_score,               # higher => more spoof-like
        }
