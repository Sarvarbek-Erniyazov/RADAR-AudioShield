"""Prototype losses (P3 + P6 / Huang eqs. 2-4).

- AAM classification: bona-fide samples pulled to the bona prototype, spoof
  samples to their nearest spoof prototype, with an additive angular margin.
- intra: spread spoof prototypes apart (prevent collapse).
- inter: separate spoof prototypes from the bona prototype.
Operates on cosine outputs from CosinePrototypes.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def aam_prototype_loss(
    bona_cos: torch.Tensor,    # [B, num_bona]
    spoof_cos: torch.Tensor,   # [B, K]
    target: torch.Tensor,      # [B] 1=spoof 0=bona
    margin: float = 0.2,
    scale: float = 20.0,
) -> torch.Tensor:
    B = target.shape[0]
    bona_best = bona_cos.max(dim=1).values            # [B]
    spoof_best = spoof_cos.max(dim=1).values          # [B]
    is_spoof = target > 0.5

    # positive cosine = similarity to the correct class's nearest prototype
    pos = torch.where(is_spoof, spoof_best, bona_best)
    neg = torch.where(is_spoof, bona_best, spoof_best)

    # additive angular margin on the positive
    pos_m = torch.cos(torch.acos(pos.clamp(-1 + 1e-6, 1 - 1e-6)) + margin)
    logits = scale * torch.stack([pos_m, neg], dim=1)  # [B, 2], index 0 = correct
    labels = torch.zeros(B, dtype=torch.long, device=target.device)
    return F.cross_entropy(logits, labels)


def spoof_prototype_regularizers(spoof_protos: torch.Tensor, bona_protos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """spoof_protos [K, D], bona_protos [num_bona, D] (already normalized upstream)."""
    sp = F.normalize(spoof_protos, dim=1)
    bp = F.normalize(bona_protos, dim=1)
    K = sp.shape[0]
    # intra: mean pairwise cosine among spoof prototypes (minimize => spread)
    sim = sp @ sp.t()
    off = sim - torch.eye(K, device=sp.device)
    intra = off.sum() / max(1, K * (K - 1))
    # inter: spoof prototypes should be far from the bona prototype(s)
    inter = (sp @ bp.t()).mean()
    return intra, inter
