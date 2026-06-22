"""CALAS: corpus-adaptive latent-space augmentation on spoof embeddings (P6 + ours).

LSA ops (Huang): additive noise (AN), affine (AT), batch mixup (BM),
linear extrapolation toward/away spoof centroid (LE).
CALAS: per-corpus strength beta_c driven by an EMA of each corpus's spoof
margin -- corpora the model already separates get MORE extrapolative aug.
"""

from __future__ import annotations

import torch


class CalasController:
    """Tracks an EMA spoof-margin per corpus and maps it to a per-corpus beta."""

    def __init__(self, beta_max: float = 1.0, ema: float = 0.95, temp: float = 4.0):
        self.beta_max = beta_max
        self.ema = ema
        self.temp = temp
        self.margin: dict[str, float] = {}

    def update(self, corpus_margins: dict[str, float]) -> None:
        for c, m in corpus_margins.items():
            prev = self.margin.get(c, m)
            self.margin[c] = self.ema * prev + (1 - self.ema) * m

    def beta(self, corpus: str) -> float:
        if not self.margin:
            return 0.5 * self.beta_max
        mbar = sum(self.margin.values()) / len(self.margin)
        m = self.margin.get(corpus, mbar)
        # higher margin (already saturated) => higher beta
        import math
        return self.beta_max / (1.0 + math.exp(-self.temp * (m - mbar)))


def augment_spoof_embeddings(
    z: torch.Tensor,          # [B, D] spoof embeddings only
    beta: torch.Tensor,       # [B] per-sample strength in [0, beta_max]
) -> torch.Tensor:
    if z.shape[0] < 2:
        return z
    b = beta.view(-1, 1)
    out = z.clone()
    # AN: additive Gaussian noise scaled by beta and feature std
    std = z.std(dim=0, keepdim=True).clamp_min(1e-6)
    out = out + b * torch.randn_like(z) * std
    # AT: affine scale around 1
    out = out * (1.0 + 0.1 * b * torch.randn_like(z))
    # BM: batch mixup with a permuted spoof sample
    perm = torch.randperm(z.shape[0], device=z.device)
    lam = (0.5 * b).clamp(0, 0.5)
    out = (1 - lam) * out + lam * z[perm]
    # LE: extrapolate away from spoof centroid (push boundary)
    centroid = z.mean(dim=0, keepdim=True)
    out = out + b * 0.3 * (out - centroid)
    return out
