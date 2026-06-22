"""BMI: Bona-fide Manifold Invariance (primary novelty).

Three terms, computed on BONA-FIDE samples only:
  L_cent : pull bona-fide embeddings to the shared bona prototype (P3).
  L_grl  : domain critic CE (detector defeats it via gradient reversal).
  L_kwok : match bona-fide SCORE distributions across domains. Two forms:
           score-MMD (main) and sorted-score Wasserstein-1 (ablation).

The Kwok term is the differentiable form of "bona-fide cross-testing EER should
be near chance" -- prior work (Kwok) used this only as an evaluation metric.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def centroid_loss(bona_cos: torch.Tensor) -> torch.Tensor:
    # bona_cos [Nb, num_bona]; pull toward best bona prototype
    return (1.0 - bona_cos.max(dim=1).values).mean()


def grl_domain_loss(domain_logits: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
    # domain_logits already passed through GRL upstream; CE trains the critic,
    # GRL flips the sign for the embedding so it becomes domain-uninformative.
    return F.cross_entropy(domain_logits.float(), domain_id)


def _rbf_mmd2(a: torch.Tensor, b: torch.Tensor, sigmas=(0.01, 0.05, 0.1, 0.5)) -> torch.Tensor:
    a = a.view(-1, 1); b = b.view(-1, 1)
    def k(x, y):
        d2 = (x - y.t()) ** 2
        out = 0.0
        for s in sigmas:
            out = out + torch.exp(-d2 / (2 * s * s))
        return out / len(sigmas)
    return k(a, a).mean() + k(b, b).mean() - 2 * k(a, b).mean()


def kwok_score_mmd(scores: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
    """Pairwise score-MMD across bona-fide domains present in the batch."""
    doms = torch.unique(domain_id)
    if doms.numel() < 2:
        return scores.new_tensor(0.0)
    groups = [scores[domain_id == d] for d in doms]
    groups = [g for g in groups if g.numel() >= 2]
    if len(groups) < 2:
        return scores.new_tensor(0.0)
    total = scores.new_tensor(0.0); n = 0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            total = total + _rbf_mmd2(groups[i], groups[j]); n += 1
    return total / max(1, n)


def _w1_sorted(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = min(a.numel(), b.numel())
    if n < 2:
        return a.new_tensor(0.0)
    qs = torch.linspace(0, 1, n, device=a.device)
    aq = torch.quantile(a, qs)
    bq = torch.quantile(b, qs)
    return (aq - bq).abs().mean()


def kwok_wasserstein(scores: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
    """Pairwise sorted-score Wasserstein-1 across bona-fide domains."""
    doms = torch.unique(domain_id)
    if doms.numel() < 2:
        return scores.new_tensor(0.0)
    groups = [scores[domain_id == d] for d in doms]
    groups = [g for g in groups if g.numel() >= 2]
    if len(groups) < 2:
        return scores.new_tensor(0.0)
    total = scores.new_tensor(0.0); n = 0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            total = total + _w1_sorted(groups[i], groups[j]); n += 1
    return total / max(1, n)


def bmi_loss(
    bona_cos: torch.Tensor,
    domain_logits: torch.Tensor,
    bona_scores: torch.Tensor,
    domain_id: torch.Tensor,
    kwok_kind: str = "mmd",
    w_cent: float = 0.3,
    w_grl: float = 0.3,
    w_kwok: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    l_cent = centroid_loss(bona_cos)
    l_grl = grl_domain_loss(domain_logits, domain_id)
    if kwok_kind == "mmd":
        l_kwok = kwok_score_mmd(bona_scores, domain_id)
    elif kwok_kind == "wasserstein":
        l_kwok = kwok_wasserstein(bona_scores, domain_id)
    else:
        raise ValueError(f"unknown kwok_kind {kwok_kind}")
    total = w_cent * l_cent + w_grl * l_grl + w_kwok * l_kwok
    return total, {
        "bmi_cent": float(l_cent.detach()),
        "bmi_grl": float(l_grl.detach()),
        "bmi_kwok": float(l_kwok.detach()),
    }
