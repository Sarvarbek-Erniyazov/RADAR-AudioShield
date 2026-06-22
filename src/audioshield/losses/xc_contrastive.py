"""Cross-corpus class-conditional supervised contrastive loss.

Goal (grounded in the probe-decoupling finding): make the bona/spoof
separation corpus-AGNOSTIC without trying to erase corpus identity.
Positives are same-CLASS embeddings; with cross_corpus_only=True they are
restricted to same-class-DIFFERENT-corpus pairs, so the loss explicitly
rewards a class manifold that is stable across corpora while leaving
corpus structure free to remain (testing, not assuming, invariance).

Reusable in both frozen and fine-tuned regimes: operates on the model's
embedding output only; no backbone assumptions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_corpus_supcon(
    embeddings: torch.Tensor,      # [B, D]
    labels: torch.Tensor,          # [B]  (0=bona, 1=spoof)
    corpus_ids: torch.Tensor,      # [B]
    temperature: float = 0.1,
    cross_corpus_only: bool = True,
    min_corpora_per_class: int = 2,
):
    """Returns (loss_scalar, log_dict). Loss is 0 (with grad-safe path) if no
    valid positive pairs exist in the batch (degenerate-batch guard)."""
    device = embeddings.device
    B = embeddings.shape[0]
    z = F.normalize(embeddings, dim=1)
    sim = z @ z.t() / temperature                      # [B, B]

    # numerical stability
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    labels = labels.view(-1, 1)
    corpus_ids = corpus_ids.view(-1, 1)
    same_class = labels.eq(labels.t())                 # [B, B]
    same_corpus = corpus_ids.eq(corpus_ids.t())
    eye = torch.eye(B, dtype=torch.bool, device=device)

    # positives: same class, not self; optionally different corpus
    pos_mask = same_class & ~eye
    if cross_corpus_only:
        pos_mask = pos_mask & ~same_corpus

    # denominator: all non-self pairs
    denom_mask = ~eye

    # guard: need at least one valid positive overall
    n_pos = pos_mask.sum().item()
    log = {"xc_npos": float(n_pos)}
    if n_pos == 0:
        # grad-safe zero (keeps graph; contributes nothing)
        return (embeddings.sum() * 0.0), {**log, "xc_skipped": 1.0}

    exp_sim = torch.exp(sim) * denom_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    # mean log-prob over positives, per anchor that HAS positives
    pos_per_anchor = pos_mask.sum(dim=1)
    has_pos = pos_per_anchor > 0
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)[has_pos] / pos_per_anchor[has_pos]
    loss = -mean_log_prob_pos.mean()
    log["xc_skipped"] = 0.0
    log["xc_anchors"] = float(has_pos.sum().item())
    return loss, log
