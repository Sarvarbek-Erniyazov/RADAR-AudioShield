"""Clean<->degraded channel-consistency loss (e002).

Clean view is a DETACHED teacher; the degraded view is pulled toward it. This
prevents both views collapsing to satisfy consistency, and matches the design.
lambda_kl / lambda_emb make the augmentation-only arm a zero-weight special case.
Backward-compatible with the e001 call (4 positional args, teacher defaults on).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _bkl(a, b, eps=1e-6):
    a = a.clamp(eps, 1 - eps)
    b = b.clamp(eps, 1 - eps)
    return (a * (a / b).log() + (1 - a) * ((1 - a) / (1 - b)).log()).mean()


def consistency_loss(
    logit_clean: torch.Tensor,
    logit_aug: torch.Tensor,
    emb_clean: torch.Tensor,
    emb_aug: torch.Tensor,
    lambda_kl: float = 1.0,
    lambda_emb: float = 0.5,
    teacher: bool = True,
) -> torch.Tensor:
    if teacher:
        logit_t = logit_clean.detach()
        emb_t = emb_clean.detach()
        p_t = torch.sigmoid(logit_t)
        p_s = torch.sigmoid(logit_aug)
        kl = _bkl(p_s, p_t)
        cos = 1.0 - F.cosine_similarity(emb_aug, emb_t, dim=1).mean()
    else:
        p_clean = torch.sigmoid(logit_clean)
        p_aug = torch.sigmoid(logit_aug)
        kl = 0.5 * (_bkl(p_clean, p_aug) + _bkl(p_aug, p_clean))
        cos = 1.0 - F.cosine_similarity(emb_clean, emb_aug, dim=1).mean()
    return lambda_kl * kl + lambda_emb * cos
