"""Multi-objective losses for BioPhys-HyperRADAR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .hyperbolic import prototype_ce_loss


@dataclass
class LossWeights:
    spoof: float = 1.0
    method: float = 0.2
    media_state: float = 0.2
    target_prototype: float = 0.3
    method_prototype: float = 0.2
    bona_fide_compactness: float = 0.1
    energy: float = 0.02
    consistency: float = 0.1


class MultiObjectiveLoss(nn.Module):
    def __init__(
        self,
        weights: Optional[LossWeights] = None,
        energy_margin: float = -4.0,
        focal_gamma: float = 0.0,
        spoof_pos_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.energy_margin = energy_margin
        self.focal_gamma = focal_gamma
        self.spoof_pos_weight = spoof_pos_weight

    def forward(
        self,
        outputs_a: dict[str, torch.Tensor],
        batch_a: dict[str, torch.Tensor],
        outputs_b: Optional[dict[str, torch.Tensor]] = None,
        batch_b: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        target = batch_a["target"].float()
        target_long = batch_a["target_long"].long()
        method = batch_a["method_id"].long()
        media = batch_a["media_state"].long()

        terms: dict[str, torch.Tensor] = {}
        terms["spoof"] = self.spoof_loss(outputs_a["spoof_logit"], target)
        terms["method"] = F.cross_entropy(outputs_a["method_logits"], method)
        terms["media_state"] = F.cross_entropy(outputs_a["media_logits"], media)
        terms["target_prototype"] = prototype_ce_loss(outputs_a["target_distances"], target_long)
        terms["method_prototype"] = prototype_ce_loss(outputs_a["method_distances"], method)

        real_mask = target_long == 0
        if real_mask.any():
            terms["bona_fide_compactness"] = outputs_a["target_distances"][real_mask, 0].mean()
        else:
            terms["bona_fide_compactness"] = outputs_a["embedding"].new_tensor(0.0)

        method_energy = -torch.logsumexp(outputs_a["method_logits"], dim=1)
        terms["energy"] = F.softplus(method_energy - self.energy_margin).mean()

        if outputs_b is not None and batch_b is not None:
            prob_a = torch.sigmoid(outputs_a["spoof_logit"])
            prob_b = torch.sigmoid(outputs_b["spoof_logit"])
            media_a = F.log_softmax(outputs_a["media_logits"], dim=1)
            media_b = F.softmax(outputs_b["media_logits"], dim=1)
            embed_cos = 1.0 - F.cosine_similarity(outputs_a["embedding"], outputs_b["embedding"], dim=1).mean()
            terms["consistency"] = (
                F.mse_loss(prob_a, prob_b)
                + F.kl_div(media_a, media_b, reduction="batchmean")
                + embed_cos
            )
        else:
            terms["consistency"] = outputs_a["embedding"].new_tensor(0.0)

        total = outputs_a["embedding"].new_tensor(0.0)
        scalar_terms: dict[str, float] = {}
        for name, term in terms.items():
            weight = float(getattr(self.weights, name))
            total = total + weight * term
            scalar_terms[name] = float(term.detach().cpu())
        scalar_terms["total"] = float(total.detach().cpu())
        return total, scalar_terms

    def spoof_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if self.spoof_pos_weight != 1.0:
            weights = torch.where(
                target > 0.5,
                target.new_tensor(self.spoof_pos_weight),
                target.new_tensor(1.0),
            )
            loss = loss * weights
        if self.focal_gamma > 0.0:
            probs = torch.sigmoid(logits)
            pt = torch.where(target > 0.5, probs, 1.0 - probs)
            loss = loss * (1.0 - pt).clamp_min(1e-6).pow(self.focal_gamma)
        return loss.mean()
