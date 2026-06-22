"""Poincare-ball prototype memory."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def project(x: torch.Tensor, curvature: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    max_norm = (1.0 - eps) / (curvature ** 0.5)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return x * scale


def expmap0(u: torch.Tensor, curvature: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    sqrt_c = curvature ** 0.5
    norm = u.norm(dim=-1, keepdim=True).clamp_min(eps)
    mapped = torch.tanh(sqrt_c * norm) * u / (sqrt_c * norm)
    return project(mapped, curvature=curvature)


def poincare_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    curvature: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    x = project(x, curvature=curvature)
    y = project(y, curvature=curvature)
    c = curvature
    x2 = x.pow(2).sum(dim=-1)
    y2 = y.pow(2).sum(dim=-1)
    diff2 = (x - y).pow(2).sum(dim=-1)
    denom = ((1.0 - c * x2) * (1.0 - c * y2)).clamp_min(eps)
    z = 1.0 + 2.0 * c * diff2 / denom
    return torch.acosh(z.clamp_min(1.0 + eps)) / (c ** 0.5)


class HyperbolicPrototypeMemory(nn.Module):
    """Class prototypes inside a Poincare ball."""

    def __init__(self, num_prototypes: int, dim: int, curvature: float = 1.0) -> None:
        super().__init__()
        self.num_prototypes = num_prototypes
        self.dim = dim
        self.curvature = curvature
        self.raw = nn.Parameter(torch.empty(num_prototypes, dim))
        nn.init.normal_(self.raw, mean=0.0, std=0.02)

    def prototypes(self) -> torch.Tensor:
        return expmap0(self.raw, curvature=self.curvature)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return expmap0(x, curvature=self.curvature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embed(x)
        prototypes = self.prototypes()
        z_expanded = z[:, None, :].expand(-1, self.num_prototypes, -1)
        p_expanded = prototypes[None, :, :].expand(z.shape[0], -1, -1)
        return poincare_distance(z_expanded, p_expanded, curvature=self.curvature)


def prototype_ce_loss(distances: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(-distances, labels)

