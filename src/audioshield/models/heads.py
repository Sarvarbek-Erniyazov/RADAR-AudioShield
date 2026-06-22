"""Small heads: binary classifier + BMI domain critic with gradient reversal."""

from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


class BinaryHead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z).squeeze(-1)   # [B] logit


class DomainCritic(nn.Module):
    """Predicts bona-fide source domain from embedding (through GRL at call site)."""

    def __init__(self, dim: int, num_domains: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_domains),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)              # [B, num_domains] logits
