"""Frozen SSL backbone with a learnable softmax layer-weighted sum.

Replaces the old last-layer mean pool. The e000 probe selected WavLM-large
layer 10 (plateau 8-11), so the layer weights are initialized as a soft band
over that range and are free to redistribute during training.
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from torch import nn
from transformers import AutoModel


class LayerWeightedSSL(nn.Module):
    def __init__(
        self,
        backbone_name: str = "microsoft/wavlm-large",
        num_hidden_states: int = 25,
        init_center: int = 10,
        init_band: tuple[int, int] = (8, 11),
        init_temp: float = 1.0,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.hidden_size = int(self.backbone.config.hidden_size)
        # backbone-agnostic: trust the loaded model's actual layer count
        # (num_hidden_layers + 1 for the embedding output) over the config value,
        # so swapping WavLM-large -> XLS-R/other needs no manual num_hidden_states edit.
        actual_hidden_states = int(getattr(self.backbone.config, "num_hidden_layers", num_hidden_states - 1)) + 1
        if actual_hidden_states != num_hidden_states:
            print(f"[ssl] num_hidden_states {num_hidden_states} -> {actual_hidden_states} "
                  f"(from {backbone_name} config)")
        self.num_hidden_states = actual_hidden_states
        self.layer_weight_init_center = int(init_center)
        self.layer_weight_init_band = tuple(int(x) for x in init_band)
        self.frozen = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # softmax layer weights, initialized as a Gaussian-ish band around center
        w0 = torch.full((actual_hidden_states,), -10.0)
        lo, hi = init_band
        for i in range(actual_hidden_states):
            if lo <= i <= hi:
                w0[i] = -((i - init_center) ** 2) / (2.0 * init_temp ** 2)
        self.layer_logits = nn.Parameter(w0)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()  # keep frozen backbone in eval (no dropout/BN updates)
        return self

    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits, dim=0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform [B, T] float -> sequence features [B, T', H]."""
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            out = self.backbone(waveform, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)  # [L, B, T', H]
        w = self.layer_weights().view(-1, 1, 1, 1)
        return (w * hs).sum(dim=0)                   # [B, T', H]
