"""AudioShield-X detector: frozen layer-weighted SSL -> attentive pooling ->
projection -> {binary head, cosine-AAM prototypes, BMI domain critic}.
"""

from __future__ import annotations

import torch
from torch import nn

from .ssl_backbone import LayerWeightedSSL
from .pooling import AttentiveStatsPooling
from .prototypes import CosinePrototypes
from .heads import BinaryHead, DomainCritic, grad_reverse


class AudioShieldX(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        m = cfg["model"]
        self.ssl = LayerWeightedSSL(
            backbone_name=m["backbone_name"],
            num_hidden_states=m["num_hidden_states"],
            init_center=m["layer_weight_init_center"],
            init_band=tuple(m["layer_weight_init_band"]),
            init_temp=m["layer_weight_init_temp"],
            freeze=m["freeze_backbone"],
        )
        h = self.ssl.hidden_size
        self.pool = AttentiveStatsPooling(h)
        self.proj = nn.Sequential(
            nn.Linear(2 * h, m["embedding_dim"]),
            nn.LayerNorm(m["embedding_dim"]),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(m["embedding_dim"], m["embedding_dim"]),
            nn.LayerNorm(m["embedding_dim"]),
        )
        self.binary = BinaryHead(m["embedding_dim"])
        # optional projection head for contrastive learning (SupCon works better
        # on a projected space). Default off; enabled via model cfg for xc runs.
        proj_dim = m.get("contrastive_proj_dim", 0)
        if proj_dim and proj_dim > 0:
            self.contrastive_proj = nn.Sequential(
                nn.Linear(m["embedding_dim"], m["embedding_dim"]),
                nn.GELU(),
                nn.Linear(m["embedding_dim"], proj_dim),
            )
        else:
            self.contrastive_proj = None
        self.prototypes = CosinePrototypes(
            dim=m["embedding_dim"],
            num_bona=m["prototypes"]["num_bona"],
            num_spoof=m["prototypes"]["num_spoof"],
        )
        self.domain_critic = DomainCritic(
            dim=m["embedding_dim"],
            num_domains=m["bmi"]["num_bona_domains"],
            hidden=m["bmi"]["domain_critic_hidden"],
        )

    def embed(self, waveform: torch.Tensor) -> torch.Tensor:
        seq = self.ssl(waveform)        # [B, T, H]
        pooled = self.pool(seq)         # [B, 2H]
        return self.proj(pooled)        # [B, D]

    def forward(self, waveform: torch.Tensor, grl_lambda: float = 0.0) -> dict:
        z = self.embed(waveform)
        proto = self.prototypes(z)
        out = {
            "embedding": z,
            "contrastive_embedding": self.contrastive_proj(z) if self.contrastive_proj is not None else z,
            "spoof_logit": self.binary(z),
            "proto_cos": proto["cos"],
            "bona_cos": proto["bona_cos"],
            "spoof_cos": proto["spoof_cos"],
            "proto_spoof_score": proto["spoof_score"],
        }
        # domain critic is computed through GRL only when requested (BMI training)
        if grl_lambda > 0.0:
            out["domain_logits"] = self.domain_critic(grad_reverse(z, grl_lambda))
        return out
