"""BioPhys-HyperRADAR model components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .features import FrontendEncoder
from .hyperbolic import HyperbolicPrototypeMemory
from .labels import KNOWN_METHODS, TRANSFORM_STATES
from .physiology import PhysiologyEncoder


@dataclass
class ModelConfig:
    sample_rate: int = 16000
    embedding_dim: int = 256
    ssl_hidden_dim: int = 256
    frontend_dim: int = 128
    physiology_dim: int = 64
    num_methods: int = len(KNOWN_METHODS)
    num_media_states: int = len(TRANSFORM_STATES)
    num_experts: int = 4
    ssl_model_name: Optional[str] = None
    freeze_ssl: bool = True


class ConvSSLFallback(nn.Module):
    """Small waveform encoder used when a Hugging Face SSL model is unavailable."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=5, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=9, stride=4, padding=4),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 192, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Conv1d(192, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.net(waveform.unsqueeze(1))
        return x.mean(dim=-1)


class HFSSLBackbone(nn.Module):
    """Optional wrapper for XLS-R/WavLM/wav2vec2-style models."""

    def __init__(self, model_name: str, output_dim: int, freeze: bool = True) -> None:
        super().__init__()
        try:
            from transformers import AutoModel  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "transformers is required for --ssl-model-name. Install it or omit the option "
                "to use the compact fallback encoder."
            ) from exc

        self.model = AutoModel.from_pretrained(model_name, use_safetensors=True)
        hidden = int(getattr(self.model.config, "hidden_size"))
        self.proj = nn.Linear(hidden, output_dim)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        outputs = self.model(waveform, output_hidden_states=False)
        hidden = outputs.last_hidden_state
        return self.proj(hidden.mean(dim=1))


class BioPhysHyperRADAR(nn.Module):
    """Media-state-aware, hyperbolic, physiology-guided detector."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.ssl_model_name:
            self.ssl_backbone = HFSSLBackbone(
                config.ssl_model_name,
                output_dim=config.ssl_hidden_dim,
                freeze=config.freeze_ssl,
            )
        else:
            self.ssl_backbone = ConvSSLFallback(hidden_dim=config.ssl_hidden_dim)

        self.frontend_branch = FrontendEncoder(
            sample_rate=config.sample_rate,
            output_dim=config.frontend_dim,
        )
        self.physiology_branch = PhysiologyEncoder(
            sample_rate=config.sample_rate,
            hidden_dim=config.physiology_dim,
        )

        fusion_dim = config.ssl_hidden_dim + config.frontend_dim + config.physiology_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
        )
        self.phys_gate = nn.Sequential(
            nn.Linear(config.physiology_dim, config.embedding_dim),
            nn.Sigmoid(),
        )

        self.target_prototypes = HyperbolicPrototypeMemory(2, config.embedding_dim)
        self.method_prototypes = HyperbolicPrototypeMemory(config.num_methods, config.embedding_dim)

        self.media_state_head = nn.Linear(config.embedding_dim, config.num_media_states)
        self.method_head = nn.Linear(config.embedding_dim, config.num_methods)
        self.router = nn.Sequential(
            nn.Linear(config.embedding_dim + config.num_media_states, config.embedding_dim // 2),
            nn.GELU(),
            nn.Linear(config.embedding_dim // 2, config.num_experts),
        )
        self.experts = nn.ModuleList(
            [nn.Linear(config.embedding_dim, 1) for _ in range(config.num_experts)]
        )

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        ssl_vec = self.ssl_backbone(waveform)
        frontend_vec = self.frontend_branch(waveform)
        phys_vec, breath_mask = self.physiology_branch(waveform)

        fused = self.fusion(torch.cat([ssl_vec, frontend_vec, phys_vec], dim=1))
        embedding = fused * (1.0 + 0.5 * self.phys_gate(phys_vec))

        media_logits = self.media_state_head(embedding)
        media_probs = F.softmax(media_logits, dim=1)
        route_logits = self.router(torch.cat([embedding, media_probs], dim=1))
        route_weights = F.softmax(route_logits, dim=1)
        expert_logits = torch.cat([expert(embedding) for expert in self.experts], dim=1)
        spoof_logit = (expert_logits * route_weights).sum(dim=1)

        return {
            "spoof_logit": spoof_logit,
            "method_logits": self.method_head(embedding),
            "media_logits": media_logits,
            "route_weights": route_weights,
            "expert_logits": expert_logits,
            "embedding": embedding,
            "target_distances": self.target_prototypes(embedding),
            "method_distances": self.method_prototypes(embedding),
            "breath_mask": breath_mask,
        }
