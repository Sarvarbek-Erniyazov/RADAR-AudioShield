"""Physiology-inspired proxies for breath, pause, and micro-prosody cues."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PhysiologyEncoder(nn.Module):
    """Estimate compact descriptors and a gate from waveform-level proxies.

    This is not a clinical physiology model. It gives the detector explicit
    low-energy pause, breath-like noise, energy modulation, and zero-crossing
    cues that are difficult for many generators and media transforms to match.
    """

    def __init__(self, sample_rate: int = 16000, hidden_dim: int = 64) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_length = int(0.050 * sample_rate)
        self.hop_length = int(0.020 * sample_rate)
        self.mlp = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if waveform.ndim != 2:
            raise ValueError(f"Expected [batch, samples], got {tuple(waveform.shape)}")
        frames = _safe_unfold(waveform, self.frame_length, self.hop_length)
        rms = frames.pow(2).mean(dim=-1).clamp_min(1e-9).sqrt()
        signs = torch.sign(frames)
        zcr = (signs[..., 1:] != signs[..., :-1]).float().mean(dim=-1)

        median_energy = rms.median(dim=1, keepdim=True).values.clamp_min(1e-6)
        pause_mask = rms < (0.35 * median_energy)
        breath_mask = pause_mask & (zcr > zcr.mean(dim=1, keepdim=True))

        descriptors = torch.stack(
            [
                rms.mean(dim=1),
                rms.std(dim=1),
                rms.max(dim=1).values,
                (rms.std(dim=1) / rms.mean(dim=1).clamp_min(1e-6)),
                zcr.mean(dim=1),
                zcr.std(dim=1),
                pause_mask.float().mean(dim=1),
                breath_mask.float().mean(dim=1),
                _mean_run_length(pause_mask.float()),
                _modulation_energy(rms),
            ],
            dim=1,
        )
        descriptors = torch.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)
        return self.mlp(descriptors), breath_mask.float()


def _safe_unfold(waveform: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor:
    if waveform.shape[1] < frame_length:
        waveform = F.pad(waveform, (0, frame_length - waveform.shape[1]))
    return waveform.unfold(dimension=1, size=frame_length, step=hop_length)


def _mean_run_length(mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] == 0:
        return torch.zeros(mask.shape[0], device=mask.device)
    transitions = torch.diff(F.pad(mask, (1, 1)), dim=1).abs()
    run_counts = (transitions.sum(dim=1) / 2.0).clamp_min(1.0)
    return mask.sum(dim=1) / run_counts


def _modulation_energy(rms: torch.Tensor) -> torch.Tensor:
    centered = rms - rms.mean(dim=1, keepdim=True)
    if centered.shape[1] < 2:
        return torch.zeros(rms.shape[0], device=rms.device)
    spectrum = torch.fft.rfft(centered, dim=1).abs()
    low_bins = spectrum[:, 1 : min(8, spectrum.shape[1])]
    return low_bins.mean(dim=1) if low_bins.numel() else torch.zeros(rms.shape[0], device=rms.device)

