"""Trainable frontend initialized near log-mel features."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


def hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq / 700.0)


def mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mels / 2595.0) - 1.0)


def build_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 20.0,
    f_max: Optional[float] = None,
) -> torch.Tensor:
    f_max = float(f_max or sample_rate / 2)
    min_mel = hz_to_mel(torch.tensor(float(f_min)))
    max_mel = hz_to_mel(torch.tensor(float(f_max)))
    mel_points = torch.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = torch.floor((n_fft + 1) * hz_points / sample_rate).long()
    n_freqs = n_fft // 2 + 1

    fb = torch.zeros(n_mels, n_freqs)
    for idx in range(n_mels):
        left, center, right = bins[idx].item(), bins[idx + 1].item(), bins[idx + 2].item()
        center = max(center, left + 1)
        right = max(right, center + 1)
        left = min(left, n_freqs - 1)
        center = min(center, n_freqs - 1)
        right = min(right, n_freqs)
        if center > left:
            fb[idx, left:center] = torch.linspace(0.0, 1.0, center - left)
        if right > center:
            fb[idx, center:right] = torch.linspace(1.0, 0.0, right - center)

    fb = fb / fb.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return fb.clamp_min(1e-6)


class TrainableMelFrontend(nn.Module):
    """Log-mel frontend whose filterbank is learnable after mel initialization."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        fb = build_mel_filterbank(sample_rate, n_fft, n_mels)
        self.filterbank_logits = nn.Parameter(_softplus_inverse(fb))
        self.log_compression = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 2:
            raise ValueError(f"Expected [batch, samples], got {tuple(waveform.shape)}")

        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(waveform.device),
            center=True,
            return_complex=True,
        )
        power = spec.abs().pow(2.0)
        fb = F.softplus(self.filterbank_logits).to(power.device)
        mel = torch.einsum("mf,bft->bmt", fb, power)
        return torch.log1p(F.softplus(self.log_compression) * mel.clamp_min(1e-8))


class FrontendEncoder(nn.Module):
    """Compact CNN over trainable log-mel features."""

    def __init__(self, sample_rate: int = 16000, output_dim: int = 128) -> None:
        super().__init__()
        self.frontend = TrainableMelFrontend(sample_rate=sample_rate)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(1, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
        self.proj = nn.Linear(96, output_dim)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        logmel = self.frontend(waveform).unsqueeze(1)
        encoded = self.encoder(logmel)
        pooled = encoded.mean(dim=(-2, -1))
        return self.proj(pooled)


def _softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(x.clamp_min(1e-6)))


def sinusoidal_position(length: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device).float().unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe
