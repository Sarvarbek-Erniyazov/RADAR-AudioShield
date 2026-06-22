"""Online media transformations used for RADAR-style consistency training."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from .audio_io import resample_linear
from .labels import TRANSFORM_STATE_TO_ID


def make_augmented_batch(
    waveform: torch.Tensor,
    sample_rate: int,
    enabled: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one random media transform per item.

    Returns transformed waveform and transform-state labels.
    """

    if not enabled:
        states = torch.full(
            (waveform.shape[0],),
            TRANSFORM_STATE_TO_ID["clean"],
            dtype=torch.long,
            device=waveform.device,
        )
        return waveform, states

    outputs = []
    states = []
    choices = list(TRANSFORM_STATE_TO_ID.keys())
    for item in waveform:
        state = random.choice(choices)
        outputs.append(apply_transform(item, sample_rate, state))
        states.append(TRANSFORM_STATE_TO_ID[state])
    return torch.stack(outputs), torch.tensor(states, dtype=torch.long, device=waveform.device)


def apply_transform(waveform: torch.Tensor, sample_rate: int, state: str) -> torch.Tensor:
    if state == "clean":
        return waveform
    if state == "codec_proxy":
        return codec_proxy(waveform)
    if state == "resampled":
        return resample_cycle(waveform, sample_rate)
    if state == "rir_convolved":
        return rir_convolve(waveform, sample_rate)
    if state == "replay_simulated":
        return replay_simulate(waveform, sample_rate)
    if state == "noise_mixed":
        return noise_mix(waveform, sample_rate)
    raise KeyError(f"Unknown transform state: {state}")


def codec_proxy(waveform: torch.Tensor, levels: int = 256) -> torch.Tensor:
    """Mu-law companding plus quantization as a codec/laundering proxy."""

    mu = float(levels - 1)
    x = waveform.clamp(-1.0, 1.0)
    encoded = torch.sign(x) * torch.log1p(mu * x.abs()) / torch.log1p(torch.tensor(mu, device=x.device))
    quantized = torch.round((encoded + 1.0) * 0.5 * mu) / mu * 2.0 - 1.0
    decoded = torch.sign(quantized) * (torch.pow(torch.tensor(1.0 + mu, device=waveform.device), quantized.abs()) - 1.0) / mu
    return decoded.clamp(-1.0, 1.0)


def resample_cycle(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    down_sr = random.choice([8000, 11025, 12000])
    down = resample_linear(waveform, sample_rate, down_sr)
    up = resample_linear(down, down_sr, sample_rate)
    return _match_length(up, waveform.shape[-1])


def rir_convolve(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    length = max(16, int(random.uniform(0.04, 0.16) * sample_rate))
    t = torch.arange(length, device=waveform.device).float()
    decay = torch.exp(-t / random.uniform(0.015, 0.060) / sample_rate)
    noise = torch.randn(length, device=waveform.device) * 0.02
    kernel = decay + noise
    kernel[0] += 1.0
    kernel = kernel / kernel.abs().sum().clamp_min(1e-6)
    y = F.conv1d(waveform[None, None, :], kernel.flip(0)[None, None, :], padding=length - 1)
    y = y[0, 0, : waveform.shape[-1]]
    return y.clamp(-1.0, 1.0)


def replay_simulate(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    echo_delay = int(random.uniform(0.018, 0.055) * sample_rate)
    echo = F.pad(waveform[:-echo_delay], (echo_delay, 0)) if echo_delay < waveform.numel() else waveform
    low = _moving_average(waveform, kernel_size=random.choice([5, 7, 9]))
    y = 0.78 * low + 0.18 * echo + 0.02 * torch.randn_like(waveform)
    return y.clamp(-1.0, 1.0)


def noise_mix(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    snr_db = random.uniform(5.0, 20.0)
    noise = torch.randn_like(waveform)
    if random.random() < 0.5:
        freq = random.uniform(80.0, 650.0)
        t = torch.arange(waveform.numel(), device=waveform.device).float() / sample_rate
        noise = noise + 0.5 * torch.sin(2.0 * torch.pi * freq * t)
    signal_power = waveform.pow(2).mean().clamp_min(1e-9)
    noise_power = noise.pow(2).mean().clamp_min(1e-9)
    scale = torch.sqrt(signal_power / (10.0 ** (snr_db / 10.0) * noise_power))
    return (waveform + scale * noise).clamp(-1.0, 1.0)


def _moving_average(waveform: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel = torch.ones(kernel_size, device=waveform.device) / kernel_size
    y = F.conv1d(waveform[None, None, :], kernel[None, None, :], padding=kernel_size // 2)
    return y[0, 0, : waveform.shape[-1]]


def _match_length(waveform: torch.Tensor, length: int) -> torch.Tensor:
    if waveform.numel() == length:
        return waveform
    if waveform.numel() < length:
        return F.pad(waveform, (0, length - waveform.numel()))
    return waveform[:length]
