"""Channel-degradation augmentation for e002 consistency training.

Operates on a 1-D torch.float32 mono waveform at 16 kHz (the exact tensor
UnifiedAudioDataset already produces) and returns a degraded version of the
SAME length. Every op is sampled INDEPENDENTLY OF LABEL -- the model must learn
that channel degradation does not change the bona-fide/spoof decision, so
degradation must hit both classes identically.

Targets the failure modes diagnosed from the Kwok matrix + e003 baseline:
codec/band-limit (In-the-Wild, AI4T) and replay/reverb (ReplayDF).
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from scipy.signal import fftconvolve

# openSLR-28 simulated RIRs. Root comes from config (augmentation.rir_root), set via
# configure_rir_root() -- no hard-coded path, no silent no-op on a missing/misconfigured
# asset dir (audit §5).
_RIR_ROOT: Path | None = None
_rir_cache = None
_rir_read_failures = 0


def configure_rir_root(root: str | Path) -> None:
    """Set the RIR corpus root. Call once at startup (see aug_assets.resolve_aug_assets)."""
    global _RIR_ROOT, _rir_cache
    _RIR_ROOT = Path(root)
    _rir_cache = None


def _rirs():
    global _rir_cache
    if _RIR_ROOT is None:
        raise RuntimeError(
            "[channel_aug] RIR root not configured -- call "
            "channel_aug.configure_rir_root(cfg['augmentation']['rir_root']) at startup "
            "before enabling reverb augmentation (audit §5: hard-coded-path silent no-op "
            "is forbidden)."
        )
    if _rir_cache is None:
        if not _RIR_ROOT.is_dir():
            raise RuntimeError(f"[channel_aug] configured RIR root does not exist: {_RIR_ROOT}")
        _rir_cache = sorted(_RIR_ROOT.rglob("*.wav"))
        if not _rir_cache:
            raise RuntimeError(f"[channel_aug] configured RIR root has no .wav files: {_RIR_ROOT}")
    return _rir_cache


def _norm(x: np.ndarray) -> np.ndarray:
    m = float(np.max(np.abs(x))) + 1e-9
    return (x / m) if m > 1.0 else x


def conv_noise(x, rng):
    n_taps = int(rng.integers(2, 6))
    h = rng.uniform(-0.25, 0.25, n_taps).astype(np.float32); h[0] = 1.0
    return _norm(fftconvolve(x, h, mode="same").astype(np.float32))


def impulsive_noise(x, rng):
    n = max(1, int(len(x) * rng.uniform(0.0005, 0.005)))
    idx = rng.integers(0, len(x), n)
    y = x.copy()
    y[idx] += (rng.uniform(-1, 1, n).astype(np.float32)
               * np.abs(y[idx]).clip(0.05, None) * rng.uniform(2, 6))
    return _norm(y)


def stationary_noise(x, rng, snr_lo=10, snr_hi=40):
    snr = rng.uniform(snr_lo, snr_hi)
    noise = rng.standard_normal(len(x)).astype(np.float32)
    px, pn = float(np.mean(x ** 2)) + 1e-9, float(np.mean(noise ** 2)) + 1e-9
    noise *= np.sqrt(px / (pn * 10 ** (snr / 10)))
    return _norm(x + noise)


def band_limit(x, rng):
    cutoff = rng.uniform(3000.0, 4000.0)
    n = len(x)
    X = np.fft.rfft(np.asarray(x, dtype=np.float64))
    freqs = np.fft.rfftfreq(n, d=1.0 / 16000.0)
    X[freqs > cutoff] = 0.0
    y = np.fft.irfft(X, n=n).astype(np.float32)
    return _norm(y)


def mu_law(x, rng, mu=255.0):
    xc = np.clip(x, -1, 1)
    y = np.sign(xc) * np.log1p(mu * np.abs(xc)) / np.log1p(mu)
    q = np.round((y + 1) * 127.5) / 127.5 - 1
    out = np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu
    return out.astype(np.float32)


def rir_reverb(x, rng):
    global _rir_read_failures
    rirs = _rirs()
    rir_path = rirs[int(rng.integers(len(rirs)))]
    try:
        import soundfile as sf
        rir, _ = sf.read(str(rir_path), dtype="float32")
    except Exception as e:
        _rir_read_failures += 1
        raise RuntimeError(
            f"[channel_aug] RIR read failed ({_rir_read_failures} total failures so far): "
            f"{rir_path}: {e}"
        ) from e
    if rir.ndim > 1:
        rir = rir[:, 0]
    rir = rir / (np.max(np.abs(rir)) + 1e-9)
    y = fftconvolve(x, rir)[: len(x)].astype(np.float32)
    if rng.random() < 0.5:
        y = stationary_noise(y, rng, 5, 20)
    return _norm(y)


PIPELINE = [
    (conv_noise, 0.5),
    (impulsive_noise, 0.3),
    (stationary_noise, 0.3),
    (band_limit, 0.2),
    (mu_law, 0.2),
    (rir_reverb, 0.3),
]


def degrade_np(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    fired = False
    for fn, p in PIPELINE:
        if rng.random() < p:
            x = fn(x, rng); fired = True
    if not fired:
        fn, _ = PIPELINE[int(rng.integers(len(PIPELINE)))]
        x = fn(x, rng)
    return _norm(x.astype(np.float32))


def degrade_waveform(wav: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    x = wav.detach().cpu().numpy().astype(np.float32)
    y = degrade_np(x, rng)
    return torch.from_numpy(np.ascontiguousarray(y)).to(wav.dtype)
