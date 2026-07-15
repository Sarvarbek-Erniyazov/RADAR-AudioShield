"""Resampling round-trip counterfactual: downsample to an intermediate rate
and back up, which is dependency-free but not information-free -- the
downsample step's anti-aliasing low-pass irreversibly discards energy above
the intermediate rate's Nyquist frequency. `dose` = intermediate sample rate
in Hz -- LOWER dose means MORE severe bandlimiting.

Uses `scipy.signal.resample_poly` (polyphase FIR filtering, matches the
project's own embedding-extraction resampling convention) rather than a naive
linear interpolation, since the anti-aliasing behavior is exactly the
degradation this transform models.
"""
from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

from ._align import align_to_reference
from .provenance import make_provenance


def _resample(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return x.astype(np.float32)
    g = gcd(int(orig_sr), int(target_sr))
    return resample_poly(x, target_sr // g, orig_sr // g).astype(np.float32)


def resample_round_trip(
    waveform: np.ndarray,
    sr: int,
    dose: float,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Downsample `waveform` from `sr` to `dose` Hz, then back up to `sr`.

    Args:
        waveform: (n,) mono float32.
        sr: original sample rate (Hz).
        dose: intermediate sample rate (Hz), 0 < dose <= sr. Lower = more
            severe bandlimiting/aliasing artifacts.
        seed: accepted for API uniformity; unused (this transform has no RNG).

    Returns:
        (aligned_waveform (n,) float32, provenance dict).
    """
    if not (0 < dose <= sr):
        raise ValueError(f"dose (intermediate sample rate) must be in (0, {sr}] Hz, got {dose}")
    waveform = np.asarray(waveform, dtype=np.float32)
    dose_int = int(round(dose))

    down = _resample(waveform, sr, dose_int)
    back = _resample(down, dose_int, sr)
    aligned = align_to_reference(waveform, back)

    provenance = make_provenance(
        transform="resample_round_trip", family="resampling", dose=float(dose_int),
        dose_unit="hz_intermediate_sr", seed=seed, sr=sr, orig_sr=sr,
    )
    return aligned, provenance
