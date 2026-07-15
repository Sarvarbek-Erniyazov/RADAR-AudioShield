"""Additive-noise counterfactual at a controlled SNR. `dose` = target SNR in
dB -- LOWER dose means MORE severe (more noise relative to signal).

Purely additive, so this transform is trivially sample-aligned (no length or
timing change) and does not need `_align.align_to_reference`.
"""
from __future__ import annotations

import numpy as np

from .provenance import make_provenance

NOISE_GENERATORS = {
    "white": lambda rng, n: rng.standard_normal(n),
    # Pink (1/f) noise via FFT shaping of white noise -- deterministic given the
    # same white-noise draw, no extra RNG state.
    "pink": lambda rng, n: _pink_from_white(rng.standard_normal(n)),
}


def _pink_from_white(white: np.ndarray) -> np.ndarray:
    n = len(white)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1] if n > 1 else 1.0  # avoid divide-by-zero at DC
    spectrum = spectrum / np.sqrt(freqs)
    return np.fft.irfft(spectrum, n=n)


def additive_noise_snr(
    waveform: np.ndarray,
    sr: int,
    dose: float,
    seed: int,
    noise_type: str = "white",
) -> tuple[np.ndarray, dict]:
    """Add `noise_type` noise to `waveform` scaled to hit the target SNR exactly
    (given the waveform's own measured power -- not an assumed reference level).

    Args:
        waveform: (n,) mono float32.
        sr: sample rate (Hz) -- carried into provenance only; the transform
            itself is rate-agnostic.
        dose: target SNR in dB. Lower = more noise = more severe.
        seed: RNG seed. REQUIRED (not optional) -- this transform's output is
            random, so a caller always needs to pin it.
        noise_type: one of NOISE_GENERATORS' keys.

    Returns:
        (waveform + scaled noise (n,) float32, provenance dict).
    """
    if noise_type not in NOISE_GENERATORS:
        raise ValueError(f"noise_type must be one of {sorted(NOISE_GENERATORS)}, got {noise_type!r}")
    waveform = np.asarray(waveform, dtype=np.float32)
    n = len(waveform)
    rng = np.random.default_rng(seed)
    noise = NOISE_GENERATORS[noise_type](rng, n).astype(np.float32)

    signal_power = float(np.mean(waveform.astype(np.float64) ** 2))
    noise_power = float(np.mean(noise.astype(np.float64) ** 2)) + 1e-12
    if signal_power <= 0:
        raise ValueError("additive_noise_snr: waveform has zero power -- target SNR is undefined")
    target_ratio = 10.0 ** (dose / 10.0)
    scale = np.sqrt(signal_power / (noise_power * target_ratio))
    noisy = (waveform + scale * noise).astype(np.float32)

    provenance = make_provenance(
        transform="additive_noise_snr", family="noise", dose=float(dose), dose_unit="db_snr",
        seed=seed, sr=sr, noise_type=noise_type,
    )
    return noisy, provenance
