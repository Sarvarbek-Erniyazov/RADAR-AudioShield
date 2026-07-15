"""RIR convolution + replay-simulation counterfactuals.

Reuses the project's existing RIR asset directory (the same OpenSLR-28-style
corpus `channel_aug.py`'s training-time augmentation draws from) via
`aug_assets.fingerprint_asset_dir` -- the exact validation the rest of the
repo already relies on (asset dir exists, has >= min_files audio files, gets a
deterministic listing-hash for provenance) rather than a second
implementation of that check.

Two transforms, both dose = wet/dry mix in [0, 1] (0 = fully dry/no effect,
1 = fully wet/maximal effect) so both share one dose semantics:

  rir_convolve      -- pure room-impulse-response convolution, wet/dry mixed.
  replay_simulate   -- rir_convolve PLUS mic self-noise and mild band-limiting
                        that scale with the same dose, modeling a full
                        play-through-a-speaker-and-re-record replay chain
                        (the failure mode ReplayDF targets), not just reverb.

`seed` deterministically selects WHICH RIR file from the asset pool is used
(logged in provenance by relative path + sha256, alongside the whole asset
directory's fingerprint) -- decoupled from `dose`, so the same RIR can be
swept across doses and the same dose can be tested against multiple RIRs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from ..data.aug_assets import fingerprint_asset_dir
from ._align import align_to_reference
from .noise import NOISE_GENERATORS
from .provenance import make_provenance, sha256_file
from .resampling import _resample

AUDIO_EXTS = (".wav", ".flac")


def _list_rirs(rir_root: str | Path) -> list[Path]:
    root = Path(rir_root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


def _load_rir(path: Path, target_sr: int) -> np.ndarray:
    rir, rir_sr = sf.read(str(path), dtype="float32", always_2d=True)
    rir = rir.mean(axis=1)  # mono
    if rir_sr != target_sr:
        rir = _resample(rir, rir_sr, target_sr)
    peak = float(np.max(np.abs(rir))) + 1e-9
    return rir / peak


def _pick_rir(rir_root: str | Path, seed: int) -> tuple[Path, dict]:
    # fingerprint_asset_dir is the ONE validation path (dir exists, has enough
    # files, deterministic listing hash) -- reused here rather than a second,
    # possibly-inconsistent existence/count check.
    asset_fp = fingerprint_asset_dir(rir_root)
    rirs = _list_rirs(rir_root)
    idx = np.random.default_rng(seed).integers(len(rirs))
    chosen = rirs[idx]
    asset_id = dict(
        rir_root_fingerprint=asset_fp,
        chosen_rir_path=str(chosen.relative_to(Path(rir_root))),
        chosen_rir_sha256=sha256_file(chosen),
    )
    return chosen, asset_id


def _clip_range(x: np.ndarray) -> np.ndarray:
    """Clip to [-1, 1] rather than rescale by peak: clipping only touches the
    (rare, localized) samples that actually exceed range -- a whole-signal
    peak-rescale would instead distort EVERY sample by a dose-dependent factor
    (whenever a reverb tail happens to push the peak over 1.0 at some doses but
    not others), silently breaking the dose-response relationship this
    transform exists to produce. Clipping doubles as a physically reasonable
    ADC/mic-clipping model for `replay_simulate`."""
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def rir_convolve(
    waveform: np.ndarray,
    sr: int,
    dose: float,
    seed: int,
    rir_root: str | Path,
) -> tuple[np.ndarray, dict]:
    """Convolve `waveform` with a seeded-chosen RIR, wet/dry mixed by `dose`.

    Args:
        waveform: (n,) mono float32.
        sr: sample rate (Hz).
        dose: wet/dry mix in [0, 1]. 0 = no effect, 1 = fully wet (only the
            convolved signal). Higher = more severe.
        seed: selects which RIR file from `rir_root` is used (REQUIRED).
        rir_root: directory containing RIR .wav/.flac files (searched
            recursively -- same convention as channel_aug.py).

    Returns:
        (aligned_waveform (n,) float32, provenance dict).
    """
    if not (0.0 <= dose <= 1.0):
        raise ValueError(f"dose (wet/dry mix) must be in [0, 1], got {dose}")
    waveform = np.asarray(waveform, dtype=np.float32)
    chosen, asset_id = _pick_rir(rir_root, seed)
    rir = _load_rir(chosen, sr)

    wet = fftconvolve(waveform, rir)[: len(waveform)].astype(np.float32)
    mixed = _clip_range((1.0 - dose) * waveform + dose * wet)
    aligned = align_to_reference(waveform, mixed)

    provenance = make_provenance(
        transform="rir_convolve", family="reverb", dose=float(dose), dose_unit="wet_dry_mix",
        seed=seed, sr=sr, **asset_id,
    )
    return aligned, provenance


def replay_simulate(
    waveform: np.ndarray,
    sr: int,
    dose: float,
    seed: int,
    rir_root: str | Path,
    max_noise_snr_db: float = 25.0,
    min_noise_snr_db: float = 5.0,
) -> tuple[np.ndarray, dict]:
    """`rir_convolve` plus mic self-noise and mild band-limiting, both scaled
    by the same `dose` -- models a full replay chain (speaker playback + room +
    re-recording), not just room reverb.

    Args:
        dose: overall replay severity in [0, 1]; also drives the RIR wet/dry
            mix and interpolates the injected noise's SNR from
            `max_noise_snr_db` (dose=0) down to `min_noise_snr_db` (dose=1).
        seed: selects the RIR (via rir_convolve) AND the noise draw.

    Returns:
        (aligned_waveform (n,) float32, provenance dict).
    """
    if not (0.0 <= dose <= 1.0):
        raise ValueError(f"dose (replay severity) must be in [0, 1], got {dose}")
    reverbed, rir_prov = rir_convolve(waveform, sr, dose, seed, rir_root)

    snr_db = max_noise_snr_db - dose * (max_noise_snr_db - min_noise_snr_db)
    rng = np.random.default_rng(seed + 1)  # distinct stream from the RIR pick
    mic_noise = NOISE_GENERATORS["white"](rng, len(reverbed)).astype(np.float32)
    signal_power = float(np.mean(reverbed.astype(np.float64) ** 2)) + 1e-12
    noise_power = float(np.mean(mic_noise.astype(np.float64) ** 2)) + 1e-12
    scale = np.sqrt(signal_power / (noise_power * 10.0 ** (snr_db / 10.0)))
    noisy = _clip_range(reverbed + scale * mic_noise)

    # mild, dose-scaled band-limiting (a cheap microphone/speaker frequency-
    # response proxy, not a full transfer-function simulation)
    cutoff_hz = sr / 2.0 - dose * (sr / 2.0 - 3000.0)  # dose=0 -> no cut, dose=1 -> cut at 3kHz
    spectrum = np.fft.rfft(noisy.astype(np.float64))
    freqs = np.fft.rfftfreq(len(noisy), d=1.0 / sr)
    spectrum[freqs > cutoff_hz] = 0.0
    band_limited = np.fft.irfft(spectrum, n=len(noisy)).astype(np.float32)

    aligned = align_to_reference(waveform, band_limited)
    provenance = make_provenance(
        transform="replay_simulate", family="reverb", dose=float(dose), dose_unit="replay_severity",
        seed=seed, sr=sr, noise_snr_db=float(snr_db), band_limit_hz=float(cutoff_hz),
        **{k: v for k, v in rir_prov.items() if k not in ("transform", "family", "dose", "dose_unit", "seed", "sr")},
    )
    return aligned, provenance
