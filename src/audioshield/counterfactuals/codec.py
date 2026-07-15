"""Codec-chain counterfactual: encode the waveform to a lossy codec at a given
bitrate, decode it back, and align it to the original. `dose` = target
bitrate in kbps -- LOWER dose means MORE severe compression artifacts.

Requires a working `ffmpeg` on PATH; encoding/decoding always runs with
`-threads 1` so a fixed (waveform, sr, dose, codec) input is bit-for-bit
reproducible regardless of the host's core count (multi-threaded lossy
encoders are not guaranteed bit-exact across thread counts).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from ._align import align_to_reference
from .provenance import make_provenance

FFMPEG_ENCODER = {"opus": "libopus", "aac": "aac", "mp3": "libmp3lame"}
FFMPEG_CONTAINER_EXT = {"opus": "ogg", "aac": "adts", "mp3": "mp3"}
# Practical bitrate range per codec (kbps). Below the low end the encoder either
# refuses or produces audio too degraded to be a meaningful "dose" point; above
# the high end there is no more compression artifact to speak of.
BITRATE_RANGE_KBPS = {"opus": (6, 128), "aac": (16, 256), "mp3": (8, 320)}


class FfmpegNotAvailableError(RuntimeError):
    pass


def check_ffmpeg_available() -> str:
    """Return the resolved ffmpeg executable path, or raise loudly."""
    path = shutil.which("ffmpeg")
    if not path:
        raise FfmpegNotAvailableError(
            "ffmpeg not found on PATH -- codec-chain counterfactuals require a working "
            "ffmpeg install. Install it (e.g. `apt install ffmpeg` / "
            "https://ffmpeg.org/download.html) and ensure it's on PATH, then retry."
        )
    return path


def _ffmpeg_version(ffmpeg_path: str) -> str:
    out = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True, check=True)
    return out.stdout.splitlines()[0].strip()


def _run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (args={args}): {result.stderr[-2000:]}")


def codec_chain(
    waveform: np.ndarray,
    sr: int,
    dose: float,
    seed: int | None = None,
    codec: str = "opus",
    ffmpeg_path: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Round-trip `waveform` through `codec` at `dose` kbps, aligned back to the
    original length/offset.

    Args:
        waveform: (n,) mono float32 in [-1, 1].
        sr: sample rate (Hz).
        dose: target bitrate in kbps. Lower = more severe compression.
        seed: accepted for API uniformity across transform families; unused --
            codec encoding at fixed (waveform, sr, dose, codec) is already
            deterministic (single-threaded, no RNG involved).
        codec: one of "opus", "aac", "mp3".
        ffmpeg_path: override the resolved ffmpeg path (mainly for tests).

    Returns:
        (aligned_waveform (n,) float32, provenance dict).
    """
    if codec not in FFMPEG_ENCODER:
        raise ValueError(f"codec must be one of {sorted(FFMPEG_ENCODER)}, got {codec!r}")
    lo, hi = BITRATE_RANGE_KBPS[codec]
    if not (lo <= dose <= hi):
        raise ValueError(f"{codec} dose (bitrate) must be in [{lo}, {hi}] kbps, got {dose}")

    ffmpeg_path = ffmpeg_path or check_ffmpeg_available()
    waveform = np.asarray(waveform, dtype=np.float32)
    encoder = FFMPEG_ENCODER[codec]
    ext = FFMPEG_CONTAINER_EXT[codec]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_wav, enc_path, dec_wav = td / "in.wav", td / f"enc.{ext}", td / "dec.wav"
        sf.write(in_wav, waveform, sr)
        _run_ffmpeg([
            ffmpeg_path, "-y", "-threads", "1", "-i", str(in_wav),
            "-c:a", encoder, "-b:a", f"{dose}k", str(enc_path),
        ])
        _run_ffmpeg([
            ffmpeg_path, "-y", "-threads", "1", "-i", str(enc_path),
            "-ar", str(sr), "-ac", "1", str(dec_wav),
        ])
        decoded, dec_sr = sf.read(dec_wav, dtype="float32")

    aligned = align_to_reference(waveform, decoded)
    provenance = make_provenance(
        transform="codec_chain", family="codec", dose=float(dose), dose_unit="kbps",
        seed=seed, sr=sr, codec=codec, ffmpeg_encoder=encoder,
        ffmpeg_version=_ffmpeg_version(ffmpeg_path),
    )
    return aligned, provenance
