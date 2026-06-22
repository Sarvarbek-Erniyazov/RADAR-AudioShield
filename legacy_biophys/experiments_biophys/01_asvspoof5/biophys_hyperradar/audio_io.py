"""Audio loading helpers with a stdlib WAV path and optional richer backends."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def load_audio(path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
    """Load an audio file as ``[channels, samples]`` float32 in [-1, 1].

    WAV files are handled with Python's stdlib so the generated DiffSSD subset
    can be read without torchaudio. FLAC/MP3 files require torchaudio or
    soundfile, which is relevant after the user adds real LibriSpeech/LJSpeech.
    """

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".wav":
        try:
            return _load_wav_stdlib(path)
        except Exception as wav_error:
            return _load_with_optional_backend(path, wav_error)

    return _load_with_optional_backend(path)


def _load_with_optional_backend(
    path: Path,
    primary_error: Optional[Exception] = None,
) -> Tuple[torch.Tensor, int]:
    try:
        import torchaudio  # type: ignore

        waveform, sample_rate = torchaudio.load(str(path))
        return waveform.to(torch.float32), int(sample_rate)
    except Exception as torchaudio_error:
        try:
            import soundfile as sf  # type: ignore

            data, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
            waveform = torch.from_numpy(data.T.copy())
            return waveform, int(sample_rate)
        except Exception as soundfile_error:
            prefix = f"primary error: {primary_error}; " if primary_error is not None else ""
            raise RuntimeError(
                f"Could not load {path}. {prefix}"
                "RIFF WAV is supported without optional deps; FLAC/MP3/mislabeled WAV "
                "requires a working torchaudio or soundfile install. "
                f"torchaudio error: {torchaudio_error}; soundfile error: {soundfile_error}"
            ) from soundfile_error


def _load_wav_stdlib(path: Path) -> Tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        frames = reader.readframes(reader.getnframes())

    if sample_width == 1:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        signed = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        data = signed.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width {sample_width} bytes for {path}")

    waveform = torch.from_numpy(data.reshape(-1, channels).T.copy())
    return waveform, int(sample_rate)


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim != 2:
        raise ValueError(f"Expected [channels, samples], got {tuple(waveform.shape)}")
    return waveform.mean(dim=0)


def resample_linear(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Dependency-free linear resampler for model input and augmentation."""

    if orig_sr == target_sr:
        return waveform
    if waveform.ndim == 1:
        shaped = waveform[None, None, :]
        squeeze = True
    elif waveform.ndim == 2:
        shaped = waveform[None, :, :]
        squeeze = False
    else:
        raise ValueError(f"Expected 1D or 2D waveform, got {tuple(waveform.shape)}")

    target_len = max(1, round(shaped.shape[-1] * float(target_sr) / float(orig_sr)))
    out = F.interpolate(shaped, size=target_len, mode="linear", align_corners=False)
    return out[0, 0] if squeeze else out[0]


def crop_or_pad(
    waveform: torch.Tensor,
    num_samples: int,
    random_crop: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Return a fixed-length 1D waveform."""

    if waveform.ndim != 1:
        raise ValueError(f"Expected mono waveform [samples], got {tuple(waveform.shape)}")

    current = waveform.numel()
    if current == num_samples:
        return waveform
    if current < num_samples:
        return F.pad(waveform, (0, num_samples - current))

    if random_crop:
        start = int(torch.randint(0, current - num_samples + 1, (1,), generator=generator).item())
    else:
        start = max(0, (current - num_samples) // 2)
    return waveform[start : start + num_samples]
