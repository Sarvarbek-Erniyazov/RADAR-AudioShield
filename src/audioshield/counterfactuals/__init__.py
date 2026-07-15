"""Paired waveform-counterfactual generation library (Roadmap v3 Step 4's
intervention battery -- waveform-level paired counterfactuals, dose-response).

Pure library -- the batch runner (scripts/build_counterfactuals.py) is a thin
CLI wrapper, not exercised against any real corpus by this package. See:

  codec.py       -- codec-chain round-trip (opus/aac/mp3), dose = bitrate (kbps)
  resampling.py  -- resample round-trip, dose = intermediate sample rate (Hz)
  reverb.py      -- RIR convolution + replay simulation, dose = wet/dry mix / severity in [0,1]
  noise.py       -- additive noise, dose = target SNR (dB)

Every transform shares one signature: `(waveform, sr, dose, seed, **kwargs) ->
(aligned_waveform, provenance_dict)`. `_align.align_to_reference` is what makes
"aligned to the original" true across families with otherwise-unrelated
internals (encoder delay, resampling filter delay, RIR direct-path offset).
"""
from __future__ import annotations

from ._align import align_to_reference
from .codec import FfmpegNotAvailableError, check_ffmpeg_available, codec_chain
from .noise import additive_noise_snr
from .provenance import make_provenance, sha256_file
from .registry import TRANSFORMS, get_transform
from .resampling import resample_round_trip
from .reverb import replay_simulate, rir_convolve

__all__ = [
    "align_to_reference",
    "check_ffmpeg_available", "FfmpegNotAvailableError", "codec_chain",
    "resample_round_trip",
    "rir_convolve", "replay_simulate",
    "additive_noise_snr",
    "make_provenance", "sha256_file",
    "TRANSFORMS", "get_transform",
]
