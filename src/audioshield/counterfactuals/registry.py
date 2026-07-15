"""Name -> transform-callable registry, so a batch runner (or any future
caller) can look a transform up generically instead of importing every
family module directly. Every entry has the shared signature
`(waveform, sr, dose, seed, **kwargs) -> (aligned_waveform, provenance_dict)`.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .codec import codec_chain
from .noise import additive_noise_snr
from .resampling import resample_round_trip
from .reverb import replay_simulate, rir_convolve

TransformFn = Callable[..., tuple[np.ndarray, dict]]

TRANSFORMS: dict[str, TransformFn] = {
    "codec": codec_chain,
    "resample": resample_round_trip,
    "rir": rir_convolve,
    "replay": replay_simulate,
    "noise": additive_noise_snr,
}


def get_transform(name: str) -> TransformFn:
    if name not in TRANSFORMS:
        raise ValueError(f"unknown transform {name!r}; available: {sorted(TRANSFORMS)}")
    return TRANSFORMS[name]
