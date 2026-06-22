"""Stable label vocabularies used by the DiffSSD implementation."""

KNOWN_METHODS = [
    "librispeech",
    "ljspeech",
    "diffgantts",
    "elevenlabs",
    "gradtts",
    "openvoicev2",
    "playht",
    "prodiff",
    "unitspeech",
    "wavegrad2",
    "xttsv2",
    "yourtts",
]

METHOD_TO_ID = {name: idx for idx, name in enumerate(KNOWN_METHODS)}
ID_TO_METHOD = {idx: name for name, idx in METHOD_TO_ID.items()}

CATEGORIES = ["real", "pretrained", "zeroshot"]
CATEGORY_TO_ID = {name: idx for idx, name in enumerate(CATEGORIES)}
ID_TO_CATEGORY = {idx: name for name, idx in CATEGORY_TO_ID.items()}

SOURCES = ["real", "opensource", "commercial"]
SOURCE_TO_ID = {name: idx for idx, name in enumerate(SOURCES)}
ID_TO_SOURCE = {idx: name for name, idx in SOURCE_TO_ID.items()}

# The router is conditioned on the media state estimated from the current view.
# DiffSSD itself is clean speech, so transformed states are created online.
TRANSFORM_STATES = [
    "clean",
    "codec_proxy",
    "resampled",
    "rir_convolved",
    "replay_simulated",
    "noise_mixed",
]

TRANSFORM_STATE_TO_ID = {name: idx for idx, name in enumerate(TRANSFORM_STATES)}
ID_TO_TRANSFORM_STATE = {idx: name for name, idx in TRANSFORM_STATE_TO_ID.items()}

