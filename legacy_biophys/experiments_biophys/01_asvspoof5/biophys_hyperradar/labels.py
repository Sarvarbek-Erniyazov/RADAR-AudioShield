"""Label vocabularies — ASVspoof5 experiment variant.

DIFFERS FROM CORE: adds 'bonafide' and 'asvspoof5spoof' to KNOWN_METHODS
(12 -> 14), 'spoof' to CATEGORIES, 'asvspoof5' to SOURCES. ASVspoof5 attack
ids differ train/dev, so a single binary spoof identity is used instead of
per-attack labels. CORE master is untouched; this copy is local to this
experiment only.
"""
KNOWN_METHODS = [
    "librispeech", "ljspeech", "diffgantts", "elevenlabs", "gradtts",
    "openvoicev2", "playht", "prodiff", "unitspeech", "wavegrad2",
    "xttsv2", "yourtts",
    "bonafide", "asvspoof5spoof",
]
METHOD_TO_ID = {name: idx for idx, name in enumerate(KNOWN_METHODS)}
ID_TO_METHOD = {idx: name for name, idx in METHOD_TO_ID.items()}

CATEGORIES = ["real", "pretrained", "zeroshot", "spoof"]
CATEGORY_TO_ID = {name: idx for idx, name in enumerate(CATEGORIES)}
ID_TO_CATEGORY = {idx: name for name, idx in CATEGORY_TO_ID.items()}

SOURCES = ["real", "opensource", "commercial", "asvspoof5"]
SOURCE_TO_ID = {name: idx for idx, name in enumerate(SOURCES)}
ID_TO_SOURCE = {idx: name for name, idx in SOURCE_TO_ID.items()}

TRANSFORM_STATES = [
    "clean", "codec_proxy", "resampled",
    "rir_convolved", "replay_simulated", "noise_mixed",
]
TRANSFORM_STATE_TO_ID = {name: idx for idx, name in enumerate(TRANSFORM_STATES)}
ID_TO_TRANSFORM_STATE = {idx: name for name, idx in TRANSFORM_STATE_TO_ID.items()}
