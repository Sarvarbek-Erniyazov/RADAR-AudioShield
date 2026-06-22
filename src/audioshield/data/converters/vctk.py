"""Convert a VCTK folder into unified bona-fide-only manifest rows.

VCTK is genuine speech only (target=0). It provides a bona-fide domain that is
distinct from DiffSSD's LibriSpeech (different speakers, mics, accents), which is
exactly what BMI needs for a well-conditioned multi-domain bona-fide batch.

Expected layout (standard VCTK-Corpus):
    <root>/wav48_silence_trimmed/pXXX/pXXX_YYY_micN.flac
or  <root>/wav48/pXXX/pXXX_YYY.wav

We split by SPEAKER into train/val/test so no speaker leaks across splits.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..manifest import ManifestRow

AUDIO_EXTS = {".wav", ".flac"}


def _speaker_split(speaker: str, val_frac: float = 0.1, test_frac: float = 0.1) -> str:
    """Deterministic per-speaker split via hashing (stable across runs)."""
    h = int(hashlib.md5(speaker.encode()).hexdigest(), 16) % 1000 / 1000.0
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


def convert(root: str | Path, path_prefix: str = "", corpus: str = "vctk") -> list[ManifestRow]:
    root = Path(root)
    audio_files = sorted(
        p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS
    )
    if not audio_files:
        raise FileNotFoundError(f"No .wav/.flac audio found under {root}")

    rows: list[ManifestRow] = []
    for p in audio_files:
        speaker = p.stem.split("_")[0]  # pXXX
        split = _speaker_split(speaker)
        rel = p.relative_to(root).as_posix()
        full_path = f"{path_prefix.rstrip('/')}/{rel}" if path_prefix else str(p)
        rows.append(
            ManifestRow(
                utt_id=f"{corpus}/{rel}",
                path=full_path,
                target=0,
                corpus=corpus,
                split=split,
                attack="bonafide",
                bona_fide_source="vctk",
            )
        )
    return rows
