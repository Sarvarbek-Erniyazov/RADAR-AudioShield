"""Provenance helpers shared by every transform family: file hashing (for
asset ids in a provenance dict) and a common provenance-dict shape so result
bundles written by different transform families are uniform."""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """sha256 of a file's bytes, chunked (safe for large RIR/audio files)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def make_provenance(
    transform: str,
    family: str,
    dose: float,
    dose_unit: str,
    seed: int | None,
    sr: int,
    **extra,
) -> dict:
    """Common provenance-dict shape. `extra` carries family-specific fields
    (codec name + ffmpeg version, chosen RIR asset id/hash + asset-dir
    fingerprint, noise type, ...) -- always included verbatim, never dropped,
    so a result bundle can always answer "exactly what produced this file"
    without guessing which extra keys a given family provides.
    """
    prov = dict(
        transform=transform,
        family=family,
        dose=dose,
        dose_unit=dose_unit,
        seed=seed,
        sr=sr,
    )
    prov.update(extra)
    return prov
