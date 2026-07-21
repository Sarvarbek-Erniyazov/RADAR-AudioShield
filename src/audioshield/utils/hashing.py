"""File-hashing helpers, deliberately stdlib-only (no torch, no audioshield.*
submodule imports) -- shared between scripts that must stay independent of
each other's heavier dependencies. scripts/extract_model_embeddings.py (GPU/
model stack) and scripts/run_reliance_modelspace.py (CPU-only reliance
analysis) both hash a checkpoint FILE with this exact function so a match
between "the hash recorded in a Phase B shard's meta" and "the hash of the
checkpoint a consumer just loaded" is meaningful and can never drift from
two independently-written implementations
(step3_modelspace_hardening_addendum.md, Finding 2).
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
