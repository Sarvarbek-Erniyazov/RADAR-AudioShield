"""Fail-fast audio loading. Audit ref: §5 — the loader previously substituted the
next rows on read failure, silently corrupting labels/subspaces. Policy now:
unreadable audio raises AudioReadError naming the row, unless the utt_id is in an
explicit, versioned allowlist (configs/known_bad.txt) — allowlisted skips are
counted and reported, never silent."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import soundfile as sf

class AudioReadError(RuntimeError):
    pass

def load_allowlist(path: str = "configs/known_bad.txt") -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    return {ln.split("#")[0].strip() for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.split("#")[0].strip()}

def load_audio_strict(filepath: str | Path, utt_id: str, allowlist: set[str]):
    """Returns (waveform float32 mono, sr) or None if allowlisted-bad.
    Raises AudioReadError on any unlisted failure."""
    try:
        x, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if x.size == 0:
            raise ValueError("zero-length audio")
        if not np.isfinite(x).all():
            raise ValueError("non-finite samples")
        return x, sr
    except Exception as e:
        if utt_id in allowlist:
            return None
        raise AudioReadError(
            f"Unreadable audio (utt_id={utt_id}, path={filepath}): {type(e).__name__}: {e}. "
            f"If genuinely corrupt at source, add utt_id to configs/known_bad.txt with a comment."
        ) from e
