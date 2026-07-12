"""RIR/MUSAN resolution + fingerprinting. Audit ref: §5 — a hard-coded RIR path
silently no-op'd augmentation. Policy: asset roots come from config, must exist
and be non-trivial, and a structural fingerprint goes into run_config so any two
runs can prove they augmented with identical assets."""
from __future__ import annotations
import hashlib
from pathlib import Path

class AugAssetError(RuntimeError):
    pass

def fingerprint_asset_dir(root: str | Path, exts=(".wav", ".flac"), min_files: int = 10) -> dict:
    root = Path(root)
    if not root.is_dir():
        raise AugAssetError(f"Augmentation asset dir missing: {root} — refusing silent identity augmentation.")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)
    if len(files) < min_files:
        raise AugAssetError(f"Asset dir {root} has only {len(files)} audio files (<{min_files}) — misconfigured?")
    h = hashlib.sha256()
    for p in files:
        h.update(f"{p.relative_to(root).as_posix()}|{p.stat().st_size}\n".encode())
    return {"root": str(root), "n_files": len(files), "listing_sha256": h.hexdigest()}

def resolve_aug_assets(cfg: dict) -> dict:
    """Call at startup when degradation/augmentation is enabled. Returns record for run_config."""
    aug = cfg.get("augmentation", {}) or {}
    out = {}
    for key in ("rir_root", "musan_root"):
        if aug.get(key):
            out[key] = fingerprint_asset_dir(aug[key])
    if not out:
        raise AugAssetError(
            "Degradation enabled but no augmentation.rir_root/musan_root in config — "
            "the old hard-coded-path silent no-op is forbidden (audit §5)."
        )
    return out
