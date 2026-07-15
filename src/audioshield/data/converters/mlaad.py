"""Convert MLAAD's `_MASTER_MANIFEST.tsv` into unified manifest rows.

MLAAD audio is pulled from the HuggingFace dataset repo `mueller91/MLAAD` and, per
the project's disk-space lifecycle (`_mlaad_pipeline.py`), deleted language-by-language
right after its frozen embedding is cached -- only the manifest/hash bookkeeping
survives on disk. We therefore build the manifest from that bookkeeping file rather
than walking a (mostly absent) audio tree:

    _MASTER_MANIFEST.tsv columns: rel_path, size_bytes, sha256, hf_revision
    rel_path looks like:          fake/<lang>/<generator>/<book>_<chap>_fNNNNNN.wav

`rel_path` is preserved verbatim in `hf_path` so a specific file can be re-fetched
later via `hf_hub_download(repo_id="mueller91/MLAAD", filename=hf_path, revision=...)`
without re-pulling the whole (multi-hundred-GB) dataset.

MLAAD is spoof-only (every row target=1); generator_id/language/source_id are derived
from the path via `extend_manifests.derive()`'s existing "mlaad" branch so this stays
consistent with the parsing already pinned by tests/test_commit3.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..manifest import ManifestRow

_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts"


def _load_derive():
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    from extend_manifests import derive  # noqa: PLC0415
    return derive


def convert(
    master_manifest_path: str | Path,
    path_prefix: str = "datasets/10_MLAAD",
    corpus: str = "mlaad",
    split: str = "train",
    revision: str | None = None,
) -> list[ManifestRow]:
    """Read MLAAD's `_MASTER_MANIFEST.tsv` and return unified ManifestRows.

    Args:
        master_manifest_path: path to `10_MLAAD/_MASTER_MANIFEST.tsv`.
        path_prefix: dataset folder the manifest `path` column resolves from.
        corpus: unified corpus id (must stay "mlaad" -- pinned by derive()'s branch).
        split: every row gets this split; MLAAD has no external train/val/test file.
        revision: expected HF dataset revision. If set, mismatched rows raise --
            guards against silently mixing manifest entries pinned to different
            upstream snapshots.
    """
    master_manifest_path = Path(master_manifest_path)
    lines = master_manifest_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{master_manifest_path}: empty -- refusing to write a 0-row manifest")

    derive = _load_derive()
    rows: list[ManifestRow] = []
    n_dropped_malformed = 0
    for line in lines:
        cells = line.split("\t")
        if len(cells) != 4:
            n_dropped_malformed += 1
            continue
        rel_path, _size, _sha256, rev = cells
        if revision is not None and rev != revision:
            raise ValueError(
                f"{master_manifest_path}: row {rel_path!r} pinned to revision {rev!r}, "
                f"expected {revision!r} -- refusing to mix snapshots"
            )
        utt_id = f"{corpus}/{rel_path}"
        full_path = f"{path_prefix.rstrip('/')}/{rel_path}"
        meta = derive(dict(corpus=corpus, utt_id=utt_id, path=full_path, target="1", attack="na"))
        rows.append(
            ManifestRow(
                utt_id=utt_id,
                path=full_path,
                target=1,
                corpus=corpus,
                split=split,
                attack="na",
                bona_fide_source="na",
                source_id=meta["source_id"],
                speaker_id=meta["speaker_id"],
                generator_id=meta["generator_id"],
                channel_id=meta["channel_id"],
                language=meta["language"],
                platform_id=meta["platform_id"],
                hf_path=rel_path,
            )
        )
    if n_dropped_malformed:
        print(f"[mlaad] {master_manifest_path}: dropped {n_dropped_malformed} malformed rows "
              f"out of {len(lines)} data rows -- counted, not silent")
    if not rows:
        raise ValueError(f"{master_manifest_path}: 0 usable rows -- refusing to write manifest")
    return rows
