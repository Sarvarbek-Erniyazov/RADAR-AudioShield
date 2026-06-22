"""Convert any corpus already in the canonical RADAR CSV format.

Source schema (DiffSSD / FakeOrReal / ASVspoof5 / In-the-Wild / ReplayDF / AI4T):

    filename,method_name,category,source,set,target

We map it to the unified manifest schema. Notes baked in from data inspection:
- DiffSSD's CSV has junk lines before the real header; we skip to the line
  whose first cell is exactly "filename".
- target is already 1=spoof / 0=bona fide in every corpus.
- bona_fide_source: for bona-fide rows (target==0) we tag the GENUINE domain.
  By default that is the corpus id, except DiffSSD-real which is LibriSpeech.
- For spoof rows, bona_fide_source = "na".
- attack: we keep method_name purely for diagnostics.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from ..manifest import ManifestRow

# Genuine-domain tag for bona-fide rows, per corpus.
# DiffSSD's real speech IS LibriSpeech, so its bona-fide domain is "librispeech",
# which keeps it distinct from (future) standalone corpora and honest for BMI.
BONA_FIDE_SOURCE = {
    "diffssd": "librispeech",
    "fakeorreal": "fakeorreal_real",
    "asvspoof5": "asvspoof5_real",
    "inthewild": "inthewild_real",
    "replaydf": "replaydf_real",
    "ai4t": "ai4t_real",
}


def _find_header_index(raw_rows: list[list[str]]) -> int:
    for idx, cells in enumerate(raw_rows):
        if cells and cells[0].strip().lower() == "filename":
            return idx
    raise ValueError("Could not find a header row starting with 'filename'")


def convert(
    csv_path: str | Path,
    corpus: str,
    path_prefix: str = "",
    split_override: Optional[str] = None,
) -> list[ManifestRow]:
    """Read a canonical RADAR CSV and return unified ManifestRows.

    Args:
        csv_path: path to the corpus's train_val_test_splits.csv
        corpus: unified corpus id (e.g. "diffssd", "asvspoof5")
        path_prefix: prepended to each filename so the manifest stores a path
            that resolves from the repo/datasets root. Use the dataset folder,
            e.g. "datasets/03_DiffSSD".
        split_override: if set (e.g. "test"), force every row to this split.
            Useful for eval-only corpora whose 'set' column may be inconsistent.
    """
    csv_path = Path(csv_path)
    bona_tag = BONA_FIDE_SOURCE.get(corpus, f"{corpus}_real")

    with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        raw_rows = list(csv.reader(handle))

    header_idx = _find_header_index(raw_rows)
    header = [c.strip().lower() for c in raw_rows[header_idx]]
    col = {name: i for i, name in enumerate(header)}
    required = {"filename", "method_name", "category", "source", "set", "target"}
    missing = required - set(col)
    if missing:
        raise ValueError(f"{csv_path}: source CSV missing columns {missing}")

    rows: list[ManifestRow] = []
    for cells in raw_rows[header_idx + 1 :]:
        if not cells or len(cells) < len(header):
            continue
        filename = cells[col["filename"]].strip()
        if not filename:
            continue
        try:
            target = int(cells[col["target"]].strip())
        except ValueError:
            continue
        method = cells[col["method_name"]].strip().lower().replace("_", "")
        split = split_override or cells[col["set"]].strip().lower()

        full_path = f"{path_prefix.rstrip('/')}/{filename}" if path_prefix else filename
        bona_fide_source = bona_tag if target == 0 else "na"
        attack = "bonafide" if target == 0 else (method or "unknown")

        rows.append(
            ManifestRow(
                utt_id=f"{corpus}/{filename}",
                path=full_path,
                target=target,
                corpus=corpus,
                split=split,
                attack=attack,
                bona_fide_source=bona_fide_source,
            )
        )
    return rows
