"""Unified manifest schema shared by every corpus.

One CSV per corpus, identical columns, so all loaders are corpus-agnostic:

    utt_id,path,target,corpus,split,attack,bona_fide_source

- target            : 1 = spoof, 0 = bona fide
- corpus            : lowercase corpus id (diffssd, fakeorreal, asvspoof5, vctk, ...)
- split             : train | val | test
- attack            : generator/attack tag (DIAGNOSTICS ONLY -- never a train label)
- bona_fide_source  : genuine-domain tag, used by BMI / Kwok cross-testing.
                      For spoof rows this is "na".
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

FIELDNAMES = ["utt_id", "path", "target", "corpus", "split", "attack", "bona_fide_source"]
VALID_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class ManifestRow:
    utt_id: str
    path: str
    target: int
    corpus: str
    split: str
    attack: str
    bona_fide_source: str
    source_id: str = "NA"
    speaker_id: str = "NA"
    generator_id: str = "NA"
    channel_id: str = "NA"
    language: str = "NA"
    platform_id: str = "NA"

    def validate(self) -> None:
        if self.target not in (0, 1):
            raise ValueError(f"{self.utt_id}: target must be 0/1, got {self.target}")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"{self.utt_id}: split must be {VALID_SPLITS}, got {self.split}")
        if self.target == 0 and self.bona_fide_source in ("", "na"):
            raise ValueError(f"{self.utt_id}: bona-fide row needs a real bona_fide_source")
        if not self.corpus:
            raise ValueError(f"{self.utt_id}: empty corpus")


def write_manifest(rows: Iterable[ManifestRow], out_path: str | Path) -> int:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            row.validate()
            writer.writerow(asdict(row))
            n += 1
    return n


def read_manifest(
    path: str | Path,
    splits: Optional[Iterable[str]] = None,
    corpora: Optional[Iterable[str]] = None,
) -> list[ManifestRow]:
    """Read a unified manifest, optionally filtering by split and/or corpus."""
    path = Path(path)
    split_set = set(splits) if splits is not None else None
    corpus_set = set(corpora) if corpora is not None else None

    rows: list[ManifestRow] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        CORE = {'utt_id','path','target','corpus','split','attack','bona_fide_source'}
        missing = CORE - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: manifest missing columns {missing}")
        for record in reader:
            if split_set is not None and record["split"] not in split_set:
                continue
            if corpus_set is not None and record["corpus"] not in corpus_set:
                continue
            rows.append(
                ManifestRow(
                    utt_id=record["utt_id"],
                    path=record["path"],
                    target=int(record["target"]),
                    corpus=record["corpus"],
                    split=record["split"],
                    attack=record["attack"],
                    bona_fide_source=record["bona_fide_source"],
                    source_id=record.get("source_id", "NA"),
                    speaker_id=record.get("speaker_id", "NA"),
                    generator_id=record.get("generator_id", "NA"),
                    channel_id=record.get("channel_id", "NA"),
                    language=record.get("language", "NA"),
                    platform_id=record.get("platform_id", "NA"),
                )
            )
    return rows


def summarize(rows: list[ManifestRow]) -> dict:
    """Quick counts for sanity checking a manifest."""
    out: dict = {"n": len(rows), "by_split": {}, "by_target": {0: 0, 1: 0}, "bona_fide_sources": {}}
    for r in rows:
        out["by_split"][r.split] = out["by_split"].get(r.split, 0) + 1
        out["by_target"][r.target] += 1
        if r.target == 0:
            out["bona_fide_sources"][r.bona_fide_source] = (
                out["bona_fide_sources"].get(r.bona_fide_source, 0) + 1
            )
    return out
