"""One Dataset for all corpora, driven entirely by unified manifests.

e002 addition: with `degrade=True`, each item also carries `waveform_deg`, a
channel-degraded copy of the SAME crop (label-independent). collate stacks it
into batch["waveform_deg"] when present. With degrade=False the output is
byte-identical to the e001 behaviour.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import Dataset

from .audio_io import resample_linear, crop_or_pad
from .manifest import ManifestRow, read_manifest
from .safe_audio import load_audio_strict, load_allowlist, AudioReadError


class UnifiedAudioDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[ManifestRow],
        data_root: str | Path,
        sample_rate: int = 16000,
        duration_seconds: float = 4.0,
        random_crop: bool = True,
        corpus_vocab: Optional[dict[str, int]] = None,
        bona_source_vocab: Optional[dict[str, int]] = None,
        degrade: bool = False,
    ) -> None:
        if not rows:
            raise ValueError("UnifiedAudioDataset received zero rows")
        self.rows = list(rows)
        self.data_root = Path(data_root)
        self.sample_rate = sample_rate
        self.num_samples = int(round(sample_rate * duration_seconds))
        self.random_crop = random_crop
        self.degrade = degrade

        self.corpus_vocab = corpus_vocab or self._build_vocab([r.corpus for r in self.rows])
        self.bona_source_vocab = bona_source_vocab or self._build_vocab(
            [r.bona_fide_source for r in self.rows]
        )
        self.allowlist = load_allowlist()
        self.allowlist_skips = 0

    @staticmethod
    def _build_vocab(values: list[str]) -> dict[str, int]:
        return {v: i for i, v in enumerate(sorted(set(values)))}

    @classmethod
    def from_manifests(
        cls,
        manifest_paths: Sequence[str | Path],
        data_root: str | Path,
        split: str,
        **kwargs,
    ) -> "UnifiedAudioDataset":
        rows: list[ManifestRow] = []
        for mp in manifest_paths:
            rows.extend(read_manifest(mp, splits=[split]))
        return cls(rows, data_root=data_root, **kwargs)

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (self.data_root / p)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        for _offset in range(0, 50):
            try_row = self.rows[(index + _offset) % len(self.rows)]
            result = load_audio_strict(self._resolve(try_row.path), try_row.utt_id, self.allowlist)
            if result is not None:
                x, orig_sr = result
                row = try_row
                break
            # allowlisted-bad file (configs/known_bad.txt): counted, deterministic
            # substitution to the next row by index. Any OTHER failure raises
            # AudioReadError out of load_audio_strict and is NOT caught here (audit §5).
            self.allowlist_skips += 1
        else:
            raise AudioReadError(
                f"[unified_dataset] 50 consecutive allowlisted-bad files from index {index}; "
                "check configs/known_bad.txt for an over-broad entry."
            )
        waveform = torch.from_numpy(x)
        mono = resample_linear(waveform, orig_sr, self.sample_rate)
        mono = crop_or_pad(mono, self.num_samples, random_crop=self.random_crop)
        mono = mono.clamp(-1.0, 1.0)

        item = {
            "waveform": mono,
            "target": torch.tensor(row.target, dtype=torch.float32),
            "target_long": torch.tensor(row.target, dtype=torch.long),
            "corpus_id": torch.tensor(self.corpus_vocab[row.corpus], dtype=torch.long),
            "bona_source_id": torch.tensor(
                self.bona_source_vocab.get(row.bona_fide_source, -1), dtype=torch.long
            ),
            "corpus": row.corpus,
            "bona_fide_source": row.bona_fide_source,
            "attack": row.attack,
            "path": str(row.path),
        }

        if self.degrade:
            from .channel_aug import degrade_waveform
            item["waveform_deg"] = degrade_waveform(mono).clamp(-1.0, 1.0)

        return item


def collate_unified(items: Sequence[dict]) -> dict:
    tensor_keys = ["waveform", "target", "target_long", "corpus_id", "bona_source_id"]
    batch: dict = {}
    for key in tensor_keys:
        batch[key] = torch.stack([it[key] for it in items])
    if "waveform_deg" in items[0]:
        batch["waveform_deg"] = torch.stack([it["waveform_deg"] for it in items])
    for key in ["corpus", "bona_fide_source", "attack", "path"]:
        batch[key] = [it[key] for it in items]
    return batch
