"""DiffSSD manifest parsing and PyTorch dataset."""

from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset

from .audio_io import crop_or_pad, load_audio, resample_linear, to_mono
from .labels import CATEGORY_TO_ID, METHOD_TO_ID, SOURCE_TO_ID, TRANSFORM_STATE_TO_ID


@dataclass(frozen=True)
class DiffSSDRow:
    filename: str
    path: Path
    method_name: str
    category: str
    source: str
    split: str
    target: int
    exists: bool
    method_id: int
    category_id: int
    source_id: int
    speaker_id: str
    accent: str


def read_diffssd_rows(root: Union[str, Path]) -> list[DiffSSDRow]:
    root = Path(root)
    split_csv = root / "train_val_test_splits.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Missing split file: {split_csv}")

    rows: list[DiffSSDRow] = []
    with split_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            filename = record["filename"].strip()
            method = normalize_method(record["method_name"])
            category = record["category"].strip().lower()
            source = record["source"].strip().lower()
            path = root / filename
            speaker_id, accent = parse_speaker_and_accent(filename)
            rows.append(
                DiffSSDRow(
                    filename=filename,
                    path=path,
                    method_name=method,
                    category=category,
                    source=source,
                    split=record["set"].strip().lower(),
                    target=int(record["target"]),
                    exists=path.exists(),
                    method_id=METHOD_TO_ID.get(method, -1),
                    category_id=CATEGORY_TO_ID.get(category, -1),
                    source_id=SOURCE_TO_ID.get(source, -1),
                    speaker_id=speaker_id,
                    accent=accent,
                )
            )
    return rows


def normalize_method(method: str) -> str:
    method = method.strip().lower().replace("_", "")
    aliases = {
        "diffgan-tts": "diffgantts",
        "diffgantts": "diffgantts",
        "grad-tts": "gradtts",
        "gradtts": "gradtts",
        "prodiff": "prodiff",
        "pro-diff": "prodiff",
        "unit-speech": "unitspeech",
        "unitspeech": "unitspeech",
        "xtts-v2": "xttsv2",
        "xttsv2": "xttsv2",
        "your-tts": "yourtts",
        "yourtts": "yourtts",
    }
    return aliases.get(method, method)


def parse_speaker_and_accent(filename: str) -> tuple[str, str]:
    speaker_match = re.search(r"speaker_([^/\\]+)", filename)
    speaker_id = speaker_match.group(1) if speaker_match else "unknown"

    accent = "default"
    accent_match = re.search(r"_(en-[a-z]+|en-default|en-india)\.", filename)
    if accent_match:
        accent = accent_match.group(1)
    return speaker_id, accent


def filter_rows(
    rows: Iterable[DiffSSDRow],
    split: Optional[str] = None,
    existing_only: bool = True,
    require_known_labels: bool = True,
) -> list[DiffSSDRow]:
    output: list[DiffSSDRow] = []
    for row in rows:
        if split is not None and row.split != split:
            continue
        if existing_only and not row.exists:
            continue
        if require_known_labels and (row.method_id < 0 or row.category_id < 0 or row.source_id < 0):
            continue
        output.append(row)
    return output


def summarize_rows(rows: Sequence[DiffSSDRow]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for key in ["split", "target", "method_name", "category", "source"]:
        counts: dict[str, int] = {}
        for row in rows:
            value = str(getattr(row, key))
            counts[value] = counts.get(value, 0) + 1
        summary[key] = dict(sorted(counts.items()))
    summary["exists"] = {
        "true": sum(1 for row in rows if row.exists),
        "false": sum(1 for row in rows if not row.exists),
    }
    return summary


class DiffSSDDataset(Dataset):
    """DiffSSD dataset returning fixed-length mono waveforms and labels."""

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        sample_rate: int = 16000,
        duration_seconds: float = 4.0,
        max_items: Optional[int] = None,
        seed: int = 13,
        require_both_classes: bool = True,
        random_crop: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.sample_rate = sample_rate
        self.num_samples = int(round(sample_rate * duration_seconds))
        self.random_crop = random_crop

        rows = filter_rows(read_diffssd_rows(self.root), split=split, existing_only=True)
        if max_items is not None:
            rng = random.Random(seed)
            rows = rows.copy()
            rng.shuffle(rows)
            rows = rows[:max_items]
        rows = sorted(rows, key=lambda row: row.filename)

        targets = {row.target for row in rows}
        if require_both_classes and targets != {0, 1}:
            missing = sorted({0, 1} - targets)
            raise ValueError(
                f"Split '{split}' has only targets {sorted(targets)} among existing files. "
                f"Missing target(s) {missing}. DiffSSD requires you to add real_speech/ "
                "from LJ Speech and LibriSpeech before training a bona fide vs spoof detector. "
                "Use --allow-single-class-debug only for pipeline debugging."
            )
        if not rows:
            raise ValueError(f"No existing DiffSSD files found for split '{split}' under {self.root}")

        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        waveform, orig_sr = load_audio(row.path)
        mono = to_mono(waveform)
        mono = resample_linear(mono, orig_sr, self.sample_rate)
        mono = crop_or_pad(mono, self.num_samples, random_crop=self.random_crop)
        mono = mono.clamp(-1.0, 1.0)

        return {
            "waveform": mono,
            "target": torch.tensor(row.target, dtype=torch.float32),
            "target_long": torch.tensor(row.target, dtype=torch.long),
            "method_id": torch.tensor(row.method_id, dtype=torch.long),
            "category_id": torch.tensor(row.category_id, dtype=torch.long),
            "source_id": torch.tensor(row.source_id, dtype=torch.long),
            "media_state": torch.tensor(TRANSFORM_STATE_TO_ID["clean"], dtype=torch.long),
            "path": str(row.path),
            "filename": row.filename,
            "method_name": row.method_name,
            "category": row.category,
            "source": row.source,
            "split": row.split,
            "speaker_id": row.speaker_id,
            "accent": row.accent,
        }


def collate_batch(items: Sequence[dict[str, object]]) -> dict[str, object]:
    tensor_keys = [
        "waveform",
        "target",
        "target_long",
        "method_id",
        "category_id",
        "source_id",
        "media_state",
    ]
    batch: dict[str, object] = {}
    for key in tensor_keys:
        batch[key] = torch.stack([item[key] for item in items])  # type: ignore[arg-type]
    for key in [
        "path",
        "filename",
        "method_name",
        "category",
        "source",
        "split",
        "speaker_id",
        "accent",
    ]:
        batch[key] = [item[key] for item in items]
    return batch
