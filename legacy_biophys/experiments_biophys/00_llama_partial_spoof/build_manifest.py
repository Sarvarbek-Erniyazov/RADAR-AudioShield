"""Build a DiffSSD-style manifest for LlamaPartialSpoof.

LlamaPartialSpoof has two archives (already extracted):
  R01TTS.0.a/  -> bonafide + fully-fake + partial(crossfade)
  R01TTS.0.b/  -> partial(cut/paste, overlap) -- spoof only

Labels: label_R01TTS.0.a.txt / label_R01TTS.0.b.txt
  <id> <duration> <utterance-label> <segments...>
We use the utterance-level label (col 3): bonafide -> 0, spoof -> 1.
(The detector is utterance-level, so partial spoofs count as spoof.)

The dataset has no train/val split, so we create a STRATIFIED 80/20 split
(bonafide and spoof split proportionally) with a fixed seed for reproducibility.

method_name uses the binary placeholder convention (librispeech / openvoicev2)
so it matches the core label vocabulary.

Usage:
    python build_manifest.py --llama-root datasets/00_LlamaPartialSpoof
"""
from __future__ import annotations
import argparse
import csv
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--llama-root", required=True)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--output", default=None)
    return p.parse_args()


def read_label_file(label_path: Path, audio_subdir: str):
    """Yield (relative_filename, utterance_label) for each labelled clip."""
    rows = []
    with label_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            clip_id = parts[0]
            utt_label = parts[2].lower()   # col 3 = utterance-level label
            if utt_label not in ("bonafide", "spoof"):
                continue
            rows.append((f"{audio_subdir}/{clip_id}.wav", utt_label))
    return rows


def main():
    args = parse_args()
    root = Path(args.llama_root)
    out = Path(args.output) if args.output else root / "train_val_test_splits.csv"

    all_rows = []
    all_rows += read_label_file(root / "label_R01TTS.0.a.txt", "R01TTS.0.a")
    all_rows += read_label_file(root / "label_R01TTS.0.b.txt", "R01TTS.0.b")

    # Split bonafide and spoof separately (stratified)
    bona = [r for r in all_rows if r[1] == "bonafide"]
    spoof = [r for r in all_rows if r[1] == "spoof"]
    rng = random.Random(args.seed)
    rng.shuffle(bona)
    rng.shuffle(spoof)

    def split(items):
        n_val = int(round(len(items) * args.val_fraction))
        return items[n_val:], items[:n_val]   # train, val

    bona_tr, bona_val = split(bona)
    spoof_tr, spoof_val = split(spoof)

    def to_record(fn, label, split_name):
        if label == "bonafide":
            return {"filename": fn, "method_name": "librispeech", "category": "real",
                    "source": "real", "set": split_name, "target": "0"}
        return {"filename": fn, "method_name": "openvoicev2", "category": "zeroshot",
                "source": "opensource", "set": split_name, "target": "1"}

    records = []
    for fn, lab in bona_tr + spoof_tr:
        records.append(to_record(fn, lab, "train"))
    for fn, lab in bona_val + spoof_val:
        records.append(to_record(fn, lab, "val"))

    fields = ["filename", "method_name", "category", "source", "set", "target"]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    print(f"Wrote manifest: {out}")
    print(f"  total: {len(records)}")
    print(f"  train: bonafide={len(bona_tr)} spoof={len(spoof_tr)}")
    print(f"  val:   bonafide={len(bona_val)} spoof={len(spoof_val)}")


if __name__ == "__main__":
    main()
