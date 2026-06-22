"""Build a DiffSSD-style test manifest for AI4T from real/ and fake/ folders.

AI4T has no split file -- just two folders of .wav:
  AI4T_dataset/real/*.wav  -> target 0 (bonafide)
  AI4T_dataset/fake/*.wav  -> target 1 (spoof)

AI4T is small (~279 clips); used as a CROSS-DATASET TEST set, not training.
method_name uses the binary placeholder convention (librispeech / openvoicev2).

Usage:
    python build_manifest.py --ai4t-root datasets/05_AI4T/AI4T_dataset
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ai4t-root", required=True)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.ai4t_root)
    out = Path(args.output) if args.output else root / "train_val_test_splits.csv"
    fields = ["filename", "method_name", "category", "source", "set", "target"]

    rows = []
    for wav in sorted((root / "real").glob("*.wav")):
        rows.append({"filename": f"real/{wav.name}", "method_name": "librispeech",
                     "category": "real", "source": "real", "set": "test", "target": "0"})
    for wav in sorted((root / "fake").glob("*.wav")):
        rows.append({"filename": f"fake/{wav.name}", "method_name": "openvoicev2",
                     "category": "zeroshot", "source": "opensource", "set": "test", "target": "1"})

    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    n0 = sum(1 for r in rows if r["target"] == "0")
    n1 = sum(1 for r in rows if r["target"] == "1")
    print(f"Wrote manifest: {out}")
    print(f"  rows: {len(rows)}  (real/target0: {n0}, fake/target1: {n1})")


if __name__ == "__main__":
    main()
