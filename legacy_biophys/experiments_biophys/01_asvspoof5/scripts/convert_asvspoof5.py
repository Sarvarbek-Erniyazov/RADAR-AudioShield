"""Convert ASVspoof5 train/dev TSV into a DiffSSD-style manifest.

ASVspoof5 TSV (space-separated), 0-indexed columns:
  col[1] FLAC_FILE_NAME
  col[7] ATTACK_TAG  (A01..A08 train / A11.. dev, 'bonafide' for real)
  col[8] KEY         (spoof / bonafide)

ASVspoof5 attack ids differ between train (A01-A08) and dev (A11, A16, ...),
so per-attack labels do NOT generalize and must NOT be used as a target.
We map method_name to the binary identity 'bonafide' / 'asvspoof5spoof'
(no underscore -- dataset.normalize_method strips underscores).

Usage:
    python convert_asvspoof5.py --asv-root datasets/01_ASVspoof5
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

REAL_MAPPING = {"method_name": "bonafide",       "category": "real",  "source": "real",      "target": "0"}
SPOOF_MAPPING = {"method_name": "asvspoof5spoof", "category": "spoof", "source": "asvspoof5", "target": "1"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asv-root", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.asv_root)
    output_path = Path(args.output) if args.output else root / "train_val_test_splits.csv"
    fieldnames = ["filename", "method_name", "category", "source", "set", "target"]

    splits = [
        (root / "ASVspoof5.train.tsv",       "flac_T", "train"),
        (root / "ASVspoof5.dev.track_1.tsv", "flac_D", "val"),
    ]

    counts = {}
    n_total = 0
    n_missing = 0

    with output_path.open("w", newline="", encoding="utf-8") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
        writer.writeheader()
        for tsv_path, flac_dir, set_value in splits:
            if not tsv_path.exists():
                print(f"WARNING: {tsv_path} not found, skipping")
                continue
            with tsv_path.open("r", encoding="utf-8") as f:
                for line in f:
                    cols = line.strip().split()
                    if len(cols) < 9:
                        continue
                    flac_name = cols[1]
                    key = cols[8].lower()
                    if key == "bonafide":
                        mapping = REAL_MAPPING
                    elif key == "spoof":
                        mapping = SPOOF_MAPPING
                    else:
                        continue
                    filename = f"{flac_dir}/{flac_name}.flac"
                    if not (root / filename).exists():
                        n_missing += 1
                    writer.writerow({
                        "filename": filename,
                        "method_name": mapping["method_name"],
                        "category": mapping["category"],
                        "source": mapping["source"],
                        "set": set_value,
                        "target": mapping["target"],
                    })
                    k = f"{set_value}/{key}"
                    counts[k] = counts.get(k, 0) + 1
                    n_total += 1

    print(f"Wrote manifest: {output_path}")
    print(f"  rows written : {n_total}")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")
    if n_missing:
        print(f"  WARNING: {n_missing} files not found on disk")


if __name__ == "__main__":
    main()
