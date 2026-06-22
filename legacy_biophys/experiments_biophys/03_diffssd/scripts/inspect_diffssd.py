"""Inspect the local DiffSSD folder without requiring torch."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default=str(Path(__file__).resolve().parents[2] / "datasets" / "03_DiffSSD"),
    )
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.dataset_root)
    split_file = root / "train_val_test_splits.csv"
    if not split_file.exists():
        raise SystemExit(f"Missing {split_file}")

    total = 0
    counters = {
        "set": Counter(),
        "target": Counter(),
        "method_name": Counter(),
        "category": Counter(),
        "source": Counter(),
        "exists": Counter(),
    }
    exists_by_split_target: dict[tuple[str, str], Counter] = defaultdict(Counter)
    missing_examples = []

    with split_file.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total += 1
            exists = (root / row["filename"]).exists()
            for key in ["set", "target", "method_name", "category", "source"]:
                counters[key][row[key]] += 1
            counters["exists"][str(exists)] += 1
            exists_by_split_target[(row["set"], row["target"])][str(exists)] += 1
            if not exists and len(missing_examples) < args.examples:
                missing_examples.append(row["filename"])

    print(f"Dataset root: {root}")
    print(f"Rows: {total}")
    for key, counter in counters.items():
        print(f"\n{key}:")
        for value, count in counter.most_common():
            print(f"  {value}: {count}")

    print("\nExisting files by split and target:")
    for key in sorted(exists_by_split_target):
        print(f"  set={key[0]} target={key[1]}: {dict(exists_by_split_target[key])}")

    if missing_examples:
        print("\nMissing examples:")
        for filename in missing_examples:
            print(f"  {filename}")


if __name__ == "__main__":
    main()

