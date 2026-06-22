"""Build unified manifests from the canonical RADAR CSVs.

Usage:
    python scripts/build_manifests.py --all
    python scripts/build_manifests.py --corpora diffssd asvspoof5
    python scripts/build_manifests.py --vctk-root datasets/09_VCTK
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audioshield.data.converters import diffssd_style, vctk
from audioshield.data.manifest import write_manifest, read_manifest, summarize

# corpus id -> (dataset folder, csv relative path, split_override)
# split_override forces eval-only corpora to "test" regardless of their 'set' column.
DIFFSSD_STYLE = {
    "diffssd":    ("datasets/03_DiffSSD",   "train_val_test_splits.csv", None),
    "fakeorreal": ("datasets/07_FakeOrReal/for-original", "train_val_test_splits.csv", None),
    "asvspoof5":  ("datasets/01_ASVspoof5", "train_val_test_splits.csv", None),
    "inthewild":  ("datasets/02_In-the-Wild/release_in_the_wild", "train_val_test_splits.csv", "test"),
    "replaydf":   ("datasets/04_ReplayDF",  "train_val_test_splits.csv", "test"),
    "ai4t":       ("datasets/05_AI4T/AI4T_dataset_seg", "train_val_test_splits.csv", "test"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="..",
                    help="Root that dataset paths resolve from (default: repo parent of new_model).")
    ap.add_argument("--out-dir", default="manifests")
    ap.add_argument("--corpora", nargs="*", default=None,
                    help="Subset of diffssd-style corpora to build (default: all).")
    ap.add_argument("--all", action="store_true", help="Build all diffssd-style corpora.")
    ap.add_argument("--vctk-root", default=None,
                    help="If set, also build the VCTK bona-fide manifest from this folder.")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    summaries: dict = {}

    targets = list(DIFFSSD_STYLE) if (args.all or not args.corpora) else args.corpora
    for corpus in targets:
        if corpus not in DIFFSSD_STYLE:
            print(f"[skip] unknown corpus {corpus}")
            continue
        folder, csv_rel, split_override = DIFFSSD_STYLE[corpus]
        csv_path = data_root / folder / csv_rel
        if not csv_path.exists():
            print(f"[skip] {corpus}: missing {csv_path}")
            continue
        rows = diffssd_style.convert(
            csv_path, corpus=corpus, path_prefix=folder, split_override=split_override
        )
        out_path = out_dir / f"{corpus}.csv"
        n = write_manifest(rows, out_path)
        summaries[corpus] = summarize(read_manifest(out_path))
        print(f"[ok] {corpus}: wrote {n} rows -> {out_path}")

    if args.vctk_root:
        vctk_rows = vctk.convert(data_root / args.vctk_root, path_prefix=args.vctk_root)
        out_path = out_dir / "vctk.csv"
        n = write_manifest(vctk_rows, out_path)
        summaries["vctk"] = summarize(read_manifest(out_path))
        print(f"[ok] vctk: wrote {n} rows -> {out_path}")

    print("\n=== SUMMARY ===")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
