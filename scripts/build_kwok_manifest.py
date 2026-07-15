"""Build the Kwok bona-fide cross-testing v2 manifest (eval-only).

Usage:
    python scripts/build_kwok_manifest.py --data-root .. --out manifests/v2/kwokbona.csv

See src/audioshield/data/converters/kwok_bona.py for the ASVspoof-derived-subset
exclusion (leakage policy) and factor derivation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audioshield.data.converters import kwok_bona
from audioshield.data.manifest import read_manifest, summarize, write_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="..",
                    help="Root that dataset paths resolve from.")
    ap.add_argument("--kwok-folder", default="13_KWOK_BONA")
    ap.add_argument("--out", default="manifests/v2/kwokbona.csv")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    kwok_data_root = data_root / args.kwok_folder / "data"
    assert kwok_data_root.exists(), f"missing {kwok_data_root} -- pull the Kwok bona-fide pack first"

    rows = kwok_bona.convert(kwok_data_root, path_prefix=f"datasets/{args.kwok_folder}/data")
    n = len(rows)
    assert n > 0, "0 input rows -- refusing to write an empty manifest"

    n_spoof = sum(r.target for r in rows)
    spoof_frac = n_spoof / n
    assert spoof_frac == 0.0, (
        f"Kwok bona-fide pool must contain zero spoof rows; got spoof_frac={spoof_frac:.4f} "
        f"({n_spoof}/{n}) -- failing loudly rather than writing a corrupted manifest"
    )
    assert all(r.split == "test" for r in rows), "Kwok is eval-only; every row must be split=test"
    assert not any("asvspoof" in r.bona_fide_source.lower() for r in rows), (
        "an ASVspoof-derived subset leaked through the exclusion filter"
    )

    out_path = Path(args.out)
    write_manifest(rows, out_path)
    summary = summarize(read_manifest(out_path))
    print(f"[ok] kwokbona: wrote {n} rows -> {out_path}")
    print(f"     spoof_frac={spoof_frac:.4f} (gate: == 0.0); split=test for all rows")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
