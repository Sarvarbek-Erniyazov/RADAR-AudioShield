"""Build the MLAAD v2 manifest directly from `_MASTER_MANIFEST.tsv`.

Usage:
    python scripts/build_mlaad_manifest.py --data-root .. --out manifests/v2/mlaad.csv.gz

MLAAD's raw audio is deleted per-language right after embedding (see
`_mlaad_pipeline.py` in the dataset root), so this reads the manifest bookkeeping
file rather than walking a (mostly absent) audio tree. See
src/audioshield/data/converters/mlaad.py for the conversion + factor-derivation logic.

Output is gzip-compressed (456k rows, ~120MB plain -> a few MB compressed) --
too large to keep as a plain-text blob in git. write_manifest_gz below writes
it deterministically (fixed mtime=0, fixed compresslevel, no filename embedded
in the gzip header since it compresses in-memory bytes rather than wrapping a
named file handle) so re-running this script against the same input rows
produces a byte-identical .gz every time.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from audioshield.data.converters import mlaad
from audioshield.data.manifest import FIELDNAMES, ManifestRow, read_manifest, summarize


def write_manifest_gz(rows: Iterable[ManifestRow], out_path: str | Path, compresslevel: int = 9) -> int:
    """Write manifest rows to a deterministically gzip-compressed .gz file.

    Builds the plain-text CSV in memory (a StringIO, never touching disk
    uncompressed) and compresses it via `gzip.compress(..., mtime=0)` --
    fixed mtime and fixed compresslevel, and no filename in the gzip header
    (gzip.compress operates on an in-memory buffer with no `.name`, unlike
    wrapping a real file handle), so the same input rows always produce the
    exact same output bytes.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=FIELDNAMES)
    writer.writeheader()
    n = 0
    for row in rows:
        row.validate()
        writer.writerow(asdict(row))
        n += 1
    compressed = gzip.compress(buf.getvalue().encode("utf-8"), compresslevel=compresslevel, mtime=0)
    out_path.write_bytes(compressed)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="..",
                    help="Root that dataset paths resolve from.")
    ap.add_argument("--mlaad-folder", default="10_MLAAD")
    ap.add_argument("--out", default="manifests/v2/mlaad.csv.gz")
    ap.add_argument("--revision", default=None,
                    help="If set, assert every manifest row is pinned to this HF revision.")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    master = data_root / args.mlaad_folder / "_MASTER_MANIFEST.tsv"
    assert master.exists(), f"missing {master} -- run the MLAAD HF pull first"

    rows = mlaad.convert(
        master,
        path_prefix=f"datasets/{args.mlaad_folder}",
        revision=args.revision,
    )
    n = len(rows)
    assert n > 0, "0 input rows -- refusing to write an empty manifest"

    n_spoof = sum(r.target for r in rows)
    spoof_frac = n_spoof / n
    assert spoof_frac == 1.0, (
        f"MLAAD must be spoof-only (target=1 for every row); got spoof_frac={spoof_frac:.4f} "
        f"({n_spoof}/{n}) -- failing loudly rather than writing a corrupted manifest"
    )

    out_path = Path(args.out)
    write_manifest_gz(rows, out_path)
    summary = summarize(read_manifest(out_path))
    print(f"[ok] mlaad: wrote {n} rows -> {out_path}")
    print(f"     spoof_frac={spoof_frac:.4f} (gate: == 1.0)")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
