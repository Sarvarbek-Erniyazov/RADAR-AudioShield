"""Build the ODSS_CLEAN v2 manifest.

Usage:
    python scripts/build_odss_manifest.py --data-root .. --out manifests/v2/odssclean.csv

See src/audioshield/data/converters/odss_clean.py for the natural/vctk exclusion
rule (bona-only confound vs. the standalone VCTK corpus) and factor derivation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audioshield.data.converters import odss_clean
from audioshield.data.manifest import read_manifest, summarize, write_manifest

# Historically observed spoof_frac for ODSS_CLEAN (see _mix_sweep_v2.log / .json,
# docs/mix_sweep_v2_findings.md): 18,993 spoof / 26,954 total = 0.7046. Used as a
# regression guard, not a hardcoded label -- both classes must be present and the
# ratio must stay in-band, or something changed under us (different exclusion,
# missing files, corrupted download).
EXPECTED_SPOOF_FRAC = 0.70
SPOOF_FRAC_TOLERANCE = 0.05


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="..",
                    help="Root that dataset paths resolve from.")
    ap.add_argument("--odss-folder", default="12_ODSS")
    ap.add_argument("--out", default="manifests/v2/odssclean.csv")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    odss_root = data_root / args.odss_folder
    assert odss_root.exists(), f"missing {odss_root} -- pull/extract ODSS first"

    rows = odss_clean.convert(odss_root, path_prefix=f"datasets/{args.odss_folder}")
    n = len(rows)
    assert n > 0, "0 input rows -- refusing to write an empty manifest"

    n_spoof = sum(r.target for r in rows)
    spoof_frac = n_spoof / n
    assert 0.0 < spoof_frac < 1.0, (
        f"ODSS_CLEAN must contain both classes for LR training; got spoof_frac={spoof_frac:.4f}"
    )
    assert abs(spoof_frac - EXPECTED_SPOOF_FRAC) < SPOOF_FRAC_TOLERANCE, (
        f"ODSS_CLEAN spoof_frac={spoof_frac:.4f} outside expected band "
        f"{EXPECTED_SPOOF_FRAC} +/- {SPOOF_FRAC_TOLERANCE} (see _mix_sweep_v2 provenance) "
        "-- failing loudly rather than writing a manifest that silently drifted"
    )
    assert not any(r.bona_fide_source == "vctk" for r in rows), (
        "odss/natural/vctk rows leaked through the exclusion filter"
    )

    out_path = Path(args.out)
    write_manifest(rows, out_path)
    summary = summarize(read_manifest(out_path))
    print(f"[ok] odssclean: wrote {n} rows -> {out_path}")
    print(f"     spoof_frac={spoof_frac:.4f} (gate: in [{EXPECTED_SPOOF_FRAC - SPOOF_FRAC_TOLERANCE:.2f}, "
          f"{EXPECTED_SPOOF_FRAC + SPOOF_FRAC_TOLERANCE:.2f}])")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
