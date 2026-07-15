"""Schema + integrity validation for manifests/v2. Audit ref: §4.6; Commit-3 gate."""
import sys
from pathlib import Path
import pandas as pd

REQ = ["utt_id","path","target","corpus","split","attack","bona_fide_source",
       "source_id","speaker_id","generator_id","channel_id","language","platform_id"]

def main(d="manifests/v2"):
    bad = 0
    # a manifest may be gzip-compressed (e.g. mlaad.csv.gz) -- pandas infers
    # decompression from the .gz suffix on its own, so only the glob needs updating.
    files = sorted({*Path(d).glob("*.csv"), *Path(d).glob("*.csv.gz")}, key=lambda p: p.name)
    for csv in files:
        df = pd.read_csv(csv, dtype=str, keep_default_na=False)
        errs = []
        missing = [c for c in REQ if c not in df.columns]
        if missing: errs.append(f"missing cols {missing}")
        else:
            if not df["target"].isin({"0","1"}).all(): errs.append("target outside {0,1}")
            if df["utt_id"].duplicated().any(): errs.append(f"{df['utt_id'].duplicated().sum()} dup utt_ids")
            if (df[REQ].fillna("") == "").any().any(): errs.append("empty cells (use NA)")
        warns = []
        if "generator_id" in df.columns:
            n_nogen = len(df[(df.target=="1") & (df.generator_id=="NA")])
            if n_nogen: warns.append(f"{n_nogen} spoof rows lack generator_id (no per-file label in source)")
        status = "FAIL" if errs else ("WARN" if warns else "OK  ")
        if errs: bad += 1
        msg = "; ".join(errs + warns)
        print(f"[{status}] {csv.name}: rows={len(df)}" + (f" — {msg}" if msg else ""))
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
