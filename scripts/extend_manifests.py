"""Backfill factor metadata onto live manifests -> manifests/v2/<name>.csv.
Adds: source_id, speaker_id, generator_id, channel_id, language, platform_id.
Rules are per-corpus, derived from utt_id/path structure; unknown -> "NA", never blank.
Idempotent; originals untouched. Audit ref: §4.6/§4.8; Roadmap v3 Step 2a Commit 3.
Known, documented limitations (refinable later without schema change):
 - asvspoof5 attack collapsed to 'asvspoof5spoof' in source manifests; per-attack codes
   need the official protocol files (hook: --asvspoof5-protocol).
 - inthewild speaker names live in its meta.csv, not the manifest (hook: --itw-meta).
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import pandas as pd

NA = "NA"

def parts_of(path: str):
    return path.replace("\\", "/").split("/")

def derive(row) -> dict:
    c, utt, path, target = row["corpus"], row["utt_id"], row["path"], int(row["target"])
    attack = str(row.get("attack", NA)).strip().lower()
    src = spk = gen = ch = lang = NA
    p = parts_of(path)
    stem = Path(utt).stem
    # Known placeholder attack labels: constant-filled at manifest creation, not real
    # per-file generators (replaydf: one value vs varied path generators; inthewild:
    # uniform across wild spoofs). Scoped PER CORPUS -- "openvoicev2" is a REAL
    # generator for diffssd (25,000 rows), not a placeholder there; a global blacklist
    # silently mislabeled every diffssd openvoicev2 row generator_id="generated_speech"
    # (masked for the other 9 diffssd generators, where gen=attack below was already
    # correct so the buggy diffssd path-fallback never ran). Never let a corpus's own
    # placeholder value populate generator_id (v3 U_generator).
    PLACEHOLDER_ATTACK_BY_CORPUS = {
        "inthewild": {"openvoicev2"},
        "replaydf": {"openvoicev2"},
    }
    placeholder_attacks = PLACEHOLDER_ATTACK_BY_CORPUS.get(c, set())
    if target == 1 and attack not in ("", "na", "bonafide") and attack not in placeholder_attacks:
        gen = attack
    if c == "ai4t":
        src = re.sub(r"_\d{3,}$", "", stem)                      # -lPqD0Kj-gA_000 -> video id
    elif c == "diffssd":
        if "librispeech" in p:
            i = p.index("librispeech")
            spk = f"ls-{p[i+2]}" if len(p) > i + 2 else NA        # .../dev-clean/1272/128104/...
            src = f"ls-{p[i+2]}-{p[i+3]}" if len(p) > i + 3 else spk
            lang = "en"
        elif "ljspeech" in path.lower():
            spk, src, lang = "ljspeech", "ljspeech", "en"
        if target == 1 and gen == NA:
            try:
                gen = p[p.index("generated_speech") + 1]          # generated_speech/<generator>/...
            except (ValueError, IndexError):
                pass
        if gen == "openvoicev2":
            # generated_speech/openvoicev2/speaker_NNN/sentence_K_en-XX.wav -- speaker
            # and accent are only derivable from the path, never from the attack label.
            try:
                i = p.index("openvoicev2")
                spk = p[i + 1]
            except (ValueError, IndexError):
                pass
            m = re.search(r"(en-[a-z]+)$", stem)
            if m:
                lang = m.group(1)
    elif c == "replaydf":
        try:
            i = p.index("wav")
            ch = src = p[i+1]                                     # session/device hash
            kind = p[i+2]
            if kind == "benign":                                  # benign/<lang>/file
                lang = p[i+3]
            elif kind == "spoof":                                 # spoof/<generator>/<lang>/file
                gen = p[i+3]                                       # path is authoritative here
                lang = p[i+4] if len(p) > i+4 else NA
        except (ValueError, IndexError):
            pass
    elif c == "vctk":
        m = re.match(r"(p\d+)", stem)
        if m: spk = src = m.group(1)
        lang = "en"
    elif c == "asvspoof5":
        lang = "en"                                               # spk needs protocol files
    elif c == "fakeorreal":
        lang = "en"
    elif c == "inthewild":
        lang = "en"                                               # spk needs meta.csv hook
    elif c == "mlaad":
        # fake/<lang>/<generator>/<book>_<chap>_fNNNNNN.wav (10_MLAAD, when manifested)
        try:
            i = p.index("fake"); lang, gen = p[i+1], p[i+2]
            src = re.sub(r"_f\d+$", "", stem)
        except (ValueError, IndexError):
            pass
    if src == NA and spk != NA:
        src = spk
    return dict(source_id=src, speaker_id=spk, generator_id=gen,
                channel_id=ch, language=lang, platform_id=NA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", default="manifests")
    ap.add_argument("--out-dir", default="manifests/v2")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    report = []
    for csv in sorted(Path(args.manifest_dir).glob("*.csv")):
        df = pd.read_csv(csv, dtype=str, keep_default_na=False)
        need = {"utt_id", "path", "target", "corpus", "split", "attack"}
        if not need.issubset(df.columns):
            print(f"[skip] {csv.name}: columns {sorted(set(df.columns))} lack {sorted(need - set(df.columns))}")
            continue
        meta = df.apply(derive, axis=1, result_type="expand")
        v2 = pd.concat([df, meta], axis=1)
        assert not v2[meta.columns].isna().any().any(), f"{csv.name}: empty factor cell produced"
        v2.to_csv(out / csv.name, index=False, lineterminator="\n")
        cov = {col: f"{(v2[col] != 'NA').mean():.0%}" for col in meta.columns}
        report.append(f"{csv.name:18s} rows={len(v2):7d} coverage={cov}")
    print("\n".join(report) if report else "no manifests processed")

if __name__ == "__main__":
    sys.exit(main())
