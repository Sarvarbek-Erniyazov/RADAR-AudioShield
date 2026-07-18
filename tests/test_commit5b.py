"""5b: ManifestRow carries factor fields from v2 manifests; original manifests still
load with fields defaulted to NA (no crash). Audit §4.6 wiring; Roadmap v3 Step 2a."""
import csv, sys
from pathlib import Path
# import the training repo's manifest module by path (adjust if layout differs)
from audioshield.data.manifest import read_manifest, ManifestRow

CORE = ["utt_id","path","target","corpus","split","attack","bona_fide_source"]
V2 = CORE + ["source_id","speaker_id","generator_id","channel_id","language","platform_id"]

def _write(p, header, row):
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerow(row)

def test_v2_manifest_exposes_factor_fields(tmp_path):
    p = tmp_path / "v2.csv"
    _write(p, V2, ["u/1.wav","d/u/1.wav","0","cA","train","bonafide","vctk","p225","p225","NA","NA","en","NA"])
    r = read_manifest(p)[0]
    assert r.source_id == "p225" and r.speaker_id == "p225" and r.language == "en"

def test_original_manifest_loads_with_na_defaults(tmp_path):
    p = tmp_path / "orig.csv"
    _write(p, CORE, ["u/1.wav","d/u/1.wav","1","cB","train","openvoicev2","na"])
    r = read_manifest(p)[0]
    assert r.source_id == "NA" and r.generator_id == "NA"      # defaulted, no crash
    assert r.corpus == "cB" and r.target == 1                   # core still parsed

def test_missing_core_column_still_raises(tmp_path):
    p = tmp_path / "broken.csv"
    _write(p, [c for c in CORE if c != "target"], ["u/1.wav","d/u/1.wav","cB","train","x","na"])
    try:
        read_manifest(p); assert False, "should have raised on missing core column"
    except (ValueError, KeyError):
        pass
