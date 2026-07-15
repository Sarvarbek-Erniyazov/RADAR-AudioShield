"""Tests for the mlaad / odssclean / kwokbona v2 manifests (Step 3/4/5 measurement pool).

Validates the CSVs actually committed under manifests/v2 -- no dependency on the raw
corpora (already deleted/relocated post-embedding for at least mlaad), so these run
anywhere the repo is checked out.
"""
import re
from pathlib import Path

import pandas as pd
import pytest

from audioshield.data.manifest import FIELDNAMES
from audioshield.data.converters.kwok_bona import _identity_for as kwok_identity_for
from audioshield.data.converters.kwok_bona import _is_asvspoof_derived

MANIFEST_DIR = Path(__file__).resolve().parents[1] / "manifests" / "v2"
NEW_MANIFESTS = ["mlaad.csv.gz", "odssclean.csv", "kwokbona.csv"]


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(MANIFEST_DIR / name, dtype=str, keep_default_na=False)


@pytest.mark.parametrize("name", NEW_MANIFESTS)
def test_manifest_file_exists(name):
    assert (MANIFEST_DIR / name).exists(), f"{name} missing from {MANIFEST_DIR}"


@pytest.mark.parametrize("name", NEW_MANIFESTS)
def test_schema_conformance(name):
    df = _load(name)
    assert list(df.columns) == FIELDNAMES, f"{name}: column set/order drifted from manifest.FIELDNAMES"
    assert df["target"].isin({"0", "1"}).all(), f"{name}: target outside {{0,1}}"
    assert df["split"].isin({"train", "val", "test"}).all(), f"{name}: split outside train/val/test"
    assert not df["utt_id"].duplicated().any(), f"{name}: duplicate utt_id rows"
    # NA is the sentinel for "not derivable"; a truly empty cell is a bug (schema
    # says never blank, always "NA").
    assert (df[FIELDNAMES].astype(str).apply(lambda c: c.str.strip()) != "").all().all(), (
        f"{name}: found a blank cell -- should be the literal string 'NA'"
    )


@pytest.mark.parametrize("name", NEW_MANIFESTS)
def test_non_zero_row_count(name):
    df = _load(name)
    assert len(df) > 0, f"{name}: 0 rows"


def test_mlaad_is_spoof_only():
    df = _load("mlaad.csv.gz")
    spoof_frac = (df["target"] == "1").mean()
    assert spoof_frac == 1.0, f"mlaad spoof_frac={spoof_frac}, expected exactly 1.0"
    assert (df["bona_fide_source"] == "na").all()


def test_mlaad_hf_path_and_revision_preserved():
    df = _load("mlaad.csv.gz")
    assert (df["hf_path"] != "NA").all(), "every mlaad row should carry its HF repo-relative path"
    # hf_path is the tail of path (path = f"{prefix}/{hf_path}")
    mismatches = df[~df.apply(lambda r: r["path"].endswith(r["hf_path"]), axis=1)]
    assert mismatches.empty, "hf_path should always be a suffix of path"


def test_mlaad_factor_coverage_above_threshold():
    df = _load("mlaad.csv.gz")
    for col, threshold in [("generator_id", 0.99), ("language", 0.99), ("source_id", 0.99)]:
        coverage = (df[col] != "NA").mean()
        assert coverage >= threshold, f"mlaad.{col} coverage={coverage:.2%} below {threshold:.0%}"


def test_odssclean_both_classes_present_in_expected_band():
    df = _load("odssclean.csv")
    spoof_frac = (df["target"] == "1").mean()
    assert 0.65 < spoof_frac < 0.75, f"odssclean spoof_frac={spoof_frac} outside expected ~0.70 band"


def test_odssclean_excludes_natural_vctk_but_keeps_spoof_vctk():
    df = _load("odssclean.csv")
    assert not (df["bona_fide_source"] == "vctk").any(), "natural/vctk bona rows should be excluded"
    vctk_spoof = df[(df["target"] == "1") & df["utt_id"].str.contains("/vctk/")]
    assert len(vctk_spoof) > 0, "vits/vctk spoof rows should be kept (not a bona-fide duplicate)"


def test_odssclean_factor_coverage_above_threshold():
    df = _load("odssclean.csv")
    assert (df["language"] != "NA").mean() >= 0.99
    assert (df["source_id"] != "NA").mean() >= 0.99
    spoof = df[df["target"] == "1"]
    assert (spoof["generator_id"] != "NA").mean() >= 0.99, "spoof rows should always carry a generator_id"
    bona = df[df["target"] == "0"]
    assert (bona["generator_id"] == "NA").all(), "bona-fide rows should never carry a generator_id"


def test_kwokbona_is_bona_fide_only():
    df = _load("kwokbona.csv")
    assert (df["target"] == "0").all(), "kwokbona must contain zero spoof rows"
    assert (df["split"] == "test").all(), "kwokbona is eval-only -- every row must be split=test"


def test_kwokbona_zero_asvspoof_derived_rows():
    df = _load("kwokbona.csv")
    assert not df["bona_fide_source"].str.contains("asvspoof", case=False).any(), (
        "an ASVspoof-derived subset (asvspoof2019_la / asvspoof2021_df) leaked into kwokbona"
    )
    # Defense in depth: also check the native ASVspoof2019/2021 eval-partition filename
    # prefixes directly, in case a future re-pull renames the subset directory itself.
    assert not df["utt_id"].str.contains(r"/(?:LA|DF)_E_\d+\.flac$", regex=True).any(), (
        "found an ASVspoof2019-LA / ASVspoof2021-DF eval-partition filename in kwokbona"
    )


def test_kwokbona_excludes_spoof_variants_bundled_in_bona_folders():
    """emofake and llamapartialspoof_r01tts0a bundle spoof variants alongside their
    bona-fide carrier in the same folder; llamapartialspoof_r01tts0b is 100% spoof.
    All must be fully filtered out by the trial_metadata.txt label, not folder identity."""
    df = _load("kwokbona.csv")
    assert "llamapartialspoof_r01tts0b" not in set(df["bona_fide_source"]), (
        "llamapartialspoof_r01tts0b has zero bonafide-labeled rows and should not appear"
    )
    counts = df["bona_fide_source"].value_counts()
    for subset in ("emofake", "llamapartialspoof_r01tts0a"):
        assert counts.get(subset, 0) == 600, (
            f"{subset}: expected exactly 600 bonafide-labeled rows, got {counts.get(subset, 0)}"
        )


def test_kwokbona_factor_coverage_above_threshold():
    df = _load("kwokbona.csv")
    assert (df["language"] != "NA").mean() >= 0.99
    assert (df["source_id"] != "NA").mean() >= 0.99
    assert (df["speaker_id"] != "NA").mean() >= 0.99
    assert (df["generator_id"] == "NA").all(), "kwokbona is bona-fide only; generator_id must stay NA"


# --- unit tests for the identity/exclusion logic that produced the fix above ---

@pytest.mark.parametrize("name,expected", [
    ("asvspoof2019_la", True),
    ("asvspoof2021_df", True),
    ("asvspoof5", True),
    ("ami_ihm", False),
    ("vctk", False),
    ("llamapartialspoof_r01tts0a", False),
])
def test_is_asvspoof_derived(name, expected):
    assert _is_asvspoof_derived(name) is expected


@pytest.mark.parametrize("subset,stem,expected", [
    ("librispeech_test_clean", "1089-134686-0002", ("ls-1089-134686", "ls-1089", "NA")),
    ("llamapartialspoof_r01tts0a", "dev-clean_1272_128104_000005_000007",
     ("ls-1272-128104", "ls-1272", "NA")),
    ("ami_ihm", "eval_ami_en2002a_h00_mee073_0000665_0000685", ("ami-en2002a-mee073", "ami-mee073", "h00")),
    ("ami_sdm", "eval_ami_ts3003b_sdm_mtd009pm_0058319_0058337", ("ami-ts3003b-mtd009pm", "ami-mtd009pm", "sdm")),
    ("emofake", "0014_Angry_000355", ("esd-0014", "esd-0014", "NA")),
    ("vctk", "p227_001_mic1", ("p227", "p227", "NA")),
    ("emofake", "unrecognized-stem", ("NA", "NA", "NA")),
])
def test_kwok_identity_for(subset, stem, expected):
    assert kwok_identity_for(subset, stem) == expected
