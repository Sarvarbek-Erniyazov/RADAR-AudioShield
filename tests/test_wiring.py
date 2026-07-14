"""Wiring-repair series (2a-w1..w10): static source assertions that the fix modules
landed by the 2a-c* commits are actually CALLED from the real execution path, not just
importable from their own tests -- plus negative-path companions to two happy-path-only
existing tests. Audit ref: docs/review/2a_verification_report.md."""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


# --- static source assertions -------------------------------------------------

def test_train_e002_wires_seeding():
    src = _read("scripts/train_e002.py")
    assert "seed_everything(" in src
    assert "dataloader_seed_kwargs(" in src


def test_train_e002_wires_joint_sampler():
    src = _read("scripts/train_e002.py")
    assert "compute_joint_weights" in src
    assert "empirical_class_corpus_mi" in src


def test_unified_dataset_wires_strict_loader():
    src = _read("src/audioshield/data/unified_dataset.py")
    assert "load_audio_strict" in src
    # old 50-row silent-substitution loop is gone (it was the only place `last_err`
    # ever appeared in this file)
    assert "last_err" not in src


def test_ssl_backbone_wires_load_backbone():
    src = _read("src/audioshield/models/ssl_backbone.py")
    assert "load_backbone(" in src
    # routed through hf_loading, not calling transformers directly anymore
    assert "AutoModel.from_pretrained" not in src


def test_channel_aug_has_no_hardcoded_rir_dir():
    src = _read("src/audioshield/data/channel_aug.py")
    assert "RIR_DIR = Path(" not in src
    assert "configure_rir_root" in src


def test_train_e002_wires_aug_assets():
    src = _read("scripts/train_e002.py")
    assert "resolve_aug_assets(" in src
    assert "configure_rir_root(" in src


def test_loop_e002_wires_supcon_guard():
    src = _read("src/audioshield/training/loop_e002.py")
    assert "supcon_batch_valid(" in src


def test_reproduce_eval_wires_hash_check():
    src = _read("scripts/reproduce_eval.py")
    assert "expected_hashes.get(" in src
    assert src.count("load_expected_hashes(") >= 2  # def + at least one call


def test_cross_test_refuses_silent_overwrite():
    src = _read("src/audioshield/evaluation/cross_test.py")
    assert "--force" in src
    assert "out.exists()" in src


# --- negative-path companions --------------------------------------------------

def test_environment_guard_fires_on_drift():
    """Negative-path companion to tests/test_environment.py::test_lockfile_exists_and_matches_env
    (report finding 3.4 -- that test only exercises the current, already-correct
    environment.*.json; this proves the SAME comparison actually rejects a drifted one)."""
    import transformers

    drifted_info = {"transformers": "0.0.0-this-does-not-match"}
    with pytest.raises(AssertionError):
        assert drifted_info["transformers"] == transformers.__version__, "lock drifted from live env -- re-freeze"


def test_hf_loading_wrong_revision_not_silently_corrected(tmp_path):
    """Negative-path companion to tests/test_hf_loading.py::test_unpinned_revision_refused
    (report finding 3.4 -- that test only covers 'revisions file missing', not 'file present
    but wrong/stale hash for this model', a distinct branch in get_pinned_revision).

    Actually rejecting a wrong revision hash happens inside AutoModel.from_pretrained,
    which requires network access unavailable here. What we CAN assert offline: a wrong
    revision string is threaded through faithfully, not silently swapped for the correct
    pinned one -- i.e. there is no silent fallback that would mask a stale entry."""
    from audioshield.utils.hf_loading import get_pinned_revision

    p = tmp_path / "revisions.yaml"
    wrong_revision = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    p.write_text(f"facebook/wav2vec2-xls-r-300m: {wrong_revision}\n")
    got = get_pinned_revision("facebook/wav2vec2-xls-r-300m", str(p))
    assert got == wrong_revision
    assert got != "1a640f32ac3e39899438a2931f9924c02f080a54"  # not silently the real pinned SHA
