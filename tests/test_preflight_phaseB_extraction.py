"""Tests for scripts/preflight_phaseB_extraction.py -- synthetic
fixtures/fake model only, per this project's established convention
(tests/test_extract_model_embeddings.py's own docstring: constructing a
real AudioShieldX needs a downloaded HF backbone, out of scope here). The
decisive correctness check (binary.fc(emb) == logit) is tested directly
with a tiny real nn.Linear + known input, per the brief's own suggestion,
AND exercised end-to-end through run_preflight_validation with a
dependency-injected fake model -- including a deliberately WRONG-hook
variant that must be caught, not just a happy path that trivially passes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from preflight_phaseB_extraction import (  # noqa: E402
    CONFIRMED_ROW_COUNTS,
    check_correctness,
    check_determinism,
    check_sanity,
    check_schema_and_dim,
    project_full_extraction_cost,
    recompute_logit_from_embedding,
    run_preflight_validation,
)
from audioshield.data.manifest import ManifestRow  # noqa: E402


# ---------------------------------------------------------------------------
# The decisive check: binary.fc(emb) == logit, on a tiny real nn.Linear.
# ---------------------------------------------------------------------------


def test_recompute_logit_from_embedding_matches_manual_linear_algebra():
    torch.manual_seed(0)
    fc = torch.nn.Linear(4, 1)
    emb = torch.randn(3, 4)
    recomputed = recompute_logit_from_embedding(fc, emb)
    expected = (emb @ fc.weight.T + fc.bias).squeeze(-1)
    assert torch.allclose(recomputed, expected, atol=1e-6)
    assert recomputed.shape == (3,)


def test_check_correctness_passes_when_embedding_really_is_the_classifier_input():
    torch.manual_seed(1)
    fc = torch.nn.Linear(6, 1)
    emb = torch.randn(10, 6)
    true_logit = fc(emb).squeeze(-1)

    result = check_correctness(fc, emb, true_logit)

    assert result.passed is True
    assert "max|binary.fc(emb) - logit|" in result.detail


def test_check_correctness_fails_when_embedding_is_the_wrong_hook_point():
    """A wrong hook (e.g. capturing pre-GELU instead of the final
    proj output) must be caught, not pass by coincidence."""
    torch.manual_seed(2)
    fc = torch.nn.Linear(6, 1)
    emb_true = torch.randn(10, 6)
    true_logit = fc(emb_true).squeeze(-1)
    emb_wrong = emb_true + 5.0  # a systematically different point

    result = check_correctness(fc, emb_wrong, true_logit)

    assert result.passed is False


def test_check_correctness_tolerates_float16_quantization_roundtrip():
    """The default on-disk dtype is float16 -- a real, small quantization
    error must not be mistaken for a wrong hook point."""
    torch.manual_seed(3)
    fc = torch.nn.Linear(8, 1)
    emb = torch.randn(20, 8)
    true_logit = fc(emb).squeeze(-1)
    emb_roundtripped = emb.half().float()  # simulates float16 storage + reload

    result = check_correctness(fc, emb_roundtripped, true_logit)

    assert result.passed is True


# ---------------------------------------------------------------------------
# Sanity / determinism / schema checks
# ---------------------------------------------------------------------------


def test_check_sanity_passes_on_reasonable_embeddings():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(50, 16)).astype(np.float32)
    result = check_sanity(emb)
    assert result.passed is True


def test_check_sanity_fails_on_nan():
    emb = np.zeros((5, 4), dtype=np.float32)
    emb[2, 1] = np.nan
    result = check_sanity(emb)
    assert result.passed is False


def test_check_sanity_fails_on_all_zero_norms():
    emb = np.zeros((5, 4), dtype=np.float32)
    result = check_sanity(emb)
    assert result.passed is False


def test_check_determinism_passes_on_identical_arrays():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(10, 6)).astype(np.float32)
    result = check_determinism(emb, emb.copy())
    assert result.passed is True


def test_check_determinism_fails_on_differing_arrays():
    rng = np.random.default_rng(0)
    emb_a = rng.normal(size=(10, 6)).astype(np.float32)
    emb_b = emb_a.copy()
    emb_b[0, 0] += 1.0
    result = check_determinism(emb_a, emb_b)
    assert result.passed is False


def test_check_schema_and_dim_passes_on_real_written_shard(tmp_path):
    shard_path = tmp_path / "shard_0000.npz"
    meta = dict(checkpoint_sha256="a", model_config_hash="b", git_sha="c", dtype="float32",
                checkpoint_path="x", corpus="diffssd", corpus_dir="03_DiffSSD", n_rows=2)
    np.savez(shard_path, paths=np.array(["a.wav", "b.wav"]),
             emb=np.zeros((2, 256), dtype=np.float32), meta=np.array(json.dumps(meta)))

    result = check_schema_and_dim(shard_path, expected_dim=256)

    assert result.passed is True


def test_check_schema_and_dim_fails_on_3d_layer_axis_shape(tmp_path):
    """The exact shape mismatch docs/phaseB_extraction_preflight_findings.md
    identifies as the root cause of run_reliance_battery.py's consumption
    gap -- a 3-D (n, n_layers, D) array where Phase B should write 2-D."""
    shard_path = tmp_path / "shard_0000.npz"
    meta = dict(checkpoint_sha256="a", model_config_hash="b", git_sha="c", dtype="float32",
                checkpoint_path="x", corpus="diffssd", corpus_dir="03_DiffSSD", n_rows=2)
    np.savez(shard_path, paths=np.array(["a.wav", "b.wav"]),
             emb=np.zeros((2, 25, 256), dtype=np.float32), meta=np.array(json.dumps(meta)))

    result = check_schema_and_dim(shard_path, expected_dim=256)

    assert result.passed is False
    assert "2-D" in result.detail


def test_check_schema_and_dim_fails_on_wrong_embedding_dim(tmp_path):
    shard_path = tmp_path / "shard_0000.npz"
    meta = dict(checkpoint_sha256="a", model_config_hash="b", git_sha="c", dtype="float32",
                checkpoint_path="x", corpus="diffssd", corpus_dir="03_DiffSSD", n_rows=2)
    np.savez(shard_path, paths=np.array(["a.wav", "b.wav"]),
             emb=np.zeros((2, 128), dtype=np.float32), meta=np.array(json.dumps(meta)))

    result = check_schema_and_dim(shard_path, expected_dim=256)

    assert result.passed is False


def test_check_schema_and_dim_fails_on_missing_key(tmp_path):
    shard_path = tmp_path / "shard_0000.npz"
    np.savez(shard_path, wrong_key=np.zeros(3))
    result = check_schema_and_dim(shard_path, expected_dim=256)
    assert result.passed is False
    assert "missing keys" in result.detail


# ---------------------------------------------------------------------------
# Cost projection
# ---------------------------------------------------------------------------


def test_project_full_extraction_cost_uses_confirmed_row_counts():
    projection = project_full_extraction_cost(clips_per_sec=100.0)
    expected_total = 3 * sum(CONFIRMED_ROW_COUNTS.values())
    assert projection["total_clips"] == expected_total
    assert projection["n_jobs"] == 6
    assert projection["projected_seconds"] == pytest.approx(expected_total / 100.0)


def test_project_full_extraction_cost_handles_zero_throughput():
    projection = project_full_extraction_cost(clips_per_sec=0.0)
    assert projection["projected_seconds"] == float("inf")


# ---------------------------------------------------------------------------
# End-to-end run_preflight_validation with a dependency-injected fake model
# -- proves the whole orchestration (extraction -> schema -> correctness ->
# sanity -> determinism -> resume -> throughput/projection) is wired
# correctly, without a real backbone.
# ---------------------------------------------------------------------------


class _FakeBinary:
    def __init__(self, dim):
        self.fc = torch.nn.Linear(dim, 1)


class _FakeModel:
    """embed() returns a deterministic function of the waveform; __call__
    (the "real" forward pass) computes its logit from that SAME embed()
    output when wrong_hook=False (the happy path: embed() truly is the
    classifier input), or from a shifted version when wrong_hook=True (a
    simulated wrong-hook-point bug that check_correctness must catch)."""

    def __init__(self, dim=6, wrong_hook=False):
        self.binary = _FakeBinary(dim)
        self._dim = dim
        self._wrong_hook = wrong_hook

    def _true_embedding(self, waveform):
        n = waveform.shape[0]
        # deterministic, non-trivial function of the waveform's own values
        return torch.stack([waveform[i, : self._dim] * (i + 1) for i in range(n)])

    def embed(self, waveform):
        z = self._true_embedding(waveform)
        if self._wrong_hook:
            return z + 3.0  # extracts the WRONG point
        return z

    def __call__(self, waveform):
        z = self._true_embedding(waveform)  # the model's REAL classifier input, always correct
        logit = self.binary.fc(z).squeeze(-1)
        return {"spoof_logit": logit}

    def to(self, device):
        return self

    def eval(self):
        return self


def _write_manifest_and_rows(tmp_path: Path, corpus: str, corpus_dir: str, n: int) -> Path:
    data_root = tmp_path / "data"
    (data_root / "datasets" / corpus_dir).mkdir(parents=True)
    rows = []
    for i in range(n):
        rel = f"datasets/{corpus_dir}/f{i:02d}.wav"
        sf.write(data_root / rel, np.full(1600, 0.01, dtype="float32"), 16000)
        rows.append(dict(utt_id=f"{corpus}/f{i:02d}", path=rel, target=1, corpus=corpus,
                          split="train", attack="na", bona_fide_source="na"))
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(manifest_dir / f"{corpus}.csv", index=False)
    return manifest_dir, data_root


def test_run_preflight_validation_all_checks_pass_with_correct_fake_model(tmp_path):
    manifest_dir, data_root = _write_manifest_and_rows(tmp_path, "diffssd", "03_DiffSSD", n=4)

    def build_model_fn(ckpt_path, device):
        return _FakeModel(dim=6, wrong_hook=False), {"experiment": {"sample_rate": 16000, "duration_seconds": 0.5}}, {}

    ckpt_path = tmp_path / "fake.pt"
    ckpt_path.write_bytes(b"fake checkpoint bytes")

    report = run_preflight_validation(
        checkpoint=ckpt_path, corpus="diffssd", manifest_dir=manifest_dir, data_root=data_root,
        out_root=tmp_path / "out", device="cpu", n_clips=4, batch_size=2, dtype="float32",
        build_model_fn=build_model_fn, run_eigh_check=False,
    )

    by_name = {r["name"]: r for r in report["results"]}
    assert by_name["correctness_binary_fc_matches_logit"]["passed"] is True
    assert by_name["sanity_finite_and_reasonable_norms"]["passed"] is True
    assert by_name["determinism_two_extractions_match"]["passed"] is True
    assert by_name["resume_skips_completed_shard"]["passed"] is True
    assert any(k.startswith("schema_and_dim") and v["passed"] for k, v in by_name.items())
    assert report["throughput"]["n_clips"] == 4
    assert report["projection"]["n_jobs"] == 6


def test_run_preflight_validation_catches_wrong_hook_point(tmp_path):
    """The whole point of this preflight: a genuinely wrong extraction
    hook must fail the correctness check end-to-end, not just in the
    isolated unit test above."""
    manifest_dir, data_root = _write_manifest_and_rows(tmp_path, "diffssd", "03_DiffSSD", n=4)

    def build_model_fn(ckpt_path, device):
        return _FakeModel(dim=6, wrong_hook=True), {"experiment": {"sample_rate": 16000, "duration_seconds": 0.5}}, {}

    ckpt_path = tmp_path / "fake.pt"
    ckpt_path.write_bytes(b"fake checkpoint bytes")

    report = run_preflight_validation(
        checkpoint=ckpt_path, corpus="diffssd", manifest_dir=manifest_dir, data_root=data_root,
        out_root=tmp_path / "out", device="cpu", n_clips=4, batch_size=2, dtype="float32",
        build_model_fn=build_model_fn, run_eigh_check=False,
    )

    by_name = {r["name"]: r for r in report["results"]}
    assert by_name["correctness_binary_fc_matches_logit"]["passed"] is False


def test_run_preflight_validation_reports_throughput_and_projection(tmp_path):
    manifest_dir, data_root = _write_manifest_and_rows(tmp_path, "replaydf", "04_ReplayDF", n=3)

    def build_model_fn(ckpt_path, device):
        return _FakeModel(dim=4, wrong_hook=False), {"experiment": {"sample_rate": 16000, "duration_seconds": 0.5}}, {}

    ckpt_path = tmp_path / "fake.pt"
    ckpt_path.write_bytes(b"fake checkpoint bytes")

    report = run_preflight_validation(
        checkpoint=ckpt_path, corpus="replaydf", manifest_dir=manifest_dir, data_root=data_root,
        out_root=tmp_path / "out", device="cpu", n_clips=3, batch_size=2, dtype="float32",
        build_model_fn=build_model_fn, run_eigh_check=False,
    )

    assert report["throughput"]["clips_per_sec"] > 0
    assert report["projection"]["measured_clips_per_sec"] == report["throughput"]["clips_per_sec"]
    assert report["projection"]["row_counts"] == CONFIRMED_ROW_COUNTS
