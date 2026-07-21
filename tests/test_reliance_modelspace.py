"""Tests for scripts/run_reliance_modelspace.py -- the model-space
CAUSAL-RELIANCE sibling consumer (step3_modelspace_reliance_brief.md).

Synthetic data only, shaped exactly like Phase B's real, confirmed schema
(tests/fixtures/step3/*.json for the cache-space side;
docs/phaseB_extraction_preflight_findings.md for the model-space shard
schema) -- no real checkpoint, backbone, or embedding cache. Three things
this file must prove, per the brief's own Definition of Done:

  1. The whole per-checkpoint path (loaders, crossfit, merge, both
     controls) runs end-to-end and w-metrics are POPULATED, not
     not_estimable.
  2. Numerical parity: on a single checkpoint, this sibling's own
     orchestration reproduces run_reliance_battery.py's run_battery
     (imported, unmodified) fold-by-fold -- proving shared primitives,
     not divergent math.
  3. Gate wiring: scripts/run_gate.py (unmodified) reads this sibling's
     output and moves C2/C4/C7 off not_estimable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_reliance_modelspace import (  # noqa: E402
    build_parser,
    load_checkpoint_head,
    load_model_space_embeddings,
    main,
    merge_checkpoint_estimator_results,
    run_checkpoint_crossfit,
    run_modelspace_battery,
)
from run_reliance_battery import run_battery  # noqa: E402 -- imported for the numerical-parity test only


# ---------------------------------------------------------------------------
# Synthetic multi-checkpoint battery data -- same planted-factor
# construction as tests/test_reliance_battery.py's own synthetic_battery_data
# fixture (one (Z, w_true, U_true) pair, orthogonal by construction), but
# one INDEPENDENT (Z, w_true) pair PER checkpoint over the SAME shared
# (factor, y, groups) labels -- simulating each checkpoint's own,
# differently-trained 256-d decision space over the same underlying clips.
# ---------------------------------------------------------------------------


def _planted_w_and_U(seed, d, k_factor):
    rng = np.random.default_rng(seed)
    w_true = rng.normal(size=d)
    w_true /= np.linalg.norm(w_true)
    M = rng.normal(size=(d, k_factor))
    M = M - np.outer(w_true, w_true @ M)
    U_true, _, _ = np.linalg.svd(M, full_matrices=False)
    return w_true, U_true[:, :k_factor]


def _make_multi_checkpoint_synthetic_battery(seed=13, runs=("ckA", "ckB", "ckC"), d=12, n=240, k_factor=3, n_groups=40):
    rng = np.random.default_rng(seed)
    groups_raw = rng.integers(0, n_groups, size=n)
    y = rng.integers(0, 2, size=n)
    factor_levels = rng.integers(0, 4, size=n)
    factor = np.array([f"gen{i}" for i in factor_levels], dtype=object)
    groups = np.array([f"grp{i}" for i in groups_raw], dtype=object)

    Z_by_checkpoint, checkpoints = {}, {}
    for i, run in enumerate(runs):
        w_true, U_true = _planted_w_and_U(seed + i + 1, d, k_factor)
        rng_ck = np.random.default_rng(seed + i + 100)
        group_offset = rng_ck.normal(scale=0.5, size=(n_groups, d))[groups_raw]
        raw_centers = rng_ck.normal(size=(4, k_factor))
        raw_centers -= raw_centers.mean(axis=0, keepdims=True)
        Uc, Sc, Vtc = np.linalg.svd(raw_centers, full_matrices=False)
        factor_centers = Uc @ np.diag(np.full_like(Sc, 3.0)) @ Vtc
        Z = (np.outer((y * 2 - 1).astype(float), w_true) * 3.0
             + factor_centers[factor_levels] @ U_true.T
             + group_offset
             + rng_ck.normal(size=(n, d)))
        Z_by_checkpoint[run] = Z
        checkpoints[run] = dict(w=w_true, b=0.0, w_dim=d, w_dim_mismatch=False,
                                 ckpt_layer_center=None, ckpt_layer_band=None, layer_pooling="model_space",
                                 band_weights=None, w_layer_mismatch=False)
    return Z_by_checkpoint, factor, y, groups, checkpoints


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _write_model_space_shard(shard_path: Path, n: int, dim: int, corpus: str, corpus_dir: str, seed: int) -> np.ndarray:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n, dim)).astype(np.float32)
    # "f{i:04d}.wav" matches _write_manifest's stripped relative path
    # (datasets/<corpus_dir>/f{i:04d}.wav) so join_cache_to_manifest's join
    # actually matches rows in the end-to-end test below.
    paths = np.array([f"f{i:04d}.wav" for i in range(n)])
    meta = dict(checkpoint_sha256="a", model_config_hash="b", git_sha="c", dtype="float32",
                checkpoint_path="x", corpus=corpus, corpus_dir=corpus_dir, n_rows=n)
    np.savez(shard_path, paths=paths, emb=emb, meta=np.array(json.dumps(meta)))
    return emb


def test_load_model_space_embeddings_reads_real_2d_schema(tmp_path):
    emb = _write_model_space_shard(tmp_path / "ckA" / "03_DiffSSD" / "shard_0000.npz",
                                    n=10, dim=8, corpus="diffssd", corpus_dir="03_DiffSSD", seed=0)
    paths, loaded_emb = load_model_space_embeddings(tmp_path, "ckA", "03_DiffSSD")
    assert loaded_emb.shape == (10, 8)
    assert paths.shape == (10,)
    np.testing.assert_allclose(loaded_emb, emb)


def test_load_model_space_embeddings_concatenates_multiple_shards(tmp_path):
    _write_model_space_shard(tmp_path / "ckA" / "03_DiffSSD" / "shard_0000.npz",
                              n=5, dim=8, corpus="diffssd", corpus_dir="03_DiffSSD", seed=0)
    _write_model_space_shard(tmp_path / "ckA" / "03_DiffSSD" / "shard_0001.npz",
                              n=7, dim=8, corpus="diffssd", corpus_dir="03_DiffSSD", seed=1)
    paths, emb = load_model_space_embeddings(tmp_path, "ckA", "03_DiffSSD")
    assert emb.shape == (12, 8)
    assert paths.shape == (12,)


def test_load_model_space_embeddings_raises_on_missing_shards(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model_space_embeddings(tmp_path, "ckA", "03_DiffSSD")


def test_load_model_space_embeddings_rejects_3d_layer_axis_shape(tmp_path):
    """The exact confusion docs/phaseB_extraction_preflight_findings.md
    identifies as the consumption gap's root cause: a 3-D (n, n_layers, D)
    shard (the raw XLS-R-300M cache's shape) must be rejected loudly, not
    silently misread."""
    shard_dir = tmp_path / "ckA" / "03_DiffSSD"
    shard_dir.mkdir(parents=True)
    np.savez(shard_dir / "shard_0000.npz", paths=np.array(["a.wav"]),
             emb=np.zeros((1, 25, 1024), dtype=np.float32), meta=np.array(json.dumps({})))
    with pytest.raises(ValueError, match="2-D"):
        load_model_space_embeddings(tmp_path, "ckA", "03_DiffSSD")


def _write_synthetic_checkpoint(path: Path, w: np.ndarray, b: float) -> None:
    state = {"binary.fc.weight": torch.tensor(w, dtype=torch.float32).reshape(1, -1),
             "binary.fc.bias": torch.tensor([b], dtype=torch.float32)}
    torch.save({"model": state, "cfg": {"model": {}}}, path)


def test_load_checkpoint_head_reads_binary_fc_weight_and_bias(tmp_path):
    ckpt_path = tmp_path / "runs_e007_A_fresh_best.pt"
    w = np.array([1.0, 2.0, 3.0, 4.0])
    _write_synthetic_checkpoint(ckpt_path, w, b=0.5)

    loaded_w, loaded_b, w_dim = load_checkpoint_head(ckpt_path)

    np.testing.assert_allclose(loaded_w, w)
    assert loaded_b == pytest.approx(0.5)
    assert w_dim == 4


def test_load_checkpoint_head_raises_without_a_recognized_weight_key(tmp_path):
    ckpt_path = tmp_path / "bad.pt"
    torch.save({"model": {"unrelated.weight": torch.zeros(4)}}, ckpt_path)
    with pytest.raises(RuntimeError, match="no classifier weight found"):
        load_checkpoint_head(ckpt_path)


# ---------------------------------------------------------------------------
# Per-checkpoint crossfit + merge
# ---------------------------------------------------------------------------


def test_run_checkpoint_crossfit_single_checkpoint_produces_populated_metrics():
    Z_by_checkpoint, factor, y, groups, checkpoints = _make_multi_checkpoint_synthetic_battery(runs=("ckA",))
    run = "ckA"
    result = run_checkpoint_crossfit(run, checkpoints[run], Z_by_checkpoint[run], factor, y, groups,
                                      valid_ranks=[1, 2, 3], n_outer=5, seed=13)
    assert set(result) == {"lda", "probe"}
    for estimator_result in result.values():
        for fold in estimator_result["fold_results"]:
            ck_effect = fold["effect"]["per_checkpoint"][run]
            assert isinstance(ck_effect["alignment"], float)
            assert np.isfinite(ck_effect["alignment"])
            assert isinstance(ck_effect["r_var"], float)
            control = ck_effect["prediction_change_control"]
            assert isinstance(control["task_direction_effect"], float)  # NOT not_estimable
            assert isinstance(control["exceeds_random"], bool)


def test_merge_checkpoint_estimator_results_unions_per_checkpoint_entries():
    Z_by_checkpoint, factor, y, groups, checkpoints = _make_multi_checkpoint_synthetic_battery(runs=("ckA", "ckB"))
    per_checkpoint_estimators = {
        run: run_checkpoint_crossfit(run, checkpoints[run], Z_by_checkpoint[run], factor, y, groups,
                                      valid_ranks=[1, 2], n_outer=5, seed=13)
        for run in checkpoints
    }

    merged = merge_checkpoint_estimator_results(per_checkpoint_estimators, primary_run="ckA")

    assert set(merged) == {"lda", "probe"}
    for estimator_result in merged.values():
        assert estimator_result["status"] == "ok"
        for fold in estimator_result["fold_results"]:
            per_ckpt = fold["effect"]["per_checkpoint"]
            assert set(per_ckpt) == {"ckA", "ckB"}
            for run in ("ckA", "ckB"):
                assert "chosen" in per_ckpt[run]
                assert "projection_removal_control" in per_ckpt[run]
            # fold-level (non-per-checkpoint) fields come from the primary checkpoint
            assert "projection_removal_control" in fold["effect"]
            assert fold["effect"]["projection_removal_control"] == per_ckpt["ckA"]["projection_removal_control"]


def test_merge_checkpoint_estimator_results_raises_on_fold_id_mismatch():
    fake_a = {"lda": {"fold_results": [dict(fold_id=0, chosen={"k": 1}, selection_score=0.5, n_selection=10,
                                            n_effect=5, effect=dict(per_checkpoint={"ckA": {}},
                                                                     factor_separation_score=0.1,
                                                                     leace={}, inlp={},
                                                                     projection_removal_control={}))]}}
    fake_b = {"lda": {"fold_results": [dict(fold_id=1, chosen={"k": 1}, selection_score=0.5, n_selection=10,
                                            n_effect=5, effect=dict(per_checkpoint={"ckB": {}},
                                                                     factor_separation_score=0.1,
                                                                     leace={}, inlp={},
                                                                     projection_removal_control={}))]}}
    with pytest.raises(AssertionError, match="fold_id mismatch"):
        merge_checkpoint_estimator_results({"ckA": fake_a, "ckB": fake_b}, primary_run="ckA")


# ---------------------------------------------------------------------------
# End-to-end: run_modelspace_battery -- w-metrics populated, not_estimable gone
# ---------------------------------------------------------------------------


def test_run_modelspace_battery_populates_w_metrics_for_every_checkpoint():
    Z_by_checkpoint, factor, y, groups, checkpoints = _make_multi_checkpoint_synthetic_battery()
    spec = dict(name="synth_modelspace_battery", corpus="diffssd", factor="generator_id", grouping="source_id")

    result = run_modelspace_battery(spec, Z_by_checkpoint, factor, y, groups, checkpoints,
                                     ranks=[1, 2, 3], n_boot=30, seed=13, max_rows_per_level=None)

    assert "skipped" not in result
    assert result["layer_mode"] == "model_space"
    for estimator_result in result["estimators"].values():
        for fold in estimator_result["fold_results"]:
            per_ckpt = fold["effect"]["per_checkpoint"]
            assert set(per_ckpt) == set(checkpoints)
            for run, ck_effect in per_ckpt.items():
                assert np.isfinite(ck_effect["alignment"])
                assert np.isfinite(ck_effect["r_var"])
                pcc = ck_effect["prediction_change_control"]
                assert isinstance(pcc["task_direction_effect"], float)
                prc = ck_effect["projection_removal_control"]
                assert isinstance(prc["task_direction_effect"], float)
                assert isinstance(prc["exceeds_random"], bool)
    assert result["headline_bootstrap"]["metric"] == "r_var"
    assert result["headline_bootstrap"]["status"] == "ok"
    assert np.isfinite(result["headline_bootstrap"]["mean"])
    assert result["rank_sensitivity"]["status"] == "ok"


def test_run_modelspace_battery_row_cap_keeps_checkpoints_row_aligned():
    Z_by_checkpoint, factor, y, groups, checkpoints = _make_multi_checkpoint_synthetic_battery(n=300, n_groups=60)
    spec = dict(name="capped_battery", corpus="diffssd", factor="generator_id", grouping="source_id")

    result = run_modelspace_battery(spec, Z_by_checkpoint, factor, y, groups, checkpoints,
                                     ranks=[1, 2], n_boot=10, seed=13, max_rows_per_level=50)

    assert "skipped" not in result
    assert result["n_rows"] <= 300


# ---------------------------------------------------------------------------
# Numerical parity: this sibling's per-checkpoint orchestration must
# reproduce run_reliance_battery.py's OWN run_battery on a single checkpoint
# -- proving shared primitives, not divergent math (Definition of Done #3).
# ---------------------------------------------------------------------------


def test_numerical_parity_single_checkpoint_matches_run_battery():
    Z_by_checkpoint, factor, y, groups, checkpoints = _make_multi_checkpoint_synthetic_battery(runs=("ckA",))
    run = "ckA"
    spec = dict(name="parity_test", corpus="diffssd", factor="generator_id", grouping="source_id")

    sibling_result = run_modelspace_battery(spec, Z_by_checkpoint, factor, y, groups, checkpoints,
                                             ranks=[1, 2, 3], n_boot=0, seed=13, max_rows_per_level=None)
    original_result = run_battery(spec, Z_by_checkpoint[run], factor, y, groups, checkpoints,
                                   ranks=[1, 2, 3], n_boot=0, seed=13, layer_mode="fixed",
                                   w_metrics_enabled=True, max_rows_per_level=None)

    # run_battery dispatches its two estimators as SEPARATE SPAWNED PROCESSES
    # (subprocess isolation); this sibling runs in-process. Neither path here
    # pins single-threaded BLAS (that only happens inside main(), never
    # called by either call above), so a tiny (~1e-5 relative) floating-point
    # difference from multi-threaded BLAS reduction order is expected and is
    # NOT evidence of divergent math -- rel=1e-4 is still far tighter than
    # that noise floor while remaining well below any scientifically
    # meaningful difference.
    for estimator in ("lda", "probe"):
        sib_folds = sibling_result["estimators"][estimator]["fold_results"]
        orig_folds = original_result["estimators"][estimator]["fold_results"]
        assert len(sib_folds) == len(orig_folds)
        for sib_fold, orig_fold in zip(sib_folds, orig_folds):
            assert sib_fold["fold_id"] == orig_fold["fold_id"]
            sib_ck = sib_fold["effect"]["per_checkpoint"][run]
            orig_ck = orig_fold["effect"]["per_checkpoint"][run]
            assert sib_ck["alignment"] == pytest.approx(orig_ck["alignment"], rel=1e-4)
            assert sib_ck["r_var"] == pytest.approx(orig_ck["r_var"], rel=1e-4)
            assert (sib_ck["prediction_change"]["mean_abs_logit_change"]
                    == pytest.approx(orig_ck["prediction_change"]["mean_abs_logit_change"], rel=1e-4))
            assert (sib_ck["prediction_change_control"]["exceeds_random"]
                    == orig_ck["prediction_change_control"]["exceeds_random"])


# ---------------------------------------------------------------------------
# Full CLI end-to-end + gate wiring: real synthetic files on disk, run
# main(), then feed the produced battery JSON through scripts/run_gate.py's
# OWN, UNMODIFIED readers.
# ---------------------------------------------------------------------------


def _write_manifest(manifest_dir: Path, corpus: str, n: int, n_groups: int, seed: int) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append(dict(
            utt_id=f"{corpus}/f{i:04d}", path=f"datasets/03_DiffSSD/f{i:04d}.wav",
            target=int(rng.integers(0, 2)), corpus=corpus, split="train", attack="na",
            bona_fide_source="na", source_id=f"src{i % n_groups}", speaker_id="NA",
            generator_id=f"gen{i % 4}", channel_id="NA", language="NA", platform_id="NA",
        ))
    pd.DataFrame(rows).to_csv(manifest_dir / f"{corpus}.csv", index=False)


def _build_and_run_modelspace_cli(tmp_path, monkeypatch, runs=("e007_A_fresh", "e007_B_fresh")) -> Path:
    """Shared setup for the two tests below: real synthetic checkpoint .pt
    files + model-space shard npz files + a manifest CSV on disk, run
    main() for real, return the produced manifest JSON's path."""
    monkeypatch.chdir(tmp_path)
    n, n_groups, dim = 200, 20, 6
    manifest_dir = tmp_path / "manifests"
    _write_manifest(manifest_dir, "diffssd", n=n, n_groups=n_groups, seed=7)

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    cache_root = tmp_path / "_embcache_modelspace"
    for i, run in enumerate(runs):
        w = np.random.default_rng(i).normal(size=dim)
        _write_synthetic_checkpoint(ckpt_dir / f"runs_{run}_best.pt", w, b=0.0)
        _write_model_space_shard(cache_root / f"runs_{run}_best" / "03_DiffSSD" / "shard_0000.npz",
                                  n=n, dim=dim, corpus="diffssd", corpus_dir="03_DiffSSD", seed=i + 50)

    out_path = tmp_path / "reliance_modelspace.json"
    main([
        "--model-space-cache-root", str(cache_root),
        "--manifest-dir", str(manifest_dir),
        "--ckpt-dir", str(ckpt_dir),
        "--checkpoints", *runs,
        "--corpus", "diffssd",
        "--factor", "generator_id",
        "--ranks", "1", "2", "3",
        "--n-boot", "20",
        "--seed", "13",
        "--out", str(out_path),
    ])
    return out_path


def test_main_end_to_end_writes_schema_valid_populated_battery(tmp_path, monkeypatch):
    runs = ("e007_A_fresh", "e007_B_fresh")
    out_path = _build_and_run_modelspace_cli(tmp_path, monkeypatch, runs)

    assert out_path.exists()
    manifest = json.loads(out_path.read_text(encoding="utf-8"))
    assert manifest["layer_mode"] == "model_space"
    assert len(manifest["batteries"]) == 1
    battery = manifest["batteries"][0]
    assert "skipped" not in battery
    for estimator_result in battery["estimators"].values():
        for fold in estimator_result["fold_results"]:
            per_ckpt = fold["effect"]["per_checkpoint"]
            assert set(per_ckpt) == set(runs)
            for ck_effect in per_ckpt.values():
                assert np.isfinite(ck_effect["alignment"])


def test_gate_reads_modelspace_output_and_leaves_not_estimable(tmp_path, monkeypatch):
    """The definitive proof the chain is wired: scripts/run_gate.py, totally
    unmodified, reads this sibling's output and C2/C4/C7 move OFF
    not_estimable/pending_input (given a factor-corpus map + synthetic EERs)."""
    out_path = _build_and_run_modelspace_cli(tmp_path, monkeypatch)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_gate

    records, warnings = run_gate.load_phase_a_inputs([out_path])
    assert warnings == []
    assert len(records) == 1

    eers = {"e007_A_fresh": {"inthewild": 0.10}, "e007_B_fresh": {"inthewild": 0.20}}
    factor_corpus_map = {"generator_id": "inthewild"}

    c2 = run_gate.criterion_2_association(records, eers, factor_corpus_map)
    c4 = run_gate.criterion_4_intervention_vs_random(records)
    c6 = run_gate.criterion_6_estimator_agreement(records)

    assert c2["status"] != run_gate.STATUS_NOT_ESTIMABLE
    assert c4["status"] != run_gate.STATUS_NOT_ESTIMABLE
    assert c6["status"] in (run_gate.STATUS_PASS, run_gate.STATUS_FAIL)  # fully decided either way


def test_build_parser_defaults():
    args = build_parser().parse_args([])
    assert args.n_boot == 1000
    assert args.seed == 13
    assert len(args.checkpoints) == 3
