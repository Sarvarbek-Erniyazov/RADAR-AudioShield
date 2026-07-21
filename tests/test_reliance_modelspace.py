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
import run_gate  # noqa: E402 -- imported for the gate-wiring tests, completely unmodified
# run_reliance_battery.run_battery is used by _reliance_modelspace_parity_worker.py
# (the numerical-parity test's subprocess), not imported directly here.


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


def _write_model_space_shard(shard_path: Path, n: int, dim: int, corpus: str, corpus_dir: str, seed: int,
                              checkpoint_sha256: str = "a") -> np.ndarray:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n, dim)).astype(np.float32)
    # "f{i:04d}.wav" matches _write_manifest's stripped relative path
    # (datasets/<corpus_dir>/f{i:04d}.wav) so join_cache_to_manifest's join
    # actually matches rows in the end-to-end test below.
    paths = np.array([f"f{i:04d}.wav" for i in range(n)])
    meta = dict(checkpoint_sha256=checkpoint_sha256, model_config_hash="b", git_sha="c", dtype="float32",
                checkpoint_path="x", corpus=corpus, corpus_dir=corpus_dir, n_rows=n)
    np.savez(shard_path, paths=paths, emb=emb, meta=np.array(json.dumps(meta)))
    return emb


def test_load_model_space_embeddings_reads_real_2d_schema(tmp_path):
    emb = _write_model_space_shard(tmp_path / "ckA" / "03_DiffSSD" / "shard_0000.npz",
                                    n=10, dim=8, corpus="diffssd", corpus_dir="03_DiffSSD", seed=0)
    paths, loaded_emb, sha = load_model_space_embeddings(tmp_path, "ckA", "03_DiffSSD")
    assert loaded_emb.shape == (10, 8)
    assert paths.shape == (10,)
    np.testing.assert_allclose(loaded_emb, emb)
    assert sha == "a"  # _write_model_space_shard's default checkpoint_sha256


def test_load_model_space_embeddings_concatenates_multiple_shards(tmp_path):
    _write_model_space_shard(tmp_path / "ckA" / "03_DiffSSD" / "shard_0000.npz",
                              n=5, dim=8, corpus="diffssd", corpus_dir="03_DiffSSD", seed=0)
    _write_model_space_shard(tmp_path / "ckA" / "03_DiffSSD" / "shard_0001.npz",
                              n=7, dim=8, corpus="diffssd", corpus_dir="03_DiffSSD", seed=1)
    paths, emb, sha = load_model_space_embeddings(tmp_path, "ckA", "03_DiffSSD")
    assert emb.shape == (12, 8)
    assert paths.shape == (12,)
    assert sha == "a"


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


def test_load_model_space_embeddings_raises_on_disagreeing_shard_sha256(tmp_path):
    """Two shards under the same directory recording DIFFERENT
    checkpoint_sha256 values means shards from two different extraction
    runs were mixed into one cache directory -- must raise, not silently
    average/concatenate them as if they were one checkpoint's output."""
    _write_model_space_shard(tmp_path / "ckA" / "03_DiffSSD" / "shard_0000.npz",
                              n=5, dim=8, corpus="diffssd", corpus_dir="03_DiffSSD", seed=0,
                              checkpoint_sha256="sha_of_checkpoint_A")
    _write_model_space_shard(tmp_path / "ckA" / "03_DiffSSD" / "shard_0001.npz",
                              n=5, dim=8, corpus="diffssd", corpus_dir="03_DiffSSD", seed=1,
                              checkpoint_sha256="sha_of_checkpoint_B")
    with pytest.raises(ValueError, match="disagrees"):
        load_model_space_embeddings(tmp_path, "ckA", "03_DiffSSD")


def _write_synthetic_checkpoint(path: Path, w: np.ndarray, b: float) -> None:
    state = {"binary.fc.weight": torch.tensor(w, dtype=torch.float32).reshape(1, -1),
             "binary.fc.bias": torch.tensor([b], dtype=torch.float32)}
    torch.save({"model": state, "cfg": {"model": {}}}, path)


def test_load_checkpoint_head_reads_binary_fc_weight_and_bias(tmp_path):
    ckpt_path = tmp_path / "runs_e007_A_fresh_best.pt"
    w = np.array([1.0, 2.0, 3.0, 4.0])
    _write_synthetic_checkpoint(ckpt_path, w, b=0.5)

    loaded_w, loaded_b, w_dim, sha = load_checkpoint_head(ckpt_path)

    np.testing.assert_allclose(loaded_w, w)
    assert loaded_b == pytest.approx(0.5)
    assert w_dim == 4
    assert isinstance(sha, str) and len(sha) == 64  # a real sha256 hex digest


def test_load_checkpoint_head_sha256_matches_the_real_checkpoint_file_bytes(tmp_path):
    """Confirms load_checkpoint_head's sha256 is the CHECKPOINT FILE's own
    hash (same value scripts/extract_model_embeddings.py's _sha256_file
    would compute over the same file), not some other placeholder --
    the exact identity the pairing guard depends on."""
    import hashlib

    ckpt_path = tmp_path / "runs_e007_A_fresh_best.pt"
    _write_synthetic_checkpoint(ckpt_path, np.array([1.0, 2.0]), b=0.0)
    _, _, _, sha = load_checkpoint_head(ckpt_path)
    assert sha == hashlib.sha256(ckpt_path.read_bytes()).hexdigest()


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
# -- proving shared primitives, not divergent math (Definition of Done #3,
# tightened per step3_modelspace_preextraction_gate_brief.md Item 2).
#
# Runs the WHOLE comparison inside a FRESH SUBPROCESS
# (_reliance_modelspace_parity_worker.py) launched with OPENBLAS_NUM_THREADS/
# MKL_NUM_THREADS/OMP_NUM_THREADS/NUMEXPR_NUM_THREADS already set to "1" --
# BLAS reads these at library-load time, so that process's own numpy
# initializes single-threaded natively, and run_battery's own subprocess-
# isolated crossfit workers (spawned FROM that process) inherit the same
# environment, single-threading the entire process tree with one
# mechanism (the brief's own "robust, order-independent way"), rather than
# needing threadpool_limits in this test process (which cannot reach
# run_battery's separately-spawned children at all) AND env vars for the
# children as two different mechanisms.
#
# PROOF-OF-PIN is asserted, not assumed: the worker reports
# threadpoolctl.threadpool_info()'s observed num_threads for every BLAS
# layer it can see, and its own effective env vars, both of which this
# test asserts are literally 1/"1" before trusting any residual as
# "identical math."
#
# MEASURED RESULT (this investigation, this machine): comparing float64 Z
# on both sides gave a max relative residual of ~6.4e-6 -- WELL above the
# brief's ~1e-9 expectation. Investigated rather than assumed benign:
# isolated with NO subprocess involved at all that this is run_battery's
# own _write_battery_npz (line 539) unconditionally casting Z to float32
# before its workers ever see it (real, pre-existing, protected
# behavior) -- a precision difference, amplified into a large RELATIVE
# residual specifically on a near-zero alignment value (~4e-6), not
# subprocess/BLAS non-determinism and not divergent code. Casting Z to
# float32 before EITHER side computes anything (matching what both paths
# actually operate on in real production use: extract_model_embeddings.py
# and this sibling's own load_model_space_embeddings both cast to
# float32 at load, same as run_battery's _write_battery_npz) gives an
# EXACT (0.0) relative residual on every compared field, both estimators,
# every fold -- confirmed by _reliance_modelspace_parity_worker.py, this
# test's own subprocess. Tolerance is set at rel=1e-9 (a decade+ above
# the true measured 0.0) for cross-platform/BLAS-version robustness while
# remaining tight enough to catch a genuine future divergence; the exact-
# equality assertions on discrete fields (fold_id, chosen rank,
# exceeds_random) are kept as bit-exact (== not approx).
#
# SCOPE: this is a float32 statement, matching real production precision
# on both sides -- NOT a claim about float64 behavior (float64-vs-
# run_battery's-forced-float32 comparison has a real, explained, larger
# residual purely from that precision difference, not from any divergence
# between the two implementations' math).
# ---------------------------------------------------------------------------

_PARITY_TOLERANCE_REL = 1e-9  # a decade+ above the measured 0.0 (see docstring above)


def test_numerical_parity_single_checkpoint_matches_run_battery_under_verified_blas_pin():
    import subprocess
    import sys as _sys

    worker_path = Path(__file__).resolve().parent / "_reliance_modelspace_parity_worker.py"
    env = dict(__import__("os").environ)
    env.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")

    proc = subprocess.run([_sys.executable, str(worker_path)], env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"parity worker failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"

    # The worker's heartbeat logs (from _log, reused verbatim) also go to
    # stdout -- the JSON result is always the LAST line printed.
    result = json.loads(proc.stdout.strip().splitlines()[-1])

    # PROOF-OF-PIN: don't trust the residual below unless BLAS was
    # genuinely single-threaded in the process that computed it.
    assert result["env_vars"] == {"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                   "OMP_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    assert result["blas_threads"], "no BLAS layer detected by threadpoolctl in the worker process"
    assert all(n == 1 for n in result["blas_threads"]), f"BLAS not single-threaded: {result['blas_threads']}"

    assert result["discrete_mismatches"] == []  # fold_id/chosen/exceeds_random bit-exact

    max_rel_residual = max(r["rel_residual"] for r in result["residuals"])
    worst = max(result["residuals"], key=lambda r: r["rel_residual"])
    assert max_rel_residual < _PARITY_TOLERANCE_REL, (
        f"max relative residual {max_rel_residual} exceeds tolerance {_PARITY_TOLERANCE_REL} -- "
        f"worst field: {worst}"
    )


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

    import hashlib

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    cache_root = tmp_path / "_embcache_modelspace"
    for i, run in enumerate(runs):
        w = np.random.default_rng(i).normal(size=dim)
        ckpt_path = ckpt_dir / f"runs_{run}_best.pt"
        _write_synthetic_checkpoint(ckpt_path, w, b=0.0)
        # The shard's checkpoint_sha256 must be THIS checkpoint's own real
        # file hash -- matching what extract_model_embeddings.py's real
        # _sha256_file(ckpt_path) would have recorded -- so the pairing
        # guard (Item 3b) passes for this legitimate, correctly-paired case.
        real_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
        _write_model_space_shard(cache_root / f"runs_{run}_best" / "03_DiffSSD" / "shard_0000.npz",
                                  n=n, dim=dim, corpus="diffssd", corpus_dir="03_DiffSSD", seed=i + 50,
                                  checkpoint_sha256=real_sha)

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


def test_main_raises_on_mispaired_embedding_and_head(tmp_path, monkeypatch):
    """THE PAIRING GUARANTEE, end to end: a model-space cache whose
    recorded checkpoint_sha256 does NOT match the checkpoint file the
    consumer actually loads the head from -- both still 256-d (here 6-d),
    so the dimension guard alone could never catch this -- must raise,
    never silently compute a finite, plausible, wrong reliance number."""
    monkeypatch.chdir(tmp_path)
    n, n_groups, dim = 200, 20, 6
    manifest_dir = tmp_path / "manifests"
    _write_manifest(manifest_dir, "diffssd", n=n, n_groups=n_groups, seed=7)

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    cache_root = tmp_path / "_embcache_modelspace"
    run = "e007_A_fresh"
    ckpt_path = ckpt_dir / f"runs_{run}_best.pt"
    _write_synthetic_checkpoint(ckpt_path, np.random.default_rng(0).normal(size=dim), b=0.0)
    # Deliberately WRONG checkpoint_sha256 -- simulates a naming collision
    # or a stale cache directory pairing this run's head with a DIFFERENT
    # checkpoint's embeddings.
    _write_model_space_shard(cache_root / f"runs_{run}_best" / "03_DiffSSD" / "shard_0000.npz",
                              n=n, dim=dim, corpus="diffssd", corpus_dir="03_DiffSSD", seed=99,
                              checkpoint_sha256="sha_of_a_totally_different_checkpoint")

    with pytest.raises(RuntimeError, match="MISPAIRED"):
        main([
            "--model-space-cache-root", str(cache_root),
            "--manifest-dir", str(manifest_dir),
            "--ckpt-dir", str(ckpt_dir),
            "--checkpoints", run,
            "--corpus", "diffssd",
            "--factor", "generator_id",
            "--ranks", "1", "2", "3",
            "--n-boot", "0",
            "--seed", "13",
            "--out", str(tmp_path / "out.json"),
        ])


def test_gate_reads_modelspace_output_and_leaves_not_estimable(tmp_path, monkeypatch):
    """The definitive proof the chain is wired: scripts/run_gate.py, totally
    UNMODIFIED, reads this sibling's output and C2/C4/C6/C7 move OFF
    not_estimable/pending_input, given a factor-corpus map + synthetic
    EERs for all THREE checkpoints (C7 specifically requires
    >=3 checkpoints with reliance + EER data to ever leave
    not_estimable/pending -- criterion_7_no_collapse's own
    `if len(rows) < 3` gate, run_gate.py line 785 -- so this must use the
    real e007_A/B/C-style 3-checkpoint scenario, not the 2-checkpoint one
    other tests in this file use for schema/merge checks only).

    THIS IS A DISCOVERY INSTRUMENT, NOT A FORMALITY (per the hardening
    brief): if C7 ever regresses to not_estimable here, that is a real
    finding -- the extract-consume-w-metrics-gate chain is not fully wired
    for C7 -- and must be reported, not forced green."""
    runs = ("e007_A_fresh", "e007_B_fresh", "e007_C_xlsr_fresh")
    out_path = _build_and_run_modelspace_cli(tmp_path, monkeypatch, runs=runs)

    records, warnings = run_gate.load_phase_a_inputs([out_path])
    assert warnings == []
    assert len(records) == 1

    # clean_corpora (inthewild/replaydf/ai4t) for every checkpoint, per
    # criterion_7_no_collapse's own default -- plus the generator_id factor's
    # mapped eval corpus (reusing "replaydf", also one of the clean_corpora;
    # harmless for this off-not_estimable check, which doesn't examine sign).
    eers = {run: {"inthewild": 0.10 + i * 0.05, "replaydf": 0.20 + i * 0.05, "ai4t": 0.30 + i * 0.05}
            for i, run in enumerate(runs)}
    factor_corpus_map = {"generator_id": "replaydf"}

    c2 = run_gate.criterion_2_association(records, eers, factor_corpus_map)
    c4 = run_gate.criterion_4_intervention_vs_random(records)
    c6 = run_gate.criterion_6_estimator_agreement(records)
    c7 = run_gate.criterion_7_no_collapse(records, eers, factor_corpus_map)

    assert c2["status"] != run_gate.STATUS_NOT_ESTIMABLE
    assert c4["status"] != run_gate.STATUS_NOT_ESTIMABLE
    assert c6["status"] in (run_gate.STATUS_PASS, run_gate.STATUS_FAIL)  # fully decided either way
    assert c7["status"] != run_gate.STATUS_NOT_ESTIMABLE, (
        f"C7 stayed not_estimable on the sibling's populated output -- STOP finding: "
        f"{c7['numbers'].get('per_battery')}"
    )
    assert c7["status"] in (run_gate.STATUS_PASS, run_gate.STATUS_FAIL), (
        "C7 should be fully decided (not merely pending) given 3 checkpoints' worth of "
        f"reliance + EER data: {c7['numbers'].get('per_battery')}"
    )


# ---------------------------------------------------------------------------
# C4 READ-PATH ADJUDICATION (step3_modelspace_preextraction_gate_brief.md
# Item 1, Task 1b). Verified against run_gate.py's REAL, current source
# (criterion_4_intervention_vs_random, line 633 as of this writing) --
# NOT against this sibling's own docstring, which turned out to be wrong.
#
# VERDICT: Case B. run_gate.py's criterion_4_intervention_vs_random reads
# ONLY `fold["effect"]["projection_removal_control"]` (the fold level) --
# it never iterates `per_checkpoint[ckpt]["projection_removal_control"]`
# at all, unlike _per_checkpoint_reliance (used by C2/C7), which correctly
# iterates every checkpoint. merge_checkpoint_estimator_results sets the
# fold-level projection_removal_control to the PRIMARY checkpoint's value
# only (sorted(checkpoints)[0], e.g. "e007_A_fresh") -- so in the
# model-space regime, C4's verdict IS the alphabetically-first
# checkpoint's causal-intervention result alone. Checkpoints B and C's
# controls are silently discarded. This was harmless in the cache-space
# battery (the control was checkpoint-independent by construction, so
# fold-level == every per-checkpoint value there) and becomes a genuine
# scientific defect here, at n=3 (soon n=6 with backbone #2): the paper's
# per-model-instance causal claim is not what C4 actually computes.
#
# NOT PATCHED HERE: C4 is a pre-registered criterion (docs/gate_prereg.md);
# changing its cross-checkpoint aggregation is a human/prereg decision,
# not something this session makes unilaterally. RECOMMENDED FIX (gate-side
# only, run_reliance_modelspace.py's merge already populates
# per_checkpoint[ckpt]["projection_removal_control"] correctly for every
# checkpoint): make criterion_4_intervention_vs_random iterate
# per_checkpoint[ckpt]["projection_removal_control"] across ALL
# checkpoints -- the same pattern _per_checkpoint_reliance already uses --
# instead of the fold-level field.
# ---------------------------------------------------------------------------


def _make_hand_built_modelspace_battery(weak_run="ckA"):
    """Minimal, hand-built battery record (bypassing the full crossfit
    machinery for full, deterministic control) with 3 checkpoints whose
    projection_removal_control GENUINELY differs: `weak_run` (the
    PRIMARY checkpoint under merge_checkpoint_estimator_results'
    sorted()[0] convention -- "ckA" sorts first) shows a weak/no causal
    effect; the other two show a strong one."""
    def weak_control():
        return dict(true_effect=0.01, random_effects=[0.01] * 20, random_mean=0.01, random_std=0.001,
                    task_direction_effect=0.01, exceeds_random=False)

    def strong_control():
        return dict(true_effect=0.9, random_effects=[0.05] * 20, random_mean=0.05, random_std=0.01,
                    task_direction_effect=0.9, exceeds_random=True)

    runs = ["ckA", "ckB", "ckC"]
    controls = {run: (weak_control() if run == weak_run else strong_control()) for run in runs}
    per_checkpoint = {
        run: dict(alignment=0.1, r_var=0.1, prediction_change={}, prediction_change_control={},
                  projection_removal_control=controls[run])
        for run in runs
    }
    primary_run = sorted(runs)[0]  # matches merge_checkpoint_estimator_results' own convention

    def make_fold():
        return dict(fold_id=0, chosen={"k": 1}, selection_score=0.9, n_selection=10, n_effect=5,
                    effect=dict(per_checkpoint=per_checkpoint, factor_separation_score=0.5, leace={}, inlp={},
                                projection_removal_control=controls[primary_run]))

    battery = dict(
        name="synthetic_modelspace_battery", corpus="diffssd", factor="generator_id", grouping="source_id",
        n_rows=100, n_levels=4, n_groups=10, grouping_degenerate=False,
        ranks_requested=[1], ranks_valid=[1], layer_mode="model_space",
        estimators=dict(lda=dict(fold_results=[make_fold()], status="ok", timed_out=False),
                         probe=dict(fold_results=[make_fold()], status="ok", timed_out=False)),
    )
    prereg_candidate = dict(name="synthetic_modelspace_battery", headline_metric="r_var",
                             stable_rank_window=[1], estimators_agree_sign=True, cis_overlap=True,
                             n_groups=10, grouping_degenerate=False)
    return [dict(battery=battery, prereg_candidate=prereg_candidate, checkpoints={}, w_metrics={}, join_stats={},
                 source="synthetic")]


def test_c4_current_behavior_reads_only_the_primary_checkpoint():
    """Documents the CURRENT (Case B) behavior as it stands today: with
    the primary checkpoint's ("ckA", alphabetically first) control
    deliberately weak while the other two are strong, C4 currently
    reports the WEAK verdict -- confirming it reflects ckA alone, not an
    aggregate across all three checkpoints."""
    records = _make_hand_built_modelspace_battery(weak_run="ckA")
    result = run_gate.criterion_4_intervention_vs_random(records)
    assert result["status"] == run_gate.STATUS_FAIL  # ckA's weak control, not ckB/ckC's strong ones


@pytest.mark.xfail(reason=(
    "STOP finding (step3_modelspace_preextraction_gate_brief.md Item 1, Task 1b, Case B): "
    "run_gate.py's criterion_4_intervention_vs_random (line 633, as of this writing) reads ONLY "
    "fold['effect']['projection_removal_control'] -- the merge's designated PRIMARY checkpoint "
    "('ckA'/alphabetically-first) -- and never iterates "
    "per_checkpoint[ckpt]['projection_removal_control'] for the other checkpoints at all. In the "
    "model-space regime this silently discards checkpoints B and C's causal-intervention results: "
    "C4's verdict is the alphabetically-first checkpoint's alone, at n=3 (soon n=6). C4 is a "
    "pre-registered criterion (docs/gate_prereg.md) -- fixing the cross-checkpoint aggregation is "
    "a human/prereg decision, not something this session patches unilaterally. RECOMMENDED FIX "
    "(gate-side only -- run_reliance_modelspace.py's merge already populates "
    "per_checkpoint[ckpt]['projection_removal_control'] correctly for every checkpoint, so the "
    "consumer needs no change): make criterion_4_intervention_vs_random iterate "
    "per_checkpoint[ckpt]['projection_removal_control'] across ALL checkpoints, the same pattern "
    "_per_checkpoint_reliance already uses for alignment/r_var, instead of the fold-level field. "
    "This test documents the desired (fixed) behavior and will start passing the moment that fix "
    "is ratified and lands."
))
def test_c4_should_aggregate_across_all_checkpoints_once_fixed():
    records = _make_hand_built_modelspace_battery(weak_run="ckA")
    result = run_gate.criterion_4_intervention_vs_random(records)
    # Under a corrected implementation, 2 of the 3 checkpoints (ckB, ckC)
    # show a strong, controls-exceeding effect -- a majority signal, so a
    # per-checkpoint-aggregating C4 should PASS, not reflect ckA alone.
    assert result["status"] == run_gate.STATUS_PASS


def test_build_parser_defaults():
    args = build_parser().parse_args([])
    assert args.n_boot == 1000
    assert args.seed == 13
    assert len(args.checkpoints) == 3
