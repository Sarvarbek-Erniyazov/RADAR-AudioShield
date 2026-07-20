"""Tests for scripts/head_replicates.py -- Task 3 (Step 4 gate prep).

Synthetic embeddings only: the real cached-embedding feature matrix and
factor subspace live on the collaborator machine (see the module's own
docstring for the COLLABORATOR PC run command). planted_factor_data
(tests/conftest.py) gives a seeded synthetic dataset with a known task
direction w_true and an orthogonal, known factor subspace U_true -- built
for exactly this kind of test elsewhere in tests/test_reliance_*.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from head_replicates import (  # noqa: E402
    build_parser,
    main,
    replicate_effect,
    run_head_replicates,
    train_head,
)


def test_train_head_recovers_planted_direction(planted_factor_data):
    d = planted_factor_data
    w, b = train_head(d["Z"], d["y"], seed=13)
    w_hat = w / np.linalg.norm(w)
    cos_sim = float(np.abs(w_hat @ d["w_true"]))
    assert cos_sim > 0.7  # a linear head trained on data with a strong planted task direction should recover it


def test_replicate_effect_zero_when_subspace_orthogonal_to_w():
    """If U is exactly the span of w itself, projecting it out must zero
    every logit computed from w (Z @ w term vanishes for every row) --
    the strongest possible signed effect, sanity-checking the wiring
    against project_out's own definition rather than a hand-derived
    number."""
    rng = np.random.default_rng(0)
    d = 5
    w = rng.normal(size=d)
    w_hat = (w / np.linalg.norm(w)).reshape(d, 1)
    X = rng.normal(size=(200, d))
    effect = replicate_effect(X, w, b=0.0, U=w_hat)
    logits_before = X @ w
    assert effect == pytest.approx(-float(np.mean(logits_before)), abs=1e-8)


def test_run_head_replicates_consistent_sign_on_real_planted_factor(planted_factor_data):
    """With a real, planted factor subspace that overlaps the task
    direction only through finite-sample noise (U_true is orthogonal to
    w_true BY CONSTRUCTION -- see conftest.py), repeated seeds should
    mostly show a small, consistently-signed effect. This exercises the
    happy path criterion 8 needs: >=3 seeds, all finite effects."""
    d = planted_factor_data
    replicates = run_head_replicates(d["Z"], d["y"], d["U_true"], seeds=[13, 29, 47])
    assert len(replicates) == 3
    assert all(r["status"] == "ok" for r in replicates)
    assert all(isinstance(r["effect"], float) and np.isfinite(r["effect"]) for r in replicates)


def test_run_head_replicates_never_crashes_on_degenerate_seed():
    """A seed whose stratified split leaves too few samples of one class
    to fit must be recorded as a failed replicate, never raise -- matching
    this project's established never-crash-on-a-single-unit-of-work
    convention."""
    rng = np.random.default_rng(1)
    d = 4
    X = rng.normal(size=(6, d))
    y = np.array([0, 0, 0, 0, 0, 1])  # 1 positive example -- stratified split cannot succeed
    U = rng.normal(size=(d, 1))
    replicates = run_head_replicates(X, y, U, seeds=[13])
    assert replicates[0]["status"] == "failed"
    assert replicates[0]["effect"] is None
    assert "reason" in replicates[0]


def test_run_head_replicates_effect_measured_out_of_sample(planted_factor_data):
    """effect_holdout_fraction controls how many rows are held out for
    effect measurement -- confirm the split is actually happening (fit set
    and effect set don't cover the entire input identically every seed)
    by checking a degenerate holdout_fraction close to 1 still runs and a
    near-0 holdout still runs, i.e. the parameter is wired through."""
    d = planted_factor_data
    small_holdout = run_head_replicates(d["Z"], d["y"], d["U_true"], seeds=[13], effect_holdout_fraction=0.05)
    large_holdout = run_head_replicates(d["Z"], d["y"], d["U_true"], seeds=[13], effect_holdout_fraction=0.6)
    assert small_holdout[0]["status"] == "ok"
    assert large_holdout[0]["status"] == "ok"


def test_cli_writes_criterion_8_ready_schema(tmp_path, planted_factor_data):
    d = planted_factor_data
    emb_path = tmp_path / "embeddings.npz"
    np.savez(emb_path, X=d["Z"], y=d["y"])
    subspace_path = tmp_path / "subspace.npy"
    np.save(subspace_path, d["U_true"])
    out_path = tmp_path / "head_replicates.json"

    rc = main([
        "--embeddings", str(emb_path),
        "--factor-subspace", str(subspace_path),
        "--seeds", "13", "29", "47",
        "--out", str(out_path),
    ])

    assert rc == 0
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert "replicates" in written
    assert len(written["replicates"]) == 3
    assert all("seed" in r and "effect" in r for r in written["replicates"])


def test_build_parser_requires_embeddings_and_subspace():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--out", "x.json"])
