"""Tests for scripts/reliance_mde_injection.py -- the gamma-injection
minimum-detectable-effect (MDE) tool for the model-space causal-reliance
apparatus.

SYNTHETIC ONLY -- no real Phase B embedding cache, no checkpoint .pt files,
no network (none exist on this machine; the real run is on the collaborator
machine). The detection tests build a synthetic (n x 256) Z with a planted
factor subspace ORTHOGONAL to a synthetic head w (the same planted-factor
construction tests/test_reliance_modelspace.py and tests/conftest.py's
planted_factor_data already use), inject a known reliance along the top
factor direction, and run the REAL intervention pipeline
(run_checkpoint_crossfit, imported unmodified) -- proving the tool detects a
planted effect it should detect and stays silent on the one it shouldn't.
The cache-load / sha256-pairing / never-crash paths are exercised with
in-memory monkeypatches and absent directories, never by writing a cache or
a .pt to disk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from reliance_mde_injection import (  # noqa: E402
    DEFAULT_GAMMAS,
    DEFAULT_MDE_MAX_ROWS_PER_LEVEL,
    DEFAULT_MDE_N_BOOT,
    STATUS_NOT_ESTIMABLE,
    STATUS_OK,
    build_parser,
    build_report,
    derive_gamma_star,
    derive_paper_sentence,
    estimate_checkpoint,
    estimate_checkpoint_from_arrays,
    inject_head,
    main,
    plant_top_factor_direction,
    summarize_cell,
)
import reliance_mde_injection as mde  # noqa: E402 -- for monkeypatching the imported loader


# ---------------------------------------------------------------------------
# Synthetic planted-factor checkpoint: an (n, d) Z with a rank-k_factor factor
# subspace orthogonal to the head w by construction (so the UNMODIFIED head
# does not rely on the factor -- the specificity arm), plus grouped structure
# so the crossfit's grouped folds are non-trivial. Same construction as
# tests/test_reliance_modelspace.py::_planted_w_and_U + its Z assembly.
# ---------------------------------------------------------------------------


def _planted_checkpoint(seed, d=256, n=400, k_factor=3, n_groups=5):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d)
    w /= np.linalg.norm(w)
    M = rng.normal(size=(d, k_factor))
    M = M - np.outer(w, w @ M)  # factor subspace orthogonal to w by construction
    U, _, _ = np.linalg.svd(M, full_matrices=False)
    U_true = U[:, :k_factor]

    groups_raw = rng.integers(0, n_groups, size=n)
    group_offset = rng.normal(scale=0.5, size=(n_groups, d))[groups_raw]
    y = rng.integers(0, 2, size=n)
    factor_levels = rng.integers(0, 4, size=n)
    raw_centers = rng.normal(size=(4, k_factor))
    raw_centers -= raw_centers.mean(axis=0, keepdims=True)
    Uc, Sc, Vtc = np.linalg.svd(raw_centers, full_matrices=False)
    factor_centers = Uc @ np.diag(np.full_like(Sc, 3.0)) @ Vtc

    Z = (np.outer((y * 2 - 1).astype(float), w) * 3.0
         + factor_centers[factor_levels] @ U_true.T
         + group_offset
         + rng.normal(size=(n, d)))
    factor = np.array([f"gen{i}" for i in factor_levels], dtype=object)
    groups = np.array([f"grp{i}" for i in groups_raw], dtype=object)
    head = dict(w=w, b=0.0, w_dim=d)
    return dict(Z=Z, factor=factor, y=y, groups=groups, head=head, w_true=w, U_true=U_true, d=d)


# Hand-built fold shapes (no crossfit) -- same style as
# tests/test_reliance_sensitivity.py's _prc/_pc/_fold helpers.
def _prc(true_effect, random_mean, random_std, exceeds_random, task_direction_effect=5.0):
    return dict(true_effect=true_effect, random_effects=[random_mean] * 20, random_mean=random_mean,
                random_std=random_std, task_direction_effect=task_direction_effect,
                exceeds_random=exceeds_random)


def _pc(decision_flip_rate):
    return dict(mean_abs_logit_change=0.0, rmse_logit_change=0.0, mean_prob_change=0.0,
                decision_flip_rate=decision_flip_rate)


def _fold(run, pcc, pc, fold_id=0):
    return dict(fold_id=fold_id, chosen={"k": 1}, n_effect=100,
                effect=dict(per_checkpoint={run: dict(prediction_change=pc, prediction_change_control=pcc)}))


# ---------------------------------------------------------------------------
# (d) The injected-head norm relation: ||w_gamma - w|| == gamma * ||w||
# ---------------------------------------------------------------------------


def test_inject_head_norm_relation_holds_exactly():
    rng = np.random.default_rng(0)
    w = rng.normal(size=256)
    u1 = rng.normal(size=256)
    u1 /= np.linalg.norm(u1)  # inject_head assumes u1 is unit-norm (plant_top_factor_direction guarantees it)
    for gamma in (0.0, 0.01, 0.05, 0.2, 0.5, 1.0):
        w_gamma = inject_head(w, u1, gamma)
        assert np.linalg.norm(w_gamma - w) == pytest.approx(gamma * np.linalg.norm(w), rel=1e-12, abs=1e-12)


def test_inject_head_gamma_zero_is_the_identity():
    rng = np.random.default_rng(1)
    w = rng.normal(size=64)
    u1 = rng.normal(size=64)
    u1 /= np.linalg.norm(u1)
    np.testing.assert_array_equal(inject_head(w, u1, 0.0), w)


def test_plant_top_factor_direction_is_unit_and_lies_in_the_factor_subspace():
    # Well-determined regime (small d, large n -- the same d=20 shape
    # tests/conftest.py's planted_factor_data uses) so LDA can actually
    # RECOVER the planted subspace: at the production d=256 with a few hundred
    # rows LDA is underdetermined and u1 no longer aligns with the abstract
    # U_true, yet detection still works because plant and estimator both use
    # LDA and agree -- proven separately by the end-to-end tests below.
    data = _planted_checkpoint(seed=3, d=20, n=2000)
    u1 = plant_top_factor_direction(data["Z"], data["factor"], data["y"])
    assert u1.shape == (data["d"],)
    assert np.linalg.norm(u1) == pytest.approx(1.0, abs=1e-9)
    # u1 lives (almost entirely) in the planted factor subspace, not along
    # w_true -- its squared projection onto U_true dominates its projection onto w.
    in_factor = float(np.sum((data["U_true"].T @ u1) ** 2))
    on_w = float((data["w_true"] @ u1) ** 2)
    assert in_factor > 0.9
    assert on_w < 0.1


# ---------------------------------------------------------------------------
# summarize_cell: fold aggregation + the C4 >=50%-of-folds majority rule
# ---------------------------------------------------------------------------


def test_summarize_cell_aggregates_folds_and_trips_on_majority():
    run = "ckA"
    folds = [
        _fold(run, _prc(0.9, 0.05, 0.01, exceeds_random=True), _pc(0.10), fold_id=0),
        _fold(run, _prc(0.8, 0.05, 0.01, exceeds_random=True), _pc(0.20), fold_id=1),
        _fold(run, _prc(0.02, 0.05, 0.01, exceeds_random=False), _pc(0.00), fold_id=2),
    ]
    cell = summarize_cell(folds, run, gamma=0.5)
    assert cell["n_folds"] == 3
    assert cell["true_effect"] == pytest.approx(np.mean([0.9, 0.8, 0.02]))
    assert cell["decision_flip_rate"] == pytest.approx(np.mean([0.10, 0.20, 0.00]))
    assert cell["exceeds_random_fraction"] == pytest.approx(2 / 3)
    assert cell["n_folds_exceeding"] == 2
    assert cell["exceeds_random"] is True  # 2/3 >= 0.5


def test_summarize_cell_minority_of_folds_does_not_trip():
    run = "ckA"
    folds = [
        _fold(run, _prc(0.9, 0.05, 0.01, exceeds_random=True), _pc(0.1), fold_id=0),
        _fold(run, _prc(0.02, 0.05, 0.01, exceeds_random=False), _pc(0.0), fold_id=1),
        _fold(run, _prc(0.02, 0.05, 0.01, exceeds_random=False), _pc(0.0), fold_id=2),
    ]
    cell = summarize_cell(folds, run, gamma=0.1)
    assert cell["exceeds_random_fraction"] == pytest.approx(1 / 3)
    assert cell["exceeds_random"] is False  # 1/3 < 0.5


def test_summarize_cell_pooled_flag_uses_the_reported_fold_means():
    run = "ckA"
    # Every fold individually below its own 2-sigma bar (so majority=False), but
    # the fold-mean true_effect sits above the fold-mean bar -> pooled=True.
    # Documents that the two flags can legitimately disagree, and that
    # `exceeds_random` (majority) is the authoritative one.
    folds = [_fold(run, _prc(0.11, 0.05, 0.01, exceeds_random=False), _pc(0.0), fold_id=i) for i in range(3)]
    cell = summarize_cell(folds, run, gamma=0.2)
    assert cell["exceeds_random"] is False               # majority rule (C4) -- authoritative
    assert cell["exceeds_random_pooled"] is True         # 0.11 > 0.05 + 2*0.01 on the fold-means


# ---------------------------------------------------------------------------
# (c) gamma_star: finite when something trips, None when nothing does, plus
# honest non-monotonicity reporting
# ---------------------------------------------------------------------------


def test_derive_gamma_star_is_the_smallest_tripping_gamma():
    cells = [dict(gamma=0.0, exceeds_random=False), dict(gamma=0.1, exceeds_random=False),
             dict(gamma=0.2, exceeds_random=True), dict(gamma=0.5, exceeds_random=True)]
    out = derive_gamma_star(cells)
    assert out["gamma_star"] == 0.2
    assert out["monotonic"] is True
    assert out["non_monotonic_note"] is None
    assert [f["exceeds_random"] for f in out["trip_flags"]] == [False, False, True, True]


def test_derive_gamma_star_is_none_when_nothing_trips():
    cells = [dict(gamma=0.0, exceeds_random=False), dict(gamma=0.5, exceeds_random=False)]
    out = derive_gamma_star(cells)
    assert out["gamma_star"] is None
    assert out["monotonic"] is True


def test_derive_gamma_star_reports_non_monotonicity_without_masking_it():
    cells = [dict(gamma=0.0, exceeds_random=False), dict(gamma=0.1, exceeds_random=True),
             dict(gamma=0.2, exceeds_random=False), dict(gamma=0.5, exceeds_random=True)]
    out = derive_gamma_star(cells)
    assert out["gamma_star"] == 0.1                 # still the smallest tripping gamma
    assert out["monotonic"] is False                # but the threshold is not clean
    assert out["non_monotonic_note"] is not None
    assert "0.2" in out["non_monotonic_note"]       # names the offending larger gamma


# ---------------------------------------------------------------------------
# paper_sentence reduction + the specificity-arm guard
# ---------------------------------------------------------------------------


def _ok_ckpt(gamma_star_lda, gamma_star_probe, zero_trips=False):
    def est(gs):
        return dict(gammas=[dict(gamma=0.0, exceeds_random=zero_trips, true_effect=1e-6),
                            dict(gamma=0.5, exceeds_random=gs is not None)],
                    gamma_star=gs, monotonic=True, non_monotonic_note=None)
    return dict(status=STATUS_OK, estimators=dict(lda=est(gamma_star_lda), probe=est(gamma_star_probe)))


def test_derive_paper_sentence_reports_gamma_star_range():
    per_battery = {"b1": dict(per_checkpoint=dict(ckA=_ok_ckpt(0.5, 0.2), ckB=_ok_ckpt(0.1, None)))}
    ps = derive_paper_sentence(per_battery, gammas=[0.0, 0.5])
    assert ps["gamma_star_min"] == 0.1
    assert ps["gamma_star_max"] == 0.5
    assert ps["n_tripped"] == 3          # 0.5, 0.2, 0.1
    assert ps["n_never_tripped"] == 1    # ckB probe
    assert ps["specificity_violations"] == []
    assert "minimum detectable effect" in ps["text"]


def test_derive_paper_sentence_flags_a_specificity_arm_violation():
    per_battery = {"b1": dict(per_checkpoint=dict(ckA=_ok_ckpt(0.5, 0.5, zero_trips=True)))}
    ps = derive_paper_sentence(per_battery, gammas=[0.0, 0.5])
    assert ps["specificity_violations"]  # non-empty
    assert "SPECIFICITY ARM VIOLATED" in ps["text"]


# ---------------------------------------------------------------------------
# (e) Output JSON schema keys + atomic write
# ---------------------------------------------------------------------------


def test_build_report_has_schema_keys_and_writes_atomically(tmp_path):
    run = "ckA"
    cell0 = summarize_cell([_fold(run, _prc(1e-6, 0.05, 0.01, exceeds_random=False), _pc(0.0))], run, gamma=0.0)
    cell5 = summarize_cell([_fold(run, _prc(0.9, 0.05, 0.01, exceeds_random=True), _pc(0.3))], run, gamma=0.5)
    est = dict(gammas=[cell0, cell5], **derive_gamma_star([cell0, cell5]))
    per_battery = {"b1": dict(name="b1", corpus="diffssd", factor="generator_id", grouping="source_id",
                              per_checkpoint={run: dict(status=STATUS_OK, n_rows=400, n_groups=5, n_outer=5,
                                                        valid_ranks=[1, 2, 3], w_norm=1.0,
                                                        estimators=dict(lda=est, probe=est))})}
    report = build_report(per_battery, params=dict(seed=13), gammas=[0.0, 0.5])

    for key in ("schema_version", "generated_at", "git_sha", "tool", "detection", "params",
                "per_battery", "paper_sentence", "warnings"):
        assert key in report
    for key in ("true_effect", "random_mean", "random_std", "exceeds_random", "decision_flip_rate"):
        assert key in report["per_battery"]["b1"]["per_checkpoint"]["ckA"]["estimators"]["lda"]["gammas"][0]

    out_path = tmp_path / "sub" / "mde.json"  # parent created by the atomic writer
    from reliance_mde_injection import _write_json_atomic
    _write_json_atomic(out_path, report)
    assert out_path.exists()
    assert not out_path.with_suffix(out_path.suffix + ".tmp").exists()  # no leftover tmp
    reloaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert reloaded["per_battery"]["b1"]["per_checkpoint"]["ckA"]["estimators"]["lda"]["gamma_star"] == 0.5


# ---------------------------------------------------------------------------
# (f) never-crash: a missing cache / mispaired sha256 -> not_estimable, no raise
# ---------------------------------------------------------------------------


def test_missing_cache_checkpoint_is_not_estimable_without_crashing(tmp_path):
    head = dict(w=np.zeros(6), b=0.0, w_dim=6, checkpoint_stem="ckX", checkpoint_sha256="abc")
    spec = dict(name="b1", corpus="diffssd", factor="generator_id", grouping="source_id")
    res = estimate_checkpoint(spec, "ckX", head, tmp_path / "no_such_cache", pd.DataFrame(), "03_DiffSSD",
                              gammas=[0.0, 0.5], ranks=[1, 2], seed=13)  # must not raise
    assert res["status"] == STATUS_NOT_ESTIMABLE
    assert "cache not found" in res["reason"]


def test_mispaired_sha256_is_rejected_by_the_active_pairing_guard(monkeypatch):
    """The sha256 (embedding, head) pairing guard stays active: a cache whose
    recorded checkpoint_sha256 disagrees with the head file's own sha256 is
    refused as not_estimable, never ablated. Exercised in-memory (no on-disk
    cache written) so the check is proven without a real shard."""
    monkeypatch.setattr(mde, "load_model_space_embeddings",
                        lambda cache_root, stem, corpus_dir: (np.array(["a.wav"]),
                                                              np.zeros((1, 6), dtype=np.float32), "sha_WRONG"))
    head = dict(w=np.zeros(6), b=0.0, w_dim=6, checkpoint_stem="ckX", checkpoint_sha256="sha_RIGHT")
    spec = dict(name="b1", corpus="diffssd", factor="generator_id", grouping="source_id")
    res = estimate_checkpoint(spec, "ckX", head, "unused", pd.DataFrame(), "03_DiffSSD",
                              gammas=[0.0], ranks=[1], seed=13)
    assert res["status"] == STATUS_NOT_ESTIMABLE
    assert "MISPAIRED" in res["reason"]


def test_cache_without_recorded_sha256_is_rejected(monkeypatch):
    monkeypatch.setattr(mde, "load_model_space_embeddings",
                        lambda cache_root, stem, corpus_dir: (np.array(["a.wav"]),
                                                              np.zeros((1, 6), dtype=np.float32), None))
    head = dict(w=np.zeros(6), b=0.0, w_dim=6, checkpoint_stem="ckX", checkpoint_sha256="sha")
    spec = dict(name="b1", corpus="diffssd", factor="generator_id", grouping="source_id")
    res = estimate_checkpoint(spec, "ckX", head, "unused", pd.DataFrame(), "03_DiffSSD",
                              gammas=[0.0], ranks=[1], seed=13)
    assert res["status"] == STATUS_NOT_ESTIMABLE
    assert "no recorded checkpoint_sha256" in res["reason"]


# ---------------------------------------------------------------------------
# CLI defaults + full main() never-crash (no .pt, no cache -- every checkpoint
# not_estimable, JSON still written, process returns without raising)
# ---------------------------------------------------------------------------


def test_build_parser_defaults():
    args = build_parser().parse_args([])
    assert args.n_boot == DEFAULT_MDE_N_BOOT == 0
    assert args.max_rows_per_level == DEFAULT_MDE_MAX_ROWS_PER_LEVEL == 500
    assert args.seed == 13
    assert tuple(args.gammas) == DEFAULT_GAMMAS
    assert len(args.checkpoints) == 3


def _write_manifest(manifest_dir: Path, corpus: str, n: int = 20, n_groups: int = 5) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    rows = [dict(utt_id=f"{corpus}/f{i:04d}", path=f"datasets/03_DiffSSD/f{i:04d}.wav",
                 target=int(rng.integers(0, 2)), corpus=corpus, split="train", attack="na",
                 bona_fide_source="na", source_id=f"src{i % n_groups}", speaker_id="NA",
                 generator_id=f"gen{i % 4}", channel_id="NA", language="NA", platform_id="NA")
            for i in range(n)]
    pd.DataFrame(rows).to_csv(manifest_dir / f"{corpus}.csv", index=False)


def test_main_never_crashes_and_writes_report_when_no_checkpoints_or_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest_dir = tmp_path / "manifests"
    _write_manifest(manifest_dir, "diffssd")  # a manifest CSV is not a cache/.pt
    out_path = tmp_path / "mde.json"

    main([  # ckpt-dir and cache-root are empty -> every checkpoint not_estimable, but no crash
        "--model-space-cache-root", str(tmp_path / "no_cache"),
        "--ckpt-dir", str(tmp_path / "no_ckpts"),
        "--manifest-dir", str(manifest_dir),
        "--checkpoints", "e007_A_fresh", "e007_B_fresh",
        "--corpus", "diffssd", "--factor", "generator_id",
        "--gammas", "0", "0.5",
        "--out", str(out_path),
    ])

    assert out_path.exists()
    assert not out_path.with_suffix(out_path.suffix + ".tmp").exists()
    report = json.loads(out_path.read_text(encoding="utf-8"))
    battery = report["per_battery"]["diffssd_generator_by_source"]
    assert set(battery["per_checkpoint"]) == {"e007_A_fresh", "e007_B_fresh"}
    assert all(c["status"] == STATUS_NOT_ESTIMABLE for c in battery["per_checkpoint"].values())
    assert report["paper_sentence"]["n_tripped"] == 0


# ---------------------------------------------------------------------------
# (a) + (b) + (c) END TO END on the REAL intervention pipeline: the specificity
# arm (gamma=0 never trips) and detection (a large gamma always trips, with a
# finite gamma_star) across >=2 seeds. This is the only pair of tests that
# runs the full run_checkpoint_crossfit; kept to two seeds x {0.0, 0.5} for
# speed (the mid-grid gammas are covered by the unit tests above).
# ---------------------------------------------------------------------------


_CELL_SCHEMA_KEYS = {"gamma", "true_effect", "random_mean", "random_std", "exceeds_random",
                     "exceeds_random_fraction", "exceeds_random_pooled", "n_folds", "n_folds_exceeding",
                     "decision_flip_rate", "task_direction_effect"}


@pytest.mark.parametrize("seed", [0, 1])
def test_specificity_arm_and_detection_end_to_end(seed):
    data = _planted_checkpoint(seed)
    logs: list[str] = []
    res = estimate_checkpoint_from_arrays(
        data["Z"], data["factor"], data["y"], data["groups"], data["head"], run="ckA",
        gammas=[0.0, 0.5], ranks=[1, 2, 3], seed=13, max_rows_per_level=None, log=logs.append,
    )
    assert res["status"] == STATUS_OK

    # The group-count cost-model print is emitted (kept for the operator).
    assert any("n_groups=" in m and "crossfit runs" in m for m in logs)

    for est in ("lda", "probe"):
        cells = {c["gamma"]: c for c in res["estimators"][est]["gammas"]}
        assert _CELL_SCHEMA_KEYS <= set(cells[0.0])            # (e) real-cell schema
        # (a) specificity arm: the unmodified head must NOT trip
        assert cells[0.0]["exceeds_random"] is False, f"{est} gamma=0 tripped (seed={seed})"
        # (b) a large injected reliance MUST trip
        assert cells[0.5]["exceeds_random"] is True, f"{est} gamma=0.5 did not trip (seed={seed})"
        # (c) gamma_star reported, finite, and equal to the smallest tripping gamma tested
        assert res["estimators"][est]["gamma_star"] == 0.5
        assert res["estimators"][est]["monotonic"] is True


def test_gamma_star_is_none_when_only_gamma_zero_is_swept():
    """Companion to the detection test: sweeping ONLY the unmodified head
    (gamma=0) trips nowhere, so gamma_star is None (absent), end to end."""
    data = _planted_checkpoint(seed=0)
    res = estimate_checkpoint_from_arrays(
        data["Z"], data["factor"], data["y"], data["groups"], data["head"], run="ckA",
        gammas=[0.0], ranks=[1, 2, 3], seed=13, max_rows_per_level=None, log=lambda _m: None,
    )
    assert res["status"] == STATUS_OK
    for est in ("lda", "probe"):
        assert res["estimators"][est]["gamma_star"] is None
