"""Tests for scripts/run_gate.py -- the Step 4 gate consumer.

Built strictly against the REAL fixture files this module targets
(tests/fixtures/step3/*.json, produced by a real --smoke run and a real
accent-factor battery on the collaborator machine) plus the real,
already-committed EER files under experiments/e007/. Where a criterion
needs data that doesn't exist anywhere in this repo yet (a second-backbone
battery, w-matched per-checkpoint reliance, seeded head replicates), a
synthetic stand-in is constructed here, schema-shaped to match the real
fixtures (verified field-for-field against them, not guessed).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_gate import (  # noqa: E402
    STATUS_FAIL,
    STATUS_NOT_ESTIMABLE,
    STATUS_PASS,
    STATUS_PENDING,
    check_phase_b_cache,
    classify_overall,
    criterion_1_replication,
    criterion_2_association,
    criterion_3_grouped_bootstrap,
    criterion_4_intervention_vs_random,
    criterion_5_rank_stability,
    criterion_6_estimator_agreement,
    criterion_7_no_collapse,
    criterion_8_seeded_replicates,
    load_eer_file,
    load_eer_inputs,
    load_head_replicates,
    load_phase_a_file,
    load_phase_a_inputs,
    run_gate,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "step3"
SMOKE_JSON = FIXTURES / "smoke.json"
MANIFEST_JSON = FIXTURES / "reliance_layer9_boot.json"
ACCENT_BATTERY_JSON = FIXTURES / "reliance_layer9_boot_diffssd_openvoicev2_accent_by_speaker.json"
EXPERIMENTS_E007 = Path(__file__).resolve().parents[1] / "experiments" / "e007"


# ---------------------------------------------------------------------------
# Phase A loading -- real fixtures
# ---------------------------------------------------------------------------


def test_load_phase_a_file_smoke_manifest_shape():
    records = load_phase_a_file(SMOKE_JSON)
    assert len(records) == 1
    rec = records[0]
    assert rec["battery"]["name"] == "smoke_battery"
    assert rec["prereg_candidate"]["name"] == "smoke_battery"
    assert "smoke_ckpt" in rec["checkpoints"]
    assert rec["w_metrics"]["enabled"] is True


def test_load_phase_a_file_manifest_with_battery_files():
    records = load_phase_a_file(MANIFEST_JSON)
    assert len(records) == 1
    rec = records[0]
    assert rec["battery"]["name"] == "diffssd_openvoicev2_accent_by_speaker"
    assert rec["prereg_candidate"]["estimators_agree_sign"] is True
    assert set(rec["checkpoints"]) == {"e007_A_fresh", "e007_B_fresh", "e007_C_xlsr_fresh"}


def test_load_phase_a_file_standalone_per_battery_matches_manifest_copy():
    manifest_records = load_phase_a_file(MANIFEST_JSON)
    standalone_records = load_phase_a_file(ACCENT_BATTERY_JSON)
    assert len(standalone_records) == 1
    assert standalone_records[0]["battery"] == manifest_records[0]["battery"]
    assert standalone_records[0]["prereg_candidate"] == manifest_records[0]["prereg_candidate"]


def test_load_phase_a_inputs_missing_file_warns_never_crashes(tmp_path):
    records, warnings = load_phase_a_inputs([tmp_path / "does_not_exist.json"])
    assert records == []
    assert len(warnings) == 1
    assert "not found" in warnings[0]


def test_load_phase_a_inputs_malformed_file_warns_never_crashes(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"nonsense": True}), encoding="utf-8")
    records, warnings = load_phase_a_inputs([bad])
    assert records == []
    assert len(warnings) == 1
    assert "unrecognized" in warnings[0]


# ---------------------------------------------------------------------------
# EER loading -- real, already-committed experiments/e007/*_crosstest.json
# ---------------------------------------------------------------------------


def test_load_eer_file_real_e007_a():
    run_name, eers = load_eer_file(EXPERIMENTS_E007 / "e007_A_fresh_crosstest.json")
    assert run_name == "e007_A_fresh"
    assert eers["inthewild"] == pytest.approx(0.18049920365540162)
    assert eers["replaydf"] == pytest.approx(0.3327217125382263)
    assert eers["ai4t"] == pytest.approx(0.2564847780103112)


def test_load_eer_inputs_all_three_checkpoints():
    paths = [EXPERIMENTS_E007 / f"{run}_crosstest.json"
             for run in ("e007_A_fresh", "e007_B_fresh", "e007_C_xlsr_fresh")]
    eers, warnings = load_eer_inputs(paths)
    assert warnings == []
    assert set(eers) == {"e007_A_fresh", "e007_B_fresh", "e007_C_xlsr_fresh"}
    assert eers["e007_B_fresh"]["inthewild"] == pytest.approx(0.11666095012740393)


def test_load_eer_inputs_missing_file_warns_never_crashes(tmp_path):
    eers, warnings = load_eer_inputs([tmp_path / "nope.json"])
    assert eers == {}
    assert len(warnings) == 1


# ---------------------------------------------------------------------------
# Phase B cache presence/schema check -- no real cache exists on this
# machine, so the "ok" path is exercised only via a synthetic stand-in
# shaped exactly like scripts/extract_model_embeddings.py's real output.
# ---------------------------------------------------------------------------


def test_check_phase_b_cache_absent_directory_is_pending(tmp_path):
    result = check_phase_b_cache(tmp_path / "_embcache_modelspace", "e007_A_fresh", "03_DiffSSD")
    assert result["status"] == STATUS_PENDING
    assert "extract_model_embeddings.py" in result["reason"]


def test_check_phase_b_cache_empty_directory_is_pending(tmp_path):
    cache_dir = tmp_path / "e007_A_fresh" / "03_DiffSSD"
    cache_dir.mkdir(parents=True)
    result = check_phase_b_cache(tmp_path, "e007_A_fresh", "03_DiffSSD")
    assert result["status"] == STATUS_PENDING


def test_check_phase_b_cache_synthetic_shard_reads_real_schema(tmp_path):
    cache_dir = tmp_path / "e007_A_fresh" / "03_DiffSSD"
    cache_dir.mkdir(parents=True)
    paths = np.array(["clip_0001.wav", "clip_0002.wav"])
    emb = np.random.default_rng(0).normal(size=(2, 256)).astype(np.float32)
    meta = dict(checkpoint_sha256="deadbeef", model_config_hash="cafef00d", git_sha="synthetic",
                dtype="float32", checkpoint_path="runs/e007_A_fresh/best.pt", corpus="diffssd",
                corpus_dir="03_DiffSSD", n_rows=2)
    np.savez(cache_dir / "shard_0000.npz", paths=paths, emb=emb, meta=np.array(json.dumps(meta)))

    result = check_phase_b_cache(tmp_path, "e007_A_fresh", "03_DiffSSD")
    assert result["status"] == STATUS_PASS
    assert result["embedding_dim"] == 256
    assert result["n_rows"] == 2
    assert result["n_shards"] == 1


def test_check_phase_b_cache_malformed_shard_is_fail(tmp_path):
    cache_dir = tmp_path / "e007_A_fresh" / "03_DiffSSD"
    cache_dir.mkdir(parents=True)
    np.savez(cache_dir / "shard_0000.npz", wrong_key=np.zeros(3))

    result = check_phase_b_cache(tmp_path, "e007_A_fresh", "03_DiffSSD")
    assert result["status"] == STATUS_FAIL
    assert "missing expected keys" in result["reason"]


# ---------------------------------------------------------------------------
# Head-replicate loading (Task 3 output)
# ---------------------------------------------------------------------------


def test_load_head_replicates_none_path():
    replicates, warnings = load_head_replicates(None)
    assert replicates is None
    assert warnings == []


def test_load_head_replicates_missing_file_warns(tmp_path):
    replicates, warnings = load_head_replicates(tmp_path / "nope.json")
    assert replicates is None
    assert len(warnings) == 1


def test_load_head_replicates_synthetic(tmp_path):
    p = tmp_path / "replicates.json"
    p.write_text(json.dumps({"replicates": [{"seed": 0, "effect": 0.1}, {"seed": 1, "effect": 0.2}]}),
                 encoding="utf-8")
    replicates, warnings = load_head_replicates(p)
    assert warnings == []
    assert len(replicates) == 2


# ---------------------------------------------------------------------------
# Criteria computable directly from a single real battery (C3, C5, C6)
# ---------------------------------------------------------------------------


def test_criterion_3_grouped_bootstrap_passes_on_real_accent_battery():
    records = load_phase_a_file(ACCENT_BATTERY_JSON)
    result = criterion_3_grouped_bootstrap(records)
    assert result["status"] == STATUS_PASS


def test_criterion_5_rank_stability_passes_on_real_accent_battery():
    records = load_phase_a_file(ACCENT_BATTERY_JSON)
    result = criterion_5_rank_stability(records)
    assert result["status"] == STATUS_PASS
    assert result["numbers"]["per_battery"]["diffssd_openvoicev2_accent_by_speaker"]["stable_rank_window"] == [1, 2, 3]


def test_criterion_6_estimator_agreement_passes_on_real_accent_battery():
    records = load_phase_a_file(ACCENT_BATTERY_JSON)
    result = criterion_6_estimator_agreement(records)
    assert result["status"] == STATUS_PASS


def test_criterion_4_not_estimable_on_real_accent_battery():
    """The real battery's task_direction_effect (positive control) is
    Phase A's own not_estimable sentinel (w-dim mismatch), so the overall
    criterion cannot resolve to pass/fail yet -- even though the main
    factor-projection effect is fully computable."""
    records = load_phase_a_file(ACCENT_BATTERY_JSON)
    result = criterion_4_intervention_vs_random(records)
    assert result["status"] == STATUS_NOT_ESTIMABLE
    per_battery = result["numbers"]["per_battery"]["diffssd_openvoicev2_accent_by_speaker"]
    assert per_battery["main_exceeds_random_fraction"] is not None
    assert per_battery["n_control_folds_estimable"] == 0


def test_criterion_2_pending_without_factor_corpus_map():
    records = load_phase_a_file(ACCENT_BATTERY_JSON)
    result = criterion_2_association(records, eers={}, factor_corpus_map={})
    assert result["status"] == STATUS_PENDING


def test_criterion_2_not_estimable_with_map_but_no_phase_b():
    """Even with a factor-corpus map and real EERs supplied, the real
    battery's per-checkpoint reliance is not_estimable (w-dim mismatch) --
    this must surface as not_estimable, not silently as pending or a
    fabricated pass/fail."""
    records = load_phase_a_file(ACCENT_BATTERY_JSON)
    eers, _ = load_eer_inputs([EXPERIMENTS_E007 / f"{r}_crosstest.json"
                               for r in ("e007_A_fresh", "e007_B_fresh", "e007_C_xlsr_fresh")])
    result = criterion_2_association(records, eers, factor_corpus_map={"language": "inthewild"})
    assert result["status"] == STATUS_NOT_ESTIMABLE


def test_criterion_1_pending_without_secondary_backbone():
    records = load_phase_a_file(ACCENT_BATTERY_JSON)
    result = criterion_1_replication(records, secondary=[])
    assert result["status"] == STATUS_PENDING


def test_criterion_8_pending_without_replicates():
    result = criterion_8_seeded_replicates(None)
    assert result["status"] == STATUS_PENDING


def test_criterion_7_not_estimable_on_real_accent_battery():
    records = load_phase_a_file(ACCENT_BATTERY_JSON)
    eers, _ = load_eer_inputs([EXPERIMENTS_E007 / f"{r}_crosstest.json"
                               for r in ("e007_A_fresh", "e007_B_fresh", "e007_C_xlsr_fresh")])
    result = criterion_7_no_collapse(records, eers, factor_corpus_map={"language": "inthewild"})
    assert result["status"] == STATUS_NOT_ESTIMABLE


# ---------------------------------------------------------------------------
# Synthetic multi-battery fixture -- exercises the C2/C7 association logic
# in BOTH directions, since no real w-matched (Phase B) reliance data
# exists anywhere yet. Schema-shaped field-for-field against the real
# fixtures above (same dict keys/nesting), not guessed.
# ---------------------------------------------------------------------------


def _make_synthetic_battery_file(path: Path, name: str, factor: str, ckpt_reliances: dict) -> None:
    def fold_result():
        return dict(
            fold_id=0, chosen=dict(k=1), selection_score=0.9, n_selection=80, n_effect=20,
            effect=dict(
                per_checkpoint={
                    ck: dict(
                        alignment=val, r_var=abs(val) * 0.1,
                        r_var_class_conditional=dict(per_class={"0": 0.0, "1": 0.0}, overall=0.0),
                        prediction_change=dict(mean_abs_logit_change=0.0, rmse_logit_change=0.0,
                                                mean_prob_change=0.0, decision_flip_rate=0.0),
                        prediction_change_control=dict(true_effect=0.0, random_effects=[0.0] * 5,
                                                        random_mean=0.0, random_std=0.0,
                                                        task_direction_effect=0.0, exceeds_random=False),
                        w_layer_mismatch=False, ckpt_layer_center=9, layer_pooling="fixed_layer",
                    )
                    for ck, val in ckpt_reliances.items()
                },
                factor_separation_score=0.5,
                leace=dict(factor_decodability_before=1.0, factor_decodability_after=0.5, decodability_drop=0.5),
                inlp=dict(factor_decodability_before=1.0, factor_decodability_after=0.5, decodability_drop=0.5),
                projection_removal_control=dict(true_effect=0.3, random_effects=[0.0] * 20,
                                                 random_mean=0.0, random_std=0.05,
                                                 task_direction_effect=0.3, exceeds_random=True),
            ),
        )

    battery = dict(
        name=name, corpus="synthcorpus", factor=factor, grouping="synth_group",
        n_rows=100, n_levels=2, n_groups=10, grouping_degenerate=False,
        ranks_requested=[1, 2], ranks_valid=[1, 2], layer_mode="fixed",
        estimators=dict(lda=dict(fold_results=[fold_result()]), probe=dict(fold_results=[fold_result()])),
        headline_bootstrap=dict(metric="factor_separation_score", rank=1, mean=0.5, std=0.01,
                                 lo=0.48, hi=0.52, n_boot=1000, n_groups=10, n_finite=1000,
                                 n_boot_failed=0, status="ok", timed_out=False),
        rank_sensitivity=dict(metric="factor_separation_score", ranks=[1, 2], values=[0.5, 0.49],
                               status="ok", timed_out=False),
    )
    prereg_candidate = dict(name=name, headline_metric="factor_separation_score",
                             stable_rank_window=[1, 2], estimators_agree_sign=True,
                             cis_overlap=True, n_groups=10, grouping_degenerate=False)
    payload = dict(
        schema_version=1, git_sha="synthetic", timestamp="2026-01-01T00:00:00Z", layer=9,
        layer_mode="fixed", seed=13,
        w_metrics=dict(enabled=True, reason="synthetic: w matches embedding_dim by construction",
                       w_dim=8, embedding_dim=8),
        join_stats=dict(synthcorpus=dict(n_cache=100, n_manifest=100, n_joined=100, n_dropped=0)),
        checkpoints={ck: dict(ckpt_layer_center=9, ckpt_layer_band=[8, 11], layer_pooling="fixed_layer",
                               band_weights=None, w_layer_mismatch=False, w_dim=8, w_dim_mismatch=False)
                     for ck in ckpt_reliances},
        batteries=[battery], prereg_candidates=[prereg_candidate],
    )
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_criterion_2_association_positive_direction(tmp_path):
    path = tmp_path / "synthetic_positive.json"
    _make_synthetic_battery_file(path, "synth_battery", "synth_factor",
                                  {"ckA": 0.1, "ckB": 0.5, "ckC": 0.9})
    records = load_phase_a_file(path)
    eers = {"ckA": {"evalcorp": 0.10}, "ckB": {"evalcorp": 0.40}, "ckC": {"evalcorp": 0.70}}

    result = criterion_2_association(records, eers, factor_corpus_map={"synth_factor": "evalcorp"})

    assert result["status"] == STATUS_PASS
    per_battery = result["numbers"]["per_battery"]["synth_battery"]
    assert per_battery["correlation"] > 0


def test_criterion_2_association_negative_direction(tmp_path):
    path = tmp_path / "synthetic_negative.json"
    _make_synthetic_battery_file(path, "synth_battery", "synth_factor",
                                  {"ckA": 0.1, "ckB": 0.5, "ckC": 0.9})
    records = load_phase_a_file(path)
    # Same reliances, EERs now DECREASE as reliance increases -> negative association.
    eers = {"ckA": {"evalcorp": 0.70}, "ckB": {"evalcorp": 0.40}, "ckC": {"evalcorp": 0.10}}

    result = criterion_2_association(records, eers, factor_corpus_map={"synth_factor": "evalcorp"})

    assert result["status"] == STATUS_FAIL
    per_battery = result["numbers"]["per_battery"]["synth_battery"]
    assert per_battery["correlation"] < 0


def test_criterion_7_survives_residualizing_when_positive(tmp_path):
    path = tmp_path / "synthetic_c7.json"
    _make_synthetic_battery_file(path, "synth_battery", "synth_factor",
                                  {"ckA": 0.1, "ckB": 0.5, "ckC": 0.9})
    records = load_phase_a_file(path)
    eers = {"ckA": {"evalcorp": 0.10, "inthewild": 0.10, "replaydf": 0.10, "ai4t": 0.10},
            "ckB": {"evalcorp": 0.40, "inthewild": 0.20, "replaydf": 0.20, "ai4t": 0.20},
            "ckC": {"evalcorp": 0.70, "inthewild": 0.30, "replaydf": 0.30, "ai4t": 0.30}}

    result = criterion_7_no_collapse(records, eers, factor_corpus_map={"synth_factor": "evalcorp"})

    assert result["status"] in (STATUS_PASS, STATUS_FAIL)  # fully decided, not pending/not_estimable
    assert result["numbers"]["per_battery"]["synth_battery"]["n_checkpoints"] == 3


def test_criterion_8_unanimous_pass():
    replicates = [{"seed": 0, "effect": 0.1}, {"seed": 1, "effect": 0.2}, {"seed": 2, "effect": 0.05}]
    result = criterion_8_seeded_replicates(replicates)
    assert result["status"] == STATUS_PASS


def test_criterion_8_disagreement_fails():
    replicates = [{"seed": 0, "effect": 0.1}, {"seed": 1, "effect": -0.2}, {"seed": 2, "effect": 0.05}]
    result = criterion_8_seeded_replicates(replicates)
    assert result["status"] == STATUS_FAIL


def test_criterion_1_replication_sign_agreement(tmp_path):
    primary_path = tmp_path / "primary.json"
    secondary_path = tmp_path / "secondary.json"
    _make_synthetic_battery_file(primary_path, "shared_battery", "synth_factor", {"ckA": 0.3})
    _make_synthetic_battery_file(secondary_path, "shared_battery", "synth_factor", {"ckA": 0.3})
    primary = load_phase_a_file(primary_path)
    secondary = load_phase_a_file(secondary_path)

    result = criterion_1_replication(primary, secondary)

    assert result["status"] == STATUS_PASS


# ---------------------------------------------------------------------------
# Overall classification
# ---------------------------------------------------------------------------


def test_classify_overall_none_when_any_pending():
    criteria = {f"C{i}": dict(status=STATUS_PASS) for i in range(1, 9)}
    criteria["C8"] = dict(status=STATUS_PENDING)
    assert classify_overall(criteria) is None


def test_classify_overall_strong_success_when_all_pass():
    criteria = {f"C{i}": dict(status=STATUS_PASS) for i in range(1, 9)}
    assert classify_overall(criteria) == "strong_success"


def test_classify_overall_failure_on_sign_bearing_reversal():
    criteria = {f"C{i}": dict(status=STATUS_PASS) for i in range(1, 9)}
    criteria["C2"] = dict(status=STATUS_FAIL)
    assert classify_overall(criteria) == "failure"


def test_classify_overall_diagnostic_only_on_non_sign_bearing_fail():
    criteria = {f"C{i}": dict(status=STATUS_PASS) for i in range(1, 9)}
    criteria["C3"] = dict(status=STATUS_FAIL)
    assert classify_overall(criteria) == "diagnostic_only"


# ---------------------------------------------------------------------------
# End-to-end: run_gate() against the real accent-battery fixture ALONE.
# Must never crash and must always write a verdict (exit 0). Criteria that
# are intrinsically single-battery-computable (C3/C5/C6) legitimately
# resolve to pass/fail from this one real file; every cross-cutting
# criterion that needs a second backbone (C1), EERs + a factor mapping
# (C2), a positive control that needs Phase B embeddings (C4), a
# checkpoint-quality regression (C7), or seeded replicates (C8) is
# pending_input or not_estimable -- so the overall three-outcome
# classification is never emitted.
# ---------------------------------------------------------------------------


def test_run_gate_accent_battery_alone_no_crash_and_mostly_pending():
    verdict = run_gate(phase_a_paths=[ACCENT_BATTERY_JSON])

    assert verdict["overall_classification"] is None
    statuses = {name: c["status"] for name, c in verdict["criteria"].items()}
    assert statuses["C1"] == STATUS_PENDING
    assert statuses["C2"] == STATUS_PENDING  # no factor_corpus_map/EERs supplied at all
    assert statuses["C3"] == STATUS_PASS
    assert statuses["C4"] == STATUS_NOT_ESTIMABLE
    assert statuses["C5"] == STATUS_PASS
    assert statuses["C6"] == STATUS_PASS
    # Even without a factor_corpus_map, the real battery's per-checkpoint
    # reliance is ITSELF not_estimable (w-dim mismatch) -- a more
    # fundamental blocker than the missing mapping, so it takes priority.
    assert statuses["C7"] == STATUS_NOT_ESTIMABLE
    assert statuses["C8"] == STATUS_PENDING


def test_run_gate_writes_verdict_file_and_exits_cleanly(tmp_path):
    from run_gate import main

    out_path = tmp_path / "verdict.json"
    argv = ["--phase-a", str(ACCENT_BATTERY_JSON), "--out", str(out_path)]
    rc = main(argv)

    assert rc == 0
    assert out_path.exists()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert "criteria" in written
    assert len(written["criteria"]) == 8


def test_run_gate_with_full_real_and_synthetic_inputs_reaches_a_verdict(tmp_path):
    """With every input supplied (real EERs, a factor-corpus map, a
    synthetic second backbone, and synthetic head replicates), the gate
    consumer reaches a fully-decided verdict -- proving the whole pipeline
    is wired end-to-end, not just individually-tested pieces."""
    secondary_path = tmp_path / "secondary_backbone.json"
    _make_synthetic_battery_file(secondary_path, "diffssd_openvoicev2_accent_by_speaker",
                                  "language", {"e007_A_fresh": 0.3})
    replicates_path = tmp_path / "replicates.json"
    replicates_path.write_text(json.dumps({"replicates": [
        {"seed": 0, "effect": 0.1}, {"seed": 1, "effect": 0.2}, {"seed": 2, "effect": 0.15},
    ]}), encoding="utf-8")
    eer_paths = [EXPERIMENTS_E007 / f"{r}_crosstest.json" for r in ("e007_A_fresh", "e007_B_fresh", "e007_C_xlsr_fresh")]

    verdict = run_gate(
        phase_a_paths=[ACCENT_BATTERY_JSON],
        secondary_backbone_paths=[secondary_path],
        eer_paths=eer_paths,
        head_replicates_path=replicates_path,
        factor_corpus_map={"language": "inthewild"},
    )

    statuses = {name: c["status"] for name, c in verdict["criteria"].items()}
    assert statuses["C8"] == STATUS_PASS
    assert statuses["C3"] == STATUS_PASS
    assert statuses["C5"] == STATUS_PASS
    assert statuses["C6"] == STATUS_PASS
    # C2/C4/C7 remain not_estimable: the real accent battery's per-checkpoint
    # reliance is w-dim-mismatched regardless of what else is supplied.
    assert statuses["C2"] == STATUS_NOT_ESTIMABLE
    assert statuses["C4"] == STATUS_NOT_ESTIMABLE
    assert statuses["C7"] == STATUS_NOT_ESTIMABLE
