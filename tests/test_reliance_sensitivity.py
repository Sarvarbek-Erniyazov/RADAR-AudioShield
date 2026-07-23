"""Tests for scripts/reliance_sensitivity.py -- synthetic fixtures only (no
real Phase B embedding cache or model-space battery exists on this
machine; every fold/per_checkpoint block below is hand-built, shaped
exactly like the real schema confirmed by direct read of
analysis/step3/reliance_modelspace_prereg_replaydf_language_by_channel.json).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from reliance_sensitivity import (  # noqa: E402
    DEFAULT_TARGET_FLOOR,
    STATUS_NOT_ESTIMABLE,
    STATUS_OK,
    _per_checkpoint_sensitivity,
    analyze_records,
    build_parser,
    main,
)


def _prc(true_effect, random_mean, random_std, task_direction_effect=0.0, exceeds_random=False):
    return dict(true_effect=true_effect, random_effects=[random_mean] * 20, random_mean=random_mean,
                random_std=random_std, task_direction_effect=task_direction_effect, exceeds_random=exceeds_random)


def _pc(decision_flip_rate):
    return dict(mean_abs_logit_change=0.0, rmse_logit_change=0.0, mean_prob_change=0.0,
                decision_flip_rate=decision_flip_rate)


def _ckpt_entry(flip_rate, prc):
    return dict(alignment=0.0, r_var=0.0, prediction_change=_pc(flip_rate),
                prediction_change_control={}, projection_removal_control=prc)


def _fold(fold_id, n_effect, per_checkpoint):
    return dict(fold_id=fold_id, chosen={"k": 1}, selection_score=0.9, n_selection=1000, n_effect=n_effect,
                effect=dict(per_checkpoint=per_checkpoint, factor_separation_score=0.5, leace={}, inlp={},
                            projection_removal_control=next(iter(per_checkpoint.values()))["projection_removal_control"]))


def _battery(name, estimators):
    return dict(name=name, corpus="replaydf", factor="language", grouping="channel_id",
                n_rows=3000, n_levels=2, n_groups=109, grouping_degenerate=False,
                ranks_requested=[1], ranks_valid=[1], layer_mode="model_space", estimators=estimators)


# ---------------------------------------------------------------------------
# _per_checkpoint_sensitivity: the core floor/z-score math
# ---------------------------------------------------------------------------


def test_resolution_floor_is_one_over_mean_n_effect():
    """Zero observed flips at n_effect=500 -> floor is exactly 1/500, and
    floor_multiple is 0 (the observed rate sits at/below a single flip)."""
    per_ckpt = {"ckA": _ckpt_entry(0.0, _prc(0.0, 0.0, 0.002))}
    fold_results = [_fold(0, 500, per_ckpt)]
    battery = _battery("b", dict(lda=dict(status="ok", timed_out=False, fold_results=fold_results)))

    result = _per_checkpoint_sensitivity(battery, target_floor=DEFAULT_TARGET_FLOOR)

    assert result["ckA"]["status"] == STATUS_OK
    assert result["ckA"]["mean_n_effect"] == 500.0
    assert result["ckA"]["resolution_floor"] == pytest.approx(1.0 / 500)
    assert result["ckA"]["observed_flip_rate"] == 0.0
    assert result["ckA"]["floor_multiple"] == 0.0


def test_floor_multiple_reflects_a_rate_above_the_floor():
    """One flip at n_effect=200 (rate 1/200 = 0.005) against a floor of
    1/200 -- floor_multiple should be 1.0 (the observed rate IS exactly
    the single-flip floor)."""
    per_ckpt = {"ckA": _ckpt_entry(1.0 / 200, _prc(0.01, 0.0, 0.002))}
    fold_results = [_fold(0, 200, per_ckpt)]
    battery = _battery("b", dict(lda=dict(status="ok", timed_out=False, fold_results=fold_results)))

    result = _per_checkpoint_sensitivity(battery, target_floor=DEFAULT_TARGET_FLOOR)

    assert result["ckA"]["floor_multiple"] == pytest.approx(1.0)


def test_random_control_z_mean_computed_from_true_effect_and_random_stats():
    per_ckpt = {"ckA": _ckpt_entry(0.0, _prc(true_effect=0.01, random_mean=0.0, random_std=0.002))}
    fold_results = [_fold(0, 500, per_ckpt)]
    battery = _battery("b", dict(lda=dict(status="ok", timed_out=False, fold_results=fold_results)))

    result = _per_checkpoint_sensitivity(battery, target_floor=DEFAULT_TARGET_FLOOR)

    assert result["ckA"]["random_control_z_mean"] == pytest.approx((0.01 - 0.0) / 0.002)


def test_rows_needed_for_target_floor_and_additional_rows_needed():
    per_ckpt = {"ckA": _ckpt_entry(0.0, _prc(0.0, 0.0, 0.002))}
    fold_results = [_fold(0, 400, per_ckpt)]
    battery = _battery("b", dict(lda=dict(status="ok", timed_out=False, fold_results=fold_results)))

    result = _per_checkpoint_sensitivity(battery, target_floor=0.001)  # needs 1000 rows

    assert result["ckA"]["rows_needed_for_target_floor"] == 1000
    assert result["ckA"]["additional_rows_needed"] == 600  # 1000 - 400


def test_rows_needed_is_zero_when_already_below_target_floor():
    per_ckpt = {"ckA": _ckpt_entry(0.0, _prc(0.0, 0.0, 0.002))}
    fold_results = [_fold(0, 5000, per_ckpt)]  # floor already 1/5000 = 0.0002, tighter than 0.001
    battery = _battery("b", dict(lda=dict(status="ok", timed_out=False, fold_results=fold_results)))

    result = _per_checkpoint_sensitivity(battery, target_floor=0.001)

    assert result["ckA"]["additional_rows_needed"] == 0


def test_aggregates_across_multiple_folds_by_mean():
    per_ckpt_a = {"ckA": _ckpt_entry(0.0, _prc(0.0, 0.0, 0.002))}
    per_ckpt_b = {"ckA": _ckpt_entry(0.02, _prc(0.01, 0.0, 0.002))}
    fold_results = [_fold(0, 400, per_ckpt_a), _fold(1, 600, per_ckpt_b)]
    battery = _battery("b", dict(lda=dict(status="ok", timed_out=False, fold_results=fold_results)))

    result = _per_checkpoint_sensitivity(battery, target_floor=DEFAULT_TARGET_FLOOR)

    assert result["ckA"]["n_folds"] == 2
    assert result["ckA"]["mean_n_effect"] == 500.0  # mean(400, 600)
    assert result["ckA"]["observed_flip_rate"] == pytest.approx(0.01)  # mean(0.0, 0.02)


def test_failed_estimator_contributes_nothing():
    per_ckpt = {"ckA": _ckpt_entry(0.0, _prc(0.0, 0.0, 0.002))}
    fold_results = [_fold(0, 500, per_ckpt)]
    battery = _battery("b", dict(
        lda=dict(status="failed", timed_out=True, fold_results=[]),
        probe=dict(status="ok", timed_out=False, fold_results=fold_results),
    ))

    result = _per_checkpoint_sensitivity(battery, target_floor=DEFAULT_TARGET_FLOOR)

    assert result["ckA"]["status"] == STATUS_OK
    assert result["ckA"]["n_folds"] == 1  # only probe's fold, lda's failure contributed nothing


def test_checkpoint_with_no_data_at_all_is_not_estimable():
    battery = _battery("b", dict(lda=dict(status="failed", timed_out=True, fold_results=[])))

    result = _per_checkpoint_sensitivity(battery, target_floor=DEFAULT_TARGET_FLOOR)

    assert result == {}  # no per_checkpoint data ever seen -- nothing to report, not fabricated


def test_missing_prediction_change_reports_not_estimable(monkeypatch=None):
    """A per_checkpoint entry with n_effect but no usable
    decision_flip_rate (e.g. prediction_change itself missing) must report
    not_estimable, never silently compute a floor with no rate to compare
    it to."""
    per_ckpt = {"ckA": dict(alignment=0.0, r_var=0.0, prediction_change={}, prediction_change_control={},
                             projection_removal_control=_prc(0.0, 0.0, 0.002))}
    fold_results = [_fold(0, 500, per_ckpt)]
    battery = _battery("b", dict(lda=dict(status="ok", timed_out=False, fold_results=fold_results)))

    result = _per_checkpoint_sensitivity(battery, target_floor=DEFAULT_TARGET_FLOOR)

    assert result["ckA"]["status"] == STATUS_NOT_ESTIMABLE


# ---------------------------------------------------------------------------
# analyze_records / main(): end-to-end over a written Phase A file
# ---------------------------------------------------------------------------


def test_analyze_records_reports_corpus_and_factor_per_battery():
    per_ckpt = {"ckA": _ckpt_entry(0.0, _prc(0.0, 0.0, 0.002))}
    battery = _battery("replaydf_language_by_channel", dict(
        lda=dict(status="ok", timed_out=False, fold_results=[_fold(0, 500, per_ckpt)])))
    records = [dict(battery=battery, prereg_candidate={}, checkpoints={}, w_metrics={}, join_stats={},
                     source="synthetic")]

    report = analyze_records(records, target_floor=0.001)

    per_battery = report["per_battery"]["replaydf_language_by_channel"]
    assert per_battery["corpus"] == "replaydf"
    assert per_battery["factor"] == "language"
    assert per_battery["per_checkpoint"]["ckA"]["status"] == STATUS_OK


def test_main_writes_report_json_end_to_end(tmp_path):
    per_ckpt = {"ckA": _ckpt_entry(0.0, _prc(0.0, 0.0, 0.002))}
    battery = _battery("replaydf_language_by_channel", dict(
        lda=dict(status="ok", timed_out=False, fold_results=[_fold(0, 500, per_ckpt)])))
    payload = dict(schema_version=1, git_sha="synthetic", checkpoints={}, w_metrics={}, join_stats={"replaydf": {}},
                   battery=battery, prereg_candidate=dict(name=battery["name"]))
    battery_path = tmp_path / "battery.json"
    battery_path.write_text(json.dumps(payload), encoding="utf-8")
    out_path = tmp_path / "sensitivity.json"

    main(["--phase-a", str(battery_path), "--target-floor", "0.001", "--out", str(out_path)])

    assert out_path.exists()
    assert not out_path.with_name(out_path.name + ".tmp").exists()  # atomic write, no leftover tmp
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["target_floor"] == 0.001
    per_ckpt_result = report["per_battery"]["replaydf_language_by_channel"]["per_checkpoint"]["ckA"]
    assert per_ckpt_result["status"] == STATUS_OK
    assert per_ckpt_result["resolution_floor"] == pytest.approx(1.0 / 500)


def test_main_never_crashes_on_missing_phase_a_file(tmp_path):
    out_path = tmp_path / "sensitivity.json"

    main(["--phase-a", str(tmp_path / "nope.json"), "--out", str(out_path)])  # must not raise

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["per_battery"] == {}
    assert any("not found" in w for w in report["warnings"])


def test_build_parser_defaults():
    args = build_parser().parse_args([])
    assert args.phase_a == []
    assert args.target_floor == DEFAULT_TARGET_FLOOR
    assert str(args.out) == str(Path("analysis/step3/reliance_sensitivity.json"))
