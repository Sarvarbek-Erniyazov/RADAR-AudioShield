"""Resolution-floor sensitivity report for a model-space causal-reliance
battery (Roadmap v3 Step 3/4 -- the C4 "decodability without causal
reliance" finding).

Motivation (CURRENT_STATE.md's own "tighten the resolution floor" open
item): decision_flip_rate == 0.0 across every checkpoint does not mean "no
effect exists" -- it means "no effect exceeding what a single flipped
decision would produce was observed," and that floor is set entirely by
how many held-out rows the causal intervention was measured on
(`n_effect`, fold-level). One flipped decision out of ~600 effect rows
already produces a rate of ~0.17%; the honest claim the battery supports
is "no effect detectable above this floor," not "exactly zero." This
script reads already-computed Phase A model-space battery JSON
(scripts/run_reliance_modelspace.py's output, the same schema
scripts/run_gate.py consumes for C2/C4/C7) and reports, per checkpoint,
aggregated across every fold_result of every estimator (same convention
as scripts/run_gate.py's _per_checkpoint_reliance):

  - resolution_floor: 1 / mean(n_effect across folds) -- the smallest
    non-zero decision_flip_rate this fold size could possibly have
    detected.
  - observed_flip_rate: mean decision_flip_rate actually observed.
  - floor_multiple: observed_flip_rate / resolution_floor (0 means the
    observed rate sits at or below the single-flip floor -- exactly the
    e007 situation today).
  - random_control_z_mean: mean (true_effect - random_mean) / random_std
    from projection_removal_control -- the CONTINUOUS distance from the
    random-subspace control's mean, in control-std units
    (`exceeds_random` is just this thresholded at 2).
  - rows_needed_for_target_floor / additional_rows_needed: how many
    held-out effect rows (total, and beyond what was already used) would
    be needed to push the floor below --target-floor (default 0.001, i.e.
    0.1%, tighter than today's ~0.17-0.25%).

Pure post-processing of an already-computed battery JSON -- no GPU, no
extraction, no real embedding-cache read (no Phase B cache exists on this
machine, same situation scripts/run_gate.py documents; this script never
touches it). Never crashes on missing/malformed per-checkpoint data: a
checkpoint whose n_effect or decision_flip_rate can't be found is reported
status="not_estimable" with a reason, same convention as
scripts/run_gate.py, and the process always exits 0 (the verdict/report
file is always written, temp file + os.replace).

Usage:
    python scripts/reliance_sensitivity.py \\
        --phase-a analysis/step3/reliance_modelspace_prereg_replaydf_language_by_channel.json \\
                  analysis/step3/reliance_modelspace_prereg_replaydf_generator_by_channel.json \\
        --target-floor 0.001 \\
        --out analysis/step3/reliance_sensitivity.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Read-only import -- scripts/run_gate.py is never modified by this script.
# Reused because it already handles both real Phase A on-disk shapes
# (manifest-shaped / standalone) and cross-file battery deduplication
# correctly; run_gate.py itself only imports argparse/json/sys/datetime/
# pathlib/numpy (confirmed by direct read), so this stays as lightweight
# as run_gate.py's own "no GPU" reporting-tool contract.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gate import load_phase_a_inputs  # noqa: E402

STATUS_OK = "ok"
STATUS_NOT_ESTIMABLE = "not_estimable"

DEFAULT_TARGET_FLOOR = 0.001
DEFAULT_OUT = Path("analysis/step3/reliance_sensitivity.json")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _log(msg: str) -> None:
    print(f"[{_timestamp()}] {msg}", flush=True)


def _per_checkpoint_sensitivity(battery: dict, target_floor: float) -> dict[str, dict]:
    """checkpoint_name -> sensitivity dict, aggregated (mean) across every
    fold_result of every "ok" estimator -- same aggregation convention as
    scripts/run_gate.py's _per_checkpoint_reliance (mean of finite values,
    never fabricated from a single fold)."""
    acc: dict[str, dict] = {}
    for estimator in battery.get("estimators", {}).values():
        if estimator.get("status") != "ok":
            continue  # a failed estimator contributes no fold_results anyway; explicit for clarity
        for fold in estimator.get("fold_results", []):
            n_effect = fold.get("n_effect")
            per_ckpt = fold.get("effect", {}).get("per_checkpoint", {})
            for ckpt_name, ckpt_entry in per_ckpt.items():
                bucket = acc.setdefault(ckpt_name, dict(n_effects=[], flip_rates=[], zs=[]))
                if isinstance(n_effect, (int, float)) and n_effect > 0:
                    bucket["n_effects"].append(float(n_effect))
                pc = ckpt_entry.get("prediction_change")
                if isinstance(pc, dict) and isinstance(pc.get("decision_flip_rate"), (int, float)):
                    bucket["flip_rates"].append(float(pc["decision_flip_rate"]))
                prc = ckpt_entry.get("projection_removal_control")
                if isinstance(prc, dict):
                    true_effect = prc.get("true_effect")
                    random_mean = prc.get("random_mean")
                    random_std = prc.get("random_std")
                    if (all(isinstance(v, (int, float)) for v in (true_effect, random_mean, random_std))
                            and random_std > 0):
                        bucket["zs"].append((true_effect - random_mean) / random_std)

    result: dict[str, dict] = {}
    for ckpt_name, bucket in sorted(acc.items()):
        if not bucket["n_effects"] or not bucket["flip_rates"]:
            result[ckpt_name] = dict(
                status=STATUS_NOT_ESTIMABLE,
                reason="no n_effect or decision_flip_rate data found for this checkpoint",
            )
            continue
        mean_n_effect = float(np.mean(bucket["n_effects"]))
        resolution_floor = 1.0 / mean_n_effect
        observed_flip_rate = float(np.mean(bucket["flip_rates"]))
        floor_multiple = observed_flip_rate / resolution_floor
        n_needed = math.ceil(1.0 / target_floor) if target_floor > 0 else None
        additional_rows_needed = (max(0, n_needed - int(round(mean_n_effect)))
                                   if n_needed is not None else None)
        result[ckpt_name] = dict(
            status=STATUS_OK,
            n_folds=len(bucket["n_effects"]),
            mean_n_effect=mean_n_effect,
            resolution_floor=resolution_floor,
            observed_flip_rate=observed_flip_rate,
            floor_multiple=floor_multiple,
            random_control_z_mean=float(np.mean(bucket["zs"])) if bucket["zs"] else None,
            target_floor=target_floor,
            rows_needed_for_target_floor=n_needed,
            additional_rows_needed=additional_rows_needed,
        )
    return result


def analyze_records(records: list[dict], target_floor: float = DEFAULT_TARGET_FLOOR) -> dict:
    per_battery = {}
    for rec in records:
        battery = rec["battery"]
        name = battery["name"]
        per_battery[name] = dict(
            corpus=battery.get("corpus"), factor=battery.get("factor"),
            per_checkpoint=_per_checkpoint_sensitivity(battery, target_floor),
        )
    return dict(schema_version=1, generated_at=_timestamp(), target_floor=target_floor, per_battery=per_battery)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase-a", nargs="+", type=Path, default=[],
                    help="Phase A model-space battery JSON file(s) (manifest-shaped or per-battery-shaped)")
    ap.add_argument("--target-floor", type=float, default=DEFAULT_TARGET_FLOOR,
                    help="target decision_flip_rate resolution floor to project rows-needed for "
                         "(default: %(default)s)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="sensitivity-report JSON output path "
                                                                   "(default: %(default)s)")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _log("reliance_sensitivity: loading Phase A battery input(s)")
    records, warnings = load_phase_a_inputs(args.phase_a)
    for w in warnings:
        _log(f"[WARN] {w}")
    _log(f"reliance_sensitivity: loaded {len(records)} battery record(s) from {len(args.phase_a)} file(s)")

    report = analyze_records(records, args.target_floor)
    report["warnings"] = warnings

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.out.with_name(args.out.name + ".tmp")
    tmp_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp_path, args.out)
    _log(f"reliance_sensitivity: wrote {args.out}")


if __name__ == "__main__":
    main()
