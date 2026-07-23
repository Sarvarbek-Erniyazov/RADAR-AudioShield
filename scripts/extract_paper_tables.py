"""Extracts paper-ready summary tables from already-computed Roadmap v3
Step 3/4 results: cache-space factor ENCODING (FSS), model-space causal
RELIANCE per checkpoint, and the Step 4 gate verdict -- the three results
blocks CURRENT_STATE.md's "RESULTS ON RECORD" section reports in prose.

Pure post-processing of already-committed JSON -- no GPU, no extraction,
no real embedding-cache read. Read-only imports run_gate.load_phase_a_inputs
and run_gate._per_checkpoint_reliance (both already used for exactly this
per-checkpoint aggregation by the gate consumer itself); run_gate.py pulls
in nothing heavier than numpy at module level, so this stays as
lightweight as run_gate.py's own "no GPU" contract.

Honesty note this script enforces rather than papering over: only ONE
real cache-space battery (DiffSSD openvoicev2-accent) is present anywhere
in this repository as a readable JSON file -- confirmed by a repo-wide
search for `factor_separation_score`-shaped files. The other three
cache-space batteries CURRENT_STATE.md's own prose reports (DiffSSD
generator, ReplayDF language, ReplayDF generator FSS values) exist only
as numbers in that scratch note, not as independently re-derivable files
on this machine, and are therefore NOT included in Table 1 below -- this
script only ever extracts from files it can actually open and verify,
never from a prose summary.

Outputs (both written every run, atomically):
    analysis/step3/paper_tables.md    -- human-readable Markdown tables
    analysis/step3/paper_tables.json  -- the same data, structured

Usage:
    python scripts/extract_paper_tables.py \\
        --cachespace-phase-a tests/fixtures/step3/reliance_layer9_boot_diffssd_openvoicev2_accent_by_speaker.json \\
        --modelspace-phase-a analysis/step3/reliance_modelspace_prereg_replaydf_language_by_channel.json \\
                              analysis/step3/reliance_modelspace_prereg_replaydf_generator_by_channel.json \\
        --gate-verdict analysis/step4/gate_verdict_prereg_v2.json \\
        --out-dir analysis/step3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Read-only imports -- scripts/run_gate.py is never modified by this
# script. run_gate.py itself only imports argparse/json/sys/datetime/
# pathlib/numpy (confirmed by direct read), so reusing its Phase A loader
# and per-checkpoint aggregator here does not pull torch or any GPU/model
# stack into this reporting tool.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gate import _per_checkpoint_reliance, load_phase_a_inputs  # noqa: E402

DEFAULT_CACHESPACE_PHASE_A = [
    Path("tests/fixtures/step3/reliance_layer9_boot_diffssd_openvoicev2_accent_by_speaker.json"),
]
DEFAULT_MODELSPACE_PHASE_A = [
    Path("analysis/step3/reliance_modelspace_prereg_replaydf_language_by_channel.json"),
    Path("analysis/step3/reliance_modelspace_prereg_replaydf_generator_by_channel.json"),
]
DEFAULT_GATE_VERDICT = Path("analysis/step4/gate_verdict_prereg_v2.json")
DEFAULT_OUT_DIR = Path("analysis/step3")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _log(msg: str) -> None:
    print(f"[{_timestamp()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Table 1 -- cache-space factor encoding (FSS)
# ---------------------------------------------------------------------------


def extract_encoding_table(records: list[dict]) -> list[dict]:
    """One row per battery whose headline_bootstrap reports
    factor_separation_score with status="ok" -- never fabricates a row for
    a battery this can't confirm (bootstrap failed/timed out, or a
    different headline metric)."""
    rows = []
    for rec in records:
        battery = rec["battery"]
        boot = battery.get("headline_bootstrap", {})
        if boot.get("metric") != "factor_separation_score" or boot.get("status") != "ok":
            continue
        rows.append(dict(
            battery=battery["name"], corpus=battery.get("corpus"), factor=battery.get("factor"),
            fss_mean=boot["mean"], fss_lo=boot["lo"], fss_hi=boot["hi"],
            n_boot=boot["n_boot"], n_groups=boot["n_groups"],
        ))
    return rows


# ---------------------------------------------------------------------------
# Table 2 -- model-space causal reliance per checkpoint
# ---------------------------------------------------------------------------


def _per_checkpoint_flip_and_control(battery: dict) -> dict[str, dict]:
    """checkpoint_name -> {decision_flip_rate (mean across folds),
    exceeds_random (True only if every fold with a decided flag agreed),
    n_folds}. Same per_checkpoint/fold traversal convention as
    scripts/run_gate.py's _per_checkpoint_reliance and
    scripts/reliance_sensitivity.py's _per_checkpoint_sensitivity."""
    acc: dict[str, dict] = {}
    for estimator in battery.get("estimators", {}).values():
        if estimator.get("status") != "ok":
            continue
        for fold in estimator.get("fold_results", []):
            for ckpt_name, ckpt_entry in fold.get("effect", {}).get("per_checkpoint", {}).items():
                bucket = acc.setdefault(ckpt_name, dict(flip_rates=[], exceeds_flags=[]))
                pc = ckpt_entry.get("prediction_change")
                if isinstance(pc, dict) and isinstance(pc.get("decision_flip_rate"), (int, float)):
                    bucket["flip_rates"].append(float(pc["decision_flip_rate"]))
                prc = ckpt_entry.get("projection_removal_control")
                if isinstance(prc, dict) and isinstance(prc.get("exceeds_random"), bool):
                    bucket["exceeds_flags"].append(prc["exceeds_random"])
    out = {}
    for ckpt_name, bucket in acc.items():
        out[ckpt_name] = dict(
            decision_flip_rate=float(np.mean(bucket["flip_rates"])) if bucket["flip_rates"] else None,
            exceeds_random=(all(bucket["exceeds_flags"]) if bucket["exceeds_flags"] else None),
            n_folds=len(bucket["flip_rates"]),
        )
    return out


def extract_reliance_table(records: list[dict]) -> list[dict]:
    rows = []
    for rec in records:
        battery = rec["battery"]
        r_var = _per_checkpoint_reliance(battery, metric="r_var")
        alignment = _per_checkpoint_reliance(battery, metric="alignment")
        flip_control = _per_checkpoint_flip_and_control(battery)
        checkpoints = sorted(set(r_var) | set(alignment) | set(flip_control))
        for ckpt in checkpoints:
            rv, al, fc = r_var.get(ckpt, {}), alignment.get(ckpt, {}), flip_control.get(ckpt, {})
            rows.append(dict(
                battery=battery["name"], corpus=battery.get("corpus"), factor=battery.get("factor"),
                checkpoint=ckpt,
                r_var=rv.get("value"), r_var_status=rv.get("status"),
                alignment=al.get("value"), alignment_status=al.get("status"),
                decision_flip_rate=fc.get("decision_flip_rate"), exceeds_random=fc.get("exceeds_random"),
            ))
    return rows


# ---------------------------------------------------------------------------
# Table 3 -- gate verdict summary
# ---------------------------------------------------------------------------


def extract_gate_table(verdict: dict | None) -> list[dict]:
    if verdict is None:
        return []
    rows = []
    for name, c in sorted(verdict.get("criteria", {}).items()):
        rows.append(dict(criterion=name, status=c.get("status"), evidence=c.get("evidence")))
    return rows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(v, nd=4):
    if v is None:
        return "--"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def render_markdown(encoding_rows, reliance_rows, gate_rows, overall_classification) -> str:
    lines = [f"# Paper tables -- generated {_timestamp()}", ""]

    lines += ["## Table 1 -- Cache-space factor encoding (FSS)", ""]
    if encoding_rows:
        lines += ["| Battery | Corpus | Factor | FSS (mean) | 95% CI | n_groups |",
                   "|---|---|---|---|---|---|"]
        for r in encoding_rows:
            lines.append(f"| {r['battery']} | {r['corpus']} | {r['factor']} | {_fmt(r['fss_mean'])} | "
                          f"[{_fmt(r['fss_lo'])}, {_fmt(r['fss_hi'])}] | {r['n_groups']} |")
    else:
        lines.append("*(no battery with an `ok` factor_separation_score headline_bootstrap found)*")
    lines += ["", "Only batteries with a readable, `ok` `headline_bootstrap` on this machine are listed here "
                   "-- see this script's module docstring for which cache-space batteries exist only as prose "
                   "elsewhere, not as files this script could open.", ""]

    lines += ["## Table 2 -- Model-space causal reliance per checkpoint", ""]
    if reliance_rows:
        lines += ["| Battery | Checkpoint | r_var | alignment | decision_flip_rate | exceeds_random |",
                   "|---|---|---|---|---|---|"]
        for r in reliance_rows:
            lines.append(f"| {r['battery']} | {r['checkpoint']} | {_fmt(r['r_var'], 3)} | "
                          f"{_fmt(r['alignment'], 3)} | {_fmt(r['decision_flip_rate'], 3)} | "
                          f"{_fmt(r['exceeds_random'])} |")
    else:
        lines.append("*(no model-space battery input found)*")
    lines.append("")

    lines += ["## Table 3 -- Step 4 gate verdict", ""]
    if gate_rows:
        lines += ["| Criterion | Status | Evidence |", "|---|---|---|"]
        for r in gate_rows:
            lines.append(f"| {r['criterion']} | {r['status']} | {r['evidence']} |")
        lines += ["", f"**Overall classification:** {overall_classification!r}"]
    else:
        lines.append("*(no gate verdict input found)*")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cachespace-phase-a", nargs="+", type=Path, default=DEFAULT_CACHESPACE_PHASE_A,
                    help="cache-space Phase A battery JSON file(s), for Table 1 (default: %(default)s)")
    ap.add_argument("--modelspace-phase-a", nargs="+", type=Path, default=DEFAULT_MODELSPACE_PHASE_A,
                    help="model-space Phase A battery JSON file(s), for Table 2 (default: %(default)s)")
    ap.add_argument("--gate-verdict", type=Path, default=DEFAULT_GATE_VERDICT,
                    help="gate verdict JSON (scripts/run_gate.py's --out), for Table 3 (default: %(default)s)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="directory to write paper_tables.{md,json} into (default: %(default)s)")
    return ap


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    _log("extract_paper_tables: loading cache-space Phase A input(s)")
    cachespace_records, cachespace_warnings = load_phase_a_inputs(args.cachespace_phase_a)
    _log("extract_paper_tables: loading model-space Phase A input(s)")
    modelspace_records, modelspace_warnings = load_phase_a_inputs(args.modelspace_phase_a)

    warnings = cachespace_warnings + modelspace_warnings
    for w in warnings:
        _log(f"[WARN] {w}")

    verdict = None
    if args.gate_verdict.exists():
        try:
            verdict = json.loads(args.gate_verdict.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.append(f"gate verdict {args.gate_verdict} failed to load: {exc}")
    else:
        warnings.append(f"gate verdict not found: {args.gate_verdict}")

    encoding_rows = extract_encoding_table(cachespace_records)
    reliance_rows = extract_reliance_table(modelspace_records)
    gate_rows = extract_gate_table(verdict)
    overall_classification = verdict.get("overall_classification") if verdict else None

    report = dict(
        schema_version=1, generated_at=_timestamp(), warnings=warnings,
        encoding_table=encoding_rows, reliance_table=reliance_rows, gate_table=gate_rows,
        overall_classification=overall_classification,
    )
    markdown = render_markdown(encoding_rows, reliance_rows, gate_rows, overall_classification)

    _atomic_write(args.out_dir / "paper_tables.json", json.dumps(report, indent=2))
    _atomic_write(args.out_dir / "paper_tables.md", markdown)
    _log(f"extract_paper_tables: wrote {args.out_dir / 'paper_tables.json'} and "
         f"{args.out_dir / 'paper_tables.md'}")


if __name__ == "__main__":
    main()
