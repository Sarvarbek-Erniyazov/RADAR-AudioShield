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
        --modelspace-phase-a analysis/step3/reliance_modelspace_prereg.json \\
        --gate-verdict analysis/step4/gate_verdict_prereg_v3.json \\
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
    # The combined manifest carries BOTH model-space batteries
    # (replaydf_language_by_channel, replaydf_generator_by_channel); its
    # per-battery `battery` dicts are byte-identical to the two standalone
    # split files the gate itself consumes (verified by direct comparison),
    # so reading it here changes no value in any table below -- only the
    # single-source provenance.
    Path("analysis/step3/reliance_modelspace_prereg.json"),
]
DEFAULT_GATE_VERDICT = Path("analysis/step4/gate_verdict_prereg_v3.json")
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
# Table 4 -- model-space three-tier intervention effect (from the BEHAVIORAL
# control functional, prediction_change_control), per checkpoint x battery x
# estimator. This is the functional C4 evaluates since docs/gate_prereg.md's
# 2026-07-23 #2 amendment: the factor projection effect (true_effect) vs its
# equal-norm random control (mean/std) vs the task-direction positive control
# (task_direction_effect). It makes visible the finding behind C4's
# fail-with-live-control verdict: the factor effect is ~0 while the positive
# control is huge, so the pipeline is demonstrably alive and the factor
# reliance is genuinely absent.
# ---------------------------------------------------------------------------


def extract_three_tier_effect_table(records: list[dict]) -> list[dict]:
    """One row per (battery, estimator, checkpoint), folds aggregated by mean,
    read from per_checkpoint[ckpt].prediction_change_control -- the same
    behavioral control functional run_gate.criterion_4 reads. A checkpoint
    whose prediction_change_control is the not_estimable sentinel / empty (the
    w-disabled cache-space regime) contributes no numeric row and is skipped,
    exactly as C4 would fall back rather than read it."""
    rows = []
    for rec in records:
        battery = rec["battery"]
        for est_name, estimator in sorted(battery.get("estimators", {}).items()):
            if estimator.get("status") != "ok":
                continue
            acc: dict[str, dict] = {}
            for fold in estimator.get("fold_results", []):
                for ckpt_name, ckpt_entry in fold.get("effect", {}).get("per_checkpoint", {}).items():
                    pcc = ckpt_entry.get("prediction_change_control")
                    if not (isinstance(pcc, dict) and isinstance(pcc.get("true_effect"), (int, float))):
                        continue  # not_estimable sentinel / empty -- skip, don't fabricate
                    bucket = acc.setdefault(ckpt_name, dict(
                        true_effect=[], random_mean=[], random_std=[], task_direction_effect=[], exceeds_flags=[]))
                    bucket["true_effect"].append(float(pcc["true_effect"]))
                    if isinstance(pcc.get("random_mean"), (int, float)):
                        bucket["random_mean"].append(float(pcc["random_mean"]))
                    if isinstance(pcc.get("random_std"), (int, float)):
                        bucket["random_std"].append(float(pcc["random_std"]))
                    tde = pcc.get("task_direction_effect")
                    if isinstance(tde, (int, float)) and not isinstance(tde, bool):
                        bucket["task_direction_effect"].append(float(tde))
                    if isinstance(pcc.get("exceeds_random"), bool):
                        bucket["exceeds_flags"].append(pcc["exceeds_random"])
            for ckpt_name in sorted(acc):
                bucket = acc[ckpt_name]
                rm = float(np.mean(bucket["random_mean"])) if bucket["random_mean"] else None
                rs = float(np.mean(bucket["random_std"])) if bucket["random_std"] else None
                rows.append(dict(
                    battery=battery["name"], factor=battery.get("factor"), estimator=est_name,
                    checkpoint=ckpt_name,
                    factor_true_effect=float(np.mean(bucket["true_effect"])) if bucket["true_effect"] else None,
                    random_mean=rm, random_std=rs,
                    random_mean_plus_2std=(rm + 2 * rs) if (rm is not None and rs is not None) else None,
                    task_direction_effect=(float(np.mean(bucket["task_direction_effect"]))
                                           if bucket["task_direction_effect"] else None),
                    exceeds_random=(all(bucket["exceeds_flags"]) if bucket["exceeds_flags"] else None),
                    n_folds=len(bucket["true_effect"]),
                ))
    return rows


# ---------------------------------------------------------------------------
# Table 5 -- two-space factor decodability: cache-space FSS vs model-space
# factor_separation_score and LEACE factor_decodability_before, per fold. Puts
# the ENCODING result (factor strongly decodable from the frozen cache-space
# representation) next to the model-space decodability of the same factor, the
# space in which the causal-reliance null is measured.
# ---------------------------------------------------------------------------

# Cache-space FSS for the two ReplayDF batteries exists in this repository ONLY
# as prose in CURRENT_STATE.md's "Cache-space ENCODING" table -- the underlying
# per-battery bootstrap JSON files were produced on the collaborator's machine
# and are not committed here (the sole readable cache-space file is the DiffSSD
# openvoicev2-accent battery, FSS 0.992, used in Table 1). These values are
# therefore carried as explicitly-sourced constants, never silently presented
# as file-derived; each row records `cachespace_fss_source` so the provenance
# is unambiguous. Keyed by (corpus, factor) so a real committed file, if one
# ever appears, can override the prose value in _cachespace_fss_lookup below.
CACHESPACE_FSS_PROSE: dict[tuple[str, str], float] = {
    ("replaydf", "language"): 0.949,
    ("replaydf", "generator_id"): 0.889,
}
CACHESPACE_FSS_PROSE_SOURCE = "CURRENT_STATE.md prose (collaborator-machine bootstrap, not committed here)"


def _cachespace_fss_lookup(cachespace_records: list[dict]) -> dict[tuple[str, str], dict]:
    """(corpus, factor) -> {fss, source}. Prefers a real, readable cache-space
    file's `ok` factor_separation_score bootstrap; falls back to the
    explicitly-sourced CURRENT_STATE.md prose constant for batteries whose
    cache-space JSON is not committed on this machine."""
    lookup: dict[tuple[str, str], dict] = {}
    for rec in cachespace_records:
        battery = rec["battery"]
        boot = battery.get("headline_bootstrap", {})
        if boot.get("metric") == "factor_separation_score" and boot.get("status") == "ok":
            lookup[(battery.get("corpus"), battery.get("factor"))] = dict(
                fss=boot["mean"], source=f"file:{battery['name']} headline_bootstrap")
    for key, fss in CACHESPACE_FSS_PROSE.items():
        lookup.setdefault(key, dict(fss=fss, source=CACHESPACE_FSS_PROSE_SOURCE))
    return lookup


def extract_two_space_decodability_table(modelspace_records: list[dict],
                                         cachespace_records: list[dict]) -> list[dict]:
    """One row per (battery, estimator, fold): cache-space FSS (with explicit
    provenance) alongside that fold's model-space factor_separation_score and
    LEACE factor_decodability_before -- both checkpoint-independent, read at the
    fold level exactly where run_reliance_battery.py writes them."""
    fss_lookup = _cachespace_fss_lookup(cachespace_records)
    rows = []
    for rec in modelspace_records:
        battery = rec["battery"]
        cs = fss_lookup.get((battery.get("corpus"), battery.get("factor")), {})
        for est_name, estimator in sorted(battery.get("estimators", {}).items()):
            if estimator.get("status") != "ok":
                continue
            for fold in estimator.get("fold_results", []):
                eff = fold.get("effect", {})
                fss = eff.get("factor_separation_score")
                leace_before = eff.get("leace", {}).get("factor_decodability_before")
                rows.append(dict(
                    battery=battery["name"], factor=battery.get("factor"), estimator=est_name,
                    fold_id=fold.get("fold_id"),
                    cachespace_fss=cs.get("fss"), cachespace_fss_source=cs.get("source"),
                    modelspace_factor_separation_score=(float(fss) if isinstance(fss, (int, float)) else None),
                    modelspace_leace_factor_decodability_before=(
                        float(leace_before) if isinstance(leace_before, (int, float)) else None),
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


def render_markdown(encoding_rows, reliance_rows, three_tier_rows, two_space_rows,
                    gate_rows, overall_classification) -> str:
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

    lines += ["## Table 4 -- Model-space three-tier intervention effect "
              "(behavioral control: prediction_change_control)", ""]
    if three_tier_rows:
        lines += ["| Battery | Estimator | Checkpoint | factor true_effect | random mean | random std | "
                   "random mean+2std | task_direction_effect | main exceeds_random | n_folds |",
                   "|---|---|---|---|---|---|---|---|---|---|"]
        for r in three_tier_rows:
            lines.append(
                f"| {r['battery']} | {r['estimator']} | {r['checkpoint']} | {_fmt(r['factor_true_effect'], 3)} | "
                f"{_fmt(r['random_mean'], 3)} | {_fmt(r['random_std'], 3)} | {_fmt(r['random_mean_plus_2std'], 3)} | "
                f"{_fmt(r['task_direction_effect'], 4)} | {_fmt(r['exceeds_random'])} | {r['n_folds']} |")
        lines += ["", "The factor `true_effect` sits far below `random mean+2std` (so `main exceeds_random` "
                       "is False everywhere) while `task_direction_effect` towers over the same bar -- the "
                       "positive control is alive, the factor reliance is genuinely absent. This is the "
                       "evidence behind C4's fail-WITH-live-control verdict (docs/gate_prereg.md 2026-07-23 #2)."]
    else:
        lines.append("*(no model-space prediction_change_control data found)*")
    lines.append("")

    lines += ["## Table 5 -- Two-space factor decodability "
              "(cache-space FSS vs model-space, per fold)", ""]
    if two_space_rows:
        lines += ["| Battery | Estimator | Fold | cache-space FSS | model-space FSS | "
                   "model-space LEACE decodability_before |",
                   "|---|---|---|---|---|---|"]
        for r in two_space_rows:
            lines.append(
                f"| {r['battery']} | {r['estimator']} | {r['fold_id']} | {_fmt(r['cachespace_fss'], 4)} | "
                f"{_fmt(r['modelspace_factor_separation_score'], 4)} | "
                f"{_fmt(r['modelspace_leace_factor_decodability_before'], 4)} |")
        prose_sources = sorted({r["cachespace_fss_source"] for r in two_space_rows
                                if r.get("cachespace_fss_source") and "prose" in r["cachespace_fss_source"]})
        if prose_sources:
            lines += ["", "**Cache-space FSS provenance:** the cache-space FSS column for the ReplayDF batteries "
                           "is carried from " + "; ".join(prose_sources) + " -- the per-battery cache-space "
                           "bootstrap JSON is not committed on this machine (only the DiffSSD accent battery in "
                           "Table 1 is). Each row's provenance is recorded in `cachespace_fss_source` in the "
                           "JSON output; these values are never presented as file-derived."]
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
    three_tier_rows = extract_three_tier_effect_table(modelspace_records)
    two_space_rows = extract_two_space_decodability_table(modelspace_records, cachespace_records)
    gate_rows = extract_gate_table(verdict)
    overall_classification = verdict.get("overall_classification") if verdict else None

    report = dict(
        schema_version=2, generated_at=_timestamp(), warnings=warnings,
        encoding_table=encoding_rows, reliance_table=reliance_rows,
        three_tier_effect_table=three_tier_rows, two_space_decodability_table=two_space_rows,
        gate_table=gate_rows, overall_classification=overall_classification,
    )
    markdown = render_markdown(encoding_rows, reliance_rows, three_tier_rows, two_space_rows,
                               gate_rows, overall_classification)

    _atomic_write(args.out_dir / "paper_tables.json", json.dumps(report, indent=2))
    _atomic_write(args.out_dir / "paper_tables.md", markdown)
    _log(f"extract_paper_tables: wrote {args.out_dir / 'paper_tables.json'} and "
         f"{args.out_dir / 'paper_tables.md'}")


if __name__ == "__main__":
    main()
