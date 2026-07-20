"""Step 4 gate consumer (docs/gate_prereg.md).

Evaluates the 8 pre-registered Step 4 criteria against whatever real inputs
are actually present on disk, and emits a single verdict JSON. This script
never runs the gate itself (no training, no GPU) -- it is a read-only
analysis over already-produced files:

  - Phase A battery JSON, produced by scripts/run_reliance_battery.py.
    Two real on-disk shapes, both handled (verified against
    tests/fixtures/step3/*.json, the real fixtures this module is built
    against):
      * manifest-shaped (the file passed to --out): top-level `batteries`
        (list) + `prereg_candidates` (list) + `checkpoints` + `w_metrics`
        + `join_stats`, optionally `battery_files` (name -> path of a
        standalone per-battery copy of the same content).
      * standalone per-battery file (one entry of `battery_files`):
        top-level singular `battery` + `prereg_candidate`, same shared
        `checkpoints`/`w_metrics`/`join_stats` keys.

  - Phase B model-space embedding caches, produced by
    scripts/extract_model_embeddings.py: <out-root>/<checkpoint-stem>/
    <corpus-dir>/shard_*.npz, each shard holding (paths, emb, meta) per
    that script's own docstring and `_write_shard_atomic`. This script only
    checks presence + structural schema (dir/shard exist, keys present,
    embedding_dim readable) -- no Phase B cache exists on this machine yet,
    so every check of it is exercised by a synthetic stand-in in tests and
    reported `pending_input` for real runs until the collaborator machine
    produces it.

  - Per-checkpoint EER files, produced by cross_test.py (directly, or via
    scripts/reproduce_eval.py's `repro_<run>.json`): top-level `checkpoint`
    (the .pt path, whose parent dir name is the run name, e.g.
    "e007_A_fresh") + `per_corpus.<corpus>.eer`. `experiments/e007/
    e007_{A,B,C}_fresh_crosstest.json` already exist in this repository
    today and are real, not synthetic.

  - Seeded head-replicate results, produced by scripts/head_replicates.py
    (this session's Task 3 deliverable): per-seed effect direction, for
    criterion 8.

Every criterion that needs data not present is reported as
status="pending_input" -- never crashes, never silently promoted to
pass/fail. A criterion whose underlying Phase A field is ITSELF the
`not_estimable` sentinel (e.g. the w-dim mismatch already recorded in every
real per_checkpoint block today) is reported as status="not_estimable",
distinct from `pending_input`: `not_estimable` means "Phase A already
determined this is structurally impossible with the current embedding
cache"; `pending_input` means "this script doesn't have the upstream file
yet." An overall three-outcome classification (see docs/gate_prereg.md §3)
is emitted only when every criterion has resolved to pass/fail -- i.e. none
are pending_input or not_estimable.

Follows the heartbeat/flush/incremental-output/exit-0 conventions already
established in scripts/run_reliance_battery.py: every stage is timestamped
and printed with flush=True; the verdict file is always written (temp file
+ os.replace) and the process always exits 0 -- this is a reporting tool,
not a build gate that should break a CI run while collaborator-machine work
is still in flight.

Usage:
    python scripts/run_gate.py \\
        --phase-a tests/fixtures/step3/reliance_layer9_boot.json \\
        --eer experiments/e007/e007_A_fresh_crosstest.json \\
              experiments/e007/e007_B_fresh_crosstest.json \\
              experiments/e007/e007_C_xlsr_fresh_crosstest.json \\
        --out analysis/step4/gate_verdict.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_PENDING = "pending_input"
STATUS_NOT_ESTIMABLE = "not_estimable"

DEFAULT_PHASE_B_OUT_ROOT = Path("analysis/step3/_embcache_modelspace")
DEFAULT_VERDICT_OUT = Path("analysis/step4/gate_verdict.json")


# ---------------------------------------------------------------------------
# Heartbeat logging -- same shape as scripts/run_reliance_battery.py's
# _timestamp/_log, so gate runs read consistently with Phase A runs in a
# combined log.
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _log(msg: str) -> None:
    print(f"[{_timestamp()}] {msg}", flush=True)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    tmp_path.replace(path)


def _criterion(status: str, evidence: str, numbers: dict | None = None) -> dict:
    return dict(status=status, evidence=evidence, numbers=numbers or {})


# ---------------------------------------------------------------------------
# Phase A loading -- normalizes both real on-disk shapes to one internal
# representation: a list of battery records.
# ---------------------------------------------------------------------------


def load_phase_a_file(path: Path) -> list[dict]:
    """Load one Phase A JSON file (manifest-shaped or standalone
    per-battery-shaped -- both real, observed shapes) and return a list of
    normalized battery records: {battery, prereg_candidate, checkpoints,
    w_metrics, join_stats, source}. Raises only on a file that parses but
    matches neither known shape -- callers are expected to catch this and
    degrade to pending_input, per this script's never-crash contract for
    missing/malformed *optional* inputs; a --phase-a path the caller
    explicitly supplied but that doesn't parse is a configuration error
    worth surfacing, not silently swallowing.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    shared = dict(
        checkpoints=raw.get("checkpoints", {}),
        w_metrics=raw.get("w_metrics", {}),
        join_stats=raw.get("join_stats", {}),
        schema_version=raw.get("schema_version"),
        git_sha=raw.get("git_sha"),
    )
    records = []
    if "battery" in raw and "prereg_candidate" in raw:
        records.append(dict(battery=raw["battery"], prereg_candidate=raw["prereg_candidate"],
                             source=str(path), **shared))
    elif "batteries" in raw and "prereg_candidates" in raw:
        cand_by_name = {c["name"]: c for c in raw["prereg_candidates"] if "name" in c}
        for b in raw["batteries"]:
            records.append(dict(battery=b, prereg_candidate=cand_by_name.get(b.get("name"), {}),
                                 source=str(path), **shared))
    else:
        raise ValueError(f"{path}: unrecognized Phase A schema -- no 'battery'/'prereg_candidate' "
                          "or 'batteries'/'prereg_candidates' keys found")
    return records


def load_phase_a_inputs(paths: list[Path]) -> tuple[list[dict], list[str]]:
    """Best-effort load of every --phase-a path. A path that doesn't exist
    or fails to parse is recorded as a warning and skipped, never crashes
    the run."""
    records, warnings = [], []
    for p in paths:
        p = Path(p)
        if not p.exists():
            warnings.append(f"phase-a input not found: {p}")
            continue
        try:
            records.extend(load_phase_a_file(p))
        except Exception as exc:
            warnings.append(f"phase-a input {p} failed to load: {exc}")
    return records, warnings


# ---------------------------------------------------------------------------
# EER loading (cross_test.py / reproduce_eval.py output shape)
# ---------------------------------------------------------------------------


def load_eer_file(path: Path) -> tuple[str, dict[str, float]]:
    """One cross_test.py-shaped result file -> (run_name, {corpus: eer}).
    run_name is the parent-directory name of the checkpoint path
    (result['checkpoint'], e.g. "runs/e007_A_fresh/best.pt" -> "e007_A_fresh"),
    which is exactly how checkpoints are named/keyed in Phase A battery
    output -- verified against experiments/e007/*_crosstest.json (real,
    committed files) and scripts/run_reliance_battery.py's checkpoint dict
    keys."""
    d = json.loads(path.read_text(encoding="utf-8"))
    ckpt_path = Path(d.get("checkpoint", ""))
    run_name = ckpt_path.parent.name or path.stem
    per_corpus = d.get("per_corpus", {})
    eers = {c: float(v["eer"]) for c, v in per_corpus.items() if isinstance(v, dict) and "eer" in v}
    return run_name, eers


def load_eer_inputs(paths: list[Path]) -> tuple[dict[str, dict[str, float]], list[str]]:
    out, warnings = {}, []
    for p in paths:
        p = Path(p)
        if not p.exists():
            warnings.append(f"eer input not found: {p}")
            continue
        try:
            run_name, eers = load_eer_file(p)
            out[run_name] = eers
        except Exception as exc:
            warnings.append(f"eer input {p} failed to load: {exc}")
    return out, warnings


# ---------------------------------------------------------------------------
# Phase B cache presence/schema check (extract_model_embeddings.py output)
# ---------------------------------------------------------------------------


def check_phase_b_cache(out_root: Path, checkpoint_stem: str, corpus_dir: str) -> dict:
    """Presence + structural check only, for the REAL shard layout written
    by scripts/extract_model_embeddings.py: <out_root>/<checkpoint_stem>/
    <corpus_dir>/shard_*.npz, each an atomically-written npz with
    (paths, emb, meta) keys (meta a JSON-encoded 0-d string array). Never
    raises: any read failure degrades to status="fail" with the exception
    message, an absent directory/shard set degrades to
    status="pending_input" (this is the expected, current state on this
    machine -- no Phase B cache exists here yet)."""
    cache_dir = out_root / checkpoint_stem / corpus_dir
    if not cache_dir.is_dir():
        return dict(status=STATUS_PENDING, path=str(cache_dir),
                     reason=f"no Phase B embedding cache directory at {cache_dir} -- run "
                            "scripts/extract_model_embeddings.py for this checkpoint/corpus "
                            "(collaborator GPU machine)")
    shards = sorted(cache_dir.glob("shard_*.npz"))
    if not shards:
        return dict(status=STATUS_PENDING, path=str(cache_dir),
                     reason=f"{cache_dir} exists but contains no shard_*.npz files yet")
    try:
        n_rows_total, embedding_dim, dtype = 0, None, None
        for shard_path in shards:
            with np.load(shard_path, allow_pickle=True) as npz:
                missing = {"paths", "emb", "meta"} - set(npz.files)
                if missing:
                    return dict(status=STATUS_FAIL, path=str(shard_path),
                                 reason=f"{shard_path} missing expected keys {sorted(missing)}, "
                                        f"found {sorted(npz.files)}")
                emb = npz["emb"]
                n_rows_total += int(emb.shape[0])
                embedding_dim = int(emb.shape[1])
                dtype = str(emb.dtype)
        return dict(status=STATUS_PASS, path=str(cache_dir), n_shards=len(shards),
                     n_rows=n_rows_total, embedding_dim=embedding_dim, dtype=dtype)
    except Exception as exc:
        return dict(status=STATUS_FAIL, path=str(cache_dir),
                     reason=f"failed to read shard(s) under {cache_dir}: {exc}")


# ---------------------------------------------------------------------------
# Head-replicate loading (Task 3 deliverable, scripts/head_replicates.py)
# ---------------------------------------------------------------------------


def load_head_replicates(path: Path | None) -> tuple[list[dict] | None, list[str]]:
    if path is None:
        return None, []
    path = Path(path)
    if not path.exists():
        return None, [f"head-replicate input not found: {path}"]
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        replicates = d.get("replicates")
        if not isinstance(replicates, list):
            return None, [f"{path}: expected a 'replicates' list, got {type(replicates)}"]
        return replicates, []
    except Exception as exc:
        return None, [f"head-replicate input {path} failed to load: {exc}"]


# ---------------------------------------------------------------------------
# Per-checkpoint reliance extraction shared by C2/C7 -- reads whatever
# per_checkpoint alignment/r_var values are present, honoring Phase A's own
# not_estimable sentinel rather than treating it as zero/missing.
# ---------------------------------------------------------------------------


def _per_checkpoint_reliance(battery: dict, metric: str = "alignment") -> dict[str, dict]:
    """checkpoint_name -> {status, value} aggregated (mean of finite fold
    values) across every fold_result's effect.per_checkpoint[name][metric].
    status is 'not_estimable' if every fold reports Phase A's own
    not_estimable sentinel for this checkpoint/metric, 'ok' if at least one
    finite value exists."""
    out: dict[str, dict] = {}
    for estimator in battery.get("estimators", {}).values():
        for fold in estimator.get("fold_results", []):
            per_ckpt = fold.get("effect", {}).get("per_checkpoint", {})
            for ckpt_name, ckpt_metrics in per_ckpt.items():
                entry = ckpt_metrics.get(metric)
                bucket = out.setdefault(ckpt_name, dict(values=[], any_not_estimable=False))
                if isinstance(entry, dict):
                    if entry.get("status") == STATUS_NOT_ESTIMABLE:
                        bucket["any_not_estimable"] = True
                    elif entry.get("value") is not None and np.isfinite(entry["value"]):
                        bucket["values"].append(float(entry["value"]))
                elif entry is not None and np.isfinite(entry):
                    bucket["values"].append(float(entry))
    result = {}
    for ckpt_name, bucket in out.items():
        if bucket["values"]:
            result[ckpt_name] = dict(status="ok", value=float(np.mean(bucket["values"])))
        elif bucket["any_not_estimable"]:
            result[ckpt_name] = dict(status=STATUS_NOT_ESTIMABLE, value=None)
        else:
            result[ckpt_name] = dict(status=STATUS_PENDING, value=None)
    return result


# ---------------------------------------------------------------------------
# C1 -- replication across >=2 backbones
# ---------------------------------------------------------------------------


def criterion_1_replication(primary: list[dict], secondary: list[dict]) -> dict:
    if not secondary:
        return _criterion(
            STATUS_PENDING,
            "no second-backbone (WavLM-Large) battery input provided -- pass "
            "--phase-a-secondary-backbone once the collaborator machine's WavLM-Large "
            "embedding pass produces it",
        )
    primary_by_name = {r["battery"]["name"]: r for r in primary}
    secondary_by_name = {r["battery"]["name"]: r for r in secondary}
    shared_names = sorted(set(primary_by_name) & set(secondary_by_name))
    if not shared_names:
        return _criterion(
            STATUS_PENDING,
            "primary and secondary-backbone battery sets share no battery name to compare",
            dict(primary_names=sorted(primary_by_name), secondary_names=sorted(secondary_by_name)),
        )
    per_battery = {}
    for name in shared_names:
        p_boot = primary_by_name[name]["battery"]["headline_bootstrap"]
        s_boot = secondary_by_name[name]["battery"]["headline_bootstrap"]
        if p_boot.get("status") != "ok" or s_boot.get("status") != "ok":
            per_battery[name] = dict(agree=None, reason="headline_bootstrap not ok on one or both backbones")
            continue
        same_sign = np.sign(p_boot["mean"]) == np.sign(s_boot["mean"])
        disjoint_opposite = (p_boot["hi"] < 0 < s_boot["lo"]) or (s_boot["hi"] < 0 < p_boot["lo"])
        per_battery[name] = dict(agree=bool(same_sign and not disjoint_opposite),
                                  primary_mean=p_boot["mean"], secondary_mean=s_boot["mean"])
    decided = {k: v for k, v in per_battery.items() if v["agree"] is not None}
    if not decided:
        return _criterion(STATUS_PENDING, "no shared battery had a usable headline_bootstrap on both backbones",
                           dict(per_battery=per_battery))
    all_agree = all(v["agree"] for v in decided.values())
    return _criterion(
        STATUS_PASS if all_agree else STATUS_FAIL,
        f"{sum(v['agree'] for v in decided.values())}/{len(decided)} shared batteries replicate "
        "(sign agreement, no disjoint-opposite CIs) across the two backbones",
        dict(per_battery=per_battery),
    )


# ---------------------------------------------------------------------------
# C2 -- positive reliance <-> matching-factor-degradation association
# ---------------------------------------------------------------------------


def criterion_2_association(records: list[dict], eers: dict[str, dict[str, float]],
                             factor_corpus_map: dict[str, str]) -> dict:
    if not records:
        return _criterion(STATUS_PENDING, "no Phase A battery input provided")
    per_battery = {}
    for rec in records:
        battery = rec["battery"]
        factor = battery.get("factor")
        eval_corpus = factor_corpus_map.get(factor)
        if eval_corpus is None:
            per_battery[battery["name"]] = dict(
                status=STATUS_PENDING,
                reason=f"no factor-corpus mapping supplied for factor={factor!r} -- see "
                       "docs/gate_prereg.md §4 ambiguity 2 (diffssd is dev-tier-for-thresholding "
                       "only, never itself EER-scored by cross_test.py)",
            )
            continue
        reliance = _per_checkpoint_reliance(battery, metric="alignment")
        pairs = []
        any_not_estimable = False
        for ckpt_name, rel in reliance.items():
            if rel["status"] == STATUS_NOT_ESTIMABLE:
                any_not_estimable = True
                continue
            if rel["status"] != "ok":
                continue
            eer = eers.get(ckpt_name, {}).get(eval_corpus)
            if eer is None:
                continue
            pairs.append((ckpt_name, rel["value"], eer))
        if len(pairs) < 2:
            per_battery[battery["name"]] = dict(
                status=STATUS_NOT_ESTIMABLE if any_not_estimable else STATUS_PENDING,
                reason="fewer than 2 checkpoints have both an estimable reliance value and a "
                       f"matching EER on {eval_corpus!r}" + (" (reliance not_estimable: w-dim mismatch, "
                       "pending Phase B model-space embeddings)" if any_not_estimable else ""),
                pairs=pairs,
            )
            continue
        reliances = np.array([p[1] for p in pairs])
        target_eers = np.array([p[2] for p in pairs])
        corr = float(np.corrcoef(reliances, target_eers)[0, 1]) if np.std(reliances) > 0 and np.std(target_eers) > 0 else float("nan")
        positive = np.isfinite(corr) and corr > 0
        per_battery[battery["name"]] = dict(
            status=STATUS_PASS if positive else (STATUS_FAIL if np.isfinite(corr) else STATUS_PENDING),
            correlation=corr, pairs=pairs, eval_corpus=eval_corpus,
        )
    if not per_battery:
        return _criterion(STATUS_PENDING, "no batteries produced a comparable pair")
    statuses = {v["status"] for v in per_battery.values()}
    if statuses <= {STATUS_PASS}:
        overall = STATUS_PASS
    elif STATUS_FAIL in statuses:
        overall = STATUS_FAIL
    elif STATUS_NOT_ESTIMABLE in statuses and STATUS_PENDING not in statuses:
        overall = STATUS_NOT_ESTIMABLE
    else:
        overall = STATUS_PENDING
    return _criterion(overall, f"per-battery association results: { {k: v['status'] for k, v in per_battery.items()} }",
                       dict(per_battery=per_battery))


# ---------------------------------------------------------------------------
# C3 -- grouped-bootstrap CIs (methodological precondition)
# ---------------------------------------------------------------------------


def criterion_3_grouped_bootstrap(records: list[dict], min_n_boot: int = 1000, min_n_groups: int = 8) -> dict:
    if not records:
        return _criterion(STATUS_PENDING, "no Phase A battery input provided")
    per_battery = {}
    for rec in records:
        battery = rec["battery"]
        boot = battery.get("headline_bootstrap", {})
        cand = rec.get("prereg_candidate", {})
        ok = (
            boot.get("status") == "ok"
            and boot.get("n_boot", 0) >= min_n_boot
            and cand.get("n_groups", 0) >= min_n_groups
            and cand.get("grouping_degenerate") is False
        )
        per_battery[battery["name"]] = dict(
            pass_=bool(ok), n_boot=boot.get("n_boot"), n_groups=cand.get("n_groups"),
            grouping_degenerate=cand.get("grouping_degenerate"), boot_status=boot.get("status"),
        )
    all_pass = all(v["pass_"] for v in per_battery.values())
    return _criterion(
        STATUS_PASS if all_pass else STATUS_FAIL,
        f"{sum(v['pass_'] for v in per_battery.values())}/{len(per_battery)} batteries have a usable "
        f"grouped bootstrap (status=ok, n_boot>={min_n_boot}, n_groups>={min_n_groups}, non-degenerate grouping)",
        dict(per_battery=per_battery),
    )


# ---------------------------------------------------------------------------
# C4 -- intervention effects exceed equal-norm random controls; task-
# direction removal as positive control
# ---------------------------------------------------------------------------


def criterion_4_intervention_vs_random(records: list[dict], min_fraction: float = 0.5) -> dict:
    """`projection_removal_control` reports the main factor-projection
    effect (`true_effect`, `random_effects`, `exceeds_random` -- already
    `true_effect > random_mean + 2*random_std`) alongside a SIBLING
    `task_direction_effect` field, the positive control. Verified against
    the real fixtures: both `exceeds_random` and `task_direction_effect`
    are bare floats when estimable (smoke.json) and `task_direction_effect`
    degrades to the `not_estimable` sentinel dict when w is unavailable
    (the real accent battery) -- `task_direction_effect` never carries its
    own `exceeds_random` key, so the positive control's own exceeds-random
    flag has to be computed here from the same random_mean/random_std the
    main effect uses."""
    if not records:
        return _criterion(STATUS_PENDING, "no Phase A battery input provided")
    per_battery = {}
    for rec in records:
        battery = rec["battery"]
        main_flags, control_flags, control_any_not_estimable = [], [], False
        for estimator in battery.get("estimators", {}).values():
            for fold in estimator.get("fold_results", []):
                prc = fold.get("effect", {}).get("projection_removal_control", {})
                if "exceeds_random" in prc:
                    main_flags.append(bool(prc["exceeds_random"]))
                tde = prc.get("task_direction_effect")
                random_mean, random_std = prc.get("random_mean"), prc.get("random_std")
                if isinstance(tde, dict) and tde.get("status") == STATUS_NOT_ESTIMABLE:
                    control_any_not_estimable = True
                elif (tde is not None and not isinstance(tde, dict)
                      and random_mean is not None and random_std is not None):
                    control_flags.append(bool(tde > random_mean + 2 * random_std))
        main_fraction = float(np.mean(main_flags)) if main_flags else None
        main_ok = main_fraction is not None and main_fraction >= min_fraction
        if control_flags:
            control_ok = all(control_flags)
            status = STATUS_PASS if (main_ok and control_ok) else STATUS_FAIL
        elif control_any_not_estimable:
            status = STATUS_NOT_ESTIMABLE
        else:
            status = STATUS_PENDING
        per_battery[battery["name"]] = dict(
            status=status, main_exceeds_random_fraction=main_fraction, n_main_folds=len(main_flags),
            n_control_folds_estimable=len(control_flags),
        )
    statuses = {v["status"] for v in per_battery.values()}
    if statuses <= {STATUS_PASS}:
        overall = STATUS_PASS
    elif STATUS_FAIL in statuses:
        overall = STATUS_FAIL
    elif STATUS_NOT_ESTIMABLE in statuses and STATUS_PENDING not in statuses:
        overall = STATUS_NOT_ESTIMABLE
    else:
        overall = STATUS_PENDING
    return _criterion(overall, "per-battery main-effect-vs-random and positive-control results",
                       dict(per_battery=per_battery))


# ---------------------------------------------------------------------------
# C5 -- rank stability
# ---------------------------------------------------------------------------


def criterion_5_rank_stability(records: list[dict]) -> dict:
    if not records:
        return _criterion(STATUS_PENDING, "no Phase A battery input provided")
    per_battery = {}
    for rec in records:
        cand = rec.get("prereg_candidate", {})
        window = cand.get("stable_rank_window", [])
        ok = len(window) >= 2 and 1 in window
        per_battery[rec["battery"]["name"]] = dict(pass_=bool(ok), stable_rank_window=window)
    all_pass = all(v["pass_"] for v in per_battery.values())
    return _criterion(
        STATUS_PASS if all_pass else STATUS_FAIL,
        f"{sum(v['pass_'] for v in per_battery.values())}/{len(per_battery)} batteries have a stable "
        "rank window (>=2 consecutive ranks, including headline rank 1)",
        dict(per_battery=per_battery),
    )


# ---------------------------------------------------------------------------
# C6 -- agreement across >=2 subspace estimators
# ---------------------------------------------------------------------------


def criterion_6_estimator_agreement(records: list[dict]) -> dict:
    if not records:
        return _criterion(STATUS_PENDING, "no Phase A battery input provided")
    per_battery = {
        rec["battery"]["name"]: bool(rec.get("prereg_candidate", {}).get("estimators_agree_sign", False))
        for rec in records
    }
    all_agree = all(per_battery.values())
    return _criterion(
        STATUS_PASS if all_agree else STATUS_FAIL,
        f"{sum(per_battery.values())}/{len(per_battery)} batteries show estimators_agree_sign=True "
        "(LDA-subspace vs cross-fitted linear-probe)",
        dict(per_battery=per_battery),
    )


# ---------------------------------------------------------------------------
# C7 -- no collapse after controlling for clean EER / checkpoint quality /
# training corpus
# ---------------------------------------------------------------------------


def criterion_7_no_collapse(records: list[dict], eers: dict[str, dict[str, float]],
                             factor_corpus_map: dict[str, str], clean_corpora: tuple[str, ...] = ("inthewild", "replaydf", "ai4t")) -> dict:
    if not records:
        return _criterion(STATUS_PENDING, "no Phase A battery input provided")
    per_battery = {}
    for rec in records:
        battery = rec["battery"]
        factor = battery.get("factor")
        eval_corpus = factor_corpus_map.get(factor)
        reliance = _per_checkpoint_reliance(battery, metric="alignment")
        rows = []
        any_not_estimable = any(r["status"] == STATUS_NOT_ESTIMABLE for r in reliance.values())
        for ckpt_name, rel in reliance.items():
            if rel["status"] != "ok":
                continue
            ckpt_eers = eers.get(ckpt_name, {})
            clean_vals = [ckpt_eers[c] for c in clean_corpora if c in ckpt_eers]
            if not clean_vals or eval_corpus is None or eval_corpus not in ckpt_eers:
                continue
            rows.append(dict(ckpt=ckpt_name, reliance=rel["value"], clean_eer=float(np.mean(clean_vals)),
                              target_eer=ckpt_eers[eval_corpus]))
        if len(rows) < 3:
            per_battery[battery["name"]] = dict(
                status=STATUS_NOT_ESTIMABLE if any_not_estimable else STATUS_PENDING,
                reason=f"need >=3 checkpoints with reliance + clean EER + target-corpus EER, have {len(rows)}",
                rows=rows,
            )
            continue
        reliance_arr = np.array([r["reliance"] for r in rows])
        clean_arr = np.array([r["clean_eer"] for r in rows])
        target_arr = np.array([r["target_eer"] for r in rows])
        raw_sign = np.sign(np.corrcoef(reliance_arr, target_arr)[0, 1]) if np.std(reliance_arr) > 0 else 0.0
        design = np.column_stack([np.ones_like(clean_arr), clean_arr])
        coef, *_ = np.linalg.lstsq(design, reliance_arr, rcond=None)
        reliance_resid = reliance_arr - design @ coef
        resid_sign = (np.sign(np.corrcoef(reliance_resid, target_arr)[0, 1])
                      if np.std(reliance_resid) > 1e-12 else 0.0)
        survives = bool(raw_sign != 0 and raw_sign == resid_sign)
        per_battery[battery["name"]] = dict(
            status=STATUS_PASS if survives else STATUS_FAIL,
            raw_sign=float(raw_sign), residual_sign=float(resid_sign), n_checkpoints=len(rows), rows=rows,
        )
    if not per_battery:
        return _criterion(STATUS_PENDING, "no batteries produced enough checkpoints to residualize")
    statuses = {v["status"] for v in per_battery.values()}
    if statuses <= {STATUS_PASS}:
        overall = STATUS_PASS
    elif STATUS_FAIL in statuses:
        overall = STATUS_FAIL
    elif STATUS_NOT_ESTIMABLE in statuses and STATUS_PENDING not in statuses:
        overall = STATUS_NOT_ESTIMABLE
    else:
        overall = STATUS_PENDING
    return _criterion(
        overall,
        "clean-EER-residualized sign-survival per battery (n=3-ish checkpoints -- descriptive, "
        "not a powered significance test; see docs/gate_prereg.md §4 ambiguity 3)",
        dict(per_battery=per_battery),
    )


# ---------------------------------------------------------------------------
# C8 -- consistent effect direction across >=3 independently seeded
# replicates
# ---------------------------------------------------------------------------


def criterion_8_seeded_replicates(replicates: list[dict] | None, min_replicates: int = 3) -> dict:
    if replicates is None:
        return _criterion(
            STATUS_PENDING,
            "no head-replicate result file provided -- run scripts/head_replicates.py "
            "against the real cached embeddings (collaborator machine) and pass --head-replicates",
        )
    if len(replicates) < min_replicates:
        return _criterion(
            STATUS_PENDING,
            f"only {len(replicates)} replicate(s) present, need >= {min_replicates}",
            dict(n_replicates=len(replicates)),
        )
    signs = [np.sign(r["effect"]) for r in replicates if r.get("effect") is not None]
    if len(signs) < min_replicates:
        return _criterion(STATUS_PENDING, "fewer than the minimum replicates carried a finite effect value",
                           dict(n_finite=len(signs)))
    unanimous = len(set(signs)) == 1 and signs[0] != 0
    return _criterion(
        STATUS_PASS if unanimous else STATUS_FAIL,
        f"{len(signs)} seeded replicates, signs={signs}",
        dict(seeds=[r.get("seed") for r in replicates], effects=[r.get("effect") for r in replicates]),
    )


# ---------------------------------------------------------------------------
# Overall three-outcome classification (docs/gate_prereg.md §3)
# ---------------------------------------------------------------------------


def classify_overall(criteria: dict[str, dict]) -> str | None:
    """None if any criterion is not yet a decided pass/fail (pending_input
    or not_estimable) -- an overall verdict is only ever emitted once every
    criterion has actually resolved. Otherwise: strong success if all 8
    pass; failure if any of the sign-bearing criteria (C2, C4, C6, C8) is a
    directional reversal (fail); diagnostic-only otherwise."""
    statuses = {name: c["status"] for name, c in criteria.items()}
    if any(s not in (STATUS_PASS, STATUS_FAIL) for s in statuses.values()):
        return None
    if all(s == STATUS_PASS for s in statuses.values()):
        return "strong_success"
    sign_bearing = ("C2", "C4", "C6", "C8")
    if any(statuses.get(c) == STATUS_FAIL for c in sign_bearing):
        return "failure"
    return "diagnostic_only"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_gate(
    phase_a_paths: list[Path],
    secondary_backbone_paths: list[Path] | None = None,
    eer_paths: list[Path] | None = None,
    phase_b_out_root: Path = DEFAULT_PHASE_B_OUT_ROOT,
    head_replicates_path: Path | None = None,
    factor_corpus_map: dict[str, str] | None = None,
) -> dict:
    secondary_backbone_paths = secondary_backbone_paths or []
    eer_paths = eer_paths or []
    factor_corpus_map = factor_corpus_map or {}

    _log("gate: loading Phase A battery input(s)")
    records, warnings = load_phase_a_inputs(phase_a_paths)
    _log(f"gate: loaded {len(records)} battery record(s) from {len(phase_a_paths)} file(s), "
         f"{len(warnings)} warning(s)")

    secondary_records, secondary_warnings = load_phase_a_inputs(secondary_backbone_paths)
    warnings += secondary_warnings

    _log("gate: loading EER input(s)")
    eers, eer_warnings = load_eer_inputs(eer_paths)
    warnings += eer_warnings
    _log(f"gate: loaded EERs for {len(eers)} checkpoint(s)")

    _log("gate: checking Phase B embedding-cache readiness")
    phase_b_status = {}
    seen = set()
    for rec in records:
        for ckpt_stem in rec["checkpoints"]:
            for corpus_dir in rec["join_stats"]:
                key = (ckpt_stem, corpus_dir)
                if key in seen:
                    continue
                seen.add(key)
                phase_b_status[f"{ckpt_stem}/{corpus_dir}"] = check_phase_b_cache(phase_b_out_root, ckpt_stem, corpus_dir)

    _log("gate: loading head-replicate input (criterion 8)")
    replicates, replicate_warnings = load_head_replicates(head_replicates_path)
    warnings += replicate_warnings

    _log("gate: computing criteria")
    criteria = dict(
        C1=criterion_1_replication(records, secondary_records),
        C2=criterion_2_association(records, eers, factor_corpus_map),
        C3=criterion_3_grouped_bootstrap(records),
        C4=criterion_4_intervention_vs_random(records),
        C5=criterion_5_rank_stability(records),
        C6=criterion_6_estimator_agreement(records),
        C7=criterion_7_no_collapse(records, eers, factor_corpus_map),
        C8=criterion_8_seeded_replicates(replicates),
    )
    for name, c in criteria.items():
        _log(f"gate: {name} -> {c['status']}")

    overall = classify_overall(criteria)
    _log(f"gate: overall classification = {overall!r}")

    return dict(
        schema_version=1,
        generated_at=_timestamp(),
        inputs=dict(
            phase_a_paths=[str(p) for p in phase_a_paths],
            secondary_backbone_paths=[str(p) for p in secondary_backbone_paths],
            eer_paths=[str(p) for p in eer_paths],
            phase_b_out_root=str(phase_b_out_root),
            head_replicates_path=str(head_replicates_path) if head_replicates_path else None,
        ),
        warnings=warnings,
        phase_b_cache_status=phase_b_status,
        criteria=criteria,
        overall_classification=overall,
    )


def _parse_factor_corpus_map(raw: list[str] | None) -> dict[str, str]:
    out = {}
    for item in raw or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"expected factor=corpus, got {item!r}")
        factor, corpus = item.split("=", 1)
        out[factor] = corpus
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase-a", nargs="+", type=Path, default=[],
                    help="Phase A battery JSON file(s), primary backbone (manifest-shaped or per-battery-shaped)")
    ap.add_argument("--phase-a-secondary-backbone", nargs="+", type=Path, default=[],
                    help="Phase A battery JSON file(s) for the second backbone (WavLM-Large), for criterion 1")
    ap.add_argument("--eer", nargs="+", type=Path, default=[],
                    help="cross_test.py-shaped per-checkpoint EER JSON file(s)")
    ap.add_argument("--phase-b-out-root", type=Path, default=DEFAULT_PHASE_B_OUT_ROOT,
                    help="root of scripts/extract_model_embeddings.py's output (default: %(default)s)")
    ap.add_argument("--head-replicates", type=Path, default=None,
                    help="JSON produced by scripts/head_replicates.py, for criterion 8")
    ap.add_argument("--factor-corpus-map", nargs="+", default=None,
                    help="factor=corpus pairs mapping a battery's factor to an EER-scored corpus "
                         "that shares it, for criteria 2 and 7 (e.g. language=inthewild)")
    ap.add_argument("--out", type=Path, default=DEFAULT_VERDICT_OUT, help="verdict JSON output path")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    verdict = run_gate(
        phase_a_paths=args.phase_a,
        secondary_backbone_paths=args.phase_a_secondary_backbone,
        eer_paths=args.eer,
        phase_b_out_root=args.phase_b_out_root,
        head_replicates_path=args.head_replicates,
        factor_corpus_map=_parse_factor_corpus_map(args.factor_corpus_map),
    )
    _write_json_atomic(args.out, verdict)
    _log(f"gate: wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
