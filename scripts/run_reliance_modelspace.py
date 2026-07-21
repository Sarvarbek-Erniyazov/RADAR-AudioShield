"""Roadmap v3 Step 3: model-space CAUSAL-RELIANCE consumer (Phase B).

Sibling to scripts/run_reliance_battery.py, not an extension of it (decision
made in step3_modelspace_reliance_brief.md, not re-litigated here). The two
consumers measure two DIFFERENT things, in two DIFFERENT spaces, and the
architecture says so:

  - run_reliance_battery.py (cache-space, UNCHANGED, byte-identical): factor
    ENCODING/decodability in the shared, frozen SSL representation -- the
    diagnostic leg (FSS 0.89-0.996 on the four confirmed batteries).
  - This script (model-space): factor CAUSAL RELIANCE in each checkpoint's
    OWN 256-d pre-classifier decision space -- fit the factor subspace in
    that space, ABLATE it, and measure the change in the detector's own
    output, benchmarked against equal-norm random-subspace controls and a
    task-direction positive control. This is the leg that was
    `not_estimable` in every cache-space battery (256-d classifier weight
    vs 1024-d raw pre-pooling cache -- no linear pullback across the GELU;
    see docs/phaseB_extraction_preflight_findings.md).

A factor can be strongly encoded (decodable) in the shared backbone
representation yet, once its subspace is ablated in the model's OWN
decision space, leave the detector's decisions unchanged -- decodability
without reliance. Burying the causal path as a --model-space flag inside
the diagnostic battery would blur exactly the distinction the measurement
exists to establish.

SHARED, NOT REIMPLEMENTED: every subspace/crossfit/metric/bootstrap
primitive below is imported verbatim from src/audioshield/reliance/ and
from run_reliance_battery.py itself (read-only import -- that file is
never modified, and this script proves it stays byte-identical). The ONLY
new code is the ORCHESTRATION: run_reliance_battery.py's run_battery loads
ONE embedding matrix per corpus and reuses it for every checkpoint (correct
there because the frozen XLS-R-300M backbone is checkpoint-invariant); this
script loads a SEPARATE 256-d Z per checkpoint (from that checkpoint's own
Phase B cache) and runs the ENTIRE nested crossfit -- subspace fit AND the
causal intervention -- independently per checkpoint, in that checkpoint's
own space, then merges the results into one schema-identical battery record
so scripts/run_gate.py reads it completely unchanged. Extends the intent of
run_reliance_battery.py's own --layer-mode checkpoint-band /
_recompute_band_alignment precedent (refit a subspace in a checkpoint's own
space, reusing the shared rank choice) -- but goes further, since Phase B's
embedding is not a re-pooling of one shared multi-layer cache, it is a
wholly separate space per checkpoint, so EVERY metric (not just alignment)
needs its own per-checkpoint run.

Schema-identical output, verified against scripts/run_gate.py's readers by
reading run_gate.py directly (never trust this docstring's own claim about
another file -- step3_modelspace_preextraction_gate_brief.md Item 1 caught
an earlier version of this paragraph doing exactly that). The two readers
do NOT behave the same way, and that asymmetry matters:

  - _per_checkpoint_reliance correctly iterates
    fold["effect"]["per_checkpoint"][ckpt][metric] for EVERY checkpoint --
    this is what C2/C7 (association/no-collapse) use, and it is exactly
    right for the merged, multi-checkpoint per_checkpoint dict this module
    produces.
  - criterion_4_intervention_vs_random reads ONLY the FOLD-LEVEL
    fold["effect"]["projection_removal_control"] (run_gate.py, confirmed
    by direct read) -- it never iterates per_checkpoint[ckpt]
    ["projection_removal_control"] at all. Since merge_checkpoint_
    estimator_results sets that fold-level field to the designated PRIMARY
    checkpoint's own value (first in sorted order) for schema
    compatibility, C4's verdict in the model-space regime is the
    alphabetically-first checkpoint's causal-intervention result ALONE --
    checkpoints B and C's controls are silently discarded. This was
    harmless for the cache-space battery (the control was checkpoint-
    independent by construction there, so fold-level == every per-
    checkpoint value) and is a genuine scientific defect here, since C4 is
    supposed to speak to the causal criterion across model instances, not
    one arbitrarily-chosen instance. NOT patched here: C4 is a
    pre-registered criterion (docs/gate_prereg.md); the fix (make C4
    iterate per_checkpoint[ckpt]["projection_removal_control"] the same
    way _per_checkpoint_reliance already does) is gate-side only and is a
    human/prereg decision -- see tests/test_reliance_modelspace.py's
    test_c4_should_aggregate_across_all_checkpoints_once_fixed (xfail,
    documents the desired behavior; converts to a passing regression test
    the moment the fix lands).

A merged fold's per_checkpoint[ckpt] entry carries the standard
alignment/r_var/prediction_change/prediction_change_control keys
(identical shape to run_reliance_battery.py's own per-checkpoint dict)
PLUS, additively, that checkpoint's own chosen/selection_score/
factor_separation_score/leace/inlp/projection_removal_control (genuinely
checkpoint-specific here, unlike the original schema where these are
checkpoint-independent by construction) -- this additive data is exactly
what a corrected C4 would read.

No GPU, no extraction, no backbone re-run: only binary.fc (weight 256x1 +
bias, a cheap CPU matmul) is exercised, against ALREADY-CACHED embeddings.

CHECKPOINT-NAMING COLLISION RISK (found while reading, not asked for):
scripts/extract_model_embeddings.py's output directory is
`<out-root>/<ckpt_path.stem>/<corpus_dir>/`. If extraction is invoked with
a NESTED checkpoint path like `runs/e007_A_fresh/best.pt` (the preflight
brief's own example), `ckpt_path.stem` is "best" for every checkpoint --
a directory collision. This script assumes the FLAT naming convention
run_reliance_battery.py's own load_all_checkpoints already uses
(`<ckpt-dir>/runs_<run>_best.pt`, unique stem per run) for locating BOTH
the checkpoint file and its model-space cache -- extraction must be
invoked with that same flat convention for a given checkpoint to be found
correctly by this script.

Usage (once a real Phase B model-space cache exists):
    python scripts/run_reliance_modelspace.py \\
        --model-space-cache-root analysis/step3/_embcache_modelspace \\
        --ckpt-dir /e/AI_voice_detection/checkpoint_backup \\
        --checkpoints e007_A_fresh e007_B_fresh e007_C_xlsr_fresh \\
        --manifest-dir manifests/v2 \\
        --corpus diffssd replaydf \\
        --out analysis/step3/reliance_modelspace.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audioshield.data.manifest import read_manifest
from audioshield.reliance.crossfit import run_nested_crossfit
from audioshield.reliance.metrics import r_var
from audioshield.reliance.subspaces import lda_subspace
from audioshield.reliance.uncertainty import grouped_bootstrap_ci, rank_sensitivity_curve
from audioshield.utils.hashing import sha256_file

# Read-only imports -- run_reliance_battery.py is never modified by this
# script; every name below is reused verbatim so the two consumers cannot
# silently diverge on the math or the output shape. Deliberately NOT
# importing anything from extract_model_embeddings.py (step3_modelspace_
# hardening_addendum.md Finding 2): that module is the GPU/model stack
# (torch, AudioShieldX, UnifiedAudioDataset) and sets process-global
# offline-mode env vars at import time -- this CPU-only analysis script
# must not become contingent on that stack importing cleanly just to
# reuse a five-line hash helper, which now lives in
# audioshield.utils.hashing (stdlib-only) instead.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_reliance_battery import (  # noqa: E402
    BATTERIES,
    CANDIDATE_B_KEYS,
    CANDIDATE_W_KEYS,
    CKPT_DIR,
    CORPUS_DIR,
    DEFAULT_MAX_ROWS_PER_LEVEL,
    DEFAULT_RANKS,
    RUNS,
    SCHEMA_VERSION,
    _battery_output_path,
    _checkpoints_summary,
    _find_key,
    _git_sha,
    _log,
    _make_effect_fn,
    _make_fit_subspace,
    _write_json_atomic,
    cap_rows_per_level,
    factor_separation_score,
    join_cache_to_manifest,
    ranks_for_n_levels,
    select_battery_rows,
    summarize_prereg_candidate,
)

DEFAULT_MODEL_SPACE_CACHE_ROOT = "analysis/step3/_embcache_modelspace"


# ---------------------------------------------------------------------------
# Loaders -- NEW (Phase B's cache/checkpoint shape differs from the raw
# per-layer XLS-R-300M cache load_corpus_embeddings/load_task_direction
# handle), but built entirely from already-confirmed shapes (see
# docs/phaseB_extraction_preflight_findings.md) and reused primitives
# (_find_key/CANDIDATE_W_KEYS/CANDIDATE_B_KEYS) -- never a forked key search.
# ---------------------------------------------------------------------------


def load_model_space_embeddings(
    cache_root: Path, checkpoint_stem: str, corpus_dir: str,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Load every shard_*.npz under cache_root/<checkpoint_stem>/<corpus_dir>/
    -- Phase B's real, 2-D (n, embedding_dim) shape (extract_model_embeddings.py),
    NOT the 3-D (n, n_layers, D) shape load_corpus_embeddings expects (the
    exact confusion docs/phaseB_extraction_preflight_findings.md identifies
    as the consumption gap's root cause -- guarded against explicitly here,
    not just assumed away).

    Also returns the shard(s)' recorded `checkpoint_sha256` (None if a
    shard's meta doesn't parse or the key is absent) -- the caller
    verifies this against the ACTUAL checkpoint file's own sha256 before
    trusting the (embedding, head) pairing (see main()'s pairing guard).
    Every shard under this directory must agree on checkpoint_sha256 --
    they are all supposed to be the same checkpoint's output; a
    disagreement here means shards from two different extraction runs
    were mixed into one directory, caught explicitly rather than silently
    averaged together."""
    shard_dir = cache_root / checkpoint_stem / corpus_dir
    shard_paths = sorted(shard_dir.glob("shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"no shard_*.npz files found under {shard_dir}")
    all_paths: list[np.ndarray] = []
    all_emb: list[np.ndarray] = []
    checkpoint_sha256: str | None = None
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=True) as data:
            emb = data["emb"]
            if emb.ndim != 2:
                raise ValueError(
                    f"{shard_path}: expected a 2-D model-space embedding (n, dim), got {emb.ndim}-D "
                    f"shape {emb.shape} -- this looks like the raw per-layer XLS-R-300M cache "
                    "(load_corpus_embeddings's shape), not Phase B's model-space cache"
                )
            all_paths.append(data["paths"])
            all_emb.append(emb.astype(np.float32))
            try:
                shard_sha = json.loads(str(data["meta"])).get("checkpoint_sha256")
            except Exception:
                shard_sha = None
        if checkpoint_sha256 is None:
            checkpoint_sha256 = shard_sha
        elif shard_sha is not None and shard_sha != checkpoint_sha256:
            raise ValueError(
                f"{shard_path}: checkpoint_sha256={shard_sha!r} disagrees with an earlier shard's "
                f"{checkpoint_sha256!r} under the same {shard_dir} -- shards from two different "
                "extraction runs appear to be mixed into one cache directory"
            )
    return np.concatenate(all_paths), np.concatenate(all_emb, axis=0), checkpoint_sha256


def load_checkpoint_head(ckpt_path: Path) -> tuple[np.ndarray, float, int, str]:
    """Load a checkpoint's final linear classifier weight and bias --
    binary.fc.weight / binary.fc.bias (confirmed exact attribute path,
    docs/phaseB_extraction_preflight_findings.md §1), via the SAME
    candidate-key search run_reliance_battery.py's own load_task_direction
    uses (CANDIDATE_W_KEYS/CANDIDATE_B_KEYS, imported not duplicated) --
    but without that function's layer-center-mismatch logic, which does
    not apply here: embed()'s pooled+projected output has no "layer" at
    all, so there is nothing to compare a requested layer against.

    Also returns the checkpoint FILE's own sha256 (via
    audioshield.utils.hashing.sha256_file -- a stdlib-only shared module,
    NOT extract_model_embeddings.py, so this CPU-only script never
    transitively imports that module's GPU/model stack just to reuse a
    hash helper; see that module's own docstring), the SAME hash function
    that computed the value stored in every shard's meta.checkpoint_sha256
    -- this is the other half of the pairing guarantee: the caller
    compares this against load_model_space_embeddings' returned
    checkpoint_sha256 before trusting that this w and that Z came from
    the same trained model."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = sd.get("model", sd) if isinstance(sd, dict) else sd
    w_key = _find_key(state, CANDIDATE_W_KEYS)
    if w_key is None:
        raise RuntimeError(f"{ckpt_path}: no classifier weight found (tried {CANDIDATE_W_KEYS}) -- refusing to guess")
    w = state[w_key].squeeze().float().cpu().numpy()
    b_key = _find_key(state, CANDIDATE_B_KEYS)
    b = float(state[b_key].squeeze().float().cpu().item()) if b_key is not None else 0.0
    return w, b, int(w.shape[0]), sha256_file(ckpt_path)


# ---------------------------------------------------------------------------
# Per-checkpoint nested crossfit -- the entire subspace fit AND causal
# intervention, run independently in THIS checkpoint's own Z. Reuses
# run_nested_crossfit/_make_fit_subspace/_make_effect_fn verbatim: passing
# a SINGLE-entry checkpoints dict means _make_effect_fn's existing
# per-checkpoint loop naturally produces a single-key per_checkpoint dict,
# with NO changes to that function's own code.
# ---------------------------------------------------------------------------


def run_checkpoint_crossfit(
    run_name: str, ck: dict, Z: np.ndarray, factor: np.ndarray, y: np.ndarray, groups: np.ndarray,
    valid_ranks: list[int], n_outer: int, seed: int,
) -> dict[str, dict]:
    candidates = [{"k": r} for r in valid_ranks]
    single_ckpt = {run_name: ck}
    out: dict[str, dict] = {}
    for estimator in ("lda", "probe"):
        fit_subspace = _make_fit_subspace(estimator, seed)
        effect_fn = _make_effect_fn(single_ckpt, seed, n_random=20, layer_mode="fixed",
                                     w_metrics_enabled=True, w_metrics_reason="")
        fold_results = run_nested_crossfit(
            Z, factor, y, groups, candidates, fit_subspace, factor_separation_score, effect_fn,
            n_outer_splits=n_outer, n_inner_splits=min(3, n_outer), seed=seed,
        )
        out[estimator] = dict(fold_results=fold_results)
    return out


def merge_checkpoint_estimator_results(
    per_checkpoint_estimators: dict[str, dict[str, dict]], primary_run: str,
) -> dict[str, dict]:
    """Merges each checkpoint's own, independently-run fold_results into
    ONE schema-identical estimators dict. Folds align across checkpoints
    by construction: (n, groups, n_outer_splits, seed) are identical for
    every checkpoint (same manifest join), so make_nested_folds (called
    inside each checkpoint's own run_nested_crossfit) produces the exact
    same fold assignment every time -- verified explicitly below (raises
    if fold_id sequences ever disagree, rather than silently mismatching
    checkpoint A's fold 3 with checkpoint B's fold 4).

    Per fold: `per_checkpoint[ckpt]` carries that checkpoint's own
    alignment/r_var/prediction_change/prediction_change_control (already
    produced by _make_effect_fn, unchanged) PLUS, additively, that
    checkpoint's own chosen rank/selection_score/factor_separation_score/
    leace/inlp/projection_removal_control -- all genuinely checkpoint-
    specific here. The fold-level (non-per-checkpoint) copies of those
    same fields take `primary_run`'s value, for byte-for-byte compatible
    shape with run_reliance_battery.py's own output (nothing there reads
    a checkpoint key for these fields, and never has)."""
    runs = sorted(per_checkpoint_estimators)
    merged: dict[str, dict] = {}
    for estimator in ("lda", "probe"):
        per_run_folds = {run: per_checkpoint_estimators[run][estimator]["fold_results"] for run in runs}
        n_folds = len(per_run_folds[runs[0]])
        merged_folds = []
        for i in range(n_folds):
            fold_id = per_run_folds[runs[0]][i]["fold_id"]
            for run in runs:
                fr = per_run_folds[run][i]
                if fr["fold_id"] != fold_id:
                    raise AssertionError(
                        f"estimator={estimator} fold index {i}: fold_id mismatch across checkpoints "
                        f"({run} has {fr['fold_id']}, expected {fold_id}) -- checkpoints' embeddings "
                        "are not row-aligned to the same manifest join"
                    )
            merged_per_checkpoint = {}
            for run in runs:
                fr = per_run_folds[run][i]
                ck_entry = dict(fr["effect"]["per_checkpoint"][run])
                ck_entry["chosen"] = fr["chosen"]
                ck_entry["selection_score"] = fr["selection_score"]
                ck_entry["factor_separation_score"] = fr["effect"]["factor_separation_score"]
                ck_entry["leace"] = fr["effect"]["leace"]
                ck_entry["inlp"] = fr["effect"]["inlp"]
                ck_entry["projection_removal_control"] = fr["effect"]["projection_removal_control"]
                merged_per_checkpoint[run] = ck_entry
            primary = per_run_folds[primary_run][i]
            merged_folds.append(dict(
                fold_id=fold_id, chosen=primary["chosen"], selection_score=primary["selection_score"],
                n_selection=primary["n_selection"], n_effect=primary["n_effect"],
                effect=dict(
                    per_checkpoint=merged_per_checkpoint,
                    factor_separation_score=primary["effect"]["factor_separation_score"],
                    leace=primary["effect"]["leace"], inlp=primary["effect"]["inlp"],
                    projection_removal_control=primary["effect"]["projection_removal_control"],
                ),
            ))
        merged[estimator] = dict(fold_results=merged_folds, status="ok", timed_out=False)
    return merged


# ---------------------------------------------------------------------------
# Headline bootstrap / rank-sensitivity: the ORIGINAL script's own
# _bootstrap_worker_task/_rank_curve_worker_task fit a subspace on ONE
# shared Z and average r_var(w, U, Sigma) across checkpoints -- correct
# there since Z is shared. Here each checkpoint needs its OWN Z/U/Sigma;
# these two closures are the "thin per-checkpoint entry point" the brief
# asks for -- the MATH (lda_subspace, r_var) is the unmodified import,
# only which Z feeds which checkpoint's r_var is new.
# ---------------------------------------------------------------------------


def _make_headline_metric_fn(Z_by_checkpoint, factor, y, headline_rank, checkpoints):
    def _headline_metric(row_idx):
        fs, ys = factor[row_idx], y[row_idx]
        vals = []
        for run, ck in checkpoints.items():
            Zs = Z_by_checkpoint[run][row_idx]
            try:
                U = lda_subspace(Zs, fs, ys, k=headline_rank, mode="within_class")
            except ValueError:
                continue
            if U.shape[1] == 0:
                continue
            Sigma = np.atleast_2d(np.cov(Zs, rowvar=False)) if len(Zs) > 1 else np.eye(Zs.shape[1])
            vals.append(r_var(ck["w"], U, Sigma))
        return float(np.mean(vals)) if vals else float("nan")
    return _headline_metric


def _make_rank_curve_metric_fn(Z_by_checkpoint, factor, y, checkpoints):
    def _metric_at_rank(k):
        vals = []
        for run, ck in checkpoints.items():
            Zs = Z_by_checkpoint[run]
            U = lda_subspace(Zs, factor, y, k=k, mode="within_class")
            if U.shape[1] == 0:
                continue
            Sigma = np.atleast_2d(np.cov(Zs, rowvar=False))
            vals.append(r_var(ck["w"], U, Sigma))
        return float(np.mean(vals)) if vals else float("nan")
    return _metric_at_rank


# ---------------------------------------------------------------------------
# Per-battery orchestration -- mirrors run_reliance_battery.py's run_battery
# shape exactly (same result dict keys), with the per-checkpoint loop as
# the one real difference.
# ---------------------------------------------------------------------------


def run_modelspace_battery(
    spec: dict,
    Z_by_checkpoint: dict[str, np.ndarray],
    factor: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    checkpoints: dict[str, dict],
    ranks: list[int],
    n_boot: int,
    seed: int,
    max_rows_per_level: int | None = DEFAULT_MAX_ROWS_PER_LEVEL,
    log=None,
) -> dict:
    log = log or (lambda msg: _log(f"[{spec['name']}] {msg}"))
    runs = sorted(checkpoints)
    primary_run = runs[0]

    if max_rows_per_level is not None:
        n_before = len(y)
        capped_Z = {}
        factor_ref = y_ref = groups_ref = None
        for run in runs:
            Zc, fc, yc, gc = cap_rows_per_level(Z_by_checkpoint[run], factor, y, groups, max_rows_per_level, seed)
            capped_Z[run] = Zc
            if factor_ref is None:
                factor_ref, y_ref, groups_ref = fc, yc, gc
            elif not (np.array_equal(fc, factor_ref) and np.array_equal(yc, y_ref) and np.array_equal(gc, groups_ref)):
                raise AssertionError(
                    f"{spec['name']}: cap_rows_per_level selected different rows for checkpoint {run!r} "
                    f"than {primary_run!r} -- checkpoints are not row-aligned to the same manifest join"
                )
        Z_by_checkpoint, factor, y, groups = capped_Z, factor_ref, y_ref, groups_ref
        if len(y) != n_before:
            log(f"row cap: {n_before} -> {len(y)} rows (max_rows_per_level={max_rows_per_level})")

    n_levels = int(len(np.unique(factor)))
    valid_ranks = ranks_for_n_levels(list(ranks), n_levels)
    n_groups = int(len(np.unique(groups)))
    grouping_degenerate = spec["factor"] == spec["grouping"]

    result: dict = dict(
        name=spec["name"], corpus=spec["corpus"], factor=spec["factor"], grouping=spec["grouping"],
        n_rows=int(len(y)), n_levels=n_levels, n_groups=n_groups,
        grouping_degenerate=grouping_degenerate,
        ranks_requested=list(ranks), ranks_valid=valid_ranks,
        layer_mode="model_space",
    )
    if not valid_ranks:
        result["skipped"] = f"no requested rank <= n_levels-1={n_levels - 1}"
        return result
    n_outer = min(5, n_groups)
    if n_outer < 2:
        result["skipped"] = f"only {n_groups} group(s) available -- cannot form nested folds"
        return result

    log(f"matrix assembled: n_levels={n_levels} n_groups={n_groups} n_outer={n_outer} "
        f"valid_ranks={valid_ranks} checkpoints={runs}")

    per_checkpoint_estimators = {}
    for run in runs:
        log(f"running nested crossfit (both estimators) for checkpoint={run}")
        per_checkpoint_estimators[run] = run_checkpoint_crossfit(
            run, checkpoints[run], Z_by_checkpoint[run], factor, y, groups, valid_ranks, n_outer, seed,
        )
    result["estimators"] = merge_checkpoint_estimator_results(per_checkpoint_estimators, primary_run)

    headline_rank = valid_ranks[0]
    headline_metric_name = "r_var"  # w-metrics are always enabled here: w and Z are commensurate by construction
    if n_boot > 0:
        headline_fn = _make_headline_metric_fn(Z_by_checkpoint, factor, y, headline_rank, checkpoints)
        bootstrap = grouped_bootstrap_ci(headline_fn, groups, n_boot=n_boot, seed=seed)
        bootstrap.setdefault("status", "ok")
        bootstrap.setdefault("timed_out", False)
    else:
        bootstrap = dict(mean=float("nan"), std=float("nan"), lo=float("nan"), hi=float("nan"),
                          n_boot=0, n_groups=n_groups, n_finite=0, n_boot_failed=0,
                          status="skipped", timed_out=False, note="--n-boot 0: point-estimate-only pass")
    rank_curve = rank_sensitivity_curve(_make_rank_curve_metric_fn(Z_by_checkpoint, factor, y, checkpoints), valid_ranks)
    rank_curve.setdefault("status", "ok")
    rank_curve.setdefault("timed_out", False)

    result["headline_bootstrap"] = dict(metric=headline_metric_name, rank=headline_rank, **bootstrap)
    result["rank_sensitivity"] = dict(metric=headline_metric_name, **rank_curve)
    log(f"battery complete: checkpoints={runs}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-space-cache-root", default=DEFAULT_MODEL_SPACE_CACHE_ROOT)
    ap.add_argument("--manifest-dir", default="manifests/v2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--corpus", nargs="*", default=None, help="restrict to battery corpora in this list")
    ap.add_argument("--factor", nargs="*", default=None, help="restrict to battery factors in this list")
    ap.add_argument("--ranks", nargs="+", type=int, default=list(DEFAULT_RANKS))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    ap.add_argument("--checkpoints", nargs="+", default=list(RUNS))
    ap.add_argument("--max-rows-per-level", type=int, default=DEFAULT_MAX_ROWS_PER_LEVEL)
    ap.add_argument("--require-all-checkpoints", action="store_true",
                     help="raise instead of warn-and-skip when a requested checkpoint's .pt file or "
                         "model-space cache is missing (step3_modelspace_hardening_addendum.md Finding 1: "
                         "a naming mismatch between this consumer and scripts/extract_model_embeddings.py "
                         "otherwise silently shrinks a battery's checkpoint count -- a real gate-feeding "
                         "run must use this flag so 'cache not found' can never masquerade as a completed "
                         "battery with fewer instances). Default off, so synthetic/preflight tests keep "
                         "their lenient skip-and-continue behavior.")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    out_path = Path(args.out) if args.out else Path("analysis/step3/reliance_modelspace.json")
    max_rows_per_level = args.max_rows_per_level if args.max_rows_per_level and args.max_rows_per_level > 0 else None
    cache_root = Path(args.model_space_cache_root)
    ckpt_dir = Path(args.ckpt_dir)
    manifest_dir = Path(args.manifest_dir)

    _log(f"[run_reliance_modelspace] checkpoints={args.checkpoints} cache_root={cache_root}")

    checkpoint_heads: dict[str, dict] = {}
    for run in args.checkpoints:
        ckpt_path = ckpt_dir / f"runs_{run}_best.pt"
        if not ckpt_path.exists():
            if args.require_all_checkpoints:
                raise RuntimeError(
                    f"--require-all-checkpoints is set but checkpoint file not found for run {run!r}: "
                    f"{ckpt_path}"
                )
            _log(f"[WARN] {ckpt_path}: not found -- skipping checkpoint {run!r}")
            continue
        w, b, w_dim, ckpt_sha256 = load_checkpoint_head(ckpt_path)
        checkpoint_heads[run] = dict(w=w, b=b, w_dim=w_dim, checkpoint_stem=ckpt_path.stem,
                                      checkpoint_sha256=ckpt_sha256)
    if not checkpoint_heads:
        raise ValueError("no checkpoints loaded -- nothing to do")

    batteries = [
        b for b in BATTERIES
        if (args.corpus is None or b["corpus"] in args.corpus)
        and (args.factor is None or b["factor"] in args.factor)
    ]
    if not batteries:
        raise ValueError(f"--corpus/--factor filters matched 0 of {len(BATTERIES)} batteries")

    git_sha = _git_sha()
    battery_results = []
    for spec in batteries:
        corpus = spec["corpus"]
        corpus_dir = CORPUS_DIR[corpus]
        manifest_rows = read_manifest(manifest_dir / f"{corpus}.csv")
        manifest_df = pd.DataFrame([asdict(r) for r in manifest_rows])

        Z_by_checkpoint, checkpoints, join_stats = {}, {}, {}
        factor_ref = y_ref = groups_ref = utt_ids_ref = None
        for run, head in checkpoint_heads.items():
            try:
                cache_paths, cache_emb, cache_sha256 = load_model_space_embeddings(
                    cache_root, head["checkpoint_stem"], corpus_dir
                )
            except FileNotFoundError as e:
                if args.require_all_checkpoints:
                    expected_dir = cache_root / head["checkpoint_stem"] / corpus_dir
                    raise RuntimeError(
                        f"{spec['name']}/{run}: --require-all-checkpoints is set but no model-space "
                        f"cache found at {expected_dir} ({e}). Run:\n"
                        f"    python scripts/extract_model_embeddings.py "
                        f"--checkpoint {ckpt_dir / f'runs_{run}_best.pt'} --corpus {corpus} "
                        f"--data-root .. --out-root {cache_root}"
                    ) from e
                _log(f"[WARN] {spec['name']}: {e} -- skipping checkpoint {run!r} for this battery")
                continue
            # THE PAIRING GUARANTEE (Item 3b): dimension-matching (both sides
            # 256-d) cannot catch a mispaired (embedding, head) -- checkpoint
            # A's head could be paired with checkpoint B's cache and both
            # would still be 256-d. This is the cryptographic guard that
            # catches it regardless of naming: the shard's OWN recorded
            # checkpoint_sha256 (written by extract_model_embeddings.py from
            # the checkpoint FILE it actually extracted, not a placeholder)
            # must equal THIS run's checkpoint file's own sha256, computed
            # independently here via the same sha256_file function.
            if cache_sha256 is None:
                raise RuntimeError(
                    f"{run}/{corpus}: the model-space cache under {cache_root / head['checkpoint_stem'] / corpus_dir} "
                    "has no recorded checkpoint_sha256 (missing/unparseable meta) -- refusing to pair an "
                    "unverified embedding cache with this checkpoint's head"
                )
            if cache_sha256 != head["checkpoint_sha256"]:
                raise RuntimeError(
                    f"{run}/{corpus}: MISPAIRED (embedding, head) -- the model-space cache under "
                    f"{cache_root / head['checkpoint_stem'] / corpus_dir} was extracted from a checkpoint with "
                    f"sha256={cache_sha256[:16]}..., but the head loaded for run {run!r} "
                    f"({ckpt_dir / f'runs_{run}_best.pt'}) has sha256={head['checkpoint_sha256'][:16]}... -- "
                    "refusing to ablate a factor subspace from an embedding paired with a different "
                    "checkpoint's classifier weight (this would produce a finite, plausible, and "
                    "scientifically meaningless number)"
                )
            joined_df, joined_emb, stats = join_cache_to_manifest(cache_paths, cache_emb, manifest_df, corpus_dir)
            join_stats[run] = stats
            if head["w_dim"] != joined_emb.shape[1]:
                raise ValueError(
                    f"{run}: w is {head['w_dim']}-d but the model-space cache for corpus {corpus!r} is "
                    f"{joined_emb.shape[1]}-d -- refusing to treat these as commensurate"
                )
            Z, factor, y, groups, _ = select_battery_rows(joined_df, joined_emb, spec)
            utt_ids = joined_df.loc[:, "utt_id"].to_numpy() if "utt_id" in joined_df.columns else None
            if factor_ref is None:
                factor_ref, y_ref, groups_ref, utt_ids_ref = factor, y, groups, utt_ids
            elif not (np.array_equal(factor, factor_ref) and np.array_equal(y, y_ref)
                      and np.array_equal(groups, groups_ref)):
                raise AssertionError(
                    f"{spec['name']}: checkpoint {run!r}'s joined rows disagree with an earlier "
                    "checkpoint's (factor/y/groups mismatch) -- Phase B extraction likely covered a "
                    "different row subset for this checkpoint; refusing to silently misalign embeddings"
                )
            Z_by_checkpoint[run] = Z
            checkpoints[run] = dict(
                w=head["w"], b=head["b"], w_dim=head["w_dim"], w_dim_mismatch=False,
                ckpt_layer_center=None, ckpt_layer_band=None, layer_pooling="model_space",
                band_weights=None, w_layer_mismatch=False,
            )

        if len(checkpoints) < 2:
            _log(f"[WARN] {spec['name']}: fewer than 2 checkpoints have a usable model-space cache -- skipping")
            continue

        res = run_modelspace_battery(
            spec, Z_by_checkpoint, factor_ref, y_ref, groups_ref, checkpoints,
            args.ranks, args.n_boot, args.seed, max_rows_per_level,
        )
        battery_results.append(res)

        embedding_dim = next(iter(checkpoints.values()))["w_dim"]
        w_metrics = dict(enabled=True, reason="model-space cache: w and embedding dimensionality match by construction",
                          w_dim=embedding_dim, embedding_dim=embedding_dim)
        battery_path = _battery_output_path(out_path, spec["name"])
        _write_json_atomic(battery_path, dict(
            schema_version=SCHEMA_VERSION, git_sha=git_sha, timestamp=datetime.now(timezone.utc).isoformat(),
            layer=None, layer_mode="model_space", seed=args.seed, w_metrics=w_metrics,
            join_stats={corpus: join_stats}, checkpoints=_checkpoints_summary(checkpoints),
            battery=res, prereg_candidate=summarize_prereg_candidate(res),
        ))
        _log(f"[battery] {spec['name']}: wrote -> {battery_path}")

    prereg = [summarize_prereg_candidate(r) for r in battery_results]
    manifest = dict(
        schema_version=SCHEMA_VERSION, git_sha=git_sha, timestamp=datetime.now(timezone.utc).isoformat(),
        layer=None, layer_mode="model_space", seed=args.seed,
        w_metrics=dict(enabled=True, reason="model-space cache: w and embedding dimensionality match by construction",
                        w_dim=None, embedding_dim=None),
        join_stats={}, checkpoints={},
        battery_files={r["name"]: str(_battery_output_path(out_path, r["name"])) for r in battery_results if "name" in r},
        batteries=battery_results, prereg_candidates=prereg,
    )
    _write_json_atomic(out_path, manifest)
    _log(f"[done] wrote manifest -> {out_path}")


if __name__ == "__main__":
    main()
