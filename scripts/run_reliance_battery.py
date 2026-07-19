"""Roadmap v3 Step 3, Phase B: measure factor reliance in cached XLS-R-300M
embeddings using the merged src/audioshield/reliance/ library.

Read-only over the embedding cache -- NO GPU, NO training, NO gradient
computation. Loads three checkpoints only to extract their frozen final
linear-classifier weight (the "task direction" w) for alignment/removal
metrics; the backbones themselves are never touched.

Usage:
    python scripts/run_reliance_battery.py --layer 9

Do NOT run against the real embedding cache from this repo checkout -- it
lives on the collaborator machine (see --cache-root's default). This script
is exercised here only via tests/test_reliance_battery.py's synthetic
fixtures.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from audioshield.data.manifest import read_manifest
from audioshield.reliance.crossfit import run_nested_crossfit
from audioshield.reliance.metrics import (
    alignment,
    fit_inlp,
    fit_leace,
    prediction_change,
    project_out,
    r_var,
    r_var_class_conditional,
    removal_control_report,
)
from audioshield.reliance.subspaces import crossfitted_probe_subspace, lda_subspace
from audioshield.reliance.uncertainty import grouped_bootstrap_ci, rank_sensitivity_curve

SCHEMA_VERSION = 1

# collaborator machine paths; adjust locally (same convention as scripts/reproduce_eval.py's CKPT_DIR)
DEFAULT_CACHE_ROOT = "/e/AI_voice_detection/datasets/_embcache_xlsr300m"
CKPT_DIR = Path("/e/AI_voice_detection/checkpoint_backup")
RUNS = ("e007_A_fresh", "e007_B_fresh", "e007_C_xlsr_fresh")

# The final binary spoof-classifier weight's key in the checkpoint's model state
# dict. "binary.fc.weight" is the confirmed key (scripts/compute_drs.py precedent);
# the rest are defensive fallbacks in case naming drifted -- if NONE match, loading
# fails loudly rather than silently returning a wrong vector.
CANDIDATE_W_KEYS = ("binary.fc.weight", "spoof_head.weight", "head.fc.weight", "classifier.weight")
CANDIDATE_B_KEYS = ("binary.fc.bias", "spoof_head.bias", "head.fc.bias", "classifier.bias")

# corpus id -> raw dataset-root folder name. Used BOTH to strip the manifest
# `path` prefix down to the cache's bare relative path, and as the embedding
# cache's per-corpus subdirectory name (same convention observed throughout the
# project's _embcache_xlsr300m layout, e.g. "03_DiffSSD", "04_ReplayDF").
CORPUS_DIR = {
    "diffssd": "03_DiffSSD",
    "replaydf": "04_ReplayDF",
    "vctk": "09_VCTK",
}

DEFAULT_RANKS = (1, 2, 3, 5, 8, 12, 16, 24)

# (corpus, factor, grouping) batteries -- exactly these, nothing pooled across
# corpora. `row_filter`, when present, restricts the battery to rows matching
# (column, value) before anything else (used for diffssd's openvoicev2-only
# accent battery).
BATTERIES = [
    dict(name="diffssd_generator_by_source", corpus="diffssd", factor="generator_id", grouping="source_id"),
    dict(name="replaydf_generator_by_channel", corpus="replaydf", factor="generator_id", grouping="channel_id"),
    dict(name="replaydf_language_by_channel", corpus="replaydf", factor="language", grouping="channel_id"),
    dict(name="replaydf_channel_by_channel", corpus="replaydf", factor="channel_id", grouping="channel_id"),
    dict(name="diffssd_openvoicev2_accent_by_speaker", corpus="diffssd", factor="language", grouping="speaker_id",
         row_filter=("generator_id", "openvoicev2")),
    dict(name="vctk_speaker_by_speaker", corpus="vctk", factor="speaker_id", grouping="speaker_id"),
]


# ---------------------------------------------------------------------------
# Data loading / joining
# ---------------------------------------------------------------------------


def strip_cache_prefix(manifest_path: str, corpus_dir: str) -> str | None:
    """Manifest `path` is "datasets/<CORPUS_DIR>/<rel>"; cache `paths` is "<rel>".
    Returns None (never mis-joins) if `manifest_path` lacks the expected prefix."""
    prefix = f"datasets/{corpus_dir}/"
    if not manifest_path.startswith(prefix):
        return None
    return manifest_path[len(prefix):]


def load_corpus_embeddings(cache_root: Path, corpus_dir: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Load every shard_*.npz under cache_root/<corpus_dir>/, extract
    emb[:, layer, :] (float16 -> float32) and paths, concatenated across shards.

    Fails loudly if `layer` is out of range for any individual shard's emb array
    (shards are per-corpus, per-extraction-run artifacts; a stale/mismatched
    shard should stop the run, not silently be skipped or mis-indexed).
    """
    shard_dir = cache_root / corpus_dir
    shard_paths = sorted(shard_dir.glob("shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"no shard_*.npz files found under {shard_dir}")

    all_paths: list[np.ndarray] = []
    all_emb: list[np.ndarray] = []
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as data:
            emb = data["emb"]
            paths = data["paths"]
        n_layers = emb.shape[1]
        if not (0 <= layer < n_layers):
            raise ValueError(
                f"{shard_path}: layer {layer} out of range for emb with {n_layers} layers "
                f"(valid range [0, {n_layers - 1}])"
            )
        all_paths.append(paths)
        all_emb.append(emb[:, layer, :].astype(np.float32))
    return np.concatenate(all_paths), np.concatenate(all_emb, axis=0)


def join_cache_to_manifest(
    cache_paths: np.ndarray,
    cache_emb: np.ndarray,
    manifest_df: pd.DataFrame,
    corpus_dir: str,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Inner-join cached embeddings to manifest rows on the stripped relative
    path. Cache rows absent from the manifest are dropped (cache-extras
    protocol, cf. mix sweep v2 log: asvspoof5 1184, ai4t 277, replaydf 480).
    Manifest rows with no matching cache embedding are also dropped (nothing
    to attach). Asserts n_joined > 0.

    Returns (joined_manifest_df, joined_emb (n_joined, d), stats).
    """
    manifest_df = manifest_df.copy()
    manifest_df["_rel"] = manifest_df["path"].apply(lambda p: strip_cache_prefix(p, corpus_dir))
    n_manifest = len(manifest_df)

    cache_df = pd.DataFrame({"_rel": pd.Series(cache_paths, dtype=object),
                              "_cache_idx": np.arange(len(cache_paths))})
    n_cache = len(cache_df)

    merged = manifest_df.merge(cache_df, on="_rel", how="inner")
    n_joined = len(merged)
    n_dropped = n_cache - n_joined  # cache-extras: present in cache, not in current manifest

    joined_emb = cache_emb[merged["_cache_idx"].to_numpy()]
    joined_df = merged.drop(columns=["_rel", "_cache_idx"]).reset_index(drop=True)

    stats = dict(n_cache=int(n_cache), n_manifest=int(n_manifest), n_joined=int(n_joined), n_dropped=int(n_dropped))
    assert n_joined > 0, f"0 rows joined for corpus_dir={corpus_dir!r} -- cache/manifest path prefixes don't match"
    return joined_df, joined_emb, stats


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def ranks_for_n_levels(ranks: list[int], n_levels: int) -> list[int]:
    """Cap candidate ranks at n_levels-1: a categorical factor's discriminant
    subspace has intrinsic rank <= n_levels-1 (between-group scatter matrix rank)."""
    return [r for r in ranks if r <= n_levels - 1]


def groups_from_column(values: np.ndarray) -> np.ndarray:
    """Grouping-column values, with 'NA'/empty entries replaced by unique
    per-row tokens so they never falsely co-cluster -- the same fallback
    discipline evaluation.grouped_probe._derive_groups uses for a column it
    has chosen; here the column is explicitly given per battery, so there is
    no ">50% non-NA" selection gate to pass, only the per-entry fallback."""
    out = np.empty(len(values), dtype=object)
    for i, v in enumerate(values):
        out[i] = v if v not in ("NA", "", None) else f"__ungrouped_{i}"
    return out


def select_battery_rows(df: pd.DataFrame, emb: np.ndarray, spec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply a battery's optional row_filter and its mandatory factor-NA
    exclusion, then slice out (Z, factor, y, groups) for that battery.

    Rows whose factor value is "NA" are excluded from the battery (they carry
    no information about the factor being measured); row_filter (e.g.
    diffssd's openvoicev2-only accent battery) is applied first.
    """
    mask = np.ones(len(df), dtype=bool)
    if "row_filter" in spec:
        col, val = spec["row_filter"]
        mask &= (df[col] == val).to_numpy()
    mask &= (df[spec["factor"]] != "NA").to_numpy()

    Z = emb[mask]
    factor = df.loc[mask, spec["factor"]].to_numpy()
    y = df.loc[mask, "target"].to_numpy().astype(int)
    groups = groups_from_column(df.loc[mask, spec["grouping"]].to_numpy())
    return Z, factor, y, groups


def factor_separation_score(Z_val: np.ndarray, factor_val: np.ndarray, y_val: np.ndarray, U: np.ndarray) -> float:
    """Subspace-selection score for crossfit.select_hyperparameter: the
    between-factor-group / total variance ratio of Z_val projected onto U
    (higher = the candidate rank/estimator separates the factor's levels
    better on held-out validation data). Coordinate-free, no extra classifier
    fit needed."""
    if U.shape[1] == 0:
        return float("-inf")
    proj = Z_val @ U
    total_var = float(np.var(proj, axis=0).sum())
    if total_var <= 0:
        return float("-inf")
    levels = np.unique(factor_val)
    if len(levels) < 2:
        return float("-inf")
    grand_mean = proj.mean(axis=0)
    between = 0.0
    for lvl in levels:
        m = factor_val == lvl
        if m.sum() < 2:
            continue
        diff = proj[m].mean(axis=0) - grand_mean
        between += float(m.sum()) * float(np.sum(diff ** 2))
    return between / (total_var * len(proj))


def _decodability(Z_fit, f_fit, Z_eval, f_eval) -> float:
    """Held-out balanced-accuracy of a quick linear probe -- fit on (Z_fit,
    f_fit), scored on (Z_eval, f_eval). Never fit and scored on the same
    rows (see the LEACE continuous-concept in-sample-evaluation lesson)."""
    if len(np.unique(f_fit)) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=500).fit(Z_fit, f_fit)
    return float(balanced_accuracy_score(f_eval, clf.predict(Z_eval)))


# ---------------------------------------------------------------------------
# Checkpoint / task-direction loading
# ---------------------------------------------------------------------------


def _find_key(state: dict, candidates: tuple[str, ...]) -> str | None:
    for k in candidates:
        if k in state:
            return k
    return None


def load_task_direction(ckpt_path: Path, requested_layer: int) -> dict:
    """Load a checkpoint, extract its final linear classifier weight (the
    task direction w) and bias, and compare its own layer-weighting config
    against `requested_layer`.

    IMPORTANT: these e007 checkpoints use a LEARNED SOFT WEIGHTING over a
    BAND of backbone layers (model.layer_weight_init_center /
    model.layer_weight_init_band in the training config -- e.g. center=10,
    band=[8,11] for both the WavLM and XLS-R model configs this project
    uses), not a single fixed layer. `w` therefore does not, in general,
    live in the exact coordinate space of one frozen layer's cached
    embedding -- layer_weight_init_center is the closest available scalar
    proxy for "the layer", and any difference from --layer (including "no
    band info found at all") is treated as a mismatch: warned loudly and
    recorded as w_layer_mismatch=true, never silently assumed comparable.
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = sd.get("model", sd) if isinstance(sd, dict) else sd

    w_key = _find_key(state, CANDIDATE_W_KEYS)
    if w_key is None:
        raise RuntimeError(
            f"{ckpt_path}: no classifier weight found in checkpoint state dict "
            f"(tried {CANDIDATE_W_KEYS}) -- refusing to guess"
        )
    w = state[w_key].squeeze().float().cpu().numpy()

    b_key = _find_key(state, CANDIDATE_B_KEYS)
    b = float(state[b_key].squeeze().float().cpu().item()) if b_key is not None else 0.0

    cfg = sd.get("cfg", {}) if isinstance(sd, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    ckpt_layer_center = model_cfg.get("layer_weight_init_center")
    ckpt_layer_band = model_cfg.get("layer_weight_init_band")

    mismatch = ckpt_layer_center is None or ckpt_layer_center != requested_layer
    if mismatch:
        warnings.warn(
            f"{ckpt_path.name}: checkpoint layer-weighting center is "
            f"{ckpt_layer_center!r} (band {ckpt_layer_band!r}), not the requested "
            f"--layer {requested_layer} -- w is NOT guaranteed comparable to this "
            f"cached embedding layer. Recording w_layer_mismatch=true.",
            stacklevel=2,
        )
        print(
            f"[WARN] {ckpt_path.name}: layer mismatch (ckpt center={ckpt_layer_center!r}, "
            f"band={ckpt_layer_band!r}, requested={requested_layer}) -- w_layer_mismatch=true",
            file=sys.stderr,
        )

    return dict(
        w=w, b=b,
        ckpt_layer_center=ckpt_layer_center,
        ckpt_layer_band=ckpt_layer_band,
        w_layer_mismatch=bool(mismatch),
    )


def load_all_checkpoints(ckpt_dir: Path, runs: tuple[str, ...], requested_layer: int) -> dict[str, dict]:
    out = {}
    for run in runs:
        ckpt_path = ckpt_dir / f"runs_{run}_best.pt"
        if not ckpt_path.exists():
            print(f"[WARN] {ckpt_path}: not found -- skipping this checkpoint", file=sys.stderr)
            continue
        out[run] = load_task_direction(ckpt_path, requested_layer)
    return out


# ---------------------------------------------------------------------------
# Per-battery reliance measurement
# ---------------------------------------------------------------------------


def _make_fit_subspace(estimator: str, seed: int) -> Callable:
    def fit_subspace(Z, factor, y, groups, k):
        if estimator == "lda":
            return lda_subspace(Z, factor, y, k=k, mode="within_class")
        if estimator == "probe":
            return crossfitted_probe_subspace(Z, factor, y, k=k, mode="within_class", groups=groups, seed=seed)
        raise ValueError(f"unknown estimator {estimator!r}")

    return fit_subspace


def _make_effect_fn(checkpoints: dict[str, dict], seed: int, n_random: int) -> Callable:
    def effect_fn(Z_eff, factor_eff, y_eff, U):
        out: dict = {"per_checkpoint": {}}
        Sigma = np.atleast_2d(np.cov(Z_eff, rowvar=False)) if len(Z_eff) > 1 else np.eye(Z_eff.shape[1])

        for run, ck in checkpoints.items():
            w, b = ck["w"], ck["b"]
            pc = prediction_change(Z_eff, w, U, b=b)
            pc_control = removal_control_report(
                Z_eff, w, U,
                effect_fn=lambda Z, w_, U_, _b=b: prediction_change(Z, w_, U_, b=_b)["mean_abs_logit_change"],
                n_random=n_random, seed=seed,
            )
            out["per_checkpoint"][run] = dict(
                alignment=alignment(w, U),
                r_var=r_var(w, U, Sigma),
                r_var_class_conditional=r_var_class_conditional(w, U, Z_eff, y_eff),
                prediction_change=pc,
                prediction_change_control=pc_control,
                w_layer_mismatch=ck["w_layer_mismatch"],
                ckpt_layer_center=ck["ckpt_layer_center"],
            )

        # checkpoint-independent: LEACE / INLP erasure quality, each fit and
        # evaluated on disjoint halves of this effect fold (never in-sample).
        n = len(y_eff)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        half = max(1, n // 2)
        fit_idx, eval_idx = perm[:half], perm[half:]
        acc_before = _decodability(Z_eff[fit_idx], factor_eff[fit_idx], Z_eff[eval_idx], factor_eff[eval_idx])

        if len(eval_idx) > 0 and len(np.unique(factor_eff[fit_idx])) >= 2:
            leace = fit_leace(Z_eff[fit_idx], factor_eff[fit_idx])
            acc_after_leace = _decodability(
                leace.transform(Z_eff[fit_idx]), factor_eff[fit_idx],
                leace.transform(Z_eff[eval_idx]), factor_eff[eval_idx],
            )
            inlp = fit_inlp(Z_eff[fit_idx], factor_eff[fit_idx], n_iterations=6, seed=seed)
            acc_after_inlp = _decodability(
                inlp.transform(Z_eff[fit_idx]), factor_eff[fit_idx],
                inlp.transform(Z_eff[eval_idx]), factor_eff[eval_idx],
            )
        else:
            acc_after_leace = acc_after_inlp = float("nan")

        out["leace"] = dict(
            factor_decodability_before=acc_before,
            factor_decodability_after=acc_after_leace,
            decodability_drop=acc_before - acc_after_leace,
        )
        out["inlp"] = dict(
            factor_decodability_before=acc_before,
            factor_decodability_after=acc_after_inlp,
            decodability_drop=acc_before - acc_after_inlp,
        )

        def _projection_removal_decodability_drop(Z, w_unused, U_):
            Zr = project_out(Z, U_)
            b4 = _decodability(Z[fit_idx], factor_eff[fit_idx], Z[eval_idx], factor_eff[eval_idx])
            af = _decodability(Zr[fit_idx], factor_eff[fit_idx], Zr[eval_idx], factor_eff[eval_idx])
            return b4 - af

        any_w = next(iter(checkpoints.values()))["w"] if checkpoints else np.zeros(Z_eff.shape[1])
        out["projection_removal_control"] = removal_control_report(
            Z_eff, any_w, U, effect_fn=_projection_removal_decodability_drop,
            n_random=n_random, seed=seed,
        )
        return out

    return effect_fn


def run_battery(
    spec: dict,
    Z: np.ndarray,
    factor: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    checkpoints: dict[str, dict],
    ranks: list[int],
    n_boot: int,
    seed: int,
) -> dict:
    n_levels = int(len(np.unique(factor)))
    valid_ranks = ranks_for_n_levels(list(ranks), n_levels)
    n_groups = int(len(np.unique(groups)))
    grouping_degenerate = spec["factor"] == spec["grouping"]

    result: dict = dict(
        name=spec["name"], corpus=spec["corpus"], factor=spec["factor"], grouping=spec["grouping"],
        n_rows=int(len(y)), n_levels=n_levels, n_groups=n_groups,
        grouping_degenerate=grouping_degenerate,
        ranks_requested=list(ranks), ranks_valid=valid_ranks,
    )
    if grouping_degenerate:
        print(f"[WARN] battery {spec['name']}: grouping column == factor column "
              f"({spec['grouping']!r}) -- grouping_degenerate=true", file=sys.stderr)
    if not valid_ranks:
        result["skipped"] = f"no requested rank <= n_levels-1={n_levels - 1}"
        return result

    n_outer = min(5, n_groups)
    if n_outer < 2:
        result["skipped"] = f"only {n_groups} group(s) available -- cannot form nested folds"
        return result

    estimators: dict = {}
    for estimator in ("lda", "probe"):
        fit_subspace = _make_fit_subspace(estimator, seed)
        candidates = [{"k": r} for r in valid_ranks]
        effect_fn = _make_effect_fn(checkpoints, seed, n_random=20)
        try:
            fold_results = run_nested_crossfit(
                Z, factor, y, groups, candidates, fit_subspace, factor_separation_score, effect_fn,
                n_outer_splits=n_outer, n_inner_splits=min(3, n_outer), seed=seed,
            )
            estimators[estimator] = dict(fold_results=fold_results)
        except ValueError as e:
            # Most likely cause: grouping_degenerate (grouping column == factor
            # column, or otherwise every group carries exactly one factor level)
            # -- a GroupKFold split can then leave a fold's SELECTION set missing
            # factor levels entirely, so no candidate rank produces a usable
            # subspace. Record the failure per-estimator rather than crashing the
            # whole run (and every other battery after it) -- this is precisely
            # what grouping_degenerate=true exists to warn about.
            print(f"[WARN] battery {spec['name']}: estimator={estimator} failed "
                  f"({e}); recording as an error, not crashing the run", file=sys.stderr)
            estimators[estimator] = dict(fold_results=[], error=str(e))

    # grouped bootstrap CI + rank-sensitivity curve on a single headline metric
    # (mean r_var across checkpoints, LDA estimator at the smallest valid rank --
    # cheap enough to run under bootstrap resampling; the rank-sensitivity curve
    # separately covers how the metric moves across the full --ranks grid).
    headline_rank = valid_ranks[0]

    def _headline_metric(row_idx):
        Zs, fs, ys, gs = Z[row_idx], factor[row_idx], y[row_idx], groups[row_idx]
        try:
            U = lda_subspace(Zs, fs, ys, k=headline_rank, mode="within_class")
        except ValueError:
            return float("nan")
        if U.shape[1] == 0 or not checkpoints:
            return float("nan")
        Sigma = np.atleast_2d(np.cov(Zs, rowvar=False)) if len(Zs) > 1 else np.eye(Zs.shape[1])
        vals = [r_var(ck["w"], U, Sigma) for ck in checkpoints.values()]
        return float(np.mean(vals))

    bootstrap = grouped_bootstrap_ci(_headline_metric, groups, n_boot=n_boot, seed=seed)

    def _metric_at_rank(k):
        U = lda_subspace(Z, factor, y, k=k, mode="within_class")
        if U.shape[1] == 0 or not checkpoints:
            return float("nan")
        Sigma = np.atleast_2d(np.cov(Z, rowvar=False))
        return float(np.mean([r_var(ck["w"], U, Sigma) for ck in checkpoints.values()]))

    rank_curve = rank_sensitivity_curve(_metric_at_rank, valid_ranks)

    result["estimators"] = estimators
    result["headline_bootstrap_r_var"] = dict(rank=headline_rank, **bootstrap)
    result["rank_sensitivity_r_var"] = rank_curve
    return result


# ---------------------------------------------------------------------------
# prereg_candidates summary
# ---------------------------------------------------------------------------


def summarize_prereg_candidate(battery_result: dict) -> dict:
    if battery_result.get("skipped"):
        return dict(name=battery_result["name"], skipped=battery_result["skipped"])

    curve = battery_result["rank_sensitivity_r_var"]
    finite = [(r, v) for r, v in zip(curve["ranks"], curve["values"]) if np.isfinite(v)]
    if len(finite) >= 2:
        vals = np.array([v for _, v in finite])
        spread = float(np.ptp(vals))
        stable = spread < 0.25 * (abs(float(np.mean(vals))) + 1e-9)
        stable_ranks = [r for r, v in finite]
    else:
        stable, stable_ranks = False, [r for r, _ in finite]

    lda_folds = battery_result["estimators"]["lda"]["fold_results"]
    probe_folds = battery_result["estimators"]["probe"]["fold_results"]

    def _mean_r_var(fold_results):
        vals = []
        for f in fold_results:
            eff = f["effect"]
            for ck in eff.get("per_checkpoint", {}).values():
                vals.append(ck["r_var"])
        return float(np.mean(vals)) if vals else float("nan")

    lda_mean, probe_mean = _mean_r_var(lda_folds), _mean_r_var(probe_folds)
    agree_sign = np.sign(lda_mean) == np.sign(probe_mean) if np.isfinite(lda_mean) and np.isfinite(probe_mean) else False

    boot = battery_result["headline_bootstrap_r_var"]
    overlap = (boot["lo"] <= probe_mean <= boot["hi"]) or (boot["lo"] <= lda_mean <= boot["hi"])

    return dict(
        name=battery_result["name"],
        stable_rank_window=stable_ranks if stable else [],
        estimators_agree_sign=bool(agree_sign),
        cis_overlap=bool(overlap),
        n_groups=battery_result["n_groups"],
        grouping_degenerate=battery_result["grouping_degenerate"],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    ap.add_argument("--manifest-dir", default="manifests/v2")
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--out", default=None)
    ap.add_argument("--corpus", nargs="*", default=None, help="restrict to battery corpora in this list")
    ap.add_argument("--factor", nargs="*", default=None, help="restrict to battery factors in this list")
    ap.add_argument("--ranks", nargs="+", type=int, default=list(DEFAULT_RANKS))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    args = ap.parse_args(argv)

    out_path = Path(args.out) if args.out else Path(f"analysis/step3/reliance_layer{args.layer}.json")
    cache_root = Path(args.cache_root)
    manifest_dir = Path(args.manifest_dir)

    batteries = [
        b for b in BATTERIES
        if (args.corpus is None or b["corpus"] in args.corpus)
        and (args.factor is None or b["factor"] in args.factor)
    ]
    if not batteries:
        raise ValueError(f"--corpus/--factor filters matched 0 of {len(BATTERIES)} batteries")

    print(f"[run_reliance_battery] layer={args.layer} ranks={args.ranks} seed={args.seed} "
          f"batteries={[b['name'] for b in batteries]}")

    checkpoints = load_all_checkpoints(Path(args.ckpt_dir), RUNS, args.layer)
    if not checkpoints:
        print("[WARN] no checkpoints loaded -- w-dependent metrics will be empty", file=sys.stderr)

    corpora_needed = sorted({b["corpus"] for b in batteries})
    corpus_data: dict[str, dict] = {}
    join_stats: dict[str, dict] = {}
    for corpus in corpora_needed:
        corpus_dir = CORPUS_DIR[corpus]
        manifest_rows = read_manifest(manifest_dir / f"{corpus}.csv")
        manifest_df = pd.DataFrame([asdict(r) for r in manifest_rows])
        cache_paths, cache_emb = load_corpus_embeddings(cache_root, corpus_dir, args.layer)
        joined_df, joined_emb, stats = join_cache_to_manifest(cache_paths, cache_emb, manifest_df, corpus_dir)
        corpus_data[corpus] = dict(df=joined_df, emb=joined_emb)
        join_stats[corpus] = stats
        print(f"[join] {corpus}: n_cache={stats['n_cache']} n_manifest={stats['n_manifest']} "
              f"n_joined={stats['n_joined']} n_dropped={stats['n_dropped']}")

    battery_results = []
    for spec in batteries:
        df, emb = corpus_data[spec["corpus"]]["df"], corpus_data[spec["corpus"]]["emb"]
        Z, factor, y, groups = select_battery_rows(df, emb, spec)

        print(f"[battery] {spec['name']}: n_rows={len(y)} n_levels={len(np.unique(factor))}")
        res = run_battery(spec, Z, factor, y, groups, checkpoints, args.ranks, args.n_boot, args.seed)
        battery_results.append(res)

    prereg = [summarize_prereg_candidate(r) for r in battery_results]

    output = dict(
        schema_version=SCHEMA_VERSION,
        git_sha=_git_sha(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        layer=args.layer,
        seed=args.seed,
        join_stats=join_stats,
        checkpoints={run: dict(ckpt_layer_center=ck["ckpt_layer_center"], ckpt_layer_band=ck["ckpt_layer_band"],
                                w_layer_mismatch=ck["w_layer_mismatch"]) for run, ck in checkpoints.items()},
        batteries=battery_results,
        prereg_candidates=prereg,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=_json_default), encoding="utf-8")

    print("\n=== SUMMARY ===")
    for r in prereg:
        if "skipped" in r:
            print(f"  {r['name']}: SKIPPED ({r['skipped']})")
            continue
        print(f"  {r['name']}: n_groups={r['n_groups']} degenerate={r['grouping_degenerate']} "
              f"estimators_agree_sign={r['estimators_agree_sign']} cis_overlap={r['cis_overlap']} "
              f"stable_ranks={r['stable_rank_window']}")
    print(f"\nwrote -> {out_path}")


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


if __name__ == "__main__":
    main()
