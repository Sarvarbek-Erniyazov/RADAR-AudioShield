"""Roadmap v3 Step 3, Phase B: measure factor reliance in cached XLS-R-300M
embeddings using the merged src/audioshield/reliance/ library.

Read-only over the embedding cache -- NO GPU, NO training, NO gradient
computation. Loads three checkpoints only to extract their frozen final
linear-classifier weight (the "task direction" w) for alignment/removal
metrics; the backbones themselves are never touched.

--layer-mode {fixed, checkpoint-band}: the e007 heads (src/audioshield/models/
ssl_backbone.py's LayerWeightedSSL) do not read a single hidden-state layer --
they learn a softmax-weighted sum over ALL cached layers (self.layer_logits,
wired in as AudioShieldX.ssl.layer_logits), initialized as a soft band around
layer_weight_init_center but free to redistribute during training. So `w`
lives in that pooled representation, not in any one frozen layer's embedding
space.
  - fixed (default): unchanged prior behaviour. Every metric, including
    alignment, uses the single --layer embedding for both U and w. If the
    checkpoint's own layer-weighting center differs from --layer (normally
    true for these checkpoints), w_layer_mismatch is set and a comparison
    warning is printed, but the run proceeds -- alignment(w, U) is then a
    comparison across two different representations and should be read with
    that caveat.
  - checkpoint-band: ONLY the alignment metric changes. For each checkpoint,
    the embedding matrix is re-pooled across ALL cached hidden-state layers
    using that checkpoint's own resolved layer weights (its learned
    self.layer_logits after softmax, or -- if that parameter isn't present in
    the checkpoint -- a uniform average over the config's
    layer_weight_init_band, loudly flagged as layer_pooling=
    "uniform_band_fallback"). A subspace is refit in that pooled space (same
    estimator, same rank already chosen by the shared --layer run, same
    selection/effect split -- effect rows are never touched) so alignment
    compares w and U in the same coordinate system. w_layer_mismatch is False
    whenever pooling used the real learned weights (learned_softmax); the
    uniform-band fallback still sets it True, since it is an approximation of
    the model's actual pooling, not the pooling itself.
  Every OTHER metric (r_var, r_var_class_conditional, prediction_change,
  LEACE, INLP, and their removal_control_report controls) does not involve w
  at all beyond a fixed weight vector, and always uses the single --layer
  embedding regardless of --layer-mode, so those stay comparable across
  checkpoints and across --layer-mode runs.

--w-metrics {auto, on, off} (default auto): --layer-mode above assumes w and
the embedding differ only by WHICH backbone layer w was learned over. For
these cached embeddings there is a second, more fundamental mismatch: w is
AudioShieldX's classifier weight over embed()'s output (AttentiveStatsPooling
over the sequence, then a 2-layer proj MLP with a GELU nonlinearity in
between), not over a raw backbone hidden-state layer at all -- the cache is
1024-d (one backbone layer, pre-pooling), w is 256-d (post-pooling,
post-proj). No linear map recovers one from the other, so alignment, r_var,
r_var_class_conditional, prediction_change, and the task-direction positive
control inside removal_control_report (see metrics.py -- verified from
source, not the names: r_var and r_var_class_conditional both take w as a
required argument despite "r_var" not sounding like it should) cannot be
estimated against this cache at all, regardless of --layer-mode.
  - auto: try to load w; if its dimensionality doesn't match the cache's
    embedding dimensionality, DISABLE every w-dependent metric for the whole
    run, print the reason once, and continue -- never crash on this.
  - on: the same dimension check, but a mismatch is a hard failure (for runs
    where the caller has separately verified the spaces match, e.g. against
    scripts/extract_model_embeddings.py's model-space cache).
  - off: skip loading w entirely.
  Disabled w-dependent metrics appear in the JSON as
  {"value": null, "status": "not_estimable", "reason": "..."}, never silently
  omitted. Metrics that genuinely do not take w (project_out-based
  decodability drop, LEACE, INLP, the equal-norm random-subspace controls)
  still run unchanged. The grouped-bootstrap headline metric and the
  rank-sensitivity curve fall back to a non-w metric
  (factor_separation_score) when w-metrics are disabled; which metric was
  used is recorded as "metric" in their JSON blocks.

Usage:
    python scripts/run_reliance_battery.py --layer 9
    python scripts/run_reliance_battery.py --layer 9 --layer-mode checkpoint-band
    python scripts/run_reliance_battery.py --layer 9 --w-metrics off

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
import threading
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
from audioshield.reliance.crossfit import assert_no_group_leakage, make_nested_folds, run_nested_crossfit
from audioshield.reliance.metrics import (
    alignment,
    fit_inlp,
    fit_leace,
    prediction_change,
    project_out,
    r_var,
    r_var_class_conditional,
    random_subspace,
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

# The learned layer-weighting logits' key in the checkpoint's model state dict.
# Confirmed from source: AudioShieldX.ssl (src/audioshield/models/detector.py)
# is a LayerWeightedSSL (src/audioshield/models/ssl_backbone.py), which owns
# `self.layer_logits = nn.Parameter(...)` directly -- standard nn.Module
# state-dict naming gives "ssl.layer_logits", the same attr-path convention
# CANDIDATE_W_KEYS's "binary.fc.weight" already relies on.
LAYER_LOGITS_KEY = "ssl.layer_logits"

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
# Wall-clock budget for a single (battery, estimator) crossfit call, and
# separately for the headline bootstrap / rank-sensitivity sweep -- an
# overnight run once stalled inside the first battery (constant memory, no
# error, log frozen); see _run_with_timeout. 30 min is generous for a real
# 70k-row battery at the default --ranks/--n-boot, short enough that one
# intractable cell can't eat an overnight run's whole budget.
DEFAULT_BATTERY_TIMEOUT_SECONDS = 1800.0

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


def load_corpus_embeddings_all_layers(cache_root: Path, corpus_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load every shard_*.npz under cache_root/<corpus_dir>/ WITHOUT slicing to
    a single layer -- used only in --layer-mode checkpoint-band, to pool per
    checkpoint over the full set of cached hidden-state layers. Kept float16
    (as stored on disk) to bound memory; pool_band_embeddings upcasts only the
    much smaller pooled result, not this full array."""
    shard_dir = cache_root / corpus_dir
    shard_paths = sorted(shard_dir.glob("shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"no shard_*.npz files found under {shard_dir}")

    all_paths: list[np.ndarray] = []
    all_emb: list[np.ndarray] = []
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as data:
            all_paths.append(data["paths"])
            all_emb.append(data["emb"])
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


def select_battery_rows(
    df: pd.DataFrame, emb: np.ndarray, spec: dict, emb_full: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Apply a battery's optional row_filter and its mandatory factor-NA
    exclusion, then slice out (Z, factor, y, groups, Z_full) for that battery.

    Rows whose factor value is "NA" are excluded from the battery (they carry
    no information about the factor being measured); row_filter (e.g.
    diffssd's openvoicev2-only accent battery) is applied first.

    `emb_full` (n, L, D), when given (--layer-mode checkpoint-band only), is
    masked identically to `emb` so Z_full stays row-aligned with Z/factor/y/
    groups; returned as None when emb_full is None (fixed mode).
    """
    mask = np.ones(len(df), dtype=bool)
    if "row_filter" in spec:
        col, val = spec["row_filter"]
        mask &= (df[col] == val).to_numpy()
    mask &= (df[spec["factor"]] != "NA").to_numpy()

    Z = emb[mask]
    Z_full = emb_full[mask] if emb_full is not None else None
    factor = df.loc[mask, spec["factor"]].to_numpy()
    y = df.loc[mask, "target"].to_numpy().astype(int)
    groups = groups_from_column(df.loc[mask, spec["grouping"]].to_numpy())
    return Z, factor, y, groups, Z_full


def pool_band_embeddings(Z_full: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Z_full: (n, L, D) all cached hidden-state layers. weights: (L,) softmax
    layer weights. Returns (n, D) float32: sum_l weights[l] * Z_full[:, l, :]
    -- matches LayerWeightedSSL.forward's `(w * hs).sum(dim=0)` exactly
    (batch-first here vs. layer-first there)."""
    Z_full = np.asarray(Z_full, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    return np.tensordot(Z_full, weights, axes=([1], [0]))


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


def _not_estimable(reason: str) -> dict:
    """Sentinel for a metric that requires w when --w-metrics has disabled w
    -- never silently omitted, always this exact shape."""
    return {"value": None, "status": "not_estimable", "reason": reason}


# ---------------------------------------------------------------------------
# per-(battery, estimator) timeout guard
# ---------------------------------------------------------------------------


def _run_with_timeout(fn: Callable[[], object], timeout: float | None) -> tuple[bool, object, Exception | None]:
    """Run fn() (a zero-arg closure) in a background daemon thread with a
    hard wall-clock timeout. Returns (completed, value, exc):
        completed=False            -- fn() did not finish within `timeout`.
        completed=True, exc=None   -- fn() returned `value`.
        completed=True, exc=<Exception> -- fn() raised `exc`.

    Python cannot forcibly kill a thread, so a timed-out call is NOT
    actually interrupted -- the background thread is left running (daemon=
    True, so it can never block this process from exiting once everything
    else is done) while the caller stops waiting and moves on. This is
    exactly what turns "an overnight run stalled inside the first battery,
    process pinned at constant memory, log frozen, no error" into "one
    (battery, estimator) cell is marked failed and every other cell still
    runs" -- a hung matrix must degrade one cell, not the run.
    """
    outcome: dict = {}

    def _target():
        try:
            outcome["value"] = fn()
        except Exception as e:  # noqa: BLE001 -- report ANY failure to the caller, never swallow silently
            outcome["error"] = e

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        return False, None, None
    if "error" in outcome:
        return True, None, outcome["error"]
    return True, outcome.get("value"), None


def _guarded_call(fn: Callable[[], dict], timeout: float, fallback: Callable[[str], dict]) -> dict:
    """Run fn() (expected to return a dict) under _run_with_timeout; on
    timeout or exception, call fallback(reason) to build a status="failed"
    result instead of propagating. On success, fn()'s own dict gets
    status="ok" merged in (never overwriting a status key fn() already set)."""
    completed, value, exc = _run_with_timeout(fn, timeout)
    if not completed:
        return fallback(f"timed out after {timeout}s")
    if exc is not None:
        return fallback(f"{type(exc).__name__}: {exc}")
    out = dict(value)
    out.setdefault("status", "ok")
    return out


def _removal_control_without_task_direction(
    Z: np.ndarray, U_true: np.ndarray, effect_fn: Callable[[np.ndarray, np.ndarray], float],
    n_random: int = 20, seed: int = 13,
) -> dict:
    """removal_control_report's true_effect/random_effects/random_mean/
    random_std/exceeds_random computation, WITHOUT requiring a task-direction
    w -- used when w-metrics are disabled but the underlying effect_fn is
    itself factor-only (e.g. a decodability-drop-based removal effect).
    metrics.removal_control_report cannot be called at all in that case: it
    unconditionally builds task_direction_subspace(w, ...) even when its own
    effect_fn ignores w. task_direction_effect is therefore the caller's
    responsibility to mark not_estimable -- computing it structurally
    requires a real w.

    Args:
        effect_fn: `(Z, U) -> float` -- note this is metrics.removal_control_report's
            `effect_fn` with its `w` argument already dropped.
    """
    k = U_true.shape[1]
    d = Z.shape[1]
    true_effect = float(effect_fn(Z, U_true))
    rng = np.random.default_rng(seed)
    random_effects = [
        float(effect_fn(Z, random_subspace(d, k, seed=int(rng.integers(1_000_000_000)))))
        for _ in range(n_random)
    ]
    random_mean = float(np.mean(random_effects)) if random_effects else float("nan")
    random_std = float(np.std(random_effects)) if random_effects else float("nan")
    return dict(
        true_effect=true_effect, random_effects=random_effects, random_mean=random_mean, random_std=random_std,
        exceeds_random=(bool(true_effect > random_mean + 2 * random_std) if random_effects else None),
    )


# ---------------------------------------------------------------------------
# Checkpoint / task-direction loading
# ---------------------------------------------------------------------------


def _find_key(state: dict, candidates: tuple[str, ...]) -> str | None:
    for k in candidates:
        if k in state:
            return k
    return None


def _find_layer_logits_key(state: dict) -> str | None:
    """Locate the learned layer-weighting parameter in a checkpoint's state
    dict. Prefers the exact known key (LAYER_LOGITS_KEY); falls back to a
    unique suffix match so a wrapping prefix (e.g. a DDP "module." prefix)
    doesn't cause a false "not found" -- but only when exactly one key
    matches, never an ambiguous guess among several."""
    if LAYER_LOGITS_KEY in state:
        return LAYER_LOGITS_KEY
    matches = [k for k in state if k == LAYER_LOGITS_KEY or k.endswith("." + LAYER_LOGITS_KEY)]
    return matches[0] if len(matches) == 1 else None


def _uniform_band_weights(band: tuple, num_layers: int) -> np.ndarray:
    lo, hi = max(0, int(band[0])), min(num_layers - 1, int(band[1]))
    w = np.zeros(num_layers, dtype=np.float64)
    w[lo:hi + 1] = 1.0 / (hi - lo + 1)
    return w


def _describe_pooling_mismatch(state: dict, w_key: str, w_dim: int) -> str:
    """Build a reason string naming the real shapes found in THIS checkpoint's
    state dict where available (never a hardcoded/guessed description)."""
    proj0, proj4 = state.get("proj.0.weight"), state.get("proj.4.weight")
    if proj0 is not None and proj4 is not None:
        pooling = (f"AttentiveStatsPooling over the sequence, then proj.0 {tuple(proj0.shape)} -> "
                   f"GELU -> proj.4 {tuple(proj4.shape)}")
    else:
        pooling = "AttentiveStatsPooling over the sequence, then a 2-layer proj MLP with a GELU in between"
    return (
        f"w ({w_key}) is {w_dim}-d but the embedding cache is a single raw backbone hidden-state "
        f"layer, pre-pooling -- no linear pullback exists between these spaces: w is the classifier "
        f"weight over the model's pooled+projected embedding ({pooling}, a nonlinear transform)"
    )


def load_task_direction(
    ckpt_path: Path, requested_layer: int, layer_mode: str = "fixed", num_cache_layers: int | None = None,
    w_metrics_mode: str = "auto", embedding_dim: int | None = None,
) -> dict:
    """Load a checkpoint, extract its final linear classifier weight (the
    task direction w) and bias, and resolve how its embedding should be
    pooled for the alignment metric specifically (see module docstring for
    --layer-mode and --w-metrics).

    w_metrics_mode="off": skip loading w entirely (w=None, w_dim=None); the
    checkpoint's layer-weighting metadata is still loaded (cheap, and
    independent of whether w-dependent metrics are wanted).

    w_metrics_mode="auto"/"on": load w and its dimensionality (w_dim). If
    `embedding_dim` is given and doesn't match, w_dim_mismatch=true and
    w_dim_mismatch_reason names the real shapes found in this checkpoint's
    state dict (never a guessed description). "on" raises on a mismatch;
    "auto" warns and returns the mismatch flag for the caller (main()) to
    decide whether to disable w-metrics for the whole run.

    layer_mode="fixed" (unchanged prior behaviour): compares the checkpoint's
    own layer-weighting center against `requested_layer`; any difference
    (including "no band info found at all") is treated as a mismatch --
    warned loudly and recorded as w_layer_mismatch=true, never silently
    assumed comparable.

    layer_mode="checkpoint-band": resolves the checkpoint's own learned
    softmax layer weights (LAYER_LOGITS_KEY, softmax over all
    num_cache_layers hidden states -- matching LayerWeightedSSL.forward
    exactly, NOT restricted to the config band, since the model is free to
    redistribute weight outside its init band during training). If that
    parameter isn't present in the state dict, falls back to a uniform
    average over the config's layer_weight_init_band (layer_pooling=
    "uniform_band_fallback", loudly warned, w_layer_mismatch stays true since
    this is an approximation of the model's actual pooling). If neither the
    learned parameter nor a config band is available, refuses to guess and
    raises.
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = sd.get("model", sd) if isinstance(sd, dict) else sd

    cfg = sd.get("cfg", {}) if isinstance(sd, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    ckpt_layer_center = model_cfg.get("layer_weight_init_center")
    ckpt_layer_band = model_cfg.get("layer_weight_init_band")

    if w_metrics_mode == "off":
        return dict(
            w=None, b=None, w_dim=None, w_dim_mismatch=None, w_dim_mismatch_reason=None,
            ckpt_layer_center=ckpt_layer_center, ckpt_layer_band=ckpt_layer_band,
            layer_pooling="fixed_layer", band_weights=None, w_layer_mismatch=None,
        )

    w_key = _find_key(state, CANDIDATE_W_KEYS)
    if w_key is None:
        raise RuntimeError(
            f"{ckpt_path}: no classifier weight found in checkpoint state dict "
            f"(tried {CANDIDATE_W_KEYS}) -- refusing to guess"
        )
    w = state[w_key].squeeze().float().cpu().numpy()
    w_dim = int(w.shape[0])

    b_key = _find_key(state, CANDIDATE_B_KEYS)
    b = float(state[b_key].squeeze().float().cpu().item()) if b_key is not None else 0.0

    w_dim_mismatch = embedding_dim is not None and w_dim != embedding_dim
    w_dim_mismatch_reason = None
    if w_dim_mismatch:
        w_dim_mismatch_reason = f"{ckpt_path.name}: {_describe_pooling_mismatch(state, w_key, w_dim)}"
        if w_metrics_mode == "on":
            raise RuntimeError(f"--w-metrics on: {w_dim_mismatch_reason}")
        warnings.warn(w_dim_mismatch_reason, stacklevel=2)
        print(f"[WARN] {w_dim_mismatch_reason}", file=sys.stderr)

    if layer_mode == "fixed":
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
            w=w, b=b, w_dim=w_dim, w_dim_mismatch=w_dim_mismatch, w_dim_mismatch_reason=w_dim_mismatch_reason,
            ckpt_layer_center=ckpt_layer_center, ckpt_layer_band=ckpt_layer_band,
            layer_pooling="fixed_layer", band_weights=None, w_layer_mismatch=bool(mismatch),
        )

    # layer_mode == "checkpoint-band"
    if num_cache_layers is None:
        raise ValueError("layer_mode='checkpoint-band' requires num_cache_layers (from the loaded embedding cache)")

    logits_key = _find_layer_logits_key(state)
    if logits_key is not None:
        logits = state[logits_key].detach().float().cpu()
        if logits.shape[0] != num_cache_layers:
            raise RuntimeError(
                f"{ckpt_path}: {logits_key} has {logits.shape[0]} entries but the embedding "
                f"cache has {num_cache_layers} hidden-state layers -- cannot pool"
            )
        weights = torch.softmax(logits, dim=0).numpy()
        return dict(
            w=w, b=b, w_dim=w_dim, w_dim_mismatch=w_dim_mismatch, w_dim_mismatch_reason=w_dim_mismatch_reason,
            ckpt_layer_center=ckpt_layer_center, ckpt_layer_band=ckpt_layer_band,
            layer_pooling="learned_softmax", band_weights=weights, w_layer_mismatch=False,
        )

    if ckpt_layer_band is None:
        raise RuntimeError(
            f"{ckpt_path}: --layer-mode checkpoint-band requested but no learned layer-weight "
            f"parameter ({LAYER_LOGITS_KEY!r}) found in the state dict, and no "
            f"layer_weight_init_band in cfg to fall back to -- refusing to guess"
        )
    warnings.warn(
        f"{ckpt_path.name}: no learned layer-weight parameter ({LAYER_LOGITS_KEY!r}) found in "
        f"the state dict -- falling back to a uniform average over the config band "
        f"{ckpt_layer_band!r}. This is NOT the model's actual learned pooling; alignment computed "
        f"under this checkpoint is an approximation. Recording w_layer_mismatch=true.",
        stacklevel=2,
    )
    print(
        f"[WARN] {ckpt_path.name}: {LAYER_LOGITS_KEY!r} not found -- uniform_band_fallback over "
        f"band {ckpt_layer_band!r}",
        file=sys.stderr,
    )
    weights = _uniform_band_weights(ckpt_layer_band, num_cache_layers)
    return dict(
        w=w, b=b, w_dim=w_dim, w_dim_mismatch=w_dim_mismatch, w_dim_mismatch_reason=w_dim_mismatch_reason,
        ckpt_layer_center=ckpt_layer_center, ckpt_layer_band=ckpt_layer_band,
        layer_pooling="uniform_band_fallback", band_weights=weights, w_layer_mismatch=True,
    )


def resolve_w_metrics(checkpoints: dict[str, dict], w_metrics_mode: str, embedding_dim: int) -> dict:
    """Aggregate per-checkpoint w_dim_mismatch flags into a single run-wide
    decision (module docstring: --w-metrics). A mismatch on ANY loaded
    checkpoint disables w-dependent metrics for the WHOLE run, not just that
    checkpoint -- a report mixing w-enabled and w-disabled checkpoints would
    be more confusing than informative. "on" mode never reaches here with a
    mismatch (load_task_direction already raised)."""
    if w_metrics_mode == "off":
        return dict(enabled=False, reason="--w-metrics off: w loading skipped entirely",
                     w_dim=None, embedding_dim=embedding_dim)
    if not checkpoints:
        return dict(enabled=False, reason="no checkpoints loaded -- nothing to check w against",
                     w_dim=None, embedding_dim=embedding_dim)
    mismatched = {run: ck for run, ck in checkpoints.items() if ck.get("w_dim_mismatch")}
    if mismatched:
        sample = next(iter(mismatched.values()))
        return dict(enabled=False, reason=sample["w_dim_mismatch_reason"],
                     w_dim=sample["w_dim"], embedding_dim=embedding_dim)
    any_ck = next(iter(checkpoints.values()))
    return dict(enabled=True, reason="w and embedding dimensionality match",
                 w_dim=any_ck["w_dim"], embedding_dim=embedding_dim)


def load_all_checkpoints(
    ckpt_dir: Path, runs: tuple[str, ...], requested_layer: int,
    layer_mode: str = "fixed", num_cache_layers: int | None = None,
    w_metrics_mode: str = "auto", embedding_dim: int | None = None,
) -> dict[str, dict]:
    out = {}
    for run in runs:
        ckpt_path = ckpt_dir / f"runs_{run}_best.pt"
        if not ckpt_path.exists():
            print(f"[WARN] {ckpt_path}: not found -- skipping this checkpoint", file=sys.stderr)
            continue
        out[run] = load_task_direction(ckpt_path, requested_layer, layer_mode=layer_mode,
                                        num_cache_layers=num_cache_layers, w_metrics_mode=w_metrics_mode,
                                        embedding_dim=embedding_dim)
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


def _make_effect_fn(
    checkpoints: dict[str, dict], seed: int, n_random: int, layer_mode: str = "fixed",
    w_metrics_enabled: bool = True, w_metrics_reason: str = "",
) -> Callable:
    def effect_fn(Z_eff, factor_eff, y_eff, U):
        out: dict = {"per_checkpoint": {}}
        Sigma = np.atleast_2d(np.cov(Z_eff, rowvar=False)) if w_metrics_enabled and len(Z_eff) > 1 else None

        for run, ck in checkpoints.items():
            if w_metrics_enabled:
                w, b = ck["w"], ck["b"]
                pc = prediction_change(Z_eff, w, U, b=b)
                pc_control = removal_control_report(
                    Z_eff, w, U,
                    effect_fn=lambda Z, w_, U_, _b=b: prediction_change(Z, w_, U_, b=_b)["mean_abs_logit_change"],
                    n_random=n_random, seed=seed,
                )
                out["per_checkpoint"][run] = dict(
                    # In checkpoint-band mode, U here is fit on the shared --layer
                    # embedding while w lives in the checkpoint's own pooled
                    # space -- comparing them would be exactly the mismatched-
                    # representation bug this mode exists to fix, so alignment is
                    # left for _recompute_band_alignment's post-processing pass.
                    alignment=(alignment(w, U) if layer_mode == "fixed" else None),
                    r_var=r_var(w, U, Sigma if Sigma is not None else np.eye(Z_eff.shape[1])),
                    r_var_class_conditional=r_var_class_conditional(w, U, Z_eff, y_eff),
                    prediction_change=pc,
                    prediction_change_control=pc_control,
                    w_layer_mismatch=ck["w_layer_mismatch"],
                    ckpt_layer_center=ck["ckpt_layer_center"],
                    layer_pooling=ck.get("layer_pooling", "fixed_layer"),
                )
            else:
                # w-dependent metrics require w and the embedding to live in
                # the same space -- --w-metrics has determined they do not
                # (see module docstring). Never silently omitted: every
                # affected metric gets the explicit not_estimable sentinel.
                out["per_checkpoint"][run] = dict(
                    alignment=_not_estimable(w_metrics_reason),
                    r_var=_not_estimable(w_metrics_reason),
                    r_var_class_conditional=_not_estimable(w_metrics_reason),
                    prediction_change=_not_estimable(w_metrics_reason),
                    prediction_change_control=_not_estimable(w_metrics_reason),
                    w_layer_mismatch=None,
                    ckpt_layer_center=ck.get("ckpt_layer_center"),
                    layer_pooling=ck.get("layer_pooling", "fixed_layer"),
                )

        # factor-only, checkpoint-independent -- always computed regardless of
        # --w-metrics (used as the non-w headline/rank-sensitivity metric when
        # w-metrics are disabled; see run_battery).
        out["factor_separation_score"] = factor_separation_score(Z_eff, factor_eff, y_eff, U)

        # checkpoint-independent: LEACE / INLP erasure quality, each fit and
        # evaluated on disjoint halves of this effect fold (never in-sample).
        # Neither fit_leace nor fit_inlp takes w -- these run unchanged
        # regardless of --w-metrics.
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

        def _projection_removal_decodability_drop(Z, U_):
            Zr = project_out(Z, U_)
            b4 = _decodability(Z[fit_idx], factor_eff[fit_idx], Z[eval_idx], factor_eff[eval_idx])
            af = _decodability(Zr[fit_idx], factor_eff[fit_idx], Zr[eval_idx], factor_eff[eval_idx])
            return b4 - af

        # true_effect/random_effects here are factor-only (the decodability-drop
        # effect_fn never reads w) -- only the task-direction positive control
        # genuinely needs w, so it's the only piece that becomes not_estimable.
        if w_metrics_enabled:
            any_w = next(iter(checkpoints.values()))["w"] if checkpoints else np.zeros(Z_eff.shape[1])
            out["projection_removal_control"] = removal_control_report(
                Z_eff, any_w, U, effect_fn=lambda Z, w_, U_: _projection_removal_decodability_drop(Z, U_),
                n_random=n_random, seed=seed,
            )
        else:
            out["projection_removal_control"] = _removal_control_without_task_direction(
                Z_eff, U, effect_fn=_projection_removal_decodability_drop, n_random=n_random, seed=seed,
            )
            out["projection_removal_control"]["task_direction_effect"] = _not_estimable(w_metrics_reason)
        return out

    return effect_fn


def _recompute_band_alignment(
    fold_results: list[dict],
    Z_full: np.ndarray,
    factor: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    checkpoints: dict[str, dict],
    fit_subspace: Callable,
    n_outer: int,
    seed: int,
) -> None:
    """--layer-mode checkpoint-band only: for each fold and each checkpoint,
    refit a subspace in THAT checkpoint's own band-pooled embedding space
    (never the shared --layer space) and overwrite that fold's `alignment`
    entry in place, so alignment(w, U) compares w and U in the same
    representation.

    Reuses the rank already chosen by the shared --layer nested-crossfit run
    for that fold (the estimator/rank choice is shared across the metric
    suite; only the embedding space differs for alignment) and reconstructs
    the exact same outer folds run_nested_crossfit used internally --
    deterministic given the same (n, groups, n_outer_splits, seed), so this
    never touches a fold's effect rows, only its selection rows.
    """
    outer_folds = make_nested_folds(len(y), groups, n_splits=n_outer, seed=seed)
    by_id = {fr["fold_id"]: fr for fr in fold_results}
    for fold in outer_folds:
        assert_no_group_leakage(fold, groups)
        fr = by_id.get(fold.fold_id)
        if fr is None:
            continue
        k = fr["chosen"]["k"]
        sel = fold.selection_idx
        for run, ck in checkpoints.items():
            per_ck = fr["effect"]["per_checkpoint"].get(run)
            if per_ck is None:
                continue
            Z_band_sel = pool_band_embeddings(Z_full[sel], ck["band_weights"])
            try:
                U_band = fit_subspace(Z_band_sel, factor[sel], y[sel], groups[sel], k=k)
                per_ck["alignment"] = alignment(ck["w"], U_band)
            except ValueError as e:
                per_ck["alignment"] = float("nan")
                per_ck["alignment_error"] = str(e)


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
    layer_mode: str = "fixed",
    Z_full: np.ndarray | None = None,
    w_metrics_enabled: bool = True,
    w_metrics_reason: str = "",
    battery_timeout_seconds: float = DEFAULT_BATTERY_TIMEOUT_SECONDS,
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
        layer_mode=layer_mode,
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
        effect_fn = _make_effect_fn(checkpoints, seed, n_random=20, layer_mode=layer_mode,
                                     w_metrics_enabled=w_metrics_enabled, w_metrics_reason=w_metrics_reason)

        def _crossfit_fallback(reason: str, _estimator=estimator) -> dict:
            print(f"[WARN] battery {spec['name']}: estimator={_estimator} failed ({reason}); "
                  f"recording as failed, not crashing the run", file=sys.stderr)
            return dict(fold_results=[], status="failed", error=reason)

        # Wrapped rather than a plain try/except: the failure mode this
        # guards against (an overnight run stalled here, log frozen, no
        # error) never raises, so an exception handler alone can't catch it
        # -- only a wall-clock timeout can. Most exception-raising failures
        # (e.g. grouping_degenerate leaving a fold's selection set missing
        # factor levels entirely) are still caught the same way, just via
        # the same guard instead of a separate except clause.
        estimator_result = _guarded_call(
            lambda: dict(fold_results=run_nested_crossfit(
                Z, factor, y, groups, candidates, fit_subspace, factor_separation_score, effect_fn,
                n_outer_splits=n_outer, n_inner_splits=min(3, n_outer), seed=seed,
            )),
            battery_timeout_seconds, _crossfit_fallback,
        )

        if estimator_result.get("status") == "failed":
            estimators[estimator] = estimator_result
            continue

        fold_results = estimator_result["fold_results"]
        if layer_mode == "checkpoint-band" and Z_full is not None and checkpoints and w_metrics_enabled:
            try:
                _recompute_band_alignment(fold_results, Z_full, factor, y, groups, checkpoints,
                                           fit_subspace, n_outer, seed)
            except Exception as e:
                print(f"[WARN] battery {spec['name']}: estimator={estimator} band-alignment "
                      f"recompute failed ({e})", file=sys.stderr)

        estimators[estimator] = estimator_result

    # grouped bootstrap CI + rank-sensitivity curve on a single headline metric
    # (LDA estimator at the smallest valid rank -- cheap enough to run under
    # bootstrap resampling; the rank-sensitivity curve separately covers how
    # the metric moves across the full --ranks grid). r_var needs w; when
    # w-metrics are disabled, this must not silently crash (that's exactly
    # where an earlier run died) -- fall back to a checkpoint-independent
    # factor-only metric and record which one was actually used. Both sweeps
    # refit LDA repeatedly (once per bootstrap resample / once per rank), so
    # they get the same timeout guard as the main crossfit call above.
    headline_rank = valid_ranks[0]
    headline_metric_name = "r_var" if (w_metrics_enabled and checkpoints) else "factor_separation_score"

    def _headline_metric(row_idx):
        Zs, fs, ys = Z[row_idx], factor[row_idx], y[row_idx]
        try:
            U = lda_subspace(Zs, fs, ys, k=headline_rank, mode="within_class")
        except ValueError:
            return float("nan")
        if U.shape[1] == 0:
            return float("nan")
        if headline_metric_name == "r_var":
            Sigma = np.atleast_2d(np.cov(Zs, rowvar=False)) if len(Zs) > 1 else np.eye(Zs.shape[1])
            return float(np.mean([r_var(ck["w"], U, Sigma) for ck in checkpoints.values()]))
        return factor_separation_score(Zs, fs, ys, U)

    def _bootstrap_fallback(reason: str) -> dict:
        print(f"[WARN] battery {spec['name']}: headline bootstrap failed ({reason})", file=sys.stderr)
        return dict(mean=float("nan"), std=float("nan"), lo=float("nan"), hi=float("nan"),
                    n_boot=n_boot, n_groups=n_groups, n_finite=0, n_boot_failed=n_boot,
                    status="failed", error=reason)

    bootstrap = _guarded_call(
        lambda: grouped_bootstrap_ci(_headline_metric, groups, n_boot=n_boot, seed=seed),
        battery_timeout_seconds, _bootstrap_fallback,
    )

    def _metric_at_rank(k):
        U = lda_subspace(Z, factor, y, k=k, mode="within_class")
        if U.shape[1] == 0:
            return float("nan")
        if headline_metric_name == "r_var":
            Sigma = np.atleast_2d(np.cov(Z, rowvar=False))
            return float(np.mean([r_var(ck["w"], U, Sigma) for ck in checkpoints.values()]))
        return factor_separation_score(Z, factor, y, U)

    def _rank_curve_fallback(reason: str) -> dict:
        print(f"[WARN] battery {spec['name']}: rank-sensitivity sweep failed ({reason})", file=sys.stderr)
        return dict(ranks=list(valid_ranks), values=[float("nan")] * len(valid_ranks),
                    status="failed", error=reason)

    rank_curve = _guarded_call(
        lambda: rank_sensitivity_curve(_metric_at_rank, valid_ranks),
        battery_timeout_seconds, _rank_curve_fallback,
    )

    result["estimators"] = estimators
    result["headline_bootstrap"] = dict(metric=headline_metric_name, rank=headline_rank, **bootstrap)
    result["rank_sensitivity"] = dict(metric=headline_metric_name, **rank_curve)
    return result


# ---------------------------------------------------------------------------
# prereg_candidates summary
# ---------------------------------------------------------------------------


def summarize_prereg_candidate(battery_result: dict) -> dict:
    if battery_result.get("skipped"):
        return dict(name=battery_result["name"], skipped=battery_result["skipped"])

    boot = battery_result["headline_bootstrap"]
    metric_name = boot["metric"]
    curve = battery_result["rank_sensitivity"]
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

    def _mean_headline_metric(fold_results):
        vals = []
        for f in fold_results:
            eff = f["effect"]
            if metric_name == "r_var":
                for ck in eff.get("per_checkpoint", {}).values():
                    vals.append(ck["r_var"])
            else:
                v = eff.get(metric_name)
                if v is not None and np.isfinite(v):
                    vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    lda_mean, probe_mean = _mean_headline_metric(lda_folds), _mean_headline_metric(probe_folds)
    agree_sign = np.sign(lda_mean) == np.sign(probe_mean) if np.isfinite(lda_mean) and np.isfinite(probe_mean) else False

    overlap = (boot["lo"] <= probe_mean <= boot["hi"]) or (boot["lo"] <= lda_mean <= boot["hi"])

    return dict(
        name=battery_result["name"],
        headline_metric=metric_name,
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
    ap.add_argument(
        "--layer-mode", choices=["fixed", "checkpoint-band"], default="fixed",
        help="fixed (default): alignment uses the single --layer embedding, same as every other "
             "metric -- unchanged prior behaviour. checkpoint-band: alignment additionally refits "
             "its subspace in each checkpoint's own learned softmax-pooled multi-layer embedding "
             "space, so it compares w and U in the same representation; all other metrics still "
             "use --layer regardless of this flag.",
    )
    ap.add_argument(
        "--w-metrics", choices=["auto", "on", "off"], default="auto",
        help="auto (default): load w; if its dimensionality doesn't match the cache's embedding "
             "dimensionality, disable every w-dependent metric (alignment, r_var, "
             "r_var_class_conditional, prediction_change, and the task-direction positive control) "
             "for the whole run, print why, and continue. on: the same dimension check, but a "
             "mismatch is a hard failure. off: skip loading w entirely.",
    )
    ap.add_argument(
        "--battery-timeout-seconds", type=float, default=DEFAULT_BATTERY_TIMEOUT_SECONDS,
        help=f"wall-clock budget for each (battery, estimator) crossfit call, and separately for the "
             f"headline bootstrap / rank-sensitivity sweep -- a cell that doesn't finish in time is "
             f"recorded as status='failed' and the run continues to the next configuration, rather "
             f"than hanging (default: {DEFAULT_BATTERY_TIMEOUT_SECONDS:.0f}s).",
    )
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

    print(f"[run_reliance_battery] layer={args.layer} layer_mode={args.layer_mode} ranks={args.ranks} "
          f"seed={args.seed} batteries={[b['name'] for b in batteries]}")

    # Corpus data is loaded before checkpoints: in checkpoint-band mode, the
    # checkpoints' learned layer-weighting parameter must be softmaxed over
    # exactly as many entries as the embedding cache actually has hidden-state
    # layers -- that count (num_cache_layers) is only known once real shards
    # are read, never hardcoded.
    corpora_needed = sorted({b["corpus"] for b in batteries})
    corpus_data: dict[str, dict] = {}
    join_stats: dict[str, dict] = {}
    num_cache_layers: int | None = None
    embedding_dim: int | None = None
    for corpus in corpora_needed:
        corpus_dir = CORPUS_DIR[corpus]
        manifest_rows = read_manifest(manifest_dir / f"{corpus}.csv")
        manifest_df = pd.DataFrame([asdict(r) for r in manifest_rows])
        cache_paths, cache_emb = load_corpus_embeddings(cache_root, corpus_dir, args.layer)
        joined_df, joined_emb, stats = join_cache_to_manifest(cache_paths, cache_emb, manifest_df, corpus_dir)

        if embedding_dim is None:
            embedding_dim = joined_emb.shape[1]
        elif embedding_dim != joined_emb.shape[1]:
            raise ValueError(f"{corpus}: cache embedding is {joined_emb.shape[1]}-d, expected "
                              f"{embedding_dim}-d (mismatched across corpora)")

        emb_full = None
        if args.layer_mode == "checkpoint-band":
            cache_paths_full, cache_emb_full = load_corpus_embeddings_all_layers(cache_root, corpus_dir)
            joined_df_full, joined_emb_full, _ = join_cache_to_manifest(
                cache_paths_full, cache_emb_full, manifest_df, corpus_dir
            )
            if len(joined_df_full) != len(joined_df) or not (
                joined_df_full["utt_id"].to_numpy() == joined_df["utt_id"].to_numpy()
            ).all():
                raise AssertionError(
                    f"{corpus}: fixed-layer join and all-layers join disagree on row set/order -- "
                    f"cannot align band-pooled embeddings to the main embedding matrix"
                )
            emb_full = joined_emb_full
            if num_cache_layers is None:
                num_cache_layers = cache_emb_full.shape[1]
            elif num_cache_layers != cache_emb_full.shape[1]:
                raise ValueError(f"{corpus}: cache has {cache_emb_full.shape[1]} hidden-state layers, "
                                  f"expected {num_cache_layers} (mismatched across corpora)")

        corpus_data[corpus] = dict(df=joined_df, emb=joined_emb, emb_full=emb_full)
        join_stats[corpus] = stats
        print(f"[join] {corpus}: n_cache={stats['n_cache']} n_manifest={stats['n_manifest']} "
              f"n_joined={stats['n_joined']} n_dropped={stats['n_dropped']}")

    checkpoints = load_all_checkpoints(Path(args.ckpt_dir), RUNS, args.layer,
                                        layer_mode=args.layer_mode, num_cache_layers=num_cache_layers,
                                        w_metrics_mode=args.w_metrics, embedding_dim=embedding_dim)
    if not checkpoints:
        print("[WARN] no checkpoints loaded -- w-dependent metrics will be empty", file=sys.stderr)

    w_metrics = resolve_w_metrics(checkpoints, args.w_metrics, embedding_dim)
    print(f"[w-metrics] enabled={w_metrics['enabled']} w_dim={w_metrics['w_dim']} "
          f"embedding_dim={w_metrics['embedding_dim']} reason={w_metrics['reason']!r}")

    battery_results = []
    for spec in batteries:
        cd = corpus_data[spec["corpus"]]
        Z, factor, y, groups, Z_full = select_battery_rows(cd["df"], cd["emb"], spec, emb_full=cd["emb_full"])

        print(f"[battery] {spec['name']}: n_rows={len(y)} n_levels={len(np.unique(factor))}")
        res = run_battery(spec, Z, factor, y, groups, checkpoints, args.ranks, args.n_boot, args.seed,
                           layer_mode=args.layer_mode, Z_full=Z_full,
                           w_metrics_enabled=w_metrics["enabled"], w_metrics_reason=w_metrics["reason"],
                           battery_timeout_seconds=args.battery_timeout_seconds)
        battery_results.append(res)

    prereg = [summarize_prereg_candidate(r) for r in battery_results]

    output = dict(
        schema_version=SCHEMA_VERSION,
        git_sha=_git_sha(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        layer=args.layer,
        layer_mode=args.layer_mode,
        seed=args.seed,
        w_metrics=w_metrics,
        join_stats=join_stats,
        checkpoints={run: dict(ckpt_layer_center=ck["ckpt_layer_center"], ckpt_layer_band=ck["ckpt_layer_band"],
                                layer_pooling=ck["layer_pooling"], band_weights=ck["band_weights"],
                                w_layer_mismatch=ck["w_layer_mismatch"], w_dim=ck["w_dim"],
                                w_dim_mismatch=ck["w_dim_mismatch"]) for run, ck in checkpoints.items()},
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
        print(f"  {r['name']}: headline_metric={r['headline_metric']} n_groups={r['n_groups']} "
              f"degenerate={r['grouping_degenerate']} estimators_agree_sign={r['estimators_agree_sign']} "
              f"cis_overlap={r['cis_overlap']} stable_ranks={r['stable_rank_window']}")
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
