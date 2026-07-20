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

Subprocess isolation (Roadmap Step 3 robustness repair -- four attempts on
real data never produced a results file; see step3_phaseA_repair_brief.md):
each battery's four units of work -- the lda crossfit call, the probe
crossfit call, the headline bootstrap, the rank-sensitivity sweep -- run
CONCURRENTLY, each in its own freshly-spawned worker PROCESS (never a
shared/reused pool, never a thread). A worker that doesn't finish within
--battery-timeout-seconds is genuinely TERMINATED (TerminateProcess on
Windows, unblockable even mid native-BLAS-call -- a thread-based timeout
can only stop WAITING, never stop the computation, which is the leading
suspect for a real run's segfault after its thread-based guard fired: the
abandoned background thread kept running BLAS concurrently with the main
thread's next steps). BLAS is pinned to single-threaded in every worker
(OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1, set in the parent before any
worker spawns so the environment is already in place when a worker's own
`import numpy` runs) -- parallelism comes from the process pool, never
from BLAS. Every stage boundary is logged with a timestamp and flushed
immediately (Python fully-buffers stdout by default when not attached to a
terminal, e.g. `python script.py > log.txt` -- a real, confirmed
contributor to "header then silence for hours" independent of whether
anything was actually hung). A pre-run cost gate times one real unit of
work (one subspace fit, the complete metric stack, a few bootstrap
resamples) on each battery's own actual data and refuses to start if the
projected total exceeds --wall-clock-budget. See --smoke for an end-to-end
dry run on tiny synthetic data.

Usage:
    python scripts/run_reliance_battery.py --layer 9
    python scripts/run_reliance_battery.py --layer 9 --layer-mode checkpoint-band
    python scripts/run_reliance_battery.py --layer 9 --w-metrics off
    python scripts/run_reliance_battery.py --smoke

Do NOT run against the real embedding cache from this repo checkout -- it
lives on the collaborator machine (see --cache-root's default). This script
is exercised here only via tests/test_reliance_battery.py's synthetic
fixtures and --smoke.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import time
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
# Wall-clock budget for each of a battery's four worker-process tasks (lda
# crossfit, probe crossfit, headline bootstrap, rank-sensitivity sweep) --
# an overnight run once stalled inside the first battery (constant memory,
# no error, log frozen); see _run_battery_tasks. 30 min is generous for a
# real 70k-row battery at the default --ranks/--n-boot, short enough that
# one intractable cell can't eat an overnight run's whole budget.
DEFAULT_BATTERY_TIMEOUT_SECONDS = 1800.0
# Stratified per-factor-level row cap (item 5a, repair brief): nothing in
# the battery may scale unboundedly with corpus size. Applied once, upfront,
# so subspace fitting, probe training, LEACE/INLP, and the bootstrap all see
# the same bounded data.
DEFAULT_MAX_ROWS_PER_LEVEL = 2000
# Pre-run cost gate default budget (item 4, repair brief): ~2h, generous for
# a real multi-battery grid, short enough that an infeasible configuration
# is caught before it can consume an unattended multi-hour window.
DEFAULT_WALL_CLOCK_BUDGET_SECONDS = 7200.0

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
# Heartbeat logging: timestamped, immediately flushed. Python fully-buffers
# stdout by default when not attached to a terminal (e.g. `python script.py
# > log.txt`) -- a real, confirmed contributor to "header then silence for
# hours" independent of whether the underlying computation was actually
# hung. "Header then nothing" must be structurally impossible.
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _log(msg: str) -> None:
    print(f"[{_timestamp()}] {msg}", flush=True)


def _process_rss_mb() -> float | None:
    """Resident set size of THIS process, in MB -- best-effort, Windows-only
    (ctypes + psapi.dll; psutil is not a project dependency, so this avoids
    adding one). Never raises; returns None if unavailable."""
    try:
        import ctypes
        import ctypes.wintypes as wintypes

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        # Explicit argtypes/restype required: without them, ctypes' default
        # (32-bit int) argument marshalling truncates the 64-bit HANDLE on
        # 64-bit Windows and the call silently fails (returns 0) -- caught
        # directly, not guessed (verified: GetProcessMemoryInfo returns 0
        # without this, 1 with it).
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCounters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        handle = kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return round(counters.WorkingSetSize / (1024 * 1024), 1)
    except Exception:
        pass
    return None


def _set_single_threaded_blas_env() -> None:
    """Pin BLAS/OpenMP thread pools to 1. MUST be called in the PARENT
    process before any worker is spawned, never inside a worker after it
    has already imported numpy: OpenBLAS/MKL read these at library-LOAD
    time, and under Windows `spawn`, a worker re-imports this whole module
    (hence numpy) to locate its target function -- by the time any code
    inside the worker could run, numpy/BLAS are already loaded, too late
    to matter. Setting os.environ here, before Process() is ever called,
    works because a spawned child's OS-level environment is inherited from
    the parent's os.environ AT SPAWN TIME, before the child's own `import
    numpy` runs. Parallelism comes from the process pool (multiple worker
    processes), never from BLAS multithreading within one -- concurrent
    multithreaded BLAS calls (the old thread-based guard's abandoned
    background thread running concurrently with the main thread's next
    steps) is the leading suspect for a real run's segfault after its
    timeout fired.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


# ---------------------------------------------------------------------------
# Row capping: nothing in the battery may scale unboundedly with corpus size
# ---------------------------------------------------------------------------


def cap_rows_per_level(
    Z: np.ndarray, factor: np.ndarray, y: np.ndarray, groups: np.ndarray, max_rows_per_level: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified, seeded row cap: at most `max_rows_per_level` rows per
    factor level (a 25,000-row level and a 5,000-row level both get capped
    to the same ceiling). Every level with >= 1 row is preserved; a level
    already at or under the cap is untouched. Applied once, upfront, so the
    SAME bounded data flows through subspace fitting, probe training,
    LEACE/INLP, and the bootstrap -- capping only the subspace-fitting step
    would still leave every other stage scaling with the original row
    count.
    """
    rng = np.random.default_rng(seed)
    keep_idx = []
    for lvl in np.unique(factor):
        idx = np.where(factor == lvl)[0]
        if len(idx) > max_rows_per_level:
            idx = rng.choice(idx, size=max_rows_per_level, replace=False)
        keep_idx.append(idx)
    keep_idx = np.sort(np.concatenate(keep_idx))
    return Z[keep_idx], factor[keep_idx], y[keep_idx], groups[keep_idx]


# ---------------------------------------------------------------------------
# Subprocess isolation: each battery's tasks run in their own freshly-
# spawned worker process, never a shared/reused pool, never a thread. A
# task that doesn't finish within its timeout is genuinely terminated --
# unlike a thread-based guard, this works even if the worker is stuck
# inside native BLAS code, and a crashing worker (segfault) can never take
# down this process.
# ---------------------------------------------------------------------------


def _write_battery_npz(path: Path, Z: np.ndarray, factor: np.ndarray, y: np.ndarray, groups: np.ndarray) -> None:
    """Written once per battery; every worker process spawned for that
    battery's tasks reads it from disk exactly once at task start, instead
    of the (up to tens of thousands x 1024) battery matrix being pickled
    per task. factor/groups cast to a fixed-width string dtype (never
    allow_pickle=True, consistent with this project's cache-loading
    convention) since they may be pandas-derived object arrays."""
    np.savez(
        path,
        Z=np.asarray(Z, dtype=np.float32),
        factor=np.asarray(factor, dtype=str),
        y=np.asarray(y, dtype=np.int64),
        groups=np.asarray(groups, dtype=str),
    )


def _load_battery_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return data["Z"], data["factor"], data["y"], data["groups"]


def _crossfit_worker_task(
    Z: np.ndarray, factor: np.ndarray, y: np.ndarray, groups: np.ndarray, *,
    estimator: str, seed: int, valid_ranks: list[int], n_outer: int,
    checkpoints: dict, layer_mode: str, w_metrics_enabled: bool, w_metrics_reason: str,
) -> dict:
    """Runs in a worker process: one estimator's full run_nested_crossfit
    call. fit_subspace/effect_fn are reconstructed here from picklable
    primitives (never pickled as closures themselves -- that's not
    supported across a process boundary; only the ingredients are)."""
    fit_subspace = _make_fit_subspace(estimator, seed)
    candidates = [{"k": r} for r in valid_ranks]
    effect_fn = _make_effect_fn(checkpoints, seed, n_random=20, layer_mode=layer_mode,
                                 w_metrics_enabled=w_metrics_enabled, w_metrics_reason=w_metrics_reason)
    fold_results = run_nested_crossfit(
        Z, factor, y, groups, candidates, fit_subspace, factor_separation_score, effect_fn,
        n_outer_splits=n_outer, n_inner_splits=min(3, n_outer), seed=seed,
    )
    return dict(fold_results=fold_results)


def _bootstrap_worker_task(
    Z: np.ndarray, factor: np.ndarray, y: np.ndarray, groups: np.ndarray, *,
    headline_rank: int, headline_metric_name: str, checkpoints: dict, n_boot: int, seed: int,
) -> dict:
    """Runs in a worker process: the headline grouped-bootstrap CI."""
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

    return grouped_bootstrap_ci(_headline_metric, groups, n_boot=n_boot, seed=seed)


def _rank_curve_worker_task(
    Z: np.ndarray, factor: np.ndarray, y: np.ndarray, groups: np.ndarray, *,
    valid_ranks: list[int], headline_metric_name: str, checkpoints: dict,
) -> dict:
    """Runs in a worker process: the rank-sensitivity sweep. `groups` is
    unused here but every worker task shares the same (Z, factor, y,
    groups, **kwargs) call shape for a uniform dispatch table."""
    def _metric_at_rank(k):
        U = lda_subspace(Z, factor, y, k=k, mode="within_class")
        if U.shape[1] == 0:
            return float("nan")
        if headline_metric_name == "r_var":
            Sigma = np.atleast_2d(np.cov(Z, rowvar=False))
            return float(np.mean([r_var(ck["w"], U, Sigma) for ck in checkpoints.values()]))
        return factor_separation_score(Z, factor, y, U)

    return rank_sensitivity_curve(_metric_at_rank, valid_ranks)


_WORKER_TASKS = {
    "crossfit": _crossfit_worker_task,
    "bootstrap": _bootstrap_worker_task,
    "rank_curve": _rank_curve_worker_task,
}

# Test-only hang injection: set to a comma-separated list of task labels
# (e.g. "crossfit_lda,crossfit_probe" -- see _run_battery_tasks for the
# label scheme) to make ONLY those specific workers deliberately never
# return, so the timeout-and-kill path (and "one hung cell doesn't block
# its siblings") can be exercised for real without waiting on a genuinely
# pathological fit. Read via os.environ (not a module-level flag)
# specifically because it must cross the process boundary into a
# freshly-spawned worker, which a monkeypatch cannot.
_TEST_HANG_ENV_VAR = "_RADAR_RELIANCE_TEST_HANG_TASK_LABELS"


def _worker_entrypoint(label: str, task_kind: str, npz_path: str, task_kwargs: dict, result_queue) -> None:
    """Runs in a freshly-spawned child process -- one process per task,
    never a shared/reused pool, so a wedged or crashing worker can never
    affect any other task. Loads the battery matrix from disk (never
    pickled as a task argument -- see _write_battery_npz) and dispatches to
    the requested task, putting the result (or a string description of any
    exception -- exception OBJECTS are not always picklable) onto
    result_queue."""
    if label in os.environ.get(_TEST_HANG_ENV_VAR, "").split(","):
        time.sleep(3600)  # deliberately never returns in time for any real timeout
        return
    try:
        Z, factor, y, groups = _load_battery_npz(Path(npz_path))
        task_fn = _WORKER_TASKS[task_kind]
        result = task_fn(Z, factor, y, groups, **task_kwargs)
        result_queue.put(("ok", result))
    except Exception as e:
        result_queue.put(("error", f"{type(e).__name__}: {e}"))


def _run_battery_tasks(
    tasks: list[tuple[str, str, dict]], npz_path: Path, timeout: float, log: Callable[[str], None],
) -> dict[str, dict]:
    """Runs every task CONCURRENTLY, each as its own freshly-spawned
    process (parallelism comes from the process pool, never from BLAS),
    sharing ONE wall-clock deadline. A task that doesn't finish in time is
    genuinely killed (TerminateProcess on Windows, unblockable even if the
    process is stuck inside native BLAS code -- unlike a thread-based
    guard, which can only stop WAITING, never stop the computation itself)
    rather than left running concurrently with the parent's next steps.

    Args:
        tasks: list of (label, task_kind, task_kwargs). `label` is the
            unique per-task identifier used for the returned dict's keys
            and for logging/JSON "stage" naming; `task_kind` selects which
            worker function runs (see _WORKER_TASKS) -- two tasks (e.g. the
            lda and probe crossfit calls) can share a task_kind under
            different labels.

    Returns:
        {label: result_dict}. Every result dict carries "status"
        ("ok"/"failed") and, on failure, "timed_out" (bool) and "stage"
        (== label) and "error" -- recorded in the JSON, never silently
        dropped, per the repair brief.
    """
    ctx = mp.get_context("spawn")
    procs: dict[str, "mp.process.BaseProcess"] = {}
    queues: dict[str, "mp.queues.Queue"] = {}
    for label, task_kind, task_kwargs in tasks:
        q = ctx.Queue()
        p = ctx.Process(target=_worker_entrypoint, args=(label, task_kind, str(npz_path), task_kwargs, q))
        p.start()
        log(f"worker started: stage={label} pid={p.pid}")
        procs[label] = p
        queues[label] = q

    deadline = time.monotonic() + timeout
    results: dict[str, dict] = {}
    for label, p in procs.items():
        remaining = max(0.0, deadline - time.monotonic())
        p.join(timeout=remaining)

        if p.is_alive():
            log(f"worker TIMED OUT: stage={label} pid={p.pid} after {timeout}s -- terminating")
            p.terminate()
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
                p.join()
            results[label] = dict(status="failed", error=f"timed out after {timeout}s",
                                   timed_out=True, stage=label)
            continue

        if p.exitcode != 0:
            log(f"worker CRASHED: stage={label} pid={p.pid} exitcode={p.exitcode}")
            results[label] = dict(
                status="failed",
                error=f"worker process exited with code {p.exitcode} (native crash, not a Python exception)",
                timed_out=False, stage=label,
            )
            continue

        try:
            outcome, payload = queues[label].get_nowait()
        except Exception:
            results[label] = dict(status="failed", error="worker exited cleanly but returned no result",
                                   timed_out=False, stage=label)
            continue

        if outcome == "error":
            log(f"worker FAILED: stage={label} pid={p.pid}: {payload}")
            results[label] = dict(status="failed", error=payload, timed_out=False, stage=label)
        else:
            out = dict(payload)
            out.setdefault("status", "ok")
            out.setdefault("timed_out", False)
            log(f"worker done: stage={label} pid={p.pid}")
            results[label] = out
        queues[label].close()
    return results


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
    max_rows_per_level: int | None = DEFAULT_MAX_ROWS_PER_LEVEL,
    log: Callable[[str], None] | None = None,
) -> dict:
    log = log or (lambda msg: _log(f"[{spec['name']}] {msg}"))

    if max_rows_per_level is not None:
        n_before = len(y)
        Z, factor, y, groups = cap_rows_per_level(Z, factor, y, groups, max_rows_per_level, seed)
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
        layer_mode=layer_mode,
    )
    if grouping_degenerate:
        log(f"WARN grouping column == factor column ({spec['grouping']!r}) -- grouping_degenerate=true")
    if not valid_ranks:
        result["skipped"] = f"no requested rank <= n_levels-1={n_levels - 1}"
        return result

    n_outer = min(5, n_groups)
    if n_outer < 2:
        result["skipped"] = f"only {n_groups} group(s) available -- cannot form nested folds"
        return result

    log(f"matrix assembled: Z.shape={Z.shape} dtype={Z.dtype} n_levels={n_levels} n_groups={n_groups} "
        f"n_outer={n_outer} valid_ranks={valid_ranks} rss_mb={_process_rss_mb()}")

    headline_rank = valid_ranks[0]
    # r_var needs w; when w-metrics are disabled, this must not silently
    # crash -- fall back to a checkpoint-independent factor-only metric and
    # record which one was actually used.
    headline_metric_name = "r_var" if (w_metrics_enabled and checkpoints) else "factor_separation_score"

    tasks: list[tuple[str, str, dict]] = []
    for estimator in ("lda", "probe"):
        tasks.append((f"crossfit_{estimator}", "crossfit", dict(
            estimator=estimator, seed=seed, valid_ranks=valid_ranks, n_outer=n_outer,
            checkpoints=checkpoints, layer_mode=layer_mode,
            w_metrics_enabled=w_metrics_enabled, w_metrics_reason=w_metrics_reason,
        )))
    if n_boot > 0:
        tasks.append(("bootstrap", "bootstrap", dict(
            headline_rank=headline_rank, headline_metric_name=headline_metric_name,
            checkpoints=checkpoints, n_boot=n_boot, seed=seed,
        )))
    tasks.append(("rank_curve", "rank_curve", dict(
        valid_ranks=valid_ranks, headline_metric_name=headline_metric_name, checkpoints=checkpoints,
    )))

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"reliance_{spec['name']}_"))
    npz_path = tmp_dir / "battery_data.npz"
    try:
        _write_battery_npz(npz_path, Z, factor, y, groups)
        log(f"dispatching {len(tasks)} worker processes: {[label for label, _, _ in tasks]} "
            f"(timeout={battery_timeout_seconds}s each)")
        task_results = _run_battery_tasks(tasks, npz_path, battery_timeout_seconds, log)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    estimators: dict = {}
    for estimator in ("lda", "probe"):
        task_result = task_results[f"crossfit_{estimator}"]
        if task_result.get("status") == "failed":
            log(f"WARN estimator={estimator} failed ({task_result.get('error')}); recording as failed, "
                f"not crashing the run")
            # fold_results must stay present (empty) on failure: consumers
            # -- summarize_prereg_candidate in particular -- index
            # estimators[...]["fold_results"] unconditionally.
            task_result.setdefault("fold_results", [])
            estimators[estimator] = task_result
            continue

        fold_results = task_result["fold_results"]
        # checkpoint-band mode's per-fold band-pooled alignment recompute
        # stays in this (parent) process -- it's a light, exception-guarded
        # post-processing pass over an already-completed crossfit result,
        # not a repeat of the expensive fitting work the timeout guard
        # exists for, and it isn't the scenario that produced the
        # unrepeatable stalls this fix addresses (those runs used
        # --layer-mode fixed, the default).
        if layer_mode == "checkpoint-band" and Z_full is not None and checkpoints and w_metrics_enabled:
            fit_subspace = _make_fit_subspace(estimator, seed)
            try:
                _recompute_band_alignment(fold_results, Z_full, factor, y, groups, checkpoints,
                                           fit_subspace, n_outer, seed)
            except Exception as e:
                log(f"WARN estimator={estimator} band-alignment recompute failed ({e})")

        estimators[estimator] = task_result

    if n_boot > 0:
        bootstrap = task_results["bootstrap"]
    else:
        bootstrap = dict(mean=float("nan"), std=float("nan"), lo=float("nan"), hi=float("nan"),
                          n_boot=0, n_groups=n_groups, n_finite=0, n_boot_failed=0,
                          status="skipped", timed_out=False, note="--n-boot 0: point-estimate-only pass")
    rank_curve = task_results["rank_curve"]

    result["estimators"] = estimators
    result["headline_bootstrap"] = dict(metric=headline_metric_name, rank=headline_rank, **bootstrap)
    result["rank_sensitivity"] = dict(metric=headline_metric_name, **rank_curve)
    estimator_statuses = {k: v.get("status") for k, v in estimators.items()}
    log(f"battery complete: estimators={estimator_statuses} "
        f"bootstrap_status={bootstrap.get('status')} rank_curve_status={rank_curve.get('status')}")
    return result


# ---------------------------------------------------------------------------
# prereg_candidates summary
# ---------------------------------------------------------------------------


def summarize_prereg_candidate(battery_result: dict) -> dict:
    if battery_result.get("skipped"):
        return dict(name=battery_result["name"], skipped=battery_result["skipped"])
    if battery_result.get("failed"):
        # An unexpected exception from run_battery's own orchestration (not
        # a worker timeout/crash -- those already degrade to status="failed"
        # inside estimators/headline_bootstrap/rank_sensitivity without ever
        # reaching this point). Same "nothing to summarize, here's why"
        # shape as the skipped case.
        return dict(name=battery_result["name"], skipped=battery_result["failed"])

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
# Pre-run cost gate: time ONE real unit of work on THIS run's actual data,
# project the total, refuse to start if it exceeds the budget. Feasibility
# arithmetic is never again done by a human in chat after the fact -- four
# real attempts were launched on nothing but a partial, since-falsified
# cost model (see step3_phaseA_repair_brief.md / this module's docstring).
# ---------------------------------------------------------------------------


def run_cost_probe(
    Z: np.ndarray, factor: np.ndarray, y: np.ndarray, groups: np.ndarray,
    checkpoints: dict, valid_ranks: list[int], n_outer: int, layer_mode: str,
    w_metrics_enabled: bool, w_metrics_reason: str, seed: int, n_boot_sample: int = 3,
) -> dict:
    """Times one real (estimator, rank, fold) fit PLUS the complete metric
    stack (LEACE/INLP/controls -- previously never measured; the biggest
    gap in every prior cost estimate) on THIS battery's own actual
    (already row-capped) data, plus a few bootstrap-style resamples.
    Returns wall-clock seconds for each piece; project_battery_cost turns
    these into a full-grid projection.
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    eff_size = max(10, min(n, n // n_outer))
    eff_idx = rng.choice(n, size=eff_size, replace=False)

    fit_subspace = _make_fit_subspace("lda", seed)
    effect_fn = _make_effect_fn(checkpoints, seed, n_random=20, layer_mode=layer_mode,
                                 w_metrics_enabled=w_metrics_enabled, w_metrics_reason=w_metrics_reason)

    t0 = time.monotonic()
    U = fit_subspace(Z, factor, y, groups, k=valid_ranks[0])
    t_fit = time.monotonic() - t0

    t0 = time.monotonic()
    effect_fn(Z[eff_idx], factor[eff_idx], y[eff_idx], U)
    t_effect = time.monotonic() - t0

    t0 = time.monotonic()
    n_ok = 0
    for _ in range(n_boot_sample):
        boot_idx = rng.integers(0, n, size=n)
        try:
            fit_subspace(Z[boot_idx], factor[boot_idx], y[boot_idx], groups[boot_idx], k=valid_ranks[0])
            n_ok += 1
        except ValueError:
            continue
    t_boot_total = time.monotonic() - t0
    t_boot_per_resample = t_boot_total / max(n_ok, 1)

    return dict(t_fit_seconds=t_fit, t_effect_seconds=t_effect, t_boot_per_resample_seconds=t_boot_per_resample)


def project_battery_cost(cost_probe: dict, n_candidates: int, n_outer: int, n_boot: int, n_ranks: int) -> float:
    """Projected wall-clock seconds for one battery's full grid, given a
    cost_probe measured on that battery's own data. A projection, not a
    guarantee -- e.g. the probe estimator's own internal cross-fitting
    scales differently than lda's single eigh solve, so this is
    necessarily approximate (the estimate is built from lda's cost, since
    the probe is run alongside it, not instead of it, so undercounting the
    probe's share is the main known bias -- flagged, not hidden)."""
    t_fit = cost_probe["t_fit_seconds"]
    t_effect = cost_probe["t_effect_seconds"]
    t_boot = cost_probe["t_boot_per_resample_seconds"]
    # crossfit: (n_candidates + 1 final refit) fits + 1 effect_fn call, per
    # outer fold, per estimator (2: lda + probe -- both run, so x2 even
    # though the probe's own cost is approximated from lda's here).
    crossfit_seconds = 2 * n_outer * ((n_candidates + 1) * t_fit + t_effect)
    bootstrap_seconds = n_boot * t_boot
    rank_curve_seconds = n_ranks * t_fit
    return crossfit_seconds + bootstrap_seconds + rank_curve_seconds


def run_cost_gate(
    corpus_data: dict, batteries: list[dict], checkpoints: dict, w_metrics: dict, args, log: Callable[[str], None],
) -> None:
    """Times one real unit of work per requested battery (on its own,
    already row-capped data) and projects the total wall-clock cost of the
    full grid. Raises SystemExit BEFORE any long work if the projection
    exceeds --wall-clock-budget, unless --force-run overrides it.
    """
    log(f"cost gate: probing {len(batteries)} batter{'y' if len(batteries) == 1 else 'ies'} "
        f"(wall_clock_budget={args.wall_clock_budget}s)")
    total_projected = 0.0
    per_battery: dict[str, float] = {}
    for spec in batteries:
        cd = corpus_data[spec["corpus"]]
        Z, factor, y, groups, _ = select_battery_rows(cd["df"], cd["emb"], spec, emb_full=cd["emb_full"])
        if len(y) == 0:
            continue
        if args.max_rows_per_level is not None:
            Z, factor, y, groups = cap_rows_per_level(Z, factor, y, groups, args.max_rows_per_level, args.seed)

        n_levels = int(len(np.unique(factor)))
        valid_ranks = ranks_for_n_levels(list(args.ranks), n_levels)
        n_groups = int(len(np.unique(groups)))
        if not valid_ranks or min(5, n_groups) < 2:
            continue  # run_battery will report this battery as skipped -- nothing to cost-gate
        n_outer = min(5, n_groups)

        probe = run_cost_probe(Z, factor, y, groups, checkpoints, valid_ranks, n_outer, args.layer_mode,
                                w_metrics["enabled"], w_metrics["reason"], args.seed)
        projected = project_battery_cost(probe, len(valid_ranks), n_outer, args.n_boot, len(valid_ranks))
        per_battery[spec["name"]] = projected
        total_projected += projected
        log(f"cost probe {spec['name']}: t_fit={probe['t_fit_seconds']:.2f}s t_effect={probe['t_effect_seconds']:.2f}s "
            f"t_boot={probe['t_boot_per_resample_seconds']:.2f}s -> projected {projected:.0f}s ({projected / 60:.1f} min)")

    log(f"cost gate: projected total {total_projected:.0f}s ({total_projected / 3600:.2f}h) across "
        f"{len(per_battery)} batteries; budget {args.wall_clock_budget:.0f}s ({args.wall_clock_budget / 3600:.2f}h)")
    for name, secs in per_battery.items():
        log(f"  {name}: {secs:.0f}s ({secs / 60:.1f} min)")

    if total_projected > args.wall_clock_budget and not args.force_run:
        raise SystemExit(
            f"[cost gate] projected total {total_projected:.0f}s ({total_projected / 3600:.2f}h) exceeds "
            f"--wall-clock-budget {args.wall_clock_budget:.0f}s ({args.wall_clock_budget / 3600:.2f}h). "
            f"Refusing to start -- reduce --ranks/--n-boot/--max-rows-per-level, raise "
            f"--wall-clock-budget, or pass --force-run to proceed anyway."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _battery_output_path(out_path: Path, battery_name: str) -> Path:
    return out_path.with_name(f"{out_path.stem}_{battery_name}{out_path.suffix}")


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Temp file + os.replace -- a reader never sees a partially-written
    file, and a crash mid-write never corrupts whatever was there before."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    os.replace(tmp_path, path)


def _checkpoints_summary(checkpoints: dict[str, dict]) -> dict:
    return {
        run: dict(ckpt_layer_center=ck["ckpt_layer_center"], ckpt_layer_band=ck["ckpt_layer_band"],
                  layer_pooling=ck["layer_pooling"], band_weights=ck["band_weights"],
                  w_layer_mismatch=ck["w_layer_mismatch"], w_dim=ck["w_dim"], w_dim_mismatch=ck["w_dim_mismatch"])
        for run, ck in checkpoints.items()
    }


def run_smoke_test(out_path: Path, seed: int = 13, battery_timeout_seconds: float = 60.0) -> dict:
    """--smoke: a tiny, entirely-synthetic, seeded battery through the
    ENTIRE real path -- the worker-process pool, timeouts, the complete
    metric stack, a small bootstrap, JSON write -- with no real cache/
    manifest/checkpoint files needed. This calls the real run_battery, not
    a mock of it, so it exercises exactly the machinery a real run depends
    on; finishes in well under a minute.
    """
    rng = np.random.default_rng(seed)
    d, n, k_factor, n_groups, n_levels = 64, 200, 2, 20, 3

    w_true = rng.normal(size=d)
    w_true /= np.linalg.norm(w_true)
    M = rng.normal(size=(d, k_factor))
    M = M - np.outer(w_true, w_true @ M)
    U_true, _, _ = np.linalg.svd(M, full_matrices=False)
    U_true = U_true[:, :k_factor]

    groups_raw = rng.integers(0, n_groups, size=n)
    group_offset = rng.normal(scale=0.5, size=(n_groups, d))[groups_raw]
    y = rng.integers(0, 2, size=n)
    factor_levels = rng.integers(0, n_levels, size=n)
    raw_centers = rng.normal(size=(n_levels, k_factor))
    raw_centers -= raw_centers.mean(axis=0, keepdims=True)
    Uc, Sc, Vtc = np.linalg.svd(raw_centers, full_matrices=False)
    factor_centers = Uc @ np.diag(np.full_like(Sc, 3.0)) @ Vtc

    Z = (np.outer((y * 2 - 1).astype(float), w_true) * 3.0
         + factor_centers[factor_levels] @ U_true.T + group_offset + rng.normal(size=(n, d)))
    factor = np.array([f"gen{i}" for i in factor_levels], dtype=object)
    groups = np.array([f"grp{i}" for i in groups_raw], dtype=object)

    checkpoints = {
        "smoke_ckpt": dict(w=w_true, b=0.0, ckpt_layer_center=9, ckpt_layer_band=[8, 11],
                           layer_pooling="fixed_layer", band_weights=None, w_layer_mismatch=False,
                           w_dim=d, w_dim_mismatch=False),
    }
    w_metrics = dict(enabled=True, reason="smoke test: w matches embedding_dim by construction",
                      w_dim=d, embedding_dim=d)

    spec = dict(name="smoke_battery", corpus="smoke", factor="smoke_factor", grouping="smoke_group")
    _log(f"[smoke] running synthetic battery: n_rows={n} n_levels={n_levels} n_groups={n_groups} d={d}")
    result = run_battery(
        spec, Z, factor, y, groups, checkpoints, ranks=[1, 2], n_boot=20, seed=seed,
        w_metrics_enabled=True, w_metrics_reason="", battery_timeout_seconds=battery_timeout_seconds,
        max_rows_per_level=None,
    )

    output = dict(
        schema_version=SCHEMA_VERSION, git_sha=_git_sha(), timestamp=datetime.now(timezone.utc).isoformat(),
        layer=9, layer_mode="fixed", seed=seed, w_metrics=w_metrics,
        join_stats={"smoke": dict(n_cache=n, n_manifest=n, n_joined=n, n_dropped=0)},
        checkpoints=_checkpoints_summary(checkpoints),
        batteries=[result], prereg_candidates=[summarize_prereg_candidate(result)],
    )
    _write_json_atomic(out_path, output)
    _validate_smoke_schema(output)
    _log(f"[smoke] wrote -> {out_path}")
    return output


def _validate_smoke_schema(output: dict) -> None:
    """Cheap structural check on the written JSON -- not a full schema
    validator, just the fields a downstream consumer would reach for
    first, so a --smoke pass certifies the JSON is usable, not just that
    no exception was raised along the way."""
    required_top = {"schema_version", "git_sha", "timestamp", "w_metrics", "join_stats", "checkpoints",
                     "batteries", "prereg_candidates"}
    missing = required_top - set(output)
    if missing:
        raise AssertionError(f"--smoke JSON missing top-level keys: {missing}")
    if not output["batteries"]:
        raise AssertionError("--smoke JSON has zero batteries")
    battery = output["batteries"][0]
    for key in ("estimators", "headline_bootstrap", "rank_sensitivity"):
        if key not in battery:
            raise AssertionError(f"--smoke battery missing {key!r}")
    for estimator, est_result in battery["estimators"].items():
        if "status" not in est_result:
            raise AssertionError(f"--smoke battery estimator {estimator!r} missing 'status'")


def main(argv=None) -> None:
    # Set BEFORE anything else -- see _set_single_threaded_blas_env's
    # docstring: must happen before any worker process is spawned (a
    # spawned child inherits this process's environment at spawn time),
    # and doing it this early also makes the cost gate's own in-process
    # timing representative of what a (single-threaded) worker will
    # actually see.
    _set_single_threaded_blas_env()

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    ap.add_argument("--manifest-dir", default="manifests/v2")
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--out", default=None)
    ap.add_argument("--corpus", nargs="*", default=None, help="restrict to battery corpora in this list")
    ap.add_argument("--factor", nargs="*", default=None, help="restrict to battery factors in this list")
    ap.add_argument("--ranks", nargs="+", type=int, default=list(DEFAULT_RANKS))
    ap.add_argument("--n-boot", type=int, default=1000, help="0 for a point-estimate-only pass (no bootstrap CI)")
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
        help=f"wall-clock budget for each of a battery's four worker-process tasks (lda crossfit, "
             f"probe crossfit, headline bootstrap, rank-sensitivity sweep). A task that doesn't "
             f"finish in time is genuinely killed and recorded as status='failed', timed_out=true; "
             f"the run continues to the next configuration rather than hanging "
             f"(default: {DEFAULT_BATTERY_TIMEOUT_SECONDS:.0f}s).",
    )
    ap.add_argument(
        "--max-rows-per-level", type=int, default=DEFAULT_MAX_ROWS_PER_LEVEL,
        help=f"stratified, seeded cap on rows PER FACTOR LEVEL, applied once before subspace fitting, "
             f"probe training, LEACE/INLP, and the bootstrap -- nothing in a battery may scale "
             f"unboundedly with corpus size. Pass 0 or a negative value to disable "
             f"(default: {DEFAULT_MAX_ROWS_PER_LEVEL}).",
    )
    ap.add_argument(
        "--wall-clock-budget", type=float, default=DEFAULT_WALL_CLOCK_BUDGET_SECONDS,
        help=f"pre-run cost gate: times one real unit of work per requested battery (on its own actual "
             f"data) and refuses to start if the projected total exceeds this many seconds, unless "
             f"--force-run is also given (default: {DEFAULT_WALL_CLOCK_BUDGET_SECONDS:.0f}s).",
    )
    ap.add_argument("--force-run", action="store_true",
                     help="proceed even if the pre-run cost gate projects the total over --wall-clock-budget.")
    ap.add_argument("--skip-cost-gate", action="store_true",
                     help="skip the pre-run cost gate entirely (still prints nothing, times nothing).")
    ap.add_argument("--smoke", action="store_true",
                     help="run a tiny, entirely-synthetic battery through the full real path (worker "
                          "pool, timeouts, complete metric stack, small bootstrap, JSON write) and "
                          "exit -- no cache/manifest/checkpoint files needed. For a real-data dry run "
                          "before the full grid, see the repair brief's cheapest-battery-first sequence.")
    args = ap.parse_args(argv)

    out_path = Path(args.out) if args.out else Path(f"analysis/step3/reliance_layer{args.layer}.json")
    max_rows_per_level = args.max_rows_per_level if args.max_rows_per_level and args.max_rows_per_level > 0 else None

    if args.smoke:
        # Smoke data is tiny (a few hundred rows) -- it never legitimately
        # needs anywhere near the full battery timeout, and a short budget
        # here means a genuine hang during smoke testing fails fast rather
        # than waiting out --battery-timeout-seconds's (possibly 30+ minute)
        # default. Respects an explicit --battery-timeout-seconds override.
        smoke_timeout = (
            args.battery_timeout_seconds if args.battery_timeout_seconds != DEFAULT_BATTERY_TIMEOUT_SECONDS else 60.0
        )
        run_smoke_test(out_path, seed=args.seed, battery_timeout_seconds=smoke_timeout)
        return

    cache_root = Path(args.cache_root)
    manifest_dir = Path(args.manifest_dir)

    batteries = [
        b for b in BATTERIES
        if (args.corpus is None or b["corpus"] in args.corpus)
        and (args.factor is None or b["factor"] in args.factor)
    ]
    if not batteries:
        raise ValueError(f"--corpus/--factor filters matched 0 of {len(BATTERIES)} batteries")

    _log(f"[run_reliance_battery] layer={args.layer} layer_mode={args.layer_mode} ranks={args.ranks} "
         f"seed={args.seed} max_rows_per_level={max_rows_per_level} batteries={[b['name'] for b in batteries]}")

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
        _log(f"[join] {corpus}: n_cache={stats['n_cache']} n_manifest={stats['n_manifest']} "
             f"n_joined={stats['n_joined']} n_dropped={stats['n_dropped']}")

    checkpoints = load_all_checkpoints(Path(args.ckpt_dir), RUNS, args.layer,
                                        layer_mode=args.layer_mode, num_cache_layers=num_cache_layers,
                                        w_metrics_mode=args.w_metrics, embedding_dim=embedding_dim)
    if not checkpoints:
        _log("[WARN] no checkpoints loaded -- w-dependent metrics will be empty")

    w_metrics = resolve_w_metrics(checkpoints, args.w_metrics, embedding_dim)
    _log(f"[w-metrics] enabled={w_metrics['enabled']} w_dim={w_metrics['w_dim']} "
         f"embedding_dim={w_metrics['embedding_dim']} reason={w_metrics['reason']!r}")

    if not args.skip_cost_gate:
        run_cost_gate(corpus_data, batteries, checkpoints, w_metrics, args, _log)

    git_sha = _git_sha()
    checkpoints_summary = _checkpoints_summary(checkpoints)
    battery_results = []
    for spec in batteries:
        cd = corpus_data[spec["corpus"]]
        Z, factor, y, groups, Z_full = select_battery_rows(cd["df"], cd["emb"], spec, emb_full=cd["emb_full"])

        _log(f"[battery] {spec['name']}: n_rows={len(y)} n_levels={len(np.unique(factor))}")
        try:
            res = run_battery(spec, Z, factor, y, groups, checkpoints, args.ranks, args.n_boot, args.seed,
                               layer_mode=args.layer_mode, Z_full=Z_full,
                               w_metrics_enabled=w_metrics["enabled"], w_metrics_reason=w_metrics["reason"],
                               battery_timeout_seconds=args.battery_timeout_seconds,
                               max_rows_per_level=max_rows_per_level)
        except Exception as e:
            # A battery's own worker-timeout/crash paths already degrade to
            # status="failed" without raising; this is the backstop for a
            # genuinely unexpected failure in run_battery's own orchestration
            # -- it must not prevent every OTHER battery (already written to
            # disk, or not yet attempted) from being processed.
            _log(f"[battery] {spec['name']}: UNEXPECTED FAILURE ({type(e).__name__}: {e}) -- "
                 f"recording as failed, continuing to the next battery")
            res = dict(name=spec["name"], corpus=spec["corpus"], factor=spec["factor"], grouping=spec["grouping"],
                       failed=f"{type(e).__name__}: {e}")
        battery_results.append(res)

        # Written the moment this battery completes (success, failure, or
        # timeout): a later battery's failure must never erase this.
        battery_path = _battery_output_path(out_path, spec["name"])
        _write_json_atomic(battery_path, dict(
            schema_version=SCHEMA_VERSION, git_sha=git_sha, timestamp=datetime.now(timezone.utc).isoformat(),
            layer=args.layer, layer_mode=args.layer_mode, seed=args.seed, w_metrics=w_metrics,
            join_stats={spec["corpus"]: join_stats[spec["corpus"]]}, checkpoints=checkpoints_summary,
            battery=res, prereg_candidate=summarize_prereg_candidate(res),
        ))
        _log(f"[battery] {spec['name']}: wrote -> {battery_path}")

    prereg = [summarize_prereg_candidate(r) for r in battery_results]

    manifest = dict(
        schema_version=SCHEMA_VERSION,
        git_sha=git_sha,
        timestamp=datetime.now(timezone.utc).isoformat(),
        layer=args.layer,
        layer_mode=args.layer_mode,
        seed=args.seed,
        w_metrics=w_metrics,
        join_stats=join_stats,
        checkpoints=checkpoints_summary,
        battery_files={r["name"]: str(_battery_output_path(out_path, r["name"])) for r in battery_results},
        batteries=battery_results,
        prereg_candidates=prereg,
    )
    _write_json_atomic(out_path, manifest)

    print("\n=== SUMMARY ===")
    for r in prereg:
        if "skipped" in r:
            print(f"  {r['name']}: SKIPPED ({r['skipped']})")
            continue
        print(f"  {r['name']}: headline_metric={r['headline_metric']} n_groups={r['n_groups']} "
              f"degenerate={r['grouping_degenerate']} estimators_agree_sign={r['estimators_agree_sign']} "
              f"cis_overlap={r['cis_overlap']} stable_ranks={r['stable_rank_window']}")
    print(f"\nwrote manifest -> {out_path}")
    print(f"EXIT: 0")


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
