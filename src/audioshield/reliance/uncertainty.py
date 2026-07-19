"""Uncertainty quantification for the reliance battery: grouped-bootstrap
confidence intervals (resample groups, never rows -- a row-level bootstrap
would treat correlated same-speaker/same-source rows as independent draws and
understate the true variance) and rank-sensitivity curves (how a metric moves
as the subspace rank changes, so a single-rank number is never over-read).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def grouped_bootstrap_ci(
    compute_metric: Callable[[np.ndarray], float],
    groups: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 13,
) -> dict:
    """Bootstrap CI by resampling GROUPS with replacement (not rows).

    Args:
        compute_metric: `(row_idx) -> float`, where `row_idx` is an array of row
            indices (with repeats, since whole groups are resampled with
            replacement) into whatever arrays the caller closed over.
        groups: (n,) grouping key, e.g. source_id/speaker_id.
        n_boot: number of bootstrap resamples.
        ci: central interval width (e.g. 0.95 for a 95% CI).
        seed: RNG seed.

    Returns:
        dict(mean, std, lo, hi, n_boot, n_groups, n_finite, n_boot_failed).
        `lo`/`hi` are the `ci`-coverage percentile interval over the
        bootstrap distribution. A resample whose `compute_metric` call
        raises is caught and skipped (counted in `n_boot_failed`), not
        fatal -- one bad resample must not take down the whole CI, only
        narrow the effective sample it's built from.
    """
    groups = np.asarray(groups, dtype=object)
    uniq_groups = np.unique(groups)
    if len(uniq_groups) < 2:
        raise ValueError(f"need >= 2 groups to bootstrap, got {len(uniq_groups)}")
    group_to_idx = {g: np.where(groups == g)[0] for g in uniq_groups}

    rng = np.random.default_rng(seed)
    boot_vals = np.full(n_boot, np.nan)
    n_boot_failed = 0
    for b in range(n_boot):
        sampled_groups = rng.choice(uniq_groups, size=len(uniq_groups), replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sampled_groups])
        try:
            boot_vals[b] = compute_metric(idx)
        except Exception:
            n_boot_failed += 1
            # boot_vals[b] stays NaN -- excluded below, same as a
            # non-exception NaN compute_metric might legitimately return.

    finite = boot_vals[np.isfinite(boot_vals)]
    if finite.size == 0:
        return dict(mean=float("nan"), std=float("nan"), lo=float("nan"), hi=float("nan"),
                    n_boot=n_boot, n_groups=len(uniq_groups), n_finite=0, n_boot_failed=n_boot_failed)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(finite, [100 * alpha, 100 * (1 - alpha)])
    return dict(
        mean=float(np.mean(finite)), std=float(np.std(finite)),
        lo=float(lo), hi=float(hi),
        n_boot=n_boot, n_groups=len(uniq_groups), n_finite=int(finite.size), n_boot_failed=n_boot_failed,
    )


def rank_sensitivity_curve(
    compute_metric_at_rank: Callable[[int], float],
    ranks: Sequence[int],
) -> dict:
    """Evaluate a metric across candidate subspace ranks, so no single-rank
    number gets over-read. `compute_metric_at_rank` should internally re-fit
    whatever subspace/eraser the metric depends on at the given rank (this
    function is a thin sweep, not a fitter).

    Returns:
        dict(ranks=[...], values=[...]) -- NaN entries mark a rank that failed
        (e.g. requested rank exceeded the data's effective rank); callers should
        filter those before e.g. computing a monotonicity check.
    """
    values = []
    for k in ranks:
        try:
            values.append(float(compute_metric_at_rank(k)))
        except Exception:
            values.append(float("nan"))
    return dict(ranks=list(ranks), values=values)
