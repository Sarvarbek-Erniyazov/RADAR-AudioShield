"""Tests for src/audioshield/reliance/uncertainty.py."""
import numpy as np
import pytest

from audioshield.reliance.uncertainty import grouped_bootstrap_ci, rank_sensitivity_curve
from audioshield.reliance.subspaces import lda_subspace
from audioshield.reliance.metrics import alignment


def test_grouped_bootstrap_ci_contains_point_estimate(planted_factor_data):
    d = planted_factor_data

    def compute_metric(idx):
        U = lda_subspace(d["Z"][idx], d["factor"][idx], d["y"][idx], k=d["k_factor"])
        return alignment(d["w_true"], U)

    out = grouped_bootstrap_ci(compute_metric, d["groups"], n_boot=100, ci=0.95, seed=13)
    assert out["lo"] <= out["mean"] <= out["hi"]
    assert out["n_groups"] == len(set(d["groups"].tolist()))
    assert 0.0 <= out["mean"] <= 1.0 + 1e-6


def test_grouped_bootstrap_ci_resamples_groups_not_rows():
    """A within-group-correlated statistic must show wider bootstrap spread than
    resampling would produce if rows (not groups) were the resampling unit --
    otherwise the implementation is silently doing a row-level bootstrap."""
    rng = np.random.default_rng(0)
    n_groups = 20
    rows_per_group = 25
    group_effects = rng.normal(scale=2.0, size=n_groups)  # strong between-group signal
    groups = np.repeat(np.arange(n_groups), rows_per_group).astype(str)
    values = group_effects[np.repeat(np.arange(n_groups), rows_per_group)] + rng.normal(scale=0.1, size=n_groups * rows_per_group)

    def compute_mean(idx):
        return float(values[idx].mean())

    grouped = grouped_bootstrap_ci(compute_mean, groups, n_boot=500, seed=1)

    # naive row-level bootstrap for comparison
    rng2 = np.random.default_rng(1)
    n = len(values)
    row_boot = [values[rng2.integers(0, n, size=n)].mean() for _ in range(500)]
    row_std = float(np.std(row_boot))

    assert grouped["std"] > row_std * 2, (
        f"grouped bootstrap std ({grouped['std']}) should be much wider than a "
        f"row-level bootstrap's ({row_std}) when the signal is between-group"
    )


def test_grouped_bootstrap_ci_requires_at_least_two_groups():
    with pytest.raises(ValueError):
        grouped_bootstrap_ci(lambda idx: 0.0, np.array(["only_one"] * 5))


def test_grouped_bootstrap_ci_handles_nan_metric_values():
    groups = np.array(["a", "a", "b", "b", "c", "c"])

    def all_nan(idx):
        return float("nan")

    out = grouped_bootstrap_ci(all_nan, groups, n_boot=20, seed=0)
    assert out["n_finite"] == 0
    assert np.isnan(out["mean"])


def test_rank_sensitivity_curve_tracks_metric_across_ranks(planted_factor_data):
    d = planted_factor_data

    def metric_at_rank(k):
        U = lda_subspace(d["Z"], d["factor"], d["y"], k=k)
        return alignment(d["w_true"], U)

    out = rank_sensitivity_curve(metric_at_rank, ranks=[1, 2, 3])
    assert out["ranks"] == [1, 2, 3]
    assert len(out["values"]) == 3
    assert all(np.isfinite(v) for v in out["values"])
    # w_true is orthogonal to the planted factor at every rank -- alignment should
    # stay near zero regardless of rank, not drift as rank grows.
    assert all(v < 0.1 for v in out["values"])


def test_rank_sensitivity_curve_marks_failed_ranks_as_nan():
    def always_fails(k):
        raise ValueError("no such rank")

    out = rank_sensitivity_curve(always_fails, ranks=[1, 2])
    assert all(np.isnan(v) for v in out["values"])
