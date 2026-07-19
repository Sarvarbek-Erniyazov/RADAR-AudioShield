"""Tests for src/audioshield/reliance/crossfit.py -- the nested selection/effect
harness must never let a group straddle both sides of a fold (leakage canary)."""
from dataclasses import replace

import numpy as np
import pytest

from audioshield.reliance.crossfit import (
    make_nested_folds,
    assert_no_group_leakage,
    run_nested_crossfit,
    derive_groups,
)
from audioshield.reliance.subspaces import lda_subspace
from audioshield.reliance.metrics import alignment


def _grouped_indices(n, n_groups, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_groups, size=n).astype(str)


# ---------------------------------------------------------------------------
# leakage canary
# ---------------------------------------------------------------------------


def test_make_nested_folds_never_leaks_a_group():
    groups = _grouped_indices(n=800, n_groups=150, seed=1)
    folds = make_nested_folds(n=800, groups=groups, n_splits=5, seed=13)
    assert len(folds) == 5
    for fold in folds:
        assert_no_group_leakage(fold, groups)  # must not raise
        assert set(fold.selection_idx.tolist()).isdisjoint(fold.effect_idx.tolist())
        # every row accounted for exactly once between the two sides
        assert len(fold.selection_idx) + len(fold.effect_idx) == 800


def test_make_nested_folds_covers_every_row_as_effect_exactly_once():
    n = 500
    groups = _grouped_indices(n, n_groups=100, seed=2)
    folds = make_nested_folds(n, groups, n_splits=5, seed=13)
    all_effect = np.concatenate([f.effect_idx for f in folds])
    assert sorted(all_effect.tolist()) == list(range(n))  # partition, no gaps/overlaps


def test_assert_no_group_leakage_catches_row_overlap():
    groups = _grouped_indices(n=200, n_groups=40, seed=3)
    folds = make_nested_folds(200, groups, n_splits=4, seed=13)
    bad = replace(folds[0], effect_idx=np.concatenate([folds[0].effect_idx, folds[0].selection_idx[:1]]))
    with pytest.raises(AssertionError):
        assert_no_group_leakage(bad, groups)


def test_assert_no_group_leakage_catches_group_straddling():
    """Negative-control canary: construct a fold that's row-disjoint but still
    lets one group's OTHER rows sit on the other side -- must still be caught."""
    groups = np.array(["gA"] * 4 + ["gB"] * 4)
    from audioshield.reliance.crossfit import Fold
    # gA appears in both selection (idx 0,1) and effect (idx 2,3) -- rows are
    # disjoint but the GROUP straddles, which is exactly what the canary exists for.
    bad_fold = Fold(fold_id=0, selection_idx=np.array([0, 1, 4, 5]), effect_idx=np.array([2, 3, 6, 7]))
    with pytest.raises(AssertionError):
        assert_no_group_leakage(bad_fold, groups)


def test_make_nested_folds_requires_at_least_two_groups():
    with pytest.raises(ValueError):
        make_nested_folds(10, np.array(["only_one"] * 10), n_splits=5)


def test_derive_groups_falls_back_to_unique_per_row_when_no_meta():
    g = derive_groups(None, 5)
    assert len(set(g.tolist())) == 5  # no grouping info -> conservative unique-per-row


def test_derive_groups_prefers_source_id():
    meta = dict(source_id=np.array(["s1", "s1", "s2", "s2"]))
    g = derive_groups(meta, 4)
    assert g[0] == g[1] and g[2] == g[3] and g[0] != g[2]


# ---------------------------------------------------------------------------
# harness never lets selection see effect data (structural, via run_nested_crossfit)
# ---------------------------------------------------------------------------


def test_run_nested_crossfit_selection_never_touches_effect_rows(planted_factor_data):
    """Instrument fit_subspace to record every row-index block it is ever handed
    (by content, since indices aren't threaded through the callback signature),
    then confirm none of those blocks contains a row from ANY fold's effect set --
    the direct behavioral guarantee "selection never sees effect data\" promises."""
    d = planted_factor_data
    n = len(d["y"])
    seen_blocks = []

    def fit_subspace(Z, factor, y, groups, k, mode):
        seen_blocks.append(np.round(Z, 8))
        return lda_subspace(Z, factor, y, k=k, mode=mode)

    def score_subspace(Z_val, factor_val, y_val, U):
        return alignment(d["w_true"], U)

    def effect_fn(Z_eff, factor_eff, y_eff, U):
        return alignment(d["w_true"], U)

    candidates = [dict(k=3, mode="within_class")]
    results = run_nested_crossfit(
        d["Z"], d["factor"], d["y"], d["groups"],
        candidates, fit_subspace, score_subspace, effect_fn,
        n_outer_splits=4, n_inner_splits=3, seed=13,
    )
    assert len(results) == 4

    # run_nested_crossfit calls fit_subspace once per candidate (selection scoring)
    # plus one final refit, PER outer fold, in fold order -- chunk seen_blocks
    # accordingly and check each fold's own selection-phase calls only against
    # THAT fold's effect rows (a row legitimately appears in OTHER folds'
    # selection sets; only same-fold selection/effect overlap is the bug).
    calls_per_fold = len(candidates) + 1
    outer_folds = make_nested_folds(n, d["groups"], n_splits=4, seed=13)
    assert len(seen_blocks) == calls_per_fold * len(outer_folds)
    for i, fold in enumerate(outer_folds):
        fold_blocks = seen_blocks[i * calls_per_fold : (i + 1) * calls_per_fold]
        effect_rows = set(map(tuple, np.round(d["Z"][fold.effect_idx], 8)))
        for block in fold_blocks:
            seen_rows = set(map(tuple, block))
            leaked = effect_rows & seen_rows
            assert not leaked, f"fold {fold.fold_id}: {len(leaked)} effect row(s) seen during its own selection phase"


def test_run_nested_crossfit_propagates_leakage_guard(planted_factor_data, monkeypatch):
    """If fold construction were ever broken (e.g. a future refactor swaps
    GroupKFold for a naive KFold), run_nested_crossfit must fail loudly rather
    than silently evaluate effects on leaked data. Simulate that breakage by
    monkeypatching make_nested_folds to return a deliberately-leaky fold, and
    confirm the harness's own internal guard (assert_no_group_leakage, called
    on every outer fold before it is used) catches it."""
    d = planted_factor_data
    n = len(d["y"])

    import audioshield.reliance.crossfit as crossfit_mod

    real_make_nested_folds = crossfit_mod.make_nested_folds

    def fake_make_nested_folds(n_arg, groups_arg, n_splits=5, seed=13):
        folds = real_make_nested_folds(n_arg, groups_arg, n_splits=n_splits, seed=seed)
        if n_arg != n:  # only tamper with the OUTER call (full dataset); inner
            return folds  # selection/validation split must stay untouched/valid
        leaky = replace(folds[0], effect_idx=np.concatenate(
            [folds[0].effect_idx, folds[0].selection_idx[:1]]
        ))
        return [leaky] + folds[1:]

    monkeypatch.setattr(crossfit_mod, "make_nested_folds", fake_make_nested_folds)

    def fit_subspace(Z, factor, y, groups, k, mode):
        return lda_subspace(Z, factor, y, k=k, mode=mode)

    def score_subspace(Z_val, factor_val, y_val, U):
        return alignment(d["w_true"], U)

    def effect_fn(Z_eff, factor_eff, y_eff, U):
        return alignment(d["w_true"], U)

    with pytest.raises(AssertionError):
        run_nested_crossfit(
            d["Z"], d["factor"], d["y"], d["groups"],
            [dict(k=3, mode="within_class")], fit_subspace, score_subspace, effect_fn,
            n_outer_splits=4, n_inner_splits=3, seed=13,
        )


def test_run_nested_crossfit_reports_selected_hyperparameters(planted_factor_data):
    d = planted_factor_data

    def fit_subspace(Z, factor, y, groups, k, mode):
        return lda_subspace(Z, factor, y, k=k, mode=mode)

    def score_subspace(Z_val, factor_val, y_val, U):
        return alignment(d["w_true"], U)

    def effect_fn(Z_eff, factor_eff, y_eff, U):
        return alignment(d["w_true"], U)

    candidates = [dict(k=1, mode="within_class"), dict(k=3, mode="within_class"), dict(k=3, mode="covariate")]
    results = run_nested_crossfit(
        d["Z"], d["factor"], d["y"], d["groups"],
        candidates, fit_subspace, score_subspace, effect_fn,
        n_outer_splits=3, n_inner_splits=3, seed=13,
    )
    assert len(results) == 3
    for r in results:
        assert r["chosen"] in candidates
        assert r["n_selection"] + r["n_effect"] == len(d["y"])
