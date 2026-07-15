"""Nested cross-fitting harness for the reliance battery.

Design invariant (Roadmap v3 Step 3, verbatim): "data used to select subspace
rank/estimator/intervention strength must never evaluate effects." Concretely:
split all rows into K group-disjoint folds (grouped by source_id/speaker_id, so
a recording session or speaker never straddles two folds -- reusing the exact
grouping heuristic the 2a probe protocol already uses, `_derive_groups` from
evaluation.grouped_probe, rather than a second implementation). For fold i,
that fold's rows become the EFFECT set; every other fold's rows become the
SELECTION set. Any hyperparameter choice (subspace rank, estimator, mode,
intervention strength, ...) is made by further splitting SELECTION alone --
EFFECT is never touched until the choice is frozen.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold

from ..evaluation.grouped_probe import _derive_groups


def derive_groups(meta: dict | None, n: int) -> np.ndarray:
    """Re-export of the 2a probe protocol's grouping heuristic (source_id ->
    speaker_id -> file-prefix -> unique-per-row fallback), so callers with a
    manifest-style `meta` dict don't need to reach into evaluation.grouped_probe's
    private helper directly. See grouped_probe._derive_groups for the exact rule.
    """
    return _derive_groups(meta, n)


@dataclass(frozen=True)
class Fold:
    fold_id: int
    selection_idx: np.ndarray
    effect_idx: np.ndarray


def make_nested_folds(n: int, groups: np.ndarray, n_splits: int = 5, seed: int = 13) -> list[Fold]:
    """K group-disjoint folds. Fold i: effect_idx = rows in fold i, selection_idx =
    every other fold's rows (n-1 folds' worth). A group never appears in both
    index sets for the same fold -- verified by `assert_no_group_leakage`, which
    every consumer of `make_nested_folds` should call before using a fold.
    """
    groups = np.asarray(groups, dtype=object)
    n_groups = len(np.unique(groups))
    k = int(min(n_splits, n_groups))
    if k < 2:
        raise ValueError(f"need >= 2 groups to build nested folds, got {n_groups}")
    gkf = GroupKFold(n_splits=k)
    # GroupKFold has no seed of its own; shuffle row order up front for a
    # seed-controlled (but still group-consistent) assignment.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = []
    for i, (train_pos, test_pos) in enumerate(gkf.split(np.zeros(n), groups=groups[perm])):
        effect_idx = np.sort(perm[test_pos])
        selection_idx = np.sort(perm[train_pos])
        folds.append(Fold(fold_id=i, selection_idx=selection_idx, effect_idx=effect_idx))
    return folds


def assert_no_group_leakage(fold: Fold, groups: np.ndarray) -> None:
    """Leakage canary. Raises AssertionError if any row or any group appears in
    both `selection_idx` and `effect_idx` of the given fold."""
    groups = np.asarray(groups, dtype=object)
    overlap_rows = set(fold.selection_idx.tolist()) & set(fold.effect_idx.tolist())
    if overlap_rows:
        raise AssertionError(
            f"fold {fold.fold_id}: {len(overlap_rows)} row(s) appear in both selection and effect sets"
        )
    sel_groups = set(groups[fold.selection_idx].tolist())
    eff_groups = set(groups[fold.effect_idx].tolist())
    leaked = sel_groups & eff_groups
    if leaked:
        sample = sorted(map(str, leaked))[:5]
        raise AssertionError(
            f"fold {fold.fold_id}: {len(leaked)} group(s) straddle selection/effect (e.g. {sample})"
        )


def select_hyperparameter(
    Z: np.ndarray,
    factor: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    selection_idx: np.ndarray,
    candidates: Sequence[dict],
    fit_subspace: Callable[..., np.ndarray],
    score_subspace: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
    n_inner_splits: int = 3,
    seed: int = 13,
) -> dict:
    """Pick the best candidate hyperparameter dict (e.g. {"k": 3, "mode": "within_class"})
    using ONLY rows in `selection_idx`. Internally splits selection into an
    inner fit/validation pair (grouped, via `make_nested_folds` applied to the
    selection subset alone) so the choice never sees `effect_idx` rows.

    Args:
        fit_subspace: `(Z, factor, y, groups, **candidate) -> U (d, r)`.
        score_subspace: `(Z_val, factor_val, y_val, U) -> float`, higher is better
            (e.g. validation-fold alignment/probe-recoverability of the factor).
        n_inner_splits: number of groups the inner fit/validation split is drawn
            from (capped by available groups, same as any other nested fold).

    Returns:
        dict(best=<winning candidate>, best_score=..., all_results=[...]).
    """
    sel_groups = np.asarray(groups)[selection_idx]
    inner = make_nested_folds(len(selection_idx), sel_groups, n_splits=n_inner_splits, seed=seed)
    inner_fold = inner[0]
    assert_no_group_leakage(inner_fold, sel_groups)
    fit_idx = selection_idx[inner_fold.selection_idx]
    val_idx = selection_idx[inner_fold.effect_idx]

    results = []
    for cand in candidates:
        try:
            U = fit_subspace(Z[fit_idx], factor[fit_idx], y[fit_idx], groups[fit_idx], **cand)
            score = float(score_subspace(Z[val_idx], factor[val_idx], y[val_idx], U))
        except Exception as e:  # a candidate (e.g. too-large rank) failing is not fatal
            results.append(dict(candidate=cand, score=float("-inf"), error=str(e)))
            continue
        results.append(dict(candidate=cand, score=score))

    best = max(results, key=lambda r: r["score"])
    if best["score"] == float("-inf"):
        raise ValueError(f"no candidate hyperparameter produced a usable subspace on the selection split: {results}")
    return dict(best=best["candidate"], best_score=best["score"], all_results=results)


def run_nested_crossfit(
    Z: np.ndarray,
    factor: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    candidates: Sequence[dict],
    fit_subspace: Callable[..., np.ndarray],
    score_subspace: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
    effect_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], object],
    n_outer_splits: int = 5,
    n_inner_splits: int = 3,
    seed: int = 13,
) -> list[dict]:
    """Full nested cross-fitting run over `n_outer_splits` folds.

    Per fold: hyperparameters are selected on that fold's SELECTION rows only
    (`select_hyperparameter`, itself internally split so it never sees this
    fold's EFFECT rows either); the subspace is then refit on the *full*
    selection set with the winning hyperparameters and frozen; `effect_fn` is
    evaluated once, on the EFFECT rows only, with that frozen subspace.

    Args:
        effect_fn: `(Z_effect, factor_effect, y_effect, U) -> Any` -- any of the
            metrics in metrics.py (alignment, r_var, prediction_change, a LEACE/
            INLP erasure quality metric, a removal_control_report, ...).

    Returns:
        One dict per fold: fold_id, chosen hyperparameters, selection_score,
        effect (whatever effect_fn returned), n_selection, n_effect.
    """
    n = len(y)
    outer_folds = make_nested_folds(n, groups, n_splits=n_outer_splits, seed=seed)
    results = []
    for fold in outer_folds:
        assert_no_group_leakage(fold, groups)
        sel, eff = fold.selection_idx, fold.effect_idx
        picked = select_hyperparameter(
            Z, factor, y, groups, sel, candidates, fit_subspace, score_subspace, n_inner_splits, seed
        )
        U = fit_subspace(Z[sel], factor[sel], y[sel], groups[sel], **picked["best"])
        effect = effect_fn(Z[eff], factor[eff], y[eff], U)
        results.append(dict(
            fold_id=fold.fold_id, chosen=picked["best"], selection_score=picked["best_score"],
            effect=effect, n_selection=len(sel), n_effect=len(eff),
        ))
    return results
