"""Factor-subspace estimators over (embedding, factor-label) pairs.

Two independent estimators of the same target -- an orthonormal basis U (d, r)
for the subspace of embedding-space directions a factor (generator, codec,
channel, language, speaker/source, corpus, ...) is encoded along:

  (a) `lda_subspace`             -- class-conditional Fisher LDA discriminant
                                     directions for the factor, orthonormalized.
  (b) `crossfitted_probe_subspace` -- cross-fitted linear-probe weight rows for
                                     the factor, orthonormalized.

Both are **class-controlled**: the task label `y` is required, not optional,
and is held fixed either by fitting *within* each class stratum and pooling
(`mode="within_class"`) or by residualizing the embedding on class first and
fitting once on the residual (`mode="covariate"`). Never fit on `factor` alone
-- a nuisance factor correlated with the task label would otherwise get
credited with task-relevant variance.

Used together (agreement between the two is itself evidence the recovered
subspace is a property of the data, not an estimator artifact) inside
crossfit.py's nested selection/effect split -- callers here only ever see
"selection" data. Audit ref: Roadmap v3 Step 3 ("Two subspace estimators
throughout").
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ._linalg import orthonormal_basis

MODES = ("within_class", "covariate")


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def _residualize_by_class(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Subtract the class-conditional mean from every row -- class as a covariate."""
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y)
    out = Z.copy()
    for c in np.unique(y):
        m = y == c
        out[m] = Z[m] - Z[m].mean(axis=0, keepdims=True)
    return out


def _between_within_scatter(Z: np.ndarray, factor: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Between-group and within-group (pooled) scatter matrices of Z w.r.t. discrete
    `factor` labels. Groups with < 2 members are skipped (can't estimate their
    within-group scatter). Returns (S_b, S_w, n_usable_levels)."""
    d = Z.shape[1]
    Sb = np.zeros((d, d))
    Sw = np.zeros((d, d))
    grand_mean = Z.mean(axis=0)
    n_usable = 0
    for lvl in np.unique(factor):
        m = factor == lvl
        n_g = int(m.sum())
        if n_g < 2:
            continue
        n_usable += 1
        mu_g = Z[m].mean(axis=0)
        diff = Z[m] - mu_g
        Sw += diff.T @ diff
        gd = (mu_g - grand_mean).reshape(-1, 1)
        Sb += n_g * (gd @ gd.T)
    return Sb, Sw, n_usable


def lda_subspace(
    Z: np.ndarray,
    factor: np.ndarray,
    y: np.ndarray,
    k: int,
    mode: str = "within_class",
    ridge: float = 1e-6,
) -> np.ndarray:
    """Class-conditional Fisher LDA discriminant subspace for `factor`.

    Solves the generalized eigenproblem S_b v = lambda (S_w + ridge*I) v (S_b/S_w
    computed per the chosen class-control mode), keeps the top-`k` eigenvectors by
    eigenvalue, then orthonormalizes them (LDA directions are S_w-orthogonal, not
    Euclidean-orthogonal, so this step is required to get a projection-ready basis).

    Args:
        Z: (n, d) embeddings.
        factor: (n,) discrete factor labels (the nuisance/interpretability factor).
        y: (n,) discrete task/class labels -- REQUIRED (class-controlled by design).
        k: target rank. The returned basis may have fewer than `k` columns if the
            effective discriminant rank (<= n_factor_levels - 1) is lower.
        mode: "within_class" (fit the discriminant separately within each class
            stratum, pool the scatter matrices) or "covariate" (residualize Z by
            the class-conditional mean first, then fit once on the residual).
        ridge: diagonal regularizer added to S_w before inversion (numerical
            stability when n is small relative to d).

    Returns:
        (d, r) orthonormal basis, r <= k.
    """
    _check_mode(mode)
    Z = np.asarray(Z, dtype=np.float64)
    factor = np.asarray(factor)
    y = np.asarray(y)
    d = Z.shape[1]

    # Numerical conditioning for the eigh solve below: raw XLS-R-scale
    # embeddings have wildly varying per-dimension scale, which can
    # ill-condition the scatter-matrix eigenproblem (observed: an overnight
    # run stalled here). Standardize, solve, then map the discriminant
    # directions back to the ORIGINAL embedding coordinate frame before
    # returning -- every caller downstream (alignment/r_var/task-direction
    # metrics against a checkpoint's raw w) is unaffected. Fit on whatever
    # Z is passed in; per this module's documented contract (module
    # docstring) callers only ever pass selection-fold data, so this never
    # touches effect-fold rows.
    scaler = StandardScaler()
    Z = scaler.fit_transform(Z)

    if mode == "covariate":
        Z_resid = _residualize_by_class(Z, y)
        Sb, Sw, n_usable = _between_within_scatter(Z_resid, factor)
    else:
        Sb = np.zeros((d, d))
        Sw = np.zeros((d, d))
        n_usable = 0
        for c in np.unique(y):
            m = y == c
            if m.sum() < 4:
                continue
            sb_c, sw_c, u_c = _between_within_scatter(Z[m], factor[m])
            if u_c < 2:
                continue
            Sb += sb_c
            Sw += sw_c
            n_usable += u_c

    if n_usable < 2:
        raise ValueError(
            "lda_subspace: fewer than 2 usable factor levels (>=2 members each, "
            "within at least one class stratum for mode='within_class') -- cannot "
            "estimate a discriminant subspace"
        )

    from scipy.linalg import eigh  # local import: scipy is a repo dependency, keep import lazy/cheap

    eigvals, eigvecs = eigh(Sb, Sw + ridge * np.eye(d))
    order = np.argsort(eigvals)[::-1]
    top = eigvecs[:, order[:k]]
    top = top / scaler.scale_[:, None]  # map directions back to the original embedding space
    return orthonormal_basis(top, k=k)


def _crossfitted_weight_rows(
    Z: np.ndarray,
    factor: np.ndarray,
    groups: np.ndarray | None,
    n_splits: int,
    seed: int,
    C: float,
) -> np.ndarray:
    """Fit a linear probe (factor ~ Z) on each of several folds' TRAIN split and
    return the stacked coefficient rows -- the cross-fitting step that keeps any
    single fold's fit from dominating the subspace estimate."""
    n = len(factor)
    levels, counts = np.unique(factor, return_counts=True)
    if len(levels) < 2:
        return np.zeros((0, Z.shape[1]))
    k_eff = max(2, min(n_splits, int(counts.min()), n))

    # Same numerical-conditioning rationale as lda_subspace's eigh solve:
    # raw XLS-R-scale features burn the probe's lbfgs iteration budget
    # without converging (observed). Fit once on whatever Z this call
    # received (never effect-fold data -- see module docstring) and use the
    # SAME scale for every inner cross-fitting fold below; coefficient rows
    # are mapped back to the original embedding space before being stacked,
    # so callers see subspaces in the same coordinate frame as before.
    scaler = StandardScaler()
    Z = scaler.fit_transform(Z)

    splitter = None
    if groups is not None and len(np.unique(groups)) >= k_eff:
        splitter = GroupKFold(n_splits=k_eff)
        split_args = dict(X=Z, y=factor, groups=groups)
    else:
        try:
            splitter = StratifiedKFold(n_splits=k_eff, shuffle=True, random_state=seed)
            split_args = dict(X=Z, y=factor)
        except ValueError:
            splitter = KFold(n_splits=k_eff, shuffle=True, random_state=seed)
            split_args = dict(X=Z)

    rows = []
    for train_idx, _ in splitter.split(**split_args):
        if len(np.unique(factor[train_idx])) < 2:
            continue
        clf = LogisticRegression(max_iter=3000, C=C, random_state=seed)
        clf.fit(Z[train_idx], factor[train_idx])
        rows.append(np.atleast_2d(clf.coef_) / scaler.scale_[None, :])
    return np.concatenate(rows, axis=0) if rows else np.zeros((0, Z.shape[1]))


def crossfitted_probe_subspace(
    Z: np.ndarray,
    factor: np.ndarray,
    y: np.ndarray,
    k: int,
    mode: str = "within_class",
    n_splits: int = 5,
    groups: np.ndarray | None = None,
    seed: int = 13,
    C: float = 1.0,
) -> np.ndarray:
    """Cross-fitted linear-probe-weight subspace for `factor`.

    Fits `factor ~ Z` with a logistic-regression probe across several folds
    (grouped by `groups` when given -- reuses the same grouping discipline as
    evaluation.grouped_probe), collects every fold's weight row(s), and
    orthonormalizes the stack. Cross-fitting (vs. a single full-data fit) keeps
    any one fold's overfit direction from dominating the subspace estimate.

    Args:
        Z: (n, d) embeddings.
        factor: (n,) discrete factor labels.
        y: (n,) discrete task/class labels -- REQUIRED (class-controlled by design).
        k: target rank. The returned basis may have fewer than `k` columns.
        mode: "within_class" (fit + cross-fit separately within each class
            stratum, pool the resulting weight rows) or "covariate" (residualize
            Z by the class-conditional mean first, cross-fit once on the residual).
        n_splits: number of cross-fitting folds (capped by available groups/levels).
        groups: (n,) grouping key (e.g. source_id/speaker_id) so a group's rows
            never split across a fold's train part more than once is unavoidable,
            but at least never straddle train/held-out inconsistently within a
            single probe fit -- pass this whenever the factor's carrier (e.g.
            recording session) could otherwise leak.
        seed: RNG seed for fold shuffling and the probe itself.
        C: inverse regularization strength for the LogisticRegression probe.

    Returns:
        (d, r) orthonormal basis, r <= k.
    """
    _check_mode(mode)
    Z = np.asarray(Z, dtype=np.float64)
    factor = np.asarray(factor)
    y = np.asarray(y)

    if mode == "covariate":
        Z_use = _residualize_by_class(Z, y)
        rows = _crossfitted_weight_rows(Z_use, factor, groups, n_splits, seed, C)
    else:
        collected = []
        for c in np.unique(y):
            m = y == c
            if m.sum() < 4 or len(np.unique(factor[m])) < 2:
                continue
            g_c = groups[m] if groups is not None else None
            collected.append(_crossfitted_weight_rows(Z[m], factor[m], g_c, n_splits, seed, C))
        rows = (
            np.concatenate([r for r in collected if r.shape[0] > 0], axis=0)
            if any(r.shape[0] > 0 for r in collected)
            else np.zeros((0, Z.shape[1]))
        )

    if rows.shape[0] == 0:
        raise ValueError(
            "crossfitted_probe_subspace: no class stratum/fold produced a usable "
            "probe fit (need >=2 factor levels with enough members)"
        )
    return orthonormal_basis(rows.T, k=k)
