"""The reliance-measurement battery (Roadmap v3 Step 3): given a fitted linear
task direction `w` and a factor subspace `U` (from subspaces.py, always
estimated on SELECTION data via crossfit.py), quantify how much the task
decision depends on that subspace -- several complementary, non-causal-sounding
quantities, per HV's explicit demotion of the single gradient-DRS score:

  (i)   alignment            -- geometric weight-subspace alignment. NEVER call
                                 this "causal reliance": for a linear head it is
                                 a constant geometric fact about `w`, not an
                                 intervention result.
  (ii)  r_var / r_var_class_conditional -- HV's variance-weighted reliance,
                                 R_var = (w^T U U^T Sigma_z U U^T w) / (w^T Sigma_z w).
  (iii) prediction_change      -- logit/probability/decision change under naive
                                 projection removal of U.
  (iv)  fit_leace / fit_inlp   -- two erasers (closed-form whitened-projection
                                 LEACE, iterative-nullspace-projection INLP).
                                 These double as Step 5/6 baselines, not just
                                 Step 3 diagnostics.
  (v)   conditional_mutual_information -- stub; nonlinear-predictability hook.

Every removal-style metric ((iii) and the erasers) should be reported through
`removal_control_report`, which compares the true subspace's effect against
equal-norm random-subspace controls and a task-direction positive control.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression

from ._linalg import orthonormal_basis, sym_matrix_powers

# ---------------------------------------------------------------------------
# (i) alignment
# ---------------------------------------------------------------------------


def alignment(w: np.ndarray, U: np.ndarray) -> float:
    """Geometric weight-subspace alignment: ||U^T w_hat||^2 in [0, 1], w_hat the
    unit-normalized task weight vector. NEVER call this "causal reliance" -- for
    a linear head this is a fixed geometric property of `w` and `U`, not the
    result of an intervention (HV's demotion of the single gradient-DRS score;
    see docs/review/AudioShield_Roadmap_v3_AUTHORITATIVE.md item 2)."""
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(w)
    if norm <= 0 or U.shape[1] == 0:
        return 0.0
    w_hat = w / norm
    return float(np.sum((U.T @ w_hat) ** 2))


# ---------------------------------------------------------------------------
# (ii) R_var
# ---------------------------------------------------------------------------


def r_var(w: np.ndarray, U: np.ndarray, Sigma_z: np.ndarray) -> float:
    """R_var(w, U) = (w^T U U^T Sigma_z U U^T w) / (w^T Sigma_z w) -- HV's
    variance-weighted reliance formula (Roadmap v3 Step 3, verbatim). Typically
    in [0, 1] for PSD Sigma_z; if U is exactly orthogonal to w this is exactly 0
    regardless of Sigma_z (U^T w = 0 kills the numerator identically)."""
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    Sigma_z = np.atleast_2d(np.asarray(Sigma_z, dtype=np.float64))
    denom = float(w @ Sigma_z @ w)
    if denom <= 0:
        return float("nan")
    if U.shape[1] == 0:
        return 0.0
    proj_w = U @ (U.T @ w)  # U U^T w
    numer = float(proj_w @ Sigma_z @ proj_w)
    return numer / denom


def r_var_class_conditional(w: np.ndarray, U: np.ndarray, Z: np.ndarray, y: np.ndarray) -> dict:
    """Class-conditional R_var: per-class covariance Sigma_z|y=c, then a
    size-weighted average across classes. Compute this on OUT-OF-FOLD (effect-set)
    data -- this function is agnostic to where Z/y come from; crossfit.py's
    harness is what enforces the out-of-fold discipline."""
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y)
    per_class: dict[str, float] = {}
    total_n, weighted_sum = 0, 0.0
    for c in np.unique(y):
        m = y == c
        n_c = int(m.sum())
        if n_c < 2:
            continue
        Sigma_c = np.atleast_2d(np.cov(Z[m], rowvar=False))
        val = r_var(w, U, Sigma_c)
        per_class[str(c)] = val
        if not np.isnan(val):
            weighted_sum += val * n_c
            total_n += n_c
    overall = weighted_sum / total_n if total_n > 0 else float("nan")
    return dict(per_class=per_class, overall=overall)


# ---------------------------------------------------------------------------
# (iii) projection removal / prediction change
# ---------------------------------------------------------------------------


def project_out(Z: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Naive projection removal: Z - Z U U^T. No mean-centering (contrast with
    LEACE below, which is mean-aware and whitened) -- this is the simple
    baseline removal the battery reports prediction change under."""
    Z = np.asarray(Z, dtype=np.float64)
    if U.shape[1] == 0:
        return Z.copy()
    return Z - (Z @ U) @ U.T


def prediction_change(Z: np.ndarray, w: np.ndarray, U: np.ndarray, b: float = 0.0) -> dict:
    """Change in a linear head's logit / calibrated probability / hard decision
    when `U` is projected out of `Z`. All three reported per HV's "effects on
    logits and calibrated probabilities, not only EER.\""""
    Z = np.asarray(Z, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    logit_before = Z @ w + b
    logit_after = project_out(Z, U) @ w + b
    delta = logit_after - logit_before
    p_before = 1.0 / (1.0 + np.exp(-logit_before))
    p_after = 1.0 / (1.0 + np.exp(-logit_after))
    decision_before = logit_before > 0
    decision_after = logit_after > 0
    return dict(
        mean_abs_logit_change=float(np.mean(np.abs(delta))),
        rmse_logit_change=float(np.sqrt(np.mean(delta**2))),
        mean_prob_change=float(np.mean(np.abs(p_after - p_before))),
        decision_flip_rate=float(np.mean(decision_before != decision_after)),
    )


# ---------------------------------------------------------------------------
# (iv) erasers: LEACE (closed-form whitened projection) + INLP
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearEraser:
    """A fitted linear concept-erasing map: transform(X) = mean_ + (X - mean_) @ proj.T"""
    name: str
    mean_: np.ndarray
    proj: np.ndarray
    rank_removed: int

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return self.mean_ + (X - self.mean_) @ self.proj.T


def _prepare_concept(z: np.ndarray) -> np.ndarray:
    """1D float array -> continuous concept (kept as a column). 1D non-float
    array -> discrete factor levels, one-hot encoded. 2D array -> used as-is."""
    z = np.asarray(z)
    if z.ndim == 1:
        if np.issubdtype(z.dtype, np.floating):
            return z[:, None].astype(np.float64)
        levels = np.unique(z)
        return (z[:, None] == levels[None, :]).astype(np.float64)
    return z.astype(np.float64)


def fit_leace(X: np.ndarray, z: np.ndarray, rtol: float = 1e-8) -> LinearEraser:
    """LEACE (LEAst-squares Concept Erasure, Belrose et al. 2023): the minimal
    (least-squares) linear map that makes `z` exactly linearly unpredictable
    from the transformed X, via the closed-form whitened projection:

        W       = Sigma_XX^{-1/2}                       (whitening, pseudo-inverse-
                                                           regularized for singular Sigma_XX)
        Q       = orthonormal basis of  W @ Sigma_XZ      (whitened cross-covariance directions)
        eraser  = mu_X + Sigma_XX^{1/2} (I - Q Q^T) Sigma_XX^{-1/2} (x - mu_X)

    `z` may be a 1D discrete factor (one-hot encoded internally), a 1D
    continuous score, or an already-encoded (n, c) concept matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    Z = _prepare_concept(z)
    n, d = X.shape
    mu_x = X.mean(axis=0)
    Xc = X - mu_x
    Zc = Z - Z.mean(axis=0, keepdims=True)
    denom = max(n - 1, 1)
    Sigma_xx = (Xc.T @ Xc) / denom
    Sigma_xz = (Xc.T @ Zc) / denom

    # Both powers of the SAME Sigma_xx -- one shared eigendecomposition
    # (sym_matrix_powers), not two of a (d, d) matrix (d in the low
    # thousands at embedding scale is one of the dominant per-fold costs).
    W, W_inv = sym_matrix_powers(Sigma_xx, (-0.5, 0.5), rtol=rtol)
    WZ = W @ Sigma_xz
    Q = orthonormal_basis(WZ, rtol=rtol)
    P_ws = Q @ Q.T
    proj = W_inv @ (np.eye(d) - P_ws) @ W
    return LinearEraser(name="leace", mean_=mu_x, proj=proj, rank_removed=Q.shape[1])


def fit_inlp(X: np.ndarray, z: np.ndarray, n_iterations: int = 4, seed: int = 13, C: float = 1.0) -> LinearEraser:
    """INLP (Iterative Nullspace Projection, Ravfogel et al. 2020): repeatedly
    fit a linear probe for `z`, add its weight direction(s) to a removed-span
    accumulator, and project the ORIGINAL X through the cumulative nullspace
    projector each iteration (rebuilt from scratch, not chained, to avoid
    compounding numerical drift). Stops early if a probe's directions add
    nothing new to the removed span (e.g. already fully collapsed)."""
    X = np.asarray(X, dtype=np.float64)
    z = np.asarray(z)
    n, d = X.shape
    if len(np.unique(z)) < 2:
        raise ValueError("fit_inlp needs >= 2 concept levels to fit a probe")

    directions: list[np.ndarray] = []
    P = np.eye(d)
    rank_removed = 0
    for _ in range(n_iterations):
        X_cur = X @ P.T
        clf = LogisticRegression(max_iter=1000, C=C, random_state=seed).fit(X_cur, z)
        directions.append(np.atleast_2d(clf.coef_))
        all_dirs = np.concatenate(directions, axis=0)
        Qd = orthonormal_basis(all_dirs.T)
        if Qd.shape[1] == rank_removed:  # no new direction found; converged
            break
        rank_removed = Qd.shape[1]
        P = np.eye(d) - Qd @ Qd.T
        if rank_removed >= d:
            break
    return LinearEraser(name="inlp", mean_=np.zeros(d), proj=P, rank_removed=rank_removed)


# ---------------------------------------------------------------------------
# (v) conditional-MI hook (stub)
# ---------------------------------------------------------------------------


def conditional_mutual_information(
    Z: np.ndarray, factor: np.ndarray, y: np.ndarray | None = None, estimator: str = "knn", **kwargs
) -> float:
    """Stub for a nonlinear/nonparametric estimate of I(Z; factor | y) -- catches
    reliance the linear battery above cannot see. Roadmap v3 Step 3 scopes this
    as "where practical," not a hard requirement; not implemented yet. Interface
    is stable so a real estimator (e.g. KSG kNN-based, or a neural critic) can be
    swapped in later without changing any crossfit.py effect_fn call site."""
    raise NotImplementedError(
        "conditional_mutual_information is a Step 3 hook, not yet implemented. "
        "Interface: conditional_mutual_information(Z, factor, y=None, estimator=..., **kwargs) -> float."
    )


# ---------------------------------------------------------------------------
# equal-norm random-subspace control + task-direction positive control
# ---------------------------------------------------------------------------


def random_subspace(d: int, k: int, seed: int = 13) -> np.ndarray:
    """Equal-norm random-subspace control: an orthonormal (d, k) basis. Every
    orthonormal (d, k) basis has the same Frobenius norm (sqrt(k)), so this
    matches a fitted true-factor subspace of the same rank by construction --
    "equal-norm" falls out of orthonormality, not a post-hoc rescale."""
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(d, k))
    basis = orthonormal_basis(M, k=k)
    if basis.shape[1] < k:  # exceedingly unlikely (k <= d Gaussian draw is rank-k a.s.), but stay honest
        raise ValueError(f"random_subspace: degenerate draw produced rank {basis.shape[1]} < k={k}")
    return basis


def task_direction_subspace(w: np.ndarray, k: int = 1, seed: int = 13) -> np.ndarray:
    """Positive control: the subspace spanned by the task weight vector itself
    (rank 1), padded with random directions orthogonal to it up to rank `k` for
    matched-rank comparisons. Removing this subspace should produce a LARGE
    effect on the task's own predictions -- confirms the measurement pipeline
    can detect a real effect when one truly exists."""
    w = np.asarray(w, dtype=np.float64).reshape(-1, 1)
    d = w.shape[0]
    base = orthonormal_basis(w, k=1)
    if k <= 1 or base.shape[1] == 0:
        return base
    rng = np.random.default_rng(seed)
    extra = rng.normal(size=(d, k - 1))
    extra = extra - base @ (base.T @ extra)  # project out the task direction first
    extra_basis = orthonormal_basis(extra, k=k - 1)
    return np.concatenate([base, extra_basis], axis=1)


def removal_control_report(
    Z: np.ndarray,
    w: np.ndarray,
    U_true: np.ndarray,
    effect_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    n_random: int = 20,
    seed: int = 13,
) -> dict:
    """Report any removal-style effect against: the true subspace, `n_random`
    equal-norm random-subspace controls of matched rank, and the task-direction
    positive control. Every removal metric in the battery should be reported
    this way -- Roadmap v3 Step 4's preregistered gate requires "intervention
    effects significantly exceeding equal-norm random controls (with
    task-direction removal as positive control)."

    Args:
        effect_fn: `(Z, w, U) -> float`, e.g. `lambda Z,w,U: prediction_change(Z,w,U)["mean_abs_logit_change"]`.
    """
    k = U_true.shape[1]
    d = Z.shape[1]
    true_effect = float(effect_fn(Z, w, U_true))
    rng = np.random.default_rng(seed)
    random_effects = [
        float(effect_fn(Z, w, random_subspace(d, k, seed=int(rng.integers(1_000_000_000)))))
        for _ in range(n_random)
    ]
    task_U = task_direction_subspace(w, k=k, seed=seed)
    task_effect = float(effect_fn(Z, w, task_U))
    random_mean = float(np.mean(random_effects)) if random_effects else float("nan")
    random_std = float(np.std(random_effects)) if random_effects else float("nan")
    return dict(
        true_effect=true_effect,
        random_effects=random_effects,
        random_mean=random_mean,
        random_std=random_std,
        task_direction_effect=task_effect,
        exceeds_random=(bool(true_effect > random_mean + 2 * random_std) if random_effects else None),
    )
