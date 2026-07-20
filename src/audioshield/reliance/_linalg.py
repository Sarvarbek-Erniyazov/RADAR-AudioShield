"""Shared linear-algebra helpers for the reliance battery. Internal -- not part of
the public API (see subspaces.py / metrics.py for the public functions that use these).
"""
from __future__ import annotations

import numpy as np


def orthonormal_basis(M: np.ndarray, k: int | None = None, rtol: float = 1e-10) -> np.ndarray:
    """Orthonormal basis for the column space of M, via SVD (rank-revealing --
    safe for linearly dependent columns, e.g. centered one-hot factor levels).

    Args:
        M: (d, m) matrix whose column space we want a basis for.
        k: if given, truncate to the top-k directions (by singular value). If the
            effective rank is < k, the returned basis has fewer than k columns
            (never fabricates directions with ~zero singular value).
        rtol: singular values <= rtol * max singular value are treated as zero.

    Returns:
        (d, r) orthonormal basis, r = min(k, effective_rank) if k given else effective_rank.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim == 1:
        M = M[:, None]
    if M.shape[1] == 0:
        return np.zeros((M.shape[0], 0))
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    if S.size == 0 or S[0] <= 0:
        return np.zeros((M.shape[0], 0))
    rank = int(np.sum(S > rtol * S[0]))
    if k is not None:
        rank = min(rank, k)
    return U[:, :rank]


def sym_matrix_power(S: np.ndarray, power: float, rtol: float = 1e-8) -> np.ndarray:
    """Symmetric matrix power via eigendecomposition, pseudo-inverse-regularized:
    eigenvalues at or below `rtol * max_eigenvalue` are treated as exactly zero
    (their contribution dropped) rather than raised to a possibly-huge negative
    power. This is what makes `power=-0.5` a safe *whitening* transform even when
    S (e.g. an empirical covariance with n < d) is singular.
    """
    vals, vecs = _eigh_symmetrize(S)
    return _apply_power(vals, vecs, power, rtol)


def sym_matrix_powers(S: np.ndarray, powers: tuple[float, ...], rtol: float = 1e-8) -> tuple[np.ndarray, ...]:
    """Like `sym_matrix_power`, but for several powers of the SAME matrix at
    once, sharing a single eigendecomposition -- e.g. LEACE needs both S^-0.5
    (whitening) and S^0.5 (un-whitening) of the same covariance; computing
    each via a separate `sym_matrix_power` call redundantly eigendecomposes
    an identical (d, d) matrix twice, which for d in the low thousands
    (embedding-dimension scale) is one of the dominant per-fold costs in the
    reliance battery. Mathematically identical to calling `sym_matrix_power`
    once per power -- this is purely a shared-computation optimization, not
    a change to any result.
    """
    vals, vecs = _eigh_symmetrize(S)
    return tuple(_apply_power(vals, vecs, p, rtol) for p in powers)


def _eigh_symmetrize(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    S = np.asarray(S, dtype=np.float64)
    return np.linalg.eigh((S + S.T) / 2.0)  # symmetrize defensively


def _apply_power(vals: np.ndarray, vecs: np.ndarray, power: float, rtol: float) -> np.ndarray:
    thresh = rtol * max(vals.max(), 1e-300)
    keep = vals > thresh
    powered = np.zeros_like(vals)
    powered[keep] = vals[keep] ** power
    return (vecs * powered) @ vecs.T


def principal_angles(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between the column spaces of two
    orthonormal bases U (d,p) and V (d,q). Used to compare subspace estimators
    against each other or against a known-planted ground truth."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    if U.shape[1] == 0 or V.shape[1] == 0:
        return np.array([])
    cos_theta = np.linalg.svd(U.T @ V, compute_uv=False)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)
