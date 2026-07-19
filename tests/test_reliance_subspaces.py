"""Tests for src/audioshield/reliance/subspaces.py -- both factor-subspace
estimators should recover a planted factor subspace and agree with each other."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from audioshield.reliance._linalg import orthonormal_basis, principal_angles
from audioshield.reliance.subspaces import lda_subspace, crossfitted_probe_subspace

MAX_RECOVERY_ANGLE_DEG = 25.0   # generous: synthetic noise + finite n
MAX_AGREEMENT_ANGLE_DEG = 25.0


@pytest.mark.parametrize("mode", ["within_class", "covariate"])
def test_lda_subspace_recovers_planted_factor(planted_factor_data, mode):
    d = planted_factor_data
    U_hat = lda_subspace(d["Z"], d["factor"], d["y"], k=d["k_factor"], mode=mode)
    assert U_hat.shape == (d["d"], d["k_factor"])
    angles = np.degrees(principal_angles(U_hat, d["U_true"]))
    assert angles.max() < MAX_RECOVERY_ANGLE_DEG, f"mode={mode}: angles={angles}"


@pytest.mark.parametrize("mode", ["within_class", "covariate"])
def test_crossfitted_probe_subspace_recovers_planted_factor(planted_factor_data, mode):
    d = planted_factor_data
    U_hat = crossfitted_probe_subspace(
        d["Z"], d["factor"], d["y"], k=d["k_factor"], mode=mode, n_splits=5, seed=13
    )
    assert U_hat.shape == (d["d"], d["k_factor"])
    angles = np.degrees(principal_angles(U_hat, d["U_true"]))
    assert angles.max() < MAX_RECOVERY_ANGLE_DEG, f"mode={mode}: angles={angles}"


@pytest.mark.parametrize("mode", ["within_class", "covariate"])
def test_estimators_agree_with_each_other(planted_factor_data, mode):
    """Agreement check: the two independent estimators should land on
    near-the-same subspace, not just both near the (unknown, in practice) truth."""
    d = planted_factor_data
    U_lda = lda_subspace(d["Z"], d["factor"], d["y"], k=d["k_factor"], mode=mode)
    U_probe = crossfitted_probe_subspace(
        d["Z"], d["factor"], d["y"], k=d["k_factor"], mode=mode, n_splits=5, seed=13
    )
    angles = np.degrees(principal_angles(U_lda, U_probe))
    assert angles.max() < MAX_AGREEMENT_ANGLE_DEG, f"mode={mode}: agreement angles={angles}"


def test_lda_subspace_rejects_unknown_mode(planted_factor_data):
    d = planted_factor_data
    with pytest.raises(ValueError):
        lda_subspace(d["Z"], d["factor"], d["y"], k=3, mode="bogus")


def test_crossfitted_probe_subspace_rejects_unknown_mode(planted_factor_data):
    d = planted_factor_data
    with pytest.raises(ValueError):
        crossfitted_probe_subspace(d["Z"], d["factor"], d["y"], k=3, mode="bogus")


def test_lda_subspace_raises_on_single_factor_level(planted_factor_data):
    d = planted_factor_data
    constant_factor = np.zeros_like(d["factor"])
    with pytest.raises(ValueError):
        lda_subspace(d["Z"], constant_factor, d["y"], k=3)


def test_returned_basis_is_orthonormal(planted_factor_data):
    d = planted_factor_data
    U_lda = lda_subspace(d["Z"], d["factor"], d["y"], k=d["k_factor"])
    gram = U_lda.T @ U_lda
    np.testing.assert_allclose(gram, np.eye(U_lda.shape[1]), atol=1e-8)


def test_crossfitted_probe_subspace_respects_groups(planted_factor_data):
    """Passing the real (grouped) source_id/speaker_id key should not crash and
    should still recover the planted factor -- grouping can only remove leaked
    signal, never invalidate a real one."""
    d = planted_factor_data
    U_hat = crossfitted_probe_subspace(
        d["Z"], d["factor"], d["y"], k=d["k_factor"], groups=d["groups"], n_splits=5, seed=13
    )
    angles = np.degrees(principal_angles(U_hat, d["U_true"]))
    assert angles.max() < MAX_RECOVERY_ANGLE_DEG


# ---------------------------------------------------------------------------
# StandardScaler robustness fix: fit within-fold only, not globally; probe
# max_iter=3000; still correct (not just non-crashing) on ill-conditioned,
# wildly-varying-per-dimension-scale features like real XLS-R embeddings.
# ---------------------------------------------------------------------------


def test_lda_subspace_scaler_fit_exactly_on_passed_data(monkeypatch, planted_factor_data):
    """Leakage-style check: the internal StandardScaler must be fit EXACTLY
    on the Z passed to lda_subspace (the selection fold only, by this
    module's documented contract) -- never on a larger/different pool,
    which is what 'never fit on effect data' structurally requires."""
    d = planted_factor_data
    seen = []
    original = StandardScaler.fit_transform

    def _spy(self, X, *a, **kw):
        seen.append(np.asarray(X).copy())
        return original(self, X, *a, **kw)

    monkeypatch.setattr(StandardScaler, "fit_transform", _spy)

    n_sel = 400  # a deliberate, distinctly smaller subset of the 3000-row fixture
    Z_sel, factor_sel, y_sel = d["Z"][:n_sel], d["factor"][:n_sel], d["y"][:n_sel]
    lda_subspace(Z_sel, factor_sel, y_sel, k=d["k_factor"])

    assert len(seen) == 1, f"expected exactly one StandardScaler.fit_transform call, got {len(seen)}"
    assert seen[0].shape == (n_sel, d["d"])
    np.testing.assert_array_equal(seen[0], Z_sel)


def test_crossfitted_probe_subspace_scaler_never_sees_more_rows_than_passed(monkeypatch, planted_factor_data):
    """Same leakage-style check for the probe estimator: mode='within_class'
    fits per class stratum, so multiple StandardScaler calls are expected,
    but NONE may ever see more rows than were passed to
    crossfitted_probe_subspace in the first place."""
    d = planted_factor_data
    seen_n = []
    original = StandardScaler.fit_transform

    def _spy(self, X, *a, **kw):
        seen_n.append(np.asarray(X).shape[0])
        return original(self, X, *a, **kw)

    monkeypatch.setattr(StandardScaler, "fit_transform", _spy)

    n_sel = 400
    Z_sel, factor_sel, y_sel, groups_sel = (
        d["Z"][:n_sel], d["factor"][:n_sel], d["y"][:n_sel], d["groups"][:n_sel],
    )
    crossfitted_probe_subspace(Z_sel, factor_sel, y_sel, k=d["k_factor"], groups=groups_sel, n_splits=5, seed=13)

    assert seen_n, "StandardScaler.fit_transform was never called"
    assert all(n <= n_sel for n in seen_n), (
        f"scaler saw more rows ({seen_n}) than the {n_sel} rows passed to crossfitted_probe_subspace"
    )


def test_crossfitted_probe_subspace_uses_max_iter_3000(monkeypatch, planted_factor_data):
    """Patches .fit (not __init__ -- sklearn's BaseEstimator introspects
    __init__'s signature and rejects a *args/**kwargs replacement) and reads
    the already-set self.max_iter attribute."""
    d = planted_factor_data
    seen_max_iter = []
    original_fit = LogisticRegression.fit

    def _spy_fit(self, X, y, *a, **kw):
        seen_max_iter.append(self.max_iter)
        return original_fit(self, X, y, *a, **kw)

    monkeypatch.setattr(LogisticRegression, "fit", _spy_fit)
    crossfitted_probe_subspace(d["Z"][:200], d["factor"][:200], d["y"][:200], k=2, n_splits=3, seed=13)

    assert seen_max_iter, "LogisticRegression.fit was never called"
    assert all(m == 3000 for m in seen_max_iter)


@pytest.mark.parametrize("estimator", ["lda", "probe"])
def test_subspace_estimators_recover_planted_factor_with_ill_conditioned_scale(planted_factor_data, estimator):
    """Regression guard for the robustness fix: an overnight run stalled
    with raw, wildly-varying-per-dimension-scale XLS-R embeddings (lbfgs
    burning its iteration budget without converging). Rescale the
    planted-factor fixture's dimensions by several orders of magnitude
    (comparable to real embedding-scale variation) and confirm the
    subspace is STILL correctly recovered -- proves standardization (not
    just a higher max_iter) is doing the real work, not merely papering
    over a crash."""
    d = planted_factor_data
    rng = np.random.default_rng(99)
    per_dim_scale = 10.0 ** rng.uniform(-3, 3, size=d["d"])
    Z_illscaled = d["Z"] * per_dim_scale[None, :]
    # U_true was defined in the ORIGINAL (unscaled) coordinate frame; express
    # the same true direction in the rescaled frame before comparing (a
    # per-dimension rescale is not an orthogonal transform, so this is NOT
    # just U_true itself).
    U_true_rescaled = orthonormal_basis(d["U_true"] / per_dim_scale[:, None], k=d["k_factor"])

    if estimator == "lda":
        U_hat = lda_subspace(Z_illscaled, d["factor"], d["y"], k=d["k_factor"], mode="within_class")
    else:
        U_hat = crossfitted_probe_subspace(
            Z_illscaled, d["factor"], d["y"], k=d["k_factor"], mode="within_class", n_splits=5, seed=13,
        )

    angles = np.degrees(principal_angles(U_hat, U_true_rescaled))
    assert angles.max() < MAX_RECOVERY_ANGLE_DEG + 15, f"{estimator}: angles={angles}"  # extra slack for the extreme rescale
