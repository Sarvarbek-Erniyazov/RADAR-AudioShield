"""Tests for src/audioshield/reliance/subspaces.py -- both factor-subspace
estimators should recover a planted factor subspace and agree with each other."""
import numpy as np
import pytest

from audioshield.reliance._linalg import principal_angles
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
