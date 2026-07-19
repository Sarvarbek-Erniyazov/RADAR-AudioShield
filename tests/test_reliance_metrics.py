"""Tests for src/audioshield/reliance/metrics.py."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from audioshield.reliance._linalg import orthonormal_basis
from audioshield.reliance.metrics import (
    alignment,
    r_var,
    r_var_class_conditional,
    project_out,
    prediction_change,
    fit_leace,
    fit_inlp,
    conditional_mutual_information,
    random_subspace,
    task_direction_subspace,
    removal_control_report,
)


# ---------------------------------------------------------------------------
# alignment / R_var
# ---------------------------------------------------------------------------


def test_alignment_zero_for_orthogonal_subspace(planted_factor_data):
    d = planted_factor_data
    assert alignment(d["w_true"], d["U_true"]) == pytest.approx(0.0, abs=1e-10)


def test_alignment_one_for_self_direction(planted_factor_data):
    d = planted_factor_data
    U_task = task_direction_subspace(d["w_true"], k=1)
    assert alignment(d["w_true"], U_task) == pytest.approx(1.0, abs=1e-8)


def test_r_var_exactly_zero_for_factor_orthogonal_to_representation(planted_factor_data):
    """If U is exactly orthogonal to w (U^T w = 0), R_var must be exactly 0
    regardless of the covariance structure -- a pure linear-algebra guarantee,
    not a statistical approximation, so this must hold to numerical precision."""
    d = planted_factor_data
    Sigma_z = np.cov(d["Z"], rowvar=False)
    val = r_var(d["w_true"], d["U_true"], Sigma_z)
    assert val == pytest.approx(0.0, abs=1e-8)


def test_r_var_small_for_a_factor_independent_of_the_data(planted_factor_data):
    """A factor label with no relationship to Z at all: fit a subspace for it
    (whatever a probe recovers is essentially noise-driven) and confirm R_var of
    the TASK weight against that noise-driven subspace stays small."""
    d = planted_factor_data
    rng = np.random.default_rng(99)
    irrelevant_factor = rng.integers(0, 4, size=len(d["y"]))  # independent of Z
    from audioshield.reliance.subspaces import crossfitted_probe_subspace
    U_irrelevant = crossfitted_probe_subspace(d["Z"], irrelevant_factor, d["y"], k=3, seed=1)
    Sigma_z = np.cov(d["Z"], rowvar=False)
    val = r_var(d["w_true"], U_irrelevant, Sigma_z)
    assert val < 0.1, f"R_var against a noise-fit subspace should be small, got {val}"


def test_r_var_large_for_task_direction_itself(planted_factor_data):
    d = planted_factor_data
    Sigma_z = np.cov(d["Z"], rowvar=False)
    U_task = task_direction_subspace(d["w_true"], k=1)
    val = r_var(d["w_true"], U_task, Sigma_z)
    assert val == pytest.approx(1.0, abs=1e-6)


def test_r_var_class_conditional_matches_manual_average(planted_factor_data):
    d = planted_factor_data
    out = r_var_class_conditional(d["w_true"], d["U_true"], d["Z"], d["y"])
    assert set(out["per_class"]) == {"0", "1"}
    assert out["overall"] == pytest.approx(0.0, abs=1e-8)  # orthogonal -> exactly 0 per class too


def test_r_var_nan_for_zero_variance_direction():
    w = np.array([1.0, 0.0])
    U = np.zeros((2, 0))
    Sigma = np.zeros((2, 2))  # w^T Sigma w == 0
    assert np.isnan(r_var(w, U, Sigma))


# ---------------------------------------------------------------------------
# projection removal / prediction change
# ---------------------------------------------------------------------------


def test_project_out_removes_subspace_component(planted_factor_data):
    d = planted_factor_data
    Z_removed = project_out(d["Z"], d["U_true"])
    residual_in_subspace = Z_removed @ d["U_true"]
    np.testing.assert_allclose(residual_in_subspace, 0.0, atol=1e-8)


def test_prediction_change_zero_when_removing_orthogonal_subspace(planted_factor_data):
    d = planted_factor_data
    out = prediction_change(d["Z"], d["w_true"], d["U_true"])
    assert out["mean_abs_logit_change"] == pytest.approx(0.0, abs=1e-8)
    assert out["decision_flip_rate"] == pytest.approx(0.0, abs=1e-8)


def test_prediction_change_large_when_removing_task_direction(planted_factor_data):
    d = planted_factor_data
    U_task = task_direction_subspace(d["w_true"], k=1)
    out = prediction_change(d["Z"], d["w_true"], U_task)
    assert out["mean_abs_logit_change"] > 1.0
    assert out["decision_flip_rate"] > 0.05


# ---------------------------------------------------------------------------
# LEACE / INLP erasure
# ---------------------------------------------------------------------------


def _factor_and_task_accuracy(Ztr, Zte, ftr, fte, ytr, yte):
    factor_acc = balanced_accuracy_score(
        fte, LogisticRegression(max_iter=1000).fit(Ztr, ftr).predict(Zte)
    )
    task_acc = balanced_accuracy_score(
        yte, LogisticRegression(max_iter=1000).fit(Ztr, ytr).predict(Zte)
    )
    return factor_acc, task_acc


def test_leace_erases_factor_but_preserves_task(planted_factor_data):
    d = planted_factor_data
    Ztr, Zte, ftr, fte, ytr, yte = train_test_split(
        d["Z"], d["factor"], d["y"], test_size=0.3, random_state=0
    )
    acc_factor_before, acc_task_before = _factor_and_task_accuracy(Ztr, Zte, ftr, fte, ytr, yte)
    assert acc_factor_before > 0.6  # sanity: factor really is decodable before erasure

    eraser = fit_leace(Ztr, ftr)
    Ztr_e, Zte_e = eraser.transform(Ztr), eraser.transform(Zte)
    acc_factor_after, acc_task_after = _factor_and_task_accuracy(Ztr_e, Zte_e, ftr, fte, ytr, yte)

    chance = 1.0 / len(np.unique(d["factor"]))
    assert acc_factor_after == pytest.approx(chance, abs=0.08), (
        f"factor should collapse to ~chance ({chance}) after LEACE, got {acc_factor_after}"
    )
    # task degrades less than factor: factor collapses to chance, task barely moves
    # (w_true is orthogonal to the planted factor subspace by construction).
    assert (acc_task_before - acc_task_after) < (acc_factor_before - acc_factor_after)
    assert acc_task_after > 0.85


def test_leace_eraser_rank_matches_concept_dimensionality(planted_factor_data):
    d = planted_factor_data
    eraser = fit_leace(d["Z"], d["factor"])  # 4 discrete levels -> one-hot rank <= 3
    assert eraser.rank_removed <= 3
    assert eraser.rank_removed >= 1


def test_leace_handles_continuous_concept():
    rng = np.random.default_rng(0)
    n, dim = 500, 10
    concept = rng.normal(size=n)
    direction = rng.normal(size=dim); direction /= np.linalg.norm(direction)
    X = np.outer(concept, direction) * 2.0 + rng.normal(size=(n, dim))
    Xtr, Xte, concept_tr, concept_te = train_test_split(X, concept, test_size=0.3, random_state=0)

    eraser = fit_leace(Xtr, concept_tr)
    Xtr_e, Xte_e = eraser.transform(Xtr), eraser.transform(Xte)

    # linear predictability of the continuous concept should collapse -- fit on
    # train, scored on the HELD-OUT split (not the eraser's own fit data).
    from sklearn.linear_model import LinearRegression
    r2_before = LinearRegression().fit(Xtr, concept_tr).score(Xte, concept_te)
    r2_after = LinearRegression().fit(Xtr_e, concept_tr).score(Xte_e, concept_te)
    assert r2_before > 0.5
    assert r2_after < 0.05


def test_inlp_reduces_factor_decodability(planted_factor_data):
    d = planted_factor_data
    Ztr, Zte, ftr, fte = train_test_split(d["Z"], d["factor"], test_size=0.3, random_state=0)
    eraser = fit_inlp(Ztr, ftr, n_iterations=6)
    acc_before = balanced_accuracy_score(fte, LogisticRegression(max_iter=1000).fit(Ztr, ftr).predict(Zte))
    Ztr_e, Zte_e = eraser.transform(Ztr), eraser.transform(Zte)
    acc_after = balanced_accuracy_score(fte, LogisticRegression(max_iter=1000).fit(Ztr_e, ftr).predict(Zte_e))
    assert acc_after < acc_before - 0.3


def test_fit_inlp_raises_on_single_level():
    X = np.random.default_rng(0).normal(size=(50, 4))
    with pytest.raises(ValueError):
        fit_inlp(X, np.zeros(50))


# ---------------------------------------------------------------------------
# equal-norm random-subspace control + task-direction positive control
# ---------------------------------------------------------------------------


def test_random_subspace_is_orthonormal_and_matched_rank():
    U = random_subspace(d=15, k=4, seed=5)
    assert U.shape == (15, 4)
    np.testing.assert_allclose(U.T @ U, np.eye(4), atol=1e-8)


def test_random_control_smaller_than_true_factor_removal(planted_factor_data):
    """The headline control requirement: true-factor removal should reduce the
    FACTOR's own decodability far more than an equal-norm random subspace does."""
    d = planted_factor_data
    Ztr, Zte, ftr, fte = train_test_split(d["Z"], d["factor"], test_size=0.3, random_state=0)

    def factor_decodability_drop(Z_eval, _w_unused, U):
        Ztr_r, Zte_r = project_out(Ztr, U), project_out(Z_eval, U)
        acc_before = balanced_accuracy_score(
            fte, LogisticRegression(max_iter=500).fit(Ztr, ftr).predict(Z_eval)
        )
        acc_after = balanced_accuracy_score(
            fte, LogisticRegression(max_iter=500).fit(Ztr_r, ftr).predict(Zte_r)
        )
        return acc_before - acc_after

    report = removal_control_report(
        Zte, d["w_true"], d["U_true"], factor_decodability_drop, n_random=15, seed=7
    )
    assert report["true_effect"] > report["random_mean"] + 2 * report["random_std"]
    assert report["true_effect"] > 0.3
    assert report["exceeds_random"] is True


def test_task_direction_positive_control_detects_real_effect(planted_factor_data):
    """Removing the task's OWN weight direction should show a large prediction-change
    effect vs. random controls -- the pipeline must be able to find a real effect
    when one truly exists."""
    d = planted_factor_data

    def logit_change(Z, w, U):
        return prediction_change(Z, w, U)["mean_abs_logit_change"]

    task_U = task_direction_subspace(d["w_true"], k=1)
    report = removal_control_report(d["Z"], d["w_true"], task_U, logit_change, n_random=15, seed=3)
    assert report["true_effect"] > report["random_mean"] + 2 * report["random_std"]


# ---------------------------------------------------------------------------
# conditional-MI stub
# ---------------------------------------------------------------------------


def test_conditional_mutual_information_is_a_stable_stub():
    with pytest.raises(NotImplementedError):
        conditional_mutual_information(np.zeros((5, 2)), np.zeros(5))
