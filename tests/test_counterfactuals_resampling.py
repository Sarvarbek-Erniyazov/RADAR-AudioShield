"""Tests for src/audioshield/counterfactuals/resampling.py."""
import numpy as np
import pytest
from scipy.stats import spearmanr

from audioshield.counterfactuals.resampling import resample_round_trip


def test_preserves_length_and_pairs(synthetic_broadband_audio):
    x, sr = synthetic_broadband_audio
    aligned, prov = resample_round_trip(x, sr, dose=8000)
    assert aligned.shape == x.shape
    assert aligned.dtype == np.float32
    assert np.isfinite(aligned).all()


def test_dose_equal_to_sr_is_near_identity(synthetic_broadband_audio):
    x, sr = synthetic_broadband_audio
    aligned, _ = resample_round_trip(x, sr, dose=sr)
    mse = float(np.mean((aligned - x) ** 2))
    assert mse < 1e-6


def test_lower_intermediate_rate_yields_more_distortion(synthetic_broadband_audio):
    x, sr = synthetic_broadband_audio
    doses = [16000, 8000, 4000, 2000, 1000]
    mses = [float(np.mean((resample_round_trip(x, sr, d)[0] - x) ** 2)) for d in doses]
    rho, _ = spearmanr(doses, mses)
    assert rho < -0.9, f"doses={doses} mses={mses} rho={rho}"
    assert mses == sorted(mses)  # strictly monotonic for this (clean, deterministic) transform


def test_determinism_same_call_identical_bytes(synthetic_broadband_audio):
    x, sr = synthetic_broadband_audio
    y1, _ = resample_round_trip(x, sr, 8000)
    y2, _ = resample_round_trip(x, sr, 8000)
    np.testing.assert_array_equal(y1, y2)


def test_provenance_completeness(synthetic_broadband_audio):
    x, sr = synthetic_broadband_audio
    _, prov = resample_round_trip(x, sr, 8000, seed=13)
    required = {"transform", "family", "dose", "dose_unit", "seed", "sr", "orig_sr"}
    assert required.issubset(prov)
    assert prov["transform"] == "resample_round_trip"
    assert prov["dose"] == 8000.0
    assert prov["orig_sr"] == sr


@pytest.mark.parametrize("bad_dose", [0, -100, 99999])
def test_dose_out_of_range_raises(synthetic_broadband_audio, bad_dose):
    x, sr = synthetic_broadband_audio
    with pytest.raises(ValueError):
        resample_round_trip(x, sr, bad_dose)
