"""Tests for src/audioshield/counterfactuals/noise.py -- the SNR accuracy
requirement from the task ("SNR transform yields the requested SNR +/-
tolerance") lives here."""
import numpy as np
import pytest

from audioshield.counterfactuals.noise import additive_noise_snr

SNR_TOLERANCE_DB = 0.05


def _measured_snr_db(original: np.ndarray, noisy: np.ndarray) -> float:
    noise_est = noisy.astype(np.float64) - original.astype(np.float64)
    sig_p = np.mean(original.astype(np.float64) ** 2)
    noise_p = np.mean(noise_est ** 2)
    return 10.0 * np.log10(sig_p / noise_p)


@pytest.mark.parametrize("noise_type", ["white", "pink"])
@pytest.mark.parametrize("dose_db", [30, 20, 10, 0, -10])
def test_measured_snr_matches_requested_dose(synthetic_broadband_audio, noise_type, dose_db):
    x, sr = synthetic_broadband_audio
    y, _ = additive_noise_snr(x, sr, dose_db, seed=13, noise_type=noise_type)
    measured = _measured_snr_db(x, y)
    assert measured == pytest.approx(dose_db, abs=SNR_TOLERANCE_DB)


def test_preserves_shape_and_is_trivially_aligned(synthetic_broadband_audio):
    x, sr = synthetic_broadband_audio
    y, _ = additive_noise_snr(x, sr, 10, seed=13)
    assert y.shape == x.shape
    assert y.dtype == np.float32


def test_lower_dose_yields_more_distortion(synthetic_broadband_audio):
    x, sr = synthetic_broadband_audio
    doses = [30, 20, 10, 0, -10]
    mses = [float(np.mean((additive_noise_snr(x, sr, d, seed=13)[0] - x) ** 2)) for d in doses]
    assert mses == sorted(mses)  # strictly monotonic: lower SNR dose -> more added noise power


def test_determinism_same_seed_identical_bytes(synthetic_broadband_audio):
    x, sr = synthetic_broadband_audio
    y1, _ = additive_noise_snr(x, sr, 10, seed=13)
    y2, _ = additive_noise_snr(x, sr, 10, seed=13)
    np.testing.assert_array_equal(y1, y2)


def test_different_seed_yields_different_output(synthetic_broadband_audio):
    x, sr = synthetic_broadband_audio
    y1, _ = additive_noise_snr(x, sr, 10, seed=13)
    y2, _ = additive_noise_snr(x, sr, 10, seed=99)
    assert not np.array_equal(y1, y2)


def test_provenance_completeness(synthetic_broadband_audio):
    x, sr = synthetic_broadband_audio
    _, prov = additive_noise_snr(x, sr, 10, seed=13, noise_type="pink")
    required = {"transform", "family", "dose", "dose_unit", "seed", "sr", "noise_type"}
    assert required.issubset(prov)
    assert prov["transform"] == "additive_noise_snr"
    assert prov["seed"] == 13
    assert prov["noise_type"] == "pink"


def test_unknown_noise_type_raises(synthetic_broadband_audio):
    x, sr = synthetic_broadband_audio
    with pytest.raises(ValueError):
        additive_noise_snr(x, sr, 10, seed=13, noise_type="bogus")


def test_zero_power_waveform_raises():
    x = np.zeros(1000, dtype=np.float32)
    with pytest.raises(ValueError):
        additive_noise_snr(x, 16000, 10, seed=13)
