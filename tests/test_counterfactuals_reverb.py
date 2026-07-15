"""Tests for src/audioshield/counterfactuals/reverb.py, using a tiny synthetic
RIR asset directory (never the real RIRS_NOISES corpus)."""
import numpy as np
import pytest
from scipy.stats import spearmanr

from audioshield.data.aug_assets import AugAssetError
from audioshield.counterfactuals.reverb import rir_convolve, replay_simulate


def test_rir_convolve_preserves_length_and_pairs(synthetic_broadband_audio, synthetic_rir_root):
    x, sr = synthetic_broadband_audio
    aligned, prov = rir_convolve(x, sr, dose=0.5, seed=13, rir_root=synthetic_rir_root)
    assert aligned.shape == x.shape
    assert aligned.dtype == np.float32
    assert np.isfinite(aligned).all()


def test_dose_zero_is_near_identity(synthetic_broadband_audio, synthetic_rir_root):
    x, sr = synthetic_broadband_audio
    aligned, _ = rir_convolve(x, sr, dose=0.0, seed=13, rir_root=synthetic_rir_root)
    mse = float(np.mean((aligned - x) ** 2))
    assert mse < 1e-4


def test_dose_one_more_severe_than_dose_zero(synthetic_broadband_audio, synthetic_rir_root):
    """Endpoint check rather than dense pairwise monotonicity: RIR wet/dry
    mixing plus cross-correlation alignment can have small non-monotonic
    wobble at intermediate doses (a real convolution can partially phase-
    cancel at some mix ratios), but the overall trend from dry to fully wet
    must be a large increase, and a soft (Spearman) monotonicity check across
    a modest dose grid should hold."""
    x, sr = synthetic_broadband_audio
    doses = [0.0, 0.25, 0.5, 0.75, 1.0]
    mses = [float(np.mean((rir_convolve(x, sr, d, seed=13, rir_root=synthetic_rir_root)[0] - x) ** 2))
            for d in doses]
    assert mses[-1] > mses[0] * 10
    rho, _ = spearmanr(doses, mses)
    assert rho > 0.8, f"doses={doses} mses={mses} rho={rho}"


def test_replay_simulate_dose_response_is_clean(synthetic_broadband_audio, synthetic_rir_root):
    """replay_simulate composites RIR + noise + band-limit; the noise/band-limit
    components dominate and smooth out the RIR-only wobble seen above, so a
    strict monotonicity check is appropriate here."""
    x, sr = synthetic_broadband_audio
    doses = [0.0, 0.25, 0.5, 0.75, 1.0]
    mses = [float(np.mean((replay_simulate(x, sr, d, seed=13, rir_root=synthetic_rir_root)[0] - x) ** 2))
            for d in doses]
    assert mses == sorted(mses)


def test_replay_simulate_snr_and_cutoff_are_monotonic_in_dose(synthetic_broadband_audio, synthetic_rir_root):
    x, sr = synthetic_broadband_audio
    doses = [0.0, 0.3, 0.6, 1.0]
    snrs, cutoffs = [], []
    for d in doses:
        _, prov = replay_simulate(x, sr, d, seed=13, rir_root=synthetic_rir_root)
        snrs.append(prov["noise_snr_db"])
        cutoffs.append(prov["band_limit_hz"])
    assert snrs == sorted(snrs, reverse=True)  # higher dose -> lower (harsher) SNR
    assert cutoffs == sorted(cutoffs, reverse=True)  # higher dose -> lower (harsher) cutoff


def test_determinism_same_seed_identical_bytes(synthetic_broadband_audio, synthetic_rir_root):
    x, sr = synthetic_broadband_audio
    y1, p1 = rir_convolve(x, sr, 0.5, seed=13, rir_root=synthetic_rir_root)
    y2, p2 = rir_convolve(x, sr, 0.5, seed=13, rir_root=synthetic_rir_root)
    np.testing.assert_array_equal(y1, y2)
    assert p1["chosen_rir_path"] == p2["chosen_rir_path"]

    y3, _ = replay_simulate(x, sr, 0.5, seed=13, rir_root=synthetic_rir_root)
    y4, _ = replay_simulate(x, sr, 0.5, seed=13, rir_root=synthetic_rir_root)
    np.testing.assert_array_equal(y3, y4)


def test_provenance_includes_asset_id_and_hash(synthetic_broadband_audio, synthetic_rir_root):
    x, sr = synthetic_broadband_audio
    _, prov = rir_convolve(x, sr, 0.5, seed=13, rir_root=synthetic_rir_root)
    required = {"transform", "family", "dose", "dose_unit", "seed", "sr",
                "rir_root_fingerprint", "chosen_rir_path", "chosen_rir_sha256"}
    assert required.issubset(prov)
    assert prov["rir_root_fingerprint"]["n_files"] == 12
    assert len(prov["chosen_rir_sha256"]) == 64  # sha256 hex digest


def test_different_seeds_can_choose_different_rirs(synthetic_broadband_audio, synthetic_rir_root):
    x, sr = synthetic_broadband_audio
    chosen = {rir_convolve(x, sr, 0.5, seed=s, rir_root=synthetic_rir_root)[1]["chosen_rir_path"]
              for s in range(20)}
    assert len(chosen) > 1, "20 different seeds should not all pick the same RIR out of 12"


def test_missing_rir_root_raises(synthetic_broadband_audio, tmp_path):
    x, sr = synthetic_broadband_audio
    with pytest.raises(AugAssetError):
        rir_convolve(x, sr, 0.5, seed=13, rir_root=tmp_path / "does_not_exist")


def test_too_few_rir_files_raises(synthetic_broadband_audio, tmp_path):
    import soundfile as sf
    sparse_root = tmp_path / "sparse_rirs"
    sparse_root.mkdir()
    for i in range(3):  # below aug_assets' min_files=10 default
        sf.write(sparse_root / f"r{i}.wav", np.zeros(160, dtype=np.float32), 16000)
    x, sr = synthetic_broadband_audio
    with pytest.raises(AugAssetError):
        rir_convolve(x, sr, 0.5, seed=13, rir_root=sparse_root)


@pytest.mark.parametrize("bad_dose", [-0.1, 1.1, 2.0])
def test_dose_out_of_range_raises(synthetic_broadband_audio, synthetic_rir_root, bad_dose):
    x, sr = synthetic_broadband_audio
    with pytest.raises(ValueError):
        rir_convolve(x, sr, bad_dose, seed=13, rir_root=synthetic_rir_root)
    with pytest.raises(ValueError):
        replay_simulate(x, sr, bad_dose, seed=13, rir_root=synthetic_rir_root)
