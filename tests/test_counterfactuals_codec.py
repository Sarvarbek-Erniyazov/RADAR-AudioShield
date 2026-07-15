"""Tests for src/audioshield/counterfactuals/codec.py. Requires a real ffmpeg
on PATH (skipped otherwise) -- codec encoding itself isn't mocked, since the
whole point is exercising the real round-trip + alignment."""
import shutil

import numpy as np
import pytest
from scipy.stats import spearmanr

from audioshield.counterfactuals.codec import (
    FfmpegNotAvailableError,
    check_ffmpeg_available,
    codec_chain,
)

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_check_ffmpeg_available_returns_a_path():
    path = check_ffmpeg_available()
    assert path


def test_check_ffmpeg_available_fails_loudly_when_missing(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(FfmpegNotAvailableError, match="ffmpeg"):
        check_ffmpeg_available()


@pytest.mark.parametrize("codec,dose", [("opus", 16), ("mp3", 32), ("aac", 48)])
def test_codec_chain_preserves_length_and_pairs(synthetic_broadband_audio, codec, dose):
    x, sr = synthetic_broadband_audio
    aligned, prov = codec_chain(x, sr, dose, codec=codec)
    assert aligned.shape == x.shape
    assert aligned.dtype == np.float32
    assert np.isfinite(aligned).all()
    assert prov["codec"] == codec
    assert prov["dose"] == dose


@pytest.mark.parametrize("codec,doses", [
    ("opus", [6, 16, 32, 64]),
    ("mp3", [8, 32, 96, 256]),
    ("aac", [16, 48, 128, 224]),
])
def test_lower_bitrate_yields_more_distortion(synthetic_broadband_audio, codec, doses):
    """dose-response: lower bitrate (more severe) should trend toward larger
    deviation from the original -- checked via Spearman correlation (a soft
    monotonicity check) rather than requiring every adjacent pair to be
    strictly ordered, since a real lossy codec's behavior isn't perfectly
    monotonic sample-for-sample."""
    x, sr = synthetic_broadband_audio
    mses = []
    for dose in doses:
        y, _ = codec_chain(x, sr, dose, codec=codec)
        mses.append(float(np.mean((y - x) ** 2)))
    rho, _ = spearmanr(doses, mses)
    assert rho < -0.7, f"{codec}: expected bitrate vs. distortion to trend negative, doses={doses} mses={mses} rho={rho}"


def test_determinism_same_call_identical_bytes(synthetic_broadband_audio):
    x, sr = synthetic_broadband_audio
    y1, _ = codec_chain(x, sr, 16, codec="opus")
    y2, _ = codec_chain(x, sr, 16, codec="opus")
    np.testing.assert_array_equal(y1, y2)


def test_provenance_completeness(synthetic_broadband_audio):
    x, sr = synthetic_broadband_audio
    _, prov = codec_chain(x, sr, 16, seed=13, codec="opus")
    required = {"transform", "family", "dose", "dose_unit", "seed", "sr", "codec",
                "ffmpeg_encoder", "ffmpeg_version"}
    assert required.issubset(prov)
    assert prov["transform"] == "codec_chain"
    assert prov["family"] == "codec"
    assert prov["seed"] == 13
    assert prov["sr"] == sr


def test_unknown_codec_raises(synthetic_broadband_audio):
    x, sr = synthetic_broadband_audio
    with pytest.raises(ValueError):
        codec_chain(x, sr, 16, codec="bogus")


@pytest.mark.parametrize("codec,bad_dose", [("opus", 200), ("mp3", 1), ("aac", 500)])
def test_dose_out_of_range_raises(synthetic_broadband_audio, codec, bad_dose):
    x, sr = synthetic_broadband_audio
    with pytest.raises(ValueError):
        codec_chain(x, sr, bad_dose, codec=codec)
