"""Tests for src/audioshield/counterfactuals/_align.py."""
import numpy as np

from audioshield.counterfactuals._align import align_to_reference


def test_identity_is_a_noop(synthetic_broadband_audio):
    x, _ = synthetic_broadband_audio
    aligned = align_to_reference(x, x.copy())
    np.testing.assert_array_equal(aligned, x)


def test_corrects_a_positive_delay(synthetic_broadband_audio):
    """Generated output that starts `delay` samples late (leading zeros) --
    simulates encoder priming/lookahead delay."""
    x, _ = synthetic_broadband_audio
    delay = 11
    delayed = np.concatenate([np.zeros(delay, dtype=np.float32), x])[: len(x)]
    aligned = align_to_reference(x, delayed)
    mse_before = float(np.mean((delayed - x) ** 2))
    mse_after = float(np.mean((aligned - x) ** 2))
    assert mse_after < mse_before * 0.05
    assert aligned.shape == x.shape


def test_corrects_a_negative_delay(synthetic_broadband_audio):
    """Generated output that starts `advance` samples early -- simulates a
    filter that trims leading samples."""
    x, _ = synthetic_broadband_audio
    advance = 9
    advanced = np.concatenate([x[advance:], np.zeros(advance, dtype=np.float32)])
    aligned = align_to_reference(x, advanced)
    mse_before = float(np.mean((advanced - x) ** 2))
    mse_after = float(np.mean((aligned - x) ** 2))
    assert mse_after < mse_before * 0.05
    assert aligned.shape == x.shape


def test_output_always_matches_reference_length(synthetic_broadband_audio):
    x, _ = synthetic_broadband_audio
    shorter = x[:100]
    longer = np.concatenate([x, x[:500]])
    assert align_to_reference(x, shorter).shape == x.shape
    assert align_to_reference(x, longer).shape == x.shape


def test_empty_generated_returns_zeros_of_reference_length(synthetic_broadband_audio):
    x, _ = synthetic_broadband_audio
    aligned = align_to_reference(x, np.zeros(0, dtype=np.float32))
    assert aligned.shape == x.shape
    np.testing.assert_array_equal(aligned, 0.0)


def test_empty_reference_returns_empty():
    aligned = align_to_reference(np.zeros(0, dtype=np.float32), np.array([1.0, 2.0], dtype=np.float32))
    assert aligned.shape == (0,)
