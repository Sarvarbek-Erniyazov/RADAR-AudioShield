"""Sample-alignment helper shared by every transform family.

Codec round-trips (encoder lookahead/priming), resampling round-trips (filter
group delay), and RIR convolution (a not-exactly-zero direct-path sample) can
each introduce a small, transform-specific integer-sample shift relative to
the original. `align_to_reference` finds that shift by cross-correlation and
corrects for it, then crops/zero-pads to exactly the original length -- this
is what makes "sample-aligned where the transform permits" true across
families without each transform re-deriving its own delay compensation.
Pure-additive transforms (noise) have no such shift; calling this on them is
a safe no-op (the correlation peak is at lag 0 by construction).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import correlate, correlation_lags


def align_to_reference(reference: np.ndarray, generated: np.ndarray, max_shift_frac: float = 0.05) -> np.ndarray:
    """Shift+crop/pad `generated` to align with `reference`, returning an array
    of exactly `len(reference)` samples.

    Args:
        reference: (n,) original waveform.
        generated: the transform's raw output (any length).
        max_shift_frac: search window for the alignment shift, as a fraction of
            `len(reference)`. Shifts larger than this are assumed to indicate a
            transform that isn't meaningfully "sample-aligned" (e.g. a gross
            resampling-rate mismatch) rather than a delay to compensate for, so
            the search is capped rather than unbounded.
    """
    reference = np.asarray(reference, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    n = len(reference)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if len(generated) == 0:
        return np.zeros(n, dtype=np.float32)

    max_shift = max(1, int(round(n * max_shift_frac)))
    ref_c = reference - reference.mean()
    gen_c = generated - generated.mean()

    corr = correlate(generated, reference, mode="full", method="fft")
    lags = correlation_lags(len(generated), len(reference), mode="full")
    mask = np.abs(lags) <= max_shift
    candidates = np.where(mask)[0]
    if candidates.size == 0:
        best_lag = 0
    else:
        best_lag = int(lags[candidates[np.argmax(corr[candidates])]])

    # `best_lag` is defined (scipy convention, verified empirically in
    # tests/test_counterfactuals_align.py) such that generated[i] best matches
    # reference[i - best_lag], i.e. aligned[i] = generated[i + best_lag].
    if best_lag >= 0:
        aligned = generated[best_lag:]
    else:
        aligned = np.concatenate([np.zeros(-best_lag), generated])

    if len(aligned) < n:
        aligned = np.pad(aligned, (0, n - len(aligned)))
    else:
        aligned = aligned[:n]
    return aligned.astype(np.float32)
