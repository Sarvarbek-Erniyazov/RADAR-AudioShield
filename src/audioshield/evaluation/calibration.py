"""Expected Calibration Error (ECE) for the cross-test table."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(labels, scores, n_bins: int = 15) -> float:
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(scores)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (scores > lo) & (scores <= hi) if i > 0 else (scores >= lo) & (scores <= hi)
        if m.sum() == 0:
            continue
        conf = scores[m].mean()
        acc = labels[m].mean()  # fraction positive (spoof) in bin
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)
