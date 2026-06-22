"""UPC model selection + stage-transition trigger.

Selection metric = MEAN of per-corpus dev EERs (never single-corpus), which
avoids the shortcut-selecting behavior of the old single-corpus early stop.
"""

from __future__ import annotations

import math


class MeanCrossCorpusStopper:
    def __init__(self, patience: int = 8, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.bad = 0
        self.best_epoch = 0

    def update(self, mean_dev_eer: float, epoch: int) -> tuple[bool, bool]:
        """Returns (improved, should_stop)."""
        improved = (not math.isnan(mean_dev_eer)) and (mean_dev_eer < self.best - self.min_delta)
        if improved:
            self.best = mean_dev_eer
            self.best_epoch = epoch
            self.bad = 0
        else:
            self.bad += 1
        return improved, self.bad >= self.patience


def mean_cross_corpus_eer(per_corpus_eer: dict[str, float]) -> float:
    vals = [v for v in per_corpus_eer.values() if not math.isnan(v)]
    return float(sum(vals) / len(vals)) if vals else float("nan")
