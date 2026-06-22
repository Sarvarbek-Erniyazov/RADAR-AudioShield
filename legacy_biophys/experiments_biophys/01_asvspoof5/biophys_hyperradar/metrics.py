"""Evaluation metrics and grouped reporting."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Tuple, Union

import numpy as np


def binary_metrics(labels: Iterable[int], scores: Iterable[float], ece_bins: int = 15) -> dict[str, float]:
    labels_np = np.asarray(list(labels), dtype=np.int64)
    scores_np = np.asarray(list(scores), dtype=np.float64)
    if labels_np.size == 0:
        return {"n": 0.0, "eer": float("nan"), "ece": float("nan"), "accuracy": float("nan")}

    preds = (scores_np >= 0.5).astype(np.int64)
    best_acc, best_threshold = best_accuracy_threshold(labels_np, scores_np)
    eer, eer_threshold = equal_error_rate(labels_np, scores_np, return_threshold=True)
    return {
        "n": float(labels_np.size),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "ece": float(expected_calibration_error(labels_np, scores_np, bins=ece_bins)),
        "accuracy": float((preds == labels_np).mean()),
        "balanced_accuracy": float(balanced_accuracy(labels_np, preds)),
        "best_accuracy": float(best_acc),
        "best_accuracy_threshold": float(best_threshold),
    }


def equal_error_rate(
    labels: np.ndarray,
    scores: np.ndarray,
    return_threshold: bool = False,
) -> Union[float, Tuple[float, float]]:
    positives = labels == 1
    negatives = labels == 0
    if positives.sum() == 0 or negatives.sum() == 0:
        result = (float("nan"), float("nan"))
        return result if return_threshold else result[0]

    thresholds = np.r_[-np.inf, np.sort(np.unique(scores)), np.inf]
    fpr = np.empty_like(thresholds, dtype=np.float64)
    fnr = np.empty_like(thresholds, dtype=np.float64)
    for idx, threshold in enumerate(thresholds):
        pred_pos = scores >= threshold
        fpr[idx] = (pred_pos & negatives).sum() / negatives.sum()
        fnr[idx] = ((~pred_pos) & positives).sum() / positives.sum()
    best = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[best] + fnr[best]) / 2.0
    result = (float(eer), float(thresholds[best]))
    return result if return_threshold else result[0]


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    labels = labels.astype(np.float64)
    probs = np.clip(probs.astype(np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probs >= left) & (probs < right if right < 1.0 else probs <= right)
        if not mask.any():
            continue
        confidence = probs[mask].mean()
        accuracy = labels[mask].mean()
        ece += (mask.mean()) * abs(accuracy - confidence)
    return float(ece)


def best_accuracy_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if labels.size == 0:
        return float("nan"), float("nan")
    thresholds = np.r_[-np.inf, np.sort(np.unique(scores)), np.inf]
    best_acc = -1.0
    best_threshold = 0.5
    for threshold in thresholds:
        acc = ((scores >= threshold).astype(np.int64) == labels).mean()
        if acc > best_acc:
            best_acc = float(acc)
            best_threshold = float(threshold)
    return best_acc, best_threshold


def balanced_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    positives = labels == 1
    negatives = labels == 0
    if positives.sum() == 0 or negatives.sum() == 0:
        return float("nan")
    tpr = (preds[positives] == 1).mean()
    tnr = (preds[negatives] == 0).mean()
    return float((tpr + tnr) / 2.0)


def grouped_metrics(records: list[dict[str, object]], group_keys: list[str]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        for key in group_keys:
            grouped[f"{key}={record.get(key, 'unknown')}"].append(record)

    output: dict[str, dict[str, float]] = {}
    for group, items in sorted(grouped.items()):
        output[group] = binary_metrics(
            labels=[int(item["target"]) for item in items],
            scores=[float(item["score"]) for item in items],
        )
    return output
