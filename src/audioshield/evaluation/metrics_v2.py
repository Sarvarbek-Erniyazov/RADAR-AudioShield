"""Correct EER + cluster-aware bootstrap CIs. Audit ref: §5 (EER implementation),
§4.6 (clustering: prior code used n_clusters=n -> anti-conservative CIs, esp. ReplayDF).
Roadmap v3 Step 2a Commit 6.
"""
from __future__ import annotations
import numpy as np

def equal_error_rate(y_true, scores) -> float:
    """EER via a single sort of scores (O(n log n)), higher score => more 'spoof' (label 1).
    Returns the rate where FPR == FNR (interpolated at the crossover)."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")          # descending; stable
    y = y[order]
    P = int((y == 1).sum()); N = int((y == 0).sum())
    # sweep threshold from high to low; each step admits one more positive prediction
    tp = np.cumsum(y == 1); fp = np.cumsum(y == 0)
    fnr = 1.0 - tp / P                                  # missed spoofs
    fpr = fp / N                                        # false alarms on bona
    # crossover where fpr - fnr changes sign
    diff = fpr - fnr
    k = np.argmin(np.abs(diff))
    # linear interpolation between k-1 and k for a smoother EER
    if 0 < k < len(diff) and diff[k-1] * diff[k] < 0:
        a, b = diff[k-1], diff[k]
        w = a / (a - b)
        eer = (1-w) * 0.5*(fpr[k-1]+fnr[k-1]) + w * 0.5*(fpr[k]+fnr[k])
    else:
        eer = 0.5 * (fpr[k] + fnr[k])
    return float(eer)

# Per-corpus cluster resolution: which metadata column defines an independent unit.
# Falls back to per-segment (anti-conservative) ONLY when no grouping column exists,
# and that fallback is recorded in the returned dict so it can never be silent.
CLUSTER_RESOLUTION = {
    "ai4t":     "source_id",     # one YouTube video = one cluster
    "replaydf": "source_id",     # one recording session/device (e.g. 09252b6aeda2)
    "inthewild":"source_id",     # speaker/source when available
    "diffssd":  "source_id",
}

def clustered_bootstrap_eer(y_true, scores, clusters=None, n_boot=1000, seed=13, ci=0.95):
    """Bootstrap EER CI resampling whole CLUSTERS (not segments). If clusters is None,
    falls back to per-segment resampling and flags it. Returns point, lo, hi, n_clusters,
    and clustering flag."""
    y = np.asarray(y_true).astype(int); s = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    point = equal_error_rate(y, s)
    if clusters is None:
        clusters = np.arange(len(y))                    # per-segment fallback
        clustered = False
    else:
        clusters = np.asarray(clusters); clustered = True
    uniq = np.unique(clusters)
    idx_by_cluster = {c: np.where(clusters == c)[0] for c in uniq}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in pick])
        e = equal_error_rate(y[idx], s[idx])
        if not np.isnan(e): boots.append(e)
    lo = float(np.percentile(boots, 100*(1-ci)/2))
    hi = float(np.percentile(boots, 100*(1+ci)/2))
    return {"eer": point, "ci_lo": lo, "ci_hi": hi, "n_clusters": int(len(uniq)),
            "clustered": clustered, "n_boot": len(boots)}

def resolve_clusters(rows, corpus: str):
    """Given ManifestRows for one corpus, return the cluster-id array using
    CLUSTER_RESOLUTION, or None (per-segment) if the column is all-NA/absent."""
    col = CLUSTER_RESOLUTION.get(corpus)
    if not col:
        return None
    vals = [getattr(r, col, "NA") for r in rows]
    if all(v in ("NA", "", None) for v in vals):
        return None
    # NA rows become unique singletons so they don't co-cluster
    return np.array([v if v not in ("NA","",None) else f"__uniq_{i}"
                     for i, v in enumerate(vals)], dtype=object)
