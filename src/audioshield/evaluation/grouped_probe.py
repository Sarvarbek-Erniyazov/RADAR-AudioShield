"""Group-honest linear probe for decodability measurement.
Audit ref: §4.7 — plain cross_val_score(cv=3) leaked related items (same speaker/
source recording) across folds, inflating probe accuracy and faking the
'decodability' side of the novelty claim's leg-1 contrast (reliance predicts
failure; decodability does not). Roadmap v3 Step 2a Commit 5.

Design invariant: grouping can only REMOVE leaked signal, never add it. Every
reported number is a lower-or-equal bound vs. the old ungrouped probe. We report
balanced accuracy + macro-F1 against an HONEST majority baseline (not raw accuracy
vs. an assumed-uniform chance), so class imbalance cannot masquerade as decodability.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

def _derive_groups(meta: dict | None, n: int) -> np.ndarray:
    """groups = source_id -> speaker_id -> file-prefix. meta maps key->array(len n).
    Missing/'NA' entries fall through to the next source; final fallback = unique
    per-item (i.e., no grouping for those rows, which is conservative)."""
    if meta:
        for key in ("source_id", "speaker_id"):
            if key in meta and meta[key] is not None:
                g = np.asarray(meta[key], dtype=object)
                if (g != "NA").mean() > 0.5:            # usable if majority non-NA
                    # fill NA rows with unique tokens so they never co-cluster
                    g = np.array([v if v not in ("NA", "", None) else f"__uniq_{i}"
                                  for i, v in enumerate(g)], dtype=object)
                    return g
        if "path" in meta and meta["path"] is not None:
            # file-prefix heuristic: strip trailing _NNN / index from the stem
            import re
            out = []
            for p in meta["path"]:
                stem = str(p).replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]
                out.append(re.sub(r"[_-]?\d+$", "", stem) or stem)
            return np.array(out, dtype=object)
    return np.array([f"__uniq_{i}" for i in range(n)], dtype=object)   # no grouping available

def grouped_probe(
    X: np.ndarray,
    y: np.ndarray,
    meta: dict | None = None,
    n_splits: int = 5,
    class_controlled_by: np.ndarray | None = None,
    seed: int = 13,
) -> dict:
    """Group-honest probe. Returns balanced_accuracy, macro_f1, majority_baseline,
    n_groups, and per-fold spread. If class_controlled_by is given (e.g. the true
    task label), the probe is run within each stratum and averaged — used for
    subspace-adjacent decodability where the confound must be held fixed.
    """
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y)
    n = len(y)
    groups = _derive_groups(meta, n)

    def _one(Xs, ys, gs):
        classes, counts = np.unique(ys, return_counts=True)
        majority = counts.max() / counts.sum()                 # honest baseline
        if len(classes) < 2:
            return dict(balanced_accuracy=float("nan"), macro_f1=float("nan"),
                        majority_baseline=float(majority), n_groups=len(np.unique(gs)),
                        note="single-class stratum")
        n_g = len(np.unique(gs))
        k = int(min(n_splits, n_g))
        if k < 2:
            return dict(balanced_accuracy=float("nan"), macro_f1=float("nan"),
                        majority_baseline=float(majority), n_groups=n_g,
                        note="too few groups for grouped CV")
        skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        bacc, mf1 = [], []
        for tr, te in skf.split(Xs, ys, groups=gs):
            # scaler fit on TRAIN only — no test leakage through normalization
            sc = StandardScaler().fit(Xs[tr])
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(sc.transform(Xs[tr]), ys[tr])
            pred = clf.predict(sc.transform(Xs[te]))
            bacc.append(balanced_accuracy_score(ys[te], pred))
            mf1.append(f1_score(ys[te], pred, average="macro"))
        return dict(
            balanced_accuracy=float(np.mean(bacc)), balanced_accuracy_std=float(np.std(bacc)),
            macro_f1=float(np.mean(mf1)), majority_baseline=float(majority),
            n_groups=n_g, n_splits=k,
        )

    if class_controlled_by is None:
        return _one(X, y, groups)
    strata = np.asarray(class_controlled_by)
    per = []
    for s in np.unique(strata):
        m = strata == s
        if m.sum() >= 10:
            per.append(_one(X[m], y[m], groups[m]))
    valid = [p for p in per if not np.isnan(p["balanced_accuracy"])]
    if not valid:
        return dict(balanced_accuracy=float("nan"), macro_f1=float("nan"),
                    note="no valid strata", n_strata=len(per))
    return dict(
        balanced_accuracy=float(np.mean([p["balanced_accuracy"] for p in valid])),
        macro_f1=float(np.mean([p["macro_f1"] for p in valid])),
        majority_baseline=float(np.mean([p["majority_baseline"] for p in valid])),
        class_controlled=True, n_strata=len(valid),
    )
