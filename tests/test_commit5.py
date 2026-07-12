"""Commit-5 correctness: grouped probe must NOT inflate when related items share a
group, whereas the old ungrouped cross_val_score does. Audit §4.7."""
import numpy as np, pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from audioshield.evaluation.grouped_probe import grouped_probe, _derive_groups

def _leaky_dataset(seed=0):
    """Label is memorizable per-GROUP but NOT generalizable: each group has a unique
    random 'fingerprint' correlated with its label. Rows are INTERLEAVED (like a real
    manifest where a speaker's utterances are scattered), so shuffled segment-level CV
    puts the same group's items in both train and test (leak -> high acc), while grouped
    CV holds out whole groups (no leak -> chance)."""
    rng = np.random.default_rng(seed)
    n_groups, per = 40, 12
    fps = {g: rng.normal(size=32) * 8.0 for g in range(n_groups)}
    rows = []
    for g in range(n_groups):
        for _ in range(per):
            rows.append((fps[g] + rng.normal(size=32), g % 2, f"grp{g}"))
    rng.shuffle(rows)                                     # interleave: same group scattered
    X = np.vstack([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    groups = np.array([r[2] for r in rows], dtype=object)
    return X, y, groups

def test_ungrouped_cv_inflates_on_leaky_data():
    from sklearn.model_selection import StratifiedKFold
    X, y, _ = _leaky_dataset()
    # ungrouped stratified shuffled CV — the exact pattern audit §4.7 flagged
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    acc = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=skf).mean()
    assert acc > 0.85, f"expected inflated segment-CV accuracy, got {acc:.2f}"  # the bug, reproduced

def test_grouped_probe_does_not_inflate_on_leaky_data():
    X, y, groups = _leaky_dataset()
    res = grouped_probe(X, y, meta={"source_id": groups}, n_splits=5)
    # whole groups held out -> label not recoverable -> ~chance balanced accuracy
    assert res["balanced_accuracy"] < 0.65, f"grouped probe still inflated: {res['balanced_accuracy']:.2f}"
    assert res["n_groups"] == 40

def test_grouped_probe_recovers_real_generalizable_signal():
    """Sanity: when the signal IS generalizable (not group-bound), grouped probe finds it."""
    rng = np.random.default_rng(1)
    direction = rng.normal(size=32)
    y = rng.integers(0, 2, size=320)
    X = (y[:, None] * 2 - 1) * direction[None, :] + rng.normal(size=(320, 32))
    groups = np.array([f"g{i}" for i in range(320)], dtype=object)  # all unique
    res = grouped_probe(X, y, meta={"source_id": groups})
    assert res["balanced_accuracy"] > 0.9, f"failed to recover real signal: {res['balanced_accuracy']:.2f}"

def test_majority_baseline_reported_honestly():
    y = np.array([0]*90 + [1]*10)
    X = np.random.default_rng(0).normal(size=(100, 8))
    groups = np.array([f"g{i//2}" for i in range(100)], dtype=object)
    res = grouped_probe(X, y, meta={"source_id": groups})
    assert abs(res["majority_baseline"] - 0.9) < 1e-9   # honest 0.9, not assumed 0.5

def test_group_fallback_chain():
    meta = {"source_id": np.array(["NA"]*10), "speaker_id": np.array([f"s{i%3}" for i in range(10)])}
    g = _derive_groups(meta, 10)
    assert set(g) == {"s0", "s1", "s2"}                 # fell through source_id -> speaker_id

def test_class_controlled_mode_runs():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 16)); y = rng.integers(0, 2, size=200)
    strata = rng.integers(0, 2, size=200)               # e.g. task label held fixed
    groups = np.array([f"g{i//2}" for i in range(200)], dtype=object)
    res = grouped_probe(X, y, meta={"source_id": groups}, class_controlled_by=strata)
    assert res.get("class_controlled") is True
