import numpy as np, pytest
from audioshield.evaluation.metrics_v2 import (
    equal_error_rate, clustered_bootstrap_eer, resolve_clusters, CLUSTER_RESOLUTION)
from audioshield.training.supcon_guard import supcon_batch_valid

def _sklearn_eer(y, s):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr
    return fpr[np.nanargmin(np.abs(fpr - fnr))]

def test_eer_matches_sklearn_on_random():
    rng = np.random.default_rng(0)
    for _ in range(50):
        y = rng.integers(0, 2, size=400)
        if len(set(y)) < 2: continue
        s = rng.normal(size=400) + y * rng.uniform(0, 2)   # separable-ish
        ours, ref = equal_error_rate(y, s), _sklearn_eer(y, s)
        assert abs(ours - ref) < 0.02, f"EER mismatch: ours={ours:.3f} sklearn={ref:.3f}"

def test_eer_perfect_and_chance():
    y = np.array([0,0,0,1,1,1]); 
    assert equal_error_rate(y, np.array([0,0,0,1,1,1.0])) < 0.01   # perfect separation
    # pure chance ~ 0.5
    rng = np.random.default_rng(1); y = rng.integers(0,2,size=2000); s = rng.normal(size=2000)
    assert abs(equal_error_rate(y, s) - 0.5) < 0.06

def test_clustered_ci_wider_than_segment_ci():
    """Constructed within-cluster correlation -> clustered CI must be wider (the §4.6 fix)."""
    rng = np.random.default_rng(2)
    n_clusters, per = 30, 20
    y, s, cl = [], [], []
    for c in range(n_clusters):
        base = rng.normal()                      # cluster-level shared component
        lab = c % 2
        for _ in range(per):
            y.append(lab); s.append(base + rng.normal(0, 0.3) + lab*0.5); cl.append(f"c{c}")
    y = np.array(y); s = np.array(s); cl = np.array(cl, dtype=object)
    seg = clustered_bootstrap_eer(y, s, clusters=None, n_boot=300)
    clu = clustered_bootstrap_eer(y, s, clusters=cl, n_boot=300)
    seg_w = seg["ci_hi"] - seg["ci_lo"]; clu_w = clu["ci_hi"] - clu["ci_lo"]
    assert clu["clustered"] and not seg["clustered"]
    assert clu["n_clusters"] == 30
    assert clu_w > seg_w, f"clustered CI ({clu_w:.3f}) should exceed segment CI ({seg_w:.3f})"

def test_cluster_resolution_flags_fallback():
    from collections import namedtuple
    R = namedtuple("R", ["source_id"])
    rows_na = [R("NA")] * 10
    assert resolve_clusters(rows_na, "replaydf") is None    # all-NA -> per-segment fallback
    rows_ok = [R(f"sess{i//3}") for i in range(10)]
    g = resolve_clusters(rows_ok, "replaydf")
    assert g is not None and len(set(g)) == 4

def test_supcon_guard():
    # class 1 spans corpora {0,1}, class 0 spans {0} only -> invalid at min=2
    valid, diag = supcon_batch_valid(corpus_ids=[0,1,0], labels=[1,1,0], min_corpora_per_class=2)
    assert valid is False and diag["corpora_per_class"][0] == 1
    valid2, _ = supcon_batch_valid(corpus_ids=[0,1,0,1], labels=[1,1,0,0], min_corpora_per_class=2)
    assert valid2 is True
