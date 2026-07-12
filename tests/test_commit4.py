import numpy as np, pytest
from collections import namedtuple
from audioshield.data.balanced_weighting import (
    compute_joint_weights, empirical_class_corpus_mi, SamplerConfigError)

Row = namedtuple("Row", ["target", "corpus", "bona_fide_source"])

def make_pool():
    """Mirror the real mix: 4 two-class corpora + VCTK bona-only (the confound source)."""
    rows = []
    for corp, n_bona, n_spoof in [("asvspoof5", 300, 900), ("diffssd", 400, 400),
                                  ("fakeorreal", 500, 300), ("replaydf", 200, 200),
                                  ("vctk", 800, 0)]:
        rows += [Row(0, corp, corp)] * n_bona + [Row(1, corp, "na")] * n_spoof
    return rows

def _simulate(rows, weights, n=40000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rows), size=n, replace=True, p=weights)
    t = np.array([int(rows[i].target) for i in idx])
    c = np.array([rows[i].corpus for i in idx])
    return t, c

def _old_weights(rows):
    from collections import Counter, defaultdict
    n_corpora = len(set(r.corpus for r in rows))
    cls_n = Counter((r.corpus, r.target) for r in rows)
    classes_in_corpus = defaultdict(set)
    for r in rows: classes_in_corpus[r.corpus].add(r.target)
    w = np.array([(1.0/n_corpora)*(1.0/len(classes_in_corpus[r.corpus]))*(1.0/cls_n[(r.corpus, r.target)])
                  for r in rows]); return w / w.sum()

def test_old_weighting_reproduces_skew_and_confound():
    """Proof the NEW test would have caught the audit's finding."""
    rows = make_pool(); t, c = _simulate(rows, _old_weights(rows))
    spoof_frac = t.mean()
    assert spoof_frac < 0.45, f"expected skew away from 0.5, got spoof_frac={spoof_frac:.3f}"
    assert empirical_class_corpus_mi(t, c) > 0.05, "old weighting should show corpus->class MI"

def test_new_weighting_balances_class_and_kills_confound():
    rows = make_pool()
    w = compute_joint_weights(rows, bona_only_corpus_policy="exclude_from_class_conditional")
    t, c = _simulate(rows, w)
    assert abs(t.mean() - 0.5) < 0.02, f"class balance off: spoof_frac={t.mean():.3f}"
    mi = empirical_class_corpus_mi(t, c)
    assert mi < 0.05, f"corpus still predicts class: MI={mi:.3f} bits"
    # VCTK should be nearly absent (excluded to a 1% floor), not dominant
    vctk_frac = (c == "vctk").mean()
    assert vctk_frac < 0.05, f"excluded VCTK still has mass {vctk_frac:.3f}"

def test_min_corpora_per_class_enforced():
    rows = [Row(0, "onlybona", "x")] * 100 + [Row(1, "onlyspoof", "na")] * 100
    with pytest.raises(SamplerConfigError):
        compute_joint_weights(rows, min_corpora_per_class=2)

def test_matched_synthetic_not_yet_available():
    with pytest.raises(SamplerConfigError):
        compute_joint_weights(make_pool(), bona_only_corpus_policy="matched_synthetic")
