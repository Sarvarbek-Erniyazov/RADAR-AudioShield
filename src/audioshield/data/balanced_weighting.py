"""Joint class x corpus sampling weights. Audit ref: §4.4 (62.5/37.5 skew;
corpus predicts class) + §3.3 (VCTK bona-only confound). Roadmap v3 Step 2a Commit 4.

The prior weighting used (1/n_corpora)(1/n_classes_in_corpus)(1/count(corpus,class)).
The 1/n_classes_in_corpus term gives single-class corpora (VCTK: bona-only) DOUBLE
the per-cell mass of two-class corpora, tilting the sampled class ratio away from
50/50 AND making corpus a predictor of class (MI(corpus;class) > 0). Both effects
poison every downstream factor subspace.

New policy: target an explicit joint (class, corpus) distribution:
  - overall class balance is the primary constraint (default 50/50),
  - within each class, corpora contribute equally (subject to availability),
  - bona-only / spoof-only corpora are handled by `bona_only_corpus_policy`,
  - classes must be represented by >= min_corpora_per_class distinct corpora.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from typing import Sequence
import numpy as np

class SamplerConfigError(RuntimeError):
    pass

def compute_joint_weights(
    rows: Sequence,
    class_balance: tuple[float, float] = (0.5, 0.5),   # (bona=0, spoof=1)
    bona_only_corpus_policy: str = "exclude_from_class_conditional",
    min_corpora_per_class: int = 2,
) -> np.ndarray:
    """Return per-row sampling weights realizing the joint target. Pure/testable."""
    targets = np.array([int(r.target) for r in rows])
    corpora = np.array([r.corpus for r in rows])

    cells = Counter(zip(targets.tolist(), corpora.tolist()))     # (class, corpus) -> count
    corpora_by_class: dict[int, list[str]] = defaultdict(list)
    for (cls, corp) in cells:
        corpora_by_class[cls].append(corp)

    # Identify single-class corpora (e.g. VCTK bona-only).
    classes_of_corpus = defaultdict(set)
    for (cls, corp) in cells:
        classes_of_corpus[corp].add(cls)
    single_class_corpora = {c for c, cs in classes_of_corpus.items() if len(cs) == 1}

    # Policy for bona-only / spoof-only corpora.
    excluded: set[str] = set()
    if bona_only_corpus_policy == "exclude_from_class_conditional":
        excluded = set(single_class_corpora)
    elif bona_only_corpus_policy == "matched_synthetic":
        # Placeholder: matched synthesis is a Step-5/6 data operation, not built here.
        raise SamplerConfigError("matched_synthetic policy not implemented in 2a; use exclude_from_class_conditional")
    elif bona_only_corpus_policy != "include":
        raise SamplerConfigError(f"unknown bona_only_corpus_policy: {bona_only_corpus_policy}")

    # Effective corpora per class after exclusion.
    eff_corpora_by_class = {
        cls: sorted(set(cs) - excluded) for cls, cs in corpora_by_class.items()
    }
    for cls, cs in eff_corpora_by_class.items():
        if len(cs) < min_corpora_per_class:
            raise SamplerConfigError(
                f"class {cls} has {len(cs)} usable corpora {cs} < min_corpora_per_class="
                f"{min_corpora_per_class} (after {bona_only_corpus_policy} excluded {sorted(excluded)}). "
                "Add corpora or lower the threshold."
            )

    # Target mass per (class, corpus) cell: class_balance[cls] split equally across
    # that class's effective corpora. Excluded single-class corpora get residual mass
    # only if policy == include; under exclude they get ~0 training weight.
    class_target = {0: class_balance[0], 1: class_balance[1]}
    cell_target: dict[tuple[int, str], float] = {}
    for cls, cs in eff_corpora_by_class.items():
        share = class_target[cls] / max(len(cs), 1)
        for corp in cs:
            cell_target[(cls, corp)] = share
    # Excluded corpora: tiny non-zero floor so they still contribute a few bona examples
    # (useful as bona diversity) without driving the class ratio. 1% of their class mass.
    for corp in excluded:
        cls = next(iter(classes_of_corpus[corp]))
        cell_target[(cls, corp)] = 0.01 * class_target[cls] / max(len(excluded), 1)

    # Per-row weight = cell_target / cell_count  (so each cell's rows share its target mass).
    w = np.array([
        cell_target.get((int(r.target), r.corpus), 0.0) / cells[(int(r.target), r.corpus)]
        for r in rows
    ], dtype=np.float64)
    if w.sum() <= 0:
        raise SamplerConfigError("all-zero sampling weights — check policy/inputs")
    return w / w.sum()

def empirical_class_corpus_mi(sampled_targets: np.ndarray, sampled_corpora: np.ndarray) -> float:
    """MI(corpus; class) in bits over a sampled stream. 0 => corpus tells you nothing about class."""
    n = len(sampled_targets)
    p_c = Counter(sampled_corpora); p_y = Counter(sampled_targets.tolist())
    p_cy = Counter(zip(sampled_corpora.tolist(), sampled_targets.tolist()))
    mi = 0.0
    for (c, y), n_cy in p_cy.items():
        pxy = n_cy / n
        mi += pxy * np.log2(pxy / ((p_c[c] / n) * (p_y[y] / n)))
    return float(mi)
