"""min_corpora_per_class guard for the cross-corpus SupCon term. Audit ref: §1
(declared-but-unimplemented). Roadmap v3 Step 2a Commit 6. The sampler-side joint
weighting landed in Commit 4; this is the per-batch loss-side guard: skip (and COUNT)
the contrastive term when a class in the batch spans too few corpora to form valid
cross-corpus positive pairs — never silently compute it on degenerate batches."""
from __future__ import annotations

def supcon_batch_valid(corpus_ids, labels, min_corpora_per_class: int = 2) -> tuple[bool, dict]:
    """Return (is_valid, diagnostics). Valid iff every class present in the batch spans
    >= min_corpora_per_class distinct corpora (needed for cross-corpus positives)."""
    from collections import defaultdict
    corpora_by_class = defaultdict(set)
    for cid, lab in zip(list(corpus_ids), list(labels)):
        corpora_by_class[int(lab)].add(int(cid))
    per_class = {c: len(s) for c, s in corpora_by_class.items()}
    valid = bool(per_class) and all(v >= min_corpora_per_class for v in per_class.values())
    return valid, {"corpora_per_class": per_class, "min_required": min_corpora_per_class}
