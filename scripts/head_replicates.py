"""Task 3 (Step 4 gate prep): seeded head-replicate machinery.

docs/gate_prereg.md criterion 8 requires "consistent effect direction
across >=3 independently seeded replicates." This module trains a fresh
classifier head on cached embeddings with a given seed, N times, and
reports each replicate's signed intervention effect -- the mean logit
change under projecting a given factor subspace `U` out of the
embeddings, using that replicate's own freshly-fit head weight `w`
(mirrors `audioshield.reliance.metrics.prediction_change`'s definition,
just with `w` varying by seed instead of coming from a fixed checkpoint).
The sign of that effect is the "effect direction" criterion 8 asks about.

Only the head varies by seed; the factor subspace `U` is assumed already
estimated (by scripts/run_reliance_battery.py's existing LDA/probe
subspace estimators, from the SAME cached embeddings) and held fixed
across replicates -- retraining a lightweight logistic-regression head on
top of an already-embedded, already-cached feature matrix is the
"minutes, not hours" operation Roadmap v3 Step 4 describes ("heads
retrain on cached embeddings in minutes (3 seeds x conditions)").

Effect is measured on a held-out split (train_test_split, stratified,
seeded) rather than the head's own training data -- consistent with this
project's cross-fitting discipline elsewhere (nested_selection_effect_
crossfit in audioshield.reliance.crossfit): the seed varies the head's fit,
not which rows get to see their own effect.

No real cached embeddings exist on this machine (the embedding cache lives
on the collaborator machine) -- this module is unit-tested exclusively on
synthetic embeddings (tests/test_head_replicates.py). To run for real:

    COLLABORATOR PC (after the embedding cache + factor subspace already
    exist, e.g. from scripts/run_reliance_battery.py's own subspace
    estimation for a given battery):
    python scripts/head_replicates.py \\
        --embeddings /path/to/battery_features.npz \\
        --factor-subspace /path/to/battery_subspace.npy \\
        --seeds 13 29 47 \\
        --out analysis/step4/head_replicates_<battery_name>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from audioshield.reliance.metrics import project_out

DEFAULT_SEEDS = (13, 29, 47)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _log(msg: str) -> None:
    print(f"[{_timestamp()}] {msg}", flush=True)


def train_head(X: np.ndarray, y: np.ndarray, seed: int, **lr_kwargs) -> tuple[np.ndarray, float]:
    """Fit a plain logistic-regression head on (X, y) with random_state=seed.
    Returns (w, b): the fitted weight vector and intercept. Standardizing
    X is the caller's responsibility (this function fits exactly one
    model, no pipeline, so callers control preprocessing explicitly)."""
    clf = LogisticRegression(max_iter=1000, random_state=seed, **lr_kwargs).fit(X, y)
    return clf.coef_.reshape(-1), float(clf.intercept_[0])


def replicate_effect(X: np.ndarray, w: np.ndarray, b: float, U: np.ndarray) -> float:
    """Signed mean logit change under projecting the factor subspace `U`
    out of `X`, using the freshly-trained head (w, b). Built on
    audioshield.reliance.metrics.project_out -- the same naive-removal
    primitive scripts/run_reliance_battery.py's prediction_change uses --
    so a replicate's effect is defined identically to Phase A's own
    metric, just with `w` varying by seed."""
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    logit_before = X @ w + b
    logit_after = project_out(X, U) @ w + b
    return float(np.mean(logit_after - logit_before))


def run_head_replicates(
    X: np.ndarray, y: np.ndarray, U: np.ndarray, seeds=DEFAULT_SEEDS,
    effect_holdout_fraction: float = 0.3, **lr_kwargs,
) -> list[dict]:
    """Train len(seeds) independent head replicates and report each one's
    signed effect (see replicate_effect), measured on a held-out split of
    (X, y) so the effect isn't measured on the same rows the head was
    fit on. Never raises on a single seed's training/split failure -- an
    individual replicate that can't be fit (e.g. a degenerate y after the
    split) is recorded with effect=None, status="failed", rather than
    aborting the whole run, matching this project's established
    never-crash-on-a-single-unit-of-work convention (scripts/
    run_reliance_battery.py's per-battery try/except)."""
    replicates = []
    for seed in seeds:
        try:
            X_fit, X_eff, y_fit, y_eff = train_test_split(
                X, y, test_size=effect_holdout_fraction, random_state=seed,
                stratify=y if len(np.unique(y)) > 1 else None,
            )
            w, b = train_head(X_fit, y_fit, seed, **lr_kwargs)
            effect = replicate_effect(X_eff, w, b, U)
            replicates.append(dict(seed=int(seed), effect=effect, status="ok"))
        except Exception as exc:
            replicates.append(dict(seed=int(seed), effect=None, status="failed", reason=str(exc)))
    return replicates


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", required=True, type=Path,
                     help="npz with 'X' (n, d) and 'y' (n,) arrays -- a pre-joined cached-embedding "
                          "feature matrix for one battery/corpus (collaborator machine)")
    ap.add_argument("--factor-subspace", required=True, type=Path,
                     help="npy holding the (d, k) factor subspace U already estimated for this battery "
                          "(e.g. by scripts/run_reliance_battery.py's lda_subspace/crossfitted_probe_subspace)")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument("--effect-holdout-fraction", type=float, default=0.3)
    ap.add_argument("--out", type=Path, required=True, help="output JSON path (criterion-8-ready {'replicates': [...]})")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _log(f"head_replicates: loading embeddings from {args.embeddings}")
    with np.load(args.embeddings) as npz:
        X, y = npz["X"], npz["y"]
    _log(f"head_replicates: loaded X{X.shape} y{y.shape}")
    U = np.load(args.factor_subspace)
    _log(f"head_replicates: loaded factor subspace U{U.shape}")
    _log(f"head_replicates: running {len(args.seeds)} seeded replicates: {args.seeds}")
    replicates = run_head_replicates(X, y, U, seeds=args.seeds, effect_holdout_fraction=args.effect_holdout_fraction)
    for r in replicates:
        _log(f"head_replicates: seed={r['seed']} status={r['status']} effect={r.get('effect')}")
    payload = dict(schema_version=1, generated_at=_timestamp(), seeds=list(args.seeds),
                   embeddings_path=str(args.embeddings), factor_subspace_path=str(args.factor_subspace),
                   replicates=replicates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(args.out)
    _log(f"head_replicates: wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
