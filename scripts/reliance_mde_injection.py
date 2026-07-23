"""Gamma-injection MINIMUM-DETECTABLE-EFFECT (MDE) tool for the model-space
causal-reliance apparatus (Roadmap v3 Step 3/4 -- the C4 "decodability
without causal reliance" finding).

SIBLING, NOT REPLACEMENT: scripts/reliance_sensitivity.py is the resolution-
FLOOR report (how small a decision_flip_rate a fold size could have detected,
1 / mean(n_effect)); it post-processes an already-computed battery JSON and
never runs an intervention. THIS tool is its sibling on the other axis: it
runs the REAL intervention pipeline against a KNOWN, planted effect of
controllable magnitude and asks "how large must a factor reliance be before
the apparatus's own C4 detector fires?" The floor report bounds the effect
from below by fold size; this tool bounds the smallest effect the whole
detector (subspace fit + ablation + equal-norm random controls) can
distinguish from noise -- the two answer different questions and neither
subsumes the other.

WHAT C4 ACTUALLY VERDICTS ON (verified from scripts/run_gate.py's
criterion_4_intervention_vs_random and docs/gate_prereg.md's 2026-07-23 #2
amendment -- read both directly, never trusted a docstring's claim about
another file): per checkpoint, on the BEHAVIORAL functional
per_checkpoint[ckpt].prediction_change_control (effect =
mean_abs_logit_change from removal_control_report -- the change in the head's
OWN logits when the factor subspace is projected out), the main effect fires
iff true_effect > random_mean + 2*random_std over the 20 equal-norm random
controls, and C4 requires that to hold on a MAJORITY (>=50%) of a
checkpoint's folds. This tool's per-cell `exceeds_random` is exactly that
same >=50%-of-folds majority rule (DETECTION_MIN_FRACTION), so a cell that
"trips" here is a cell C4 would count as a main-effect pass.

DESIGN (per battery x checkpoint x gamma):
  - u1: the top factor direction, fit ONCE per (battery, checkpoint) on that
    checkpoint's own assembled (row-capped) rows via lda_subspace(k=1) -- the
    PLANTED GROUND TRUTH. It is deliberately fixed once and reused for every
    gamma; the estimators UNDER TEST (both lda and probe) still refit their
    OWN subspaces inside the crossfit, on selection folds, so detection is
    never handed the answer -- it has to recover a subspace containing u1 from
    held-out data on its own.
  - injected head: w_gamma = w + gamma * ||w|| * u1 (b unchanged). u1 is unit-
    norm, so ||w_gamma - w|| == gamma * ||w|| exactly -- gamma is the injected
    reliance magnitude as a fraction of the head's own norm, a coordinate-free
    dial.
  - intervention: the IDENTICAL model-space pipeline the real battery runs --
    scripts/run_reliance_modelspace.run_checkpoint_crossfit, called with a
    single-checkpoint dict whose w is w_gamma. No forked math, no
    reimplemented metric: prediction_change / removal_control_report /
    lda_subspace / crossfitted_probe_subspace / run_nested_crossfit are the
    unmodified imports the model-space consumer itself uses. The only new code
    here is the ORCHESTRATION (the gamma sweep and the gamma* reduction).

  gamma=0 is the SPECIFICITY ARM: with the unmodified head it must NOT trip
  (in the committed data the factor true_effect is 1.4e-7-1.6e-3 against a
  random_mean+2sigma ~= 0.08-0.16 bar -- no reliance), AND its per-fold
  true_effect reproduces the committed battery's prediction_change_control
  values fold-for-fold (same seed -> same folds -> same frozen subspace ->
  same w -> same removal_control_report). That reproduction doubles as a live
  end-to-end cross-validation of this whole tool the moment it is run on the
  collaborator machine against the real cache: if gamma=0 does NOT reproduce
  analysis/step3/reliance_modelspace_prereg*.json, the tool (or its inputs)
  is wrong and gamma* is not to be trusted.

  gamma* per (battery, estimator, checkpoint) is the smallest gamma whose cell
  trips. Non-monotonicity (a larger gamma that stops tripping) is reported
  honestly (monotonic=false, a note, and the full trip-flag list), never
  masked: gamma* stays the smallest tripping gamma but the threshold is
  flagged as not clean.

WHERE THIS RUNS: authored and synthetic-tested here; the REAL run happens on
the collaborator machine, which alone has the Phase B model-space cache and
the checkpoint .pt files (neither exists in this checkout -- same situation
scripts/run_reliance_modelspace.py documents). The sha256 head/cache pairing
guard (load_model_space_embeddings' recorded checkpoint_sha256 vs the head
file's own sha256) stays ACTIVE here exactly as in the real consumer -- a
mispairing is reported not_estimable (never silently ablates a factor
subspace out of an embedding paired with a different checkpoint's classifier
weight, which would produce a finite, plausible, meaningless number).

NEVER-CRASH / ATOMIC: a checkpoint whose cache or head can't load (or is
mispaired, or is degenerate) is reported status="not_estimable" with a
reason; the process always writes its JSON (temp file + os.replace) and exits
0. This is a measurement tool, not a build gate.

Speed: --n-boot default 0 and --max-rows-per-level default 500 (the validated
sizing). This tool does NOT run the headline grouped bootstrap at all
(run_checkpoint_crossfit doesn't); the bootstrap CI is irrelevant to gamma*
-- the 20 equal-norm random controls computed inside every fold's effect_fn
are the detection reference, and they are always computed. --n-boot is
accepted only for CLI parity with run_reliance_modelspace.py.

Usage (real run -- collaborator machine, once the Phase B cache exists):
    python scripts/reliance_mde_injection.py \\
        --model-space-cache-root analysis/step3/_embcache_modelspace \\
        --ckpt-dir /e/AI_voice_detection/checkpoint_backup \\
        --checkpoints e007_A_fresh e007_B_fresh e007_C_xlsr_fresh \\
        --manifest-dir manifests/v2 \\
        --corpus replaydf --factor generator_id language \\
        --gammas 0 0.01 0.02 0.05 0.1 0.2 0.5 \\
        --out analysis/step3/reliance_mde_injection.json
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from audioshield.data.manifest import read_manifest
from audioshield.reliance.subspaces import lda_subspace

# Read-only imports -- run_reliance_modelspace.py and run_reliance_battery.py
# are NEVER modified by this script; every intervention/metric primitive is
# reused verbatim so this tool cannot silently diverge from what the real
# model-space consumer (and therefore the gate) actually computes. Same
# scripts-on-sys.path convention the model-space consumer itself uses to
# import the battery module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_reliance_modelspace import (  # noqa: E402
    DEFAULT_MODEL_SPACE_CACHE_ROOT,
    load_checkpoint_head,
    load_model_space_embeddings,
    run_checkpoint_crossfit,
)
from run_reliance_battery import (  # noqa: E402
    BATTERIES,
    CKPT_DIR,
    CORPUS_DIR,
    DEFAULT_RANKS,
    RUNS,
    _git_sha,
    _log,
    _write_json_atomic,
    cap_rows_per_level,
    join_cache_to_manifest,
    ranks_for_n_levels,
    select_battery_rows,
)

SCHEMA_VERSION = 1
STATUS_OK = "ok"
STATUS_NOT_ESTIMABLE = "not_estimable"

# The C4 detection rule (docs/gate_prereg.md 2026-07-23 #2; run_gate.py
# criterion_4_intervention_vs_random, min_fraction=0.5): a checkpoint's main
# effect fires when a MAJORITY of its folds each have
# true_effect > random_mean + 2*random_std. This tool's per-cell trip flag is
# that same rule, scoped to one estimator (finer than C4, which pools both
# estimators' folds -- but the tool reports per estimator, so gamma* can be
# read separately for lda vs probe).
DETECTION_MIN_FRACTION = 0.5

# Task defaults deliberately different from run_reliance_modelspace.py's:
#  - no bootstrap CI (irrelevant to gamma*; the 20 random controls decide it);
#  - the validated 500-rows-per-level sizing rather than the battery's 2000.
DEFAULT_MDE_N_BOOT = 0
DEFAULT_MDE_MAX_ROWS_PER_LEVEL = 500
DEFAULT_GAMMAS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
DEFAULT_OUT = Path("analysis/step3/reliance_mde_injection.json")

ESTIMATORS = ("lda", "probe")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _not_estimable(reason: str, **extra) -> dict:
    return dict(status=STATUS_NOT_ESTIMABLE, reason=reason, **extra)


# ---------------------------------------------------------------------------
# The planted ground truth + the injected head. These two functions are the
# ONLY genuinely new math in this tool; everything downstream is the
# unmodified intervention pipeline.
# ---------------------------------------------------------------------------


def plant_top_factor_direction(Z: np.ndarray, factor: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The planted ground-truth direction u1: the top (rank-1) class-conditional
    LDA factor discriminant, fit ONCE on the assembled rows. Returned as a unit
    (d,) vector (lda_subspace returns an orthonormal basis, so its single column
    already has norm 1; the eigenvector's arbitrary sign is irrelevant -- the
    removal effect depends only on the SPAN of u1, not its sign).

    LDA (not the probe) defines the plant so the plant is deterministic and
    estimator-agnostic in provenance; both estimators under test still refit
    their own subspaces inside the crossfit, so this does not hand either
    estimator the answer. Raises ValueError (propagated by the caller into a
    not_estimable) if the factor is too degenerate to admit a discriminant.
    """
    U1 = lda_subspace(Z, factor, y, k=1, mode="within_class")
    if U1.shape[1] == 0:
        raise ValueError("planted top factor direction is empty (degenerate factor subspace)")
    return np.asarray(U1[:, 0], dtype=np.float64)


def inject_head(w: np.ndarray, u1: np.ndarray, gamma: float) -> np.ndarray:
    """w_gamma = w + gamma * ||w|| * u1, with u1 unit-norm. The injected reliance
    magnitude is gamma as a fraction of the head's own norm: by construction
    ||w_gamma - w|| == gamma * ||w|| exactly (the equal-norm relation the MDE is
    reported in)."""
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    u1 = np.asarray(u1, dtype=np.float64).reshape(-1)
    return w + gamma * float(np.linalg.norm(w)) * u1


# ---------------------------------------------------------------------------
# Per-cell aggregation across a crossfit's folds. The detection functional is
# the BEHAVIORAL prediction_change_control, exactly as C4 reads it.
# ---------------------------------------------------------------------------


def summarize_cell(fold_results: list[dict], run: str, gamma: float,
                   min_fraction: float = DETECTION_MIN_FRACTION) -> dict:
    """One (estimator, checkpoint, gamma) cell, aggregated across a crossfit's
    folds. Reads per_checkpoint[run].prediction_change_control (behavioral
    functional, mean_abs_logit_change) and the sibling prediction_change.
    decision_flip_rate -- the exact fields C4 verdicts on.

    true_effect/random_mean/random_std/task_direction_effect/decision_flip_rate
    are fold-MEANS (magnitude reporting, comparable to the committed battery's
    per_checkpoint values at gamma=0). `exceeds_random` is the AUTHORITATIVE
    trip flag: C4's >=50%-of-folds majority of the per-fold
    true_effect > random_mean + 2*random_std flags. `exceeds_random_pooled` is
    the same inequality applied to the reported fold-mean aggregates (so the
    reported numbers are self-checkable) -- it can differ from the majority
    flag near the threshold and is NOT what gamma* uses.
    """
    te, rm, rs, tde, dfr, per_fold_exceeds = [], [], [], [], [], []
    for fold in fold_results:
        per_ckpt = fold.get("effect", {}).get("per_checkpoint", {}).get(run, {})
        pcc = per_ckpt.get("prediction_change_control")
        if isinstance(pcc, dict) and isinstance(pcc.get("true_effect"), (int, float)):
            te.append(float(pcc["true_effect"]))
            if isinstance(pcc.get("random_mean"), (int, float)):
                rm.append(float(pcc["random_mean"]))
            if isinstance(pcc.get("random_std"), (int, float)):
                rs.append(float(pcc["random_std"]))
            if isinstance(pcc.get("task_direction_effect"), (int, float)):
                tde.append(float(pcc["task_direction_effect"]))
            if isinstance(pcc.get("exceeds_random"), bool):
                per_fold_exceeds.append(pcc["exceeds_random"])
        pc = per_ckpt.get("prediction_change")
        if isinstance(pc, dict) and isinstance(pc.get("decision_flip_rate"), (int, float)):
            dfr.append(float(pc["decision_flip_rate"]))

    n_folds = len(te)
    mean_te = float(np.mean(te)) if te else float("nan")
    mean_rm = float(np.mean(rm)) if rm else float("nan")
    mean_rs = float(np.mean(rs)) if rs else float("nan")
    exceeds_fraction = float(np.mean(per_fold_exceeds)) if per_fold_exceeds else 0.0
    return dict(
        gamma=float(gamma),
        true_effect=mean_te,
        random_mean=mean_rm,
        random_std=mean_rs,
        exceeds_random=bool(exceeds_fraction >= min_fraction),  # C4's >=50%-of-folds rule -- authoritative
        exceeds_random_fraction=exceeds_fraction,
        exceeds_random_pooled=bool(np.isfinite(mean_te) and np.isfinite(mean_rm) and np.isfinite(mean_rs)
                                   and mean_te > mean_rm + 2 * mean_rs),
        n_folds=n_folds,
        n_folds_exceeding=int(sum(per_fold_exceeds)),
        decision_flip_rate=float(np.mean(dfr)) if dfr else float("nan"),
        task_direction_effect=float(np.mean(tde)) if tde else float("nan"),
    )


def derive_gamma_star(cells: list[dict]) -> dict:
    """gamma* = smallest gamma whose cell trips (`exceeds_random`), plus an
    honest monotonicity report. `cells` is assumed sorted ascending by gamma.

    Non-monotonicity (detection trips at some gamma but NOT at a larger one) is
    reported, never masked: gamma_star stays the smallest tripping gamma,
    monotonic=false, and a note names the offending larger gamma(s). trip_flags
    lists every gamma's flag so the raw pattern is always inspectable.
    """
    trip_flags = [dict(gamma=c["gamma"], exceeds_random=c["exceeds_random"]) for c in cells]
    tripped = [c["gamma"] for c in cells if c["exceeds_random"]]
    gamma_star = min(tripped) if tripped else None

    flags = [c["exceeds_random"] for c in cells]
    monotonic, note = True, None
    if any(flags):
        first = flags.index(True)
        if not all(flags[first:]):
            monotonic = False
            offenders = [cells[i]["gamma"] for i in range(first, len(cells)) if not flags[i]]
            note = (
                f"non-monotonic detection: first trips at gamma={cells[first]['gamma']:g} but does NOT "
                f"trip at larger gamma(s) {offenders} -- gamma_star is the smallest tripping gamma, but the "
                "detection threshold is not a clean step; reported as-is, not masked"
            )
    return dict(gamma_star=gamma_star, trip_flags=trip_flags, monotonic=monotonic, non_monotonic_note=note)


# ---------------------------------------------------------------------------
# Per-checkpoint MDE estimation. estimate_checkpoint_from_arrays is the pure,
# in-memory core (unit-testable with synthetic Z, no cache/.pt); estimate_
# checkpoint wraps it with the real cache load + sha256 pairing guard.
# ---------------------------------------------------------------------------


def estimate_checkpoint_from_arrays(
    Z: np.ndarray, factor: np.ndarray, y: np.ndarray, groups: np.ndarray, head: dict, run: str,
    gammas: list[float], ranks: list[int], seed: int,
    max_rows_per_level: int | None = DEFAULT_MDE_MAX_ROWS_PER_LEVEL,
    min_fraction: float = DETECTION_MIN_FRACTION, label: str = "", log=None,
) -> dict:
    """Run the full gamma sweep for ONE checkpoint on already-selected
    (Z, factor, y, groups). `head` is a loaded head dict (w, b, w_dim, ...).
    Returns an "ok" result (per-estimator gamma grids + gamma_star) or a
    not_estimable dict. Independent per checkpoint by construction -- unlike
    run_reliance_modelspace's battery, no cross-checkpoint row alignment is
    needed, since each checkpoint's gamma* is measured in its own space.
    """
    log = log or (lambda msg: _log(f"[mde] {msg}"))
    label = label or run

    if max_rows_per_level is not None:
        Z, factor, y, groups = cap_rows_per_level(Z, factor, y, groups, max_rows_per_level, seed)

    n_levels = int(len(np.unique(factor)))
    valid_ranks = ranks_for_n_levels(list(ranks), n_levels)
    n_groups = int(len(np.unique(groups)))
    n_outer = min(5, n_groups)
    if not valid_ranks:
        return _not_estimable(f"no requested rank <= n_levels-1={n_levels - 1}",
                              n_rows=int(len(y)), n_levels=n_levels, n_groups=n_groups)
    if n_outer < 2:
        return _not_estimable(f"only {n_groups} group(s) available -- cannot form nested folds",
                              n_rows=int(len(y)), n_levels=n_levels, n_groups=n_groups)

    try:
        u1 = plant_top_factor_direction(Z, factor, y)
    except ValueError as e:
        return _not_estimable(f"cannot fit planted top factor direction (u1): {e}",
                              n_rows=int(len(y)), n_levels=n_levels, n_groups=n_groups)

    w = np.asarray(head["w"], dtype=np.float64).reshape(-1)
    w_dim = int(head.get("w_dim", w.shape[0]))
    if w.shape[0] != Z.shape[1]:
        return _not_estimable(
            f"w is {w.shape[0]}-d but the model-space embedding is {Z.shape[1]}-d -- refusing to inject",
            n_rows=int(len(y)), n_levels=n_levels, n_groups=n_groups)
    w_norm = float(np.linalg.norm(w))
    base_ck = dict(w=w, b=float(head.get("b", 0.0)), w_dim=w_dim, w_dim_mismatch=False,
                   ckpt_layer_center=None, ckpt_layer_band=None, layer_pooling="model_space",
                   band_weights=None, w_layer_mismatch=False)

    # The group-count print the operator cost-models from: cost scales with
    # (2 estimators x n_outer folds) crossfit units per gamma.
    log(f"{label}: n_rows={len(y)} n_levels={n_levels} n_groups={n_groups} n_outer={n_outer} "
        f"valid_ranks={valid_ranks} |gammas|={len(gammas)} -> {len(gammas)} crossfit runs "
        f"(each = 2 estimators x {n_outer} folds); w_norm={w_norm:.4g}")

    est_cells: dict[str, list[dict]] = {est: [] for est in ESTIMATORS}
    for gamma in gammas:
        w_gamma = inject_head(w, u1, gamma)
        ck_gamma = dict(base_ck, w=w_gamma)
        try:
            estimator_results = run_checkpoint_crossfit(
                run, ck_gamma, Z, factor, y, groups, valid_ranks, n_outer, seed,
            )
        except Exception as e:  # a single gamma failing must not sink the others
            log(f"{label}: gamma={gamma:g} crossfit failed ({type(e).__name__}: {e}) -- recording error cell")
            for est in ESTIMATORS:
                est_cells[est].append(dict(gamma=float(gamma), exceeds_random=False, n_folds=0,
                                           error=f"{type(e).__name__}: {e}"))
            continue
        for est in ESTIMATORS:
            fold_results = estimator_results.get(est, {}).get("fold_results", [])
            est_cells[est].append(summarize_cell(fold_results, run, gamma, min_fraction))

    estimators_out = {}
    for est in ESTIMATORS:
        cells = est_cells[est]
        estimators_out[est] = dict(gammas=cells, **derive_gamma_star(cells))

    return dict(
        status=STATUS_OK, n_rows=int(len(y)), n_levels=n_levels, n_groups=n_groups, n_outer=n_outer,
        valid_ranks=valid_ranks, w_norm=w_norm,
        planted_direction=dict(estimator="lda", rank=1, mode="within_class", norm=float(np.linalg.norm(u1))),
        estimators=estimators_out,
    )


def estimate_checkpoint(
    spec: dict, run: str, head: dict, cache_root: Path, manifest_df: pd.DataFrame, corpus_dir: str,
    gammas: list[float], ranks: list[int], seed: int,
    max_rows_per_level: int | None = DEFAULT_MDE_MAX_ROWS_PER_LEVEL,
    min_fraction: float = DETECTION_MIN_FRACTION, log=None,
) -> dict:
    """Load this checkpoint's model-space cache, verify the sha256
    (embedding, head) pairing, join to the manifest, select the battery's
    rows, then run estimate_checkpoint_from_arrays. Any load/pairing/degeneracy
    failure degrades to not_estimable (never raises) -- the never-crash
    contract. The pairing guard is ACTIVE and identical to the real consumer's:
    a cache whose recorded checkpoint_sha256 does not equal the head file's own
    sha256 is refused (reported not_estimable), never ablated.
    """
    log = log or (lambda msg: _log(f"[mde] {msg}"))
    label = f"{spec['name']}/{run}"
    cache_root = Path(cache_root)  # tolerate a str path -- load_model_space_embeddings needs a Path

    try:
        cache_paths, cache_emb, cache_sha256 = load_model_space_embeddings(
            cache_root, head["checkpoint_stem"], corpus_dir
        )
    except FileNotFoundError as e:
        return _not_estimable(f"model-space cache not found: {e}")
    except Exception as e:
        return _not_estimable(f"model-space cache failed to load ({type(e).__name__}: {e})")

    # THE PAIRING GUARANTEE (kept active, never bypassed): a 256-d cache paired
    # with a 256-d head from a DIFFERENT checkpoint is dimension-clean but
    # scientifically meaningless. The shard's own recorded checkpoint_sha256
    # must equal this run's head file's own sha256.
    if cache_sha256 is None:
        return _not_estimable(
            "model-space cache has no recorded checkpoint_sha256 (missing/unparseable meta) -- refusing "
            "to pair an unverified embedding cache with this checkpoint's head")
    if cache_sha256 != head.get("checkpoint_sha256"):
        return _not_estimable(
            f"MISPAIRED (embedding, head): cache sha256={str(cache_sha256)[:16]}... != head "
            f"sha256={str(head.get('checkpoint_sha256'))[:16]}... -- refusing to inject/ablate against a "
            "cache extracted from a different checkpoint")

    try:
        joined_df, joined_emb, _ = join_cache_to_manifest(cache_paths, cache_emb, manifest_df, corpus_dir)
        if head["w_dim"] != joined_emb.shape[1]:
            return _not_estimable(
                f"w is {head['w_dim']}-d but the model-space cache is {joined_emb.shape[1]}-d -- not commensurate")
        Z, factor, y, groups, _ = select_battery_rows(joined_df, joined_emb, spec)
    except Exception as e:
        return _not_estimable(f"join/row-selection failed ({type(e).__name__}: {e})")

    if len(y) == 0:
        return _not_estimable("0 battery rows after join/row-selection")

    try:
        return estimate_checkpoint_from_arrays(
            Z, factor, y, groups, head, run, gammas, ranks, seed,
            max_rows_per_level=max_rows_per_level, min_fraction=min_fraction, label=label, log=log,
        )
    except Exception as e:  # final never-crash backstop for an unexpected orchestration failure
        return _not_estimable(f"unexpected failure in gamma sweep ({type(e).__name__}: {e})")


# ---------------------------------------------------------------------------
# Battery-level orchestration + the paper_sentence reduction
# ---------------------------------------------------------------------------


def run_battery_mde(
    spec: dict, checkpoint_heads: dict[str, dict], cache_root: Path, manifest_dir: Path,
    gammas: list[float], ranks: list[int], seed: int,
    max_rows_per_level: int | None = DEFAULT_MDE_MAX_ROWS_PER_LEVEL, log=None,
) -> dict:
    """One battery: run every requested checkpoint's gamma sweep independently."""
    log = log or _log
    corpus = spec["corpus"]
    corpus_dir = CORPUS_DIR[corpus]

    per_checkpoint: dict[str, dict] = {}
    try:
        manifest_rows = read_manifest(manifest_dir / f"{corpus}.csv")
        manifest_df = pd.DataFrame([asdict(r) for r in manifest_rows])
    except Exception as e:
        # A missing/unreadable manifest fails only THIS battery, and only into
        # a legible not_estimable per checkpoint -- never a crash.
        for run in checkpoint_heads:
            per_checkpoint[run] = _not_estimable(f"manifest for corpus {corpus!r} unreadable "
                                                 f"({type(e).__name__}: {e})")
        return dict(name=spec["name"], corpus=corpus, factor=spec["factor"], grouping=spec["grouping"],
                    per_checkpoint=per_checkpoint)

    for run, head in checkpoint_heads.items():
        if head.get("load_error"):
            per_checkpoint[run] = _not_estimable(f"checkpoint head not loaded: {head['load_error']}")
            continue
        per_checkpoint[run] = estimate_checkpoint(
            spec, run, head, cache_root, manifest_df, corpus_dir, gammas, ranks, seed,
            max_rows_per_level=max_rows_per_level, log=log,
        )
    return dict(name=spec["name"], corpus=corpus, factor=spec["factor"], grouping=spec["grouping"],
                per_checkpoint=per_checkpoint)


def derive_paper_sentence(per_battery: dict[str, dict], gammas: list[float]) -> dict:
    """One top-level summary of gamma* across every estimable
    (battery, estimator, checkpoint) cell -- the range the paper quotes for the
    apparatus's minimum detectable effect, and the specificity-arm check."""
    finite: list[dict] = []
    never_tripped: list[dict] = []
    specificity_violations: list[dict] = []
    n_not_estimable = 0

    for bname, battery in per_battery.items():
        for run, ck in battery.get("per_checkpoint", {}).items():
            if ck.get("status") != STATUS_OK:
                n_not_estimable += 1
                continue
            for est, est_res in ck.get("estimators", {}).items():
                gs = est_res.get("gamma_star")
                label = dict(battery=bname, estimator=est, checkpoint=run)
                if gs is None:
                    never_tripped.append(label)
                else:
                    finite.append(dict(**label, gamma_star=gs, monotonic=est_res.get("monotonic", True)))
                # specificity arm: the gamma=0 cell must not trip
                zero = next((c for c in est_res.get("gammas", []) if c.get("gamma") == 0.0), None)
                if zero is not None and zero.get("exceeds_random"):
                    specificity_violations.append(dict(**label, true_effect=zero.get("true_effect")))

    gamma_star_values = sorted(f["gamma_star"] for f in finite)
    gamma_star_min = gamma_star_values[0] if gamma_star_values else None
    gamma_star_max = gamma_star_values[-1] if gamma_star_values else None
    max_gamma = max(gammas) if gammas else None
    has_zero_arm = 0.0 in [float(g) for g in gammas]

    if specificity_violations:
        text = (
            "SPECIFICITY ARM VIOLATED: the unmodified head (gamma=0) trips C4's main-effect detector in "
            f"{len(specificity_violations)} cell(s) -- the tool or its inputs are suspect and gamma* is NOT to "
            "be trusted until this is explained (gamma=0 must reproduce the committed battery's "
            "non-tripping prediction_change_control values)."
        )
    elif gamma_star_min is not None:
        text = (
            f"On the model-space causal-reliance apparatus, a planted factor-aligned reliance of magnitude "
            f"gamma*.||w|| is first detected by C4's own criterion (main effect exceeds the equal-norm random "
            f"control at >2 sigma on a majority of folds) at gamma* in [{gamma_star_min:g}, {gamma_star_max:g}] "
            f"across {len(finite)} checkpoint x estimator x battery cell(s)"
            + (f", with {len(never_tripped)} cell(s) not tripping up to the largest tested gamma={max_gamma:g}"
               if never_tripped else "")
            + (". The unmodified head (gamma=0) trips in no cell, so the observed effects correspond to "
               f"gamma < {gamma_star_min:g} -- any real factor reliance is below the apparatus's minimum "
               "detectable effect." if has_zero_arm else ".")
        )
    else:
        text = (
            "No injected reliance was detected in any cell up to the largest tested "
            f"gamma={max_gamma:g}" + (" (the specificity arm holds: gamma=0 trips nowhere)" if has_zero_arm else "")
            + f" -- the apparatus's minimum detectable effect exceeds {max_gamma:g}.||w|| on this data."
        )

    return dict(
        text=text,
        gamma_star_min=gamma_star_min,
        gamma_star_max=gamma_star_max,
        gamma_star_values=gamma_star_values,
        n_estimable_cells=len(finite) + len(never_tripped),
        n_tripped=len(finite),
        n_never_tripped=len(never_tripped),
        n_not_estimable_checkpoints=n_not_estimable,
        specificity_arm_present=has_zero_arm,
        specificity_violations=specificity_violations,
    )


def build_report(per_battery: dict[str, dict], params: dict, gammas: list[float],
                 warnings: list[str] | None = None) -> dict:
    return dict(
        schema_version=SCHEMA_VERSION,
        generated_at=_timestamp(),
        git_sha=_git_sha(),
        tool="reliance_mde_injection",
        detection=dict(
            functional="prediction_change_control.true_effect (mean_abs_logit_change)",
            rule="true_effect > random_mean + 2*random_std over 20 equal-norm random controls, "
                 "on a majority (>=DETECTION_MIN_FRACTION) of a checkpoint's folds -- C4's own criterion "
                 "(docs/gate_prereg.md 2026-07-23 #2)",
            min_fraction=DETECTION_MIN_FRACTION,
            injection="w_gamma = w + gamma * ||w|| * u1; u1 = top LDA factor direction fit once per "
                      "(battery, checkpoint); estimators under test refit their own subspaces in the crossfit",
        ),
        params=params,
        per_battery=per_battery,
        paper_sentence=derive_paper_sentence(per_battery, gammas),
        warnings=warnings or [],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Mirrors scripts/run_reliance_modelspace.py's args, plus --gammas and --out.
    ap.add_argument("--model-space-cache-root", default=DEFAULT_MODEL_SPACE_CACHE_ROOT)
    ap.add_argument("--manifest-dir", default="manifests/v2")
    ap.add_argument("--corpus", nargs="*", default=None, help="restrict to battery corpora in this list")
    ap.add_argument("--factor", nargs="*", default=None, help="restrict to battery factors in this list")
    ap.add_argument("--ranks", nargs="+", type=int, default=list(DEFAULT_RANKS))
    ap.add_argument("--n-boot", type=int, default=DEFAULT_MDE_N_BOOT,
                    help="accepted for CLI parity with run_reliance_modelspace.py only -- this tool runs NO "
                         "headline bootstrap (irrelevant to gamma*); the 20 per-fold random controls decide "
                         "detection (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    ap.add_argument("--checkpoints", nargs="+", default=list(RUNS))
    ap.add_argument("--max-rows-per-level", type=int, default=DEFAULT_MDE_MAX_ROWS_PER_LEVEL,
                    help="validated model-space sizing (default: %(default)s); 0 or negative disables the cap")
    ap.add_argument("--gammas", nargs="+", type=float, default=list(DEFAULT_GAMMAS),
                    help="injected reliance magnitudes, as fractions of ||w|| (default: %(default)s). "
                         "Include 0 for the specificity arm.")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    out_path = Path(args.out)
    cache_root = Path(args.model_space_cache_root)
    ckpt_dir = Path(args.ckpt_dir)
    manifest_dir = Path(args.manifest_dir)
    max_rows_per_level = args.max_rows_per_level if args.max_rows_per_level and args.max_rows_per_level > 0 else None
    gammas = sorted({float(g) for g in args.gammas})  # ascending, deduped -- gamma_star is "smallest that trips"

    _log(f"[reliance_mde_injection] checkpoints={args.checkpoints} gammas={gammas} cache_root={cache_root}")

    warnings: list[str] = []

    # Load every requested checkpoint's head once. A head that can't load is
    # kept with a load_error marker so every battery reports it not_estimable
    # (never dropped silently, never a crash).
    checkpoint_heads: dict[str, dict] = {}
    for run in args.checkpoints:
        ckpt_path = ckpt_dir / f"runs_{run}_best.pt"
        try:
            if not ckpt_path.exists():
                raise FileNotFoundError(f"checkpoint file not found: {ckpt_path}")
            w, b, w_dim, ckpt_sha256 = load_checkpoint_head(ckpt_path)
            checkpoint_heads[run] = dict(w=w, b=b, w_dim=w_dim, checkpoint_stem=ckpt_path.stem,
                                         checkpoint_sha256=ckpt_sha256, load_error=None)
        except Exception as e:
            msg = f"{run}: {type(e).__name__}: {e}"
            warnings.append(f"checkpoint head not loaded -- {msg}")
            _log(f"[WARN] {msg} -- every battery will report {run!r} not_estimable")
            checkpoint_heads[run] = dict(load_error=f"{type(e).__name__}: {e}")

    batteries = [
        b for b in BATTERIES
        if (args.corpus is None or b["corpus"] in args.corpus)
        and (args.factor is None or b["factor"] in args.factor)
    ]
    if not batteries:
        warnings.append(f"--corpus/--factor filters matched 0 of {len(BATTERIES)} batteries")

    per_battery: dict[str, dict] = {}
    for spec in batteries:
        try:
            res = run_battery_mde(spec, checkpoint_heads, cache_root, manifest_dir, gammas, args.ranks,
                                  args.seed, max_rows_per_level=max_rows_per_level)
        except Exception as e:
            # Backstop: a battery's own orchestration failing must not stop the others.
            _log(f"[WARN] battery {spec['name']}: UNEXPECTED FAILURE ({type(e).__name__}: {e})")
            res = dict(name=spec["name"], corpus=spec["corpus"], factor=spec["factor"], grouping=spec["grouping"],
                       per_checkpoint={run: _not_estimable(f"unexpected battery failure ({type(e).__name__}: {e})")
                                       for run in checkpoint_heads})
        per_battery[spec["name"]] = res
        _log(f"[battery] {spec['name']}: {sum(1 for c in res['per_checkpoint'].values() if c.get('status') == STATUS_OK)}"
             f"/{len(res['per_checkpoint'])} checkpoint(s) estimable")

    params = dict(
        gammas=gammas, ranks=list(args.ranks), seed=args.seed, n_boot=args.n_boot,
        max_rows_per_level=max_rows_per_level, checkpoints=list(args.checkpoints),
        model_space_cache_root=str(cache_root), ckpt_dir=str(ckpt_dir), manifest_dir=str(manifest_dir),
        detection_min_fraction=DETECTION_MIN_FRACTION,
    )
    report = build_report(per_battery, params, gammas, warnings)
    _write_json_atomic(out_path, report)
    _log(f"[done] wrote MDE report -> {out_path}")
    _log(f"[done] {report['paper_sentence']['text']}")


if __name__ == "__main__":
    main()
