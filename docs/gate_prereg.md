# Step 4 Gate — Pre-registration

> **STATUS: DRAFT — NOT FINAL.** This document is a working draft prepared
> ahead of Step 4 ("THE GATE") per Roadmap v3. It is **not** the committed
> pre-registration. The user and collaborator must review, resolve every
> `THRESHOLD TBD` marker below, and commit the reviewed version themselves
> **before Step 4 is run**. Nothing in this draft has been tuned to any
> already-observed Step 3 result (see §0).

## 0. Why this document exists, and what it must not do

Roadmap v3, Step 4, requires: *"Before running, commit `docs/gate_prereg.md`
containing the quantitative pass criteria, verbatim from HV: [the 8 criteria
in §2]."* The purpose of pre-registration is to fix the pass/fail rule
**before** the result is known, so the gate cannot be tuned to whatever
Step 3 happens to show — this is the direct fix for the adaptive-test-leakage
failure mode documented in `code_issue.md` (CI) and reclassified project-wide
in Roadmap v3 item 1.

**This draft was written blind to Phase B and blind to any effect-direction
tuning.** Exploratory Phase A output exists (a single accent-factor battery
on one dev-tier corpus, one backbone, `--w-metrics` disabled by a dimension
mismatch) — this document does not reference its numbers, and no threshold
below was chosen to make that battery pass or fail. Where a candidate
default below happens to reuse a statistic Phase A already computes (e.g.
`exceeds_random`, `estimators_agree_sign`, `stable_rank_window`), that is
because the statistic is an existing, already-implemented operationalization
of the criterion's own wording — not because of what value it took on any
real battery.

Every one of the 8 criteria in the roadmap is stated **qualitatively**
(direction, existence, or a bare "≥N" count) rather than with a numeric
pass bar attached to a specific quantity. Per instruction, every criterion
below therefore carries an explicit `THRESHOLD TBD` marker and a proposed
defensible default with a one-line justification. **None of these defaults
are authoritative until reviewed and committed.**

## 1. Corpus and claim-language framing (Roadmap v3 §4, binding on this document)

- ITW / ReplayDF / AI4T are **development OOD diagnostic corpora**, not
  "honest test sets" — nothing in this document treats a result on them as
  a final-tier evaluation.
- Any reference to a single checkpoint's result (e.g. e007-B) that has not
  been replicated is described as a **promising single-seed result, pending
  replication** — never as decisive on its own.
- Alignment (weight–subspace geometric alignment) is never called "causal
  reliance." Decodability-vs-reliance is a **hypothesis under test**, related
  to TCAV/INLP/LEACE, differentiated by factor-specific, cross-fitted,
  causally-validated application to deepfake detection — not an established
  finding.
- Banned in this document and in the eventual gate verdict report:
  "breakthrough," "proved/proven," "SOTA," "statistically decisive" (for any
  single-seed comparison).

## 2. The 8 criteria

Each entry: the roadmap's verbatim wording, the proposed operationalization,
the data source the gate consumer (`scripts/run_gate.py`) will read it from,
and an explicit threshold marker.

### C1 — Replication across ≥2 backbones (XLS-R-300M + WavLM-Large)

> Verbatim: "replication across ≥2 backbones."

The count (**2**, named explicitly: XLS-R-300M + WavLM-Large) is already a
number and is not TBD. What is unspecified is what counts as "replicated"
once both backbones' batteries exist — same sign only, or sign + overlapping
CI, or sign + magnitude within a tolerance.

**`THRESHOLD TBD — decide before commit`**: proposed default — a
criterion-2/4/6/8-style effect (headline metric on the matching factor)
must agree in **sign** across both backbones, AND the two backbones'
`headline_bootstrap` CIs must not be disjoint in a way that implies opposite
population values (i.e. not "CI entirely positive on one backbone, entirely
negative on the other"). *Justification: sign agreement is the minimum
possible bar for "replication" and is symmetric with C8's own bar (sign
agreement across seeds); requiring literal CI overlap in addition would make
C1 strictly harder than C8 for no stated reason.*

**Data source:** two independent `run_reliance_battery.py` manifests, one
per backbone, joined by matching `factor`/`corpus`/`grouping`. **WavLM-Large
embeddings do not exist on this machine and are `pending_input`** until the
collaborator runs the WavLM-Large embedding pass named in the roadmap.

### C2 — Positive reliance↔matching-factor-degradation association across checkpoints

> Verbatim: "positive reliance↔matching-factor-degradation association
> across checkpoints."

**`THRESHOLD TBD — decide before commit`**: proposed default — across the
checkpoints present in a battery's `per_checkpoint` block (currently
e007_A/B/C, n=3), the sign of the association between (a) that checkpoint's
factor-subspace reliance statistic (`alignment` or `r_var`, mean over folds)
and (b) that checkpoint's EER on a factor-matched held-out corpus is
**positive** (higher reliance co-occurs with higher/worse EER under shift of
that factor). With only 3 checkpoints, a correlation coefficient is reported
for evidence but its sign, not a p-value, is the pass/fail signal —
*justification: n=3 gives a rank-correlation test negligible power; sign
agreement across all 3 pairwise comparisons is the most a 3-point series can
honestly support, and is explicit about that limitation rather than
reporting a misleadingly precise p-value.*

**Data source:** reliance side from Phase A's `per_checkpoint` metrics
(currently `not_estimable` for all three e007 checkpoints — 256-d classifier
weight vs 1024-d raw-backbone cache — until Phase B model-space embeddings
land). EER side from `experiments/e007/{run}_crosstest.json`'s `per_corpus`
block, **already present in this repository** for e007_A/B/C on
inthewild/replaydf/ai4t. **Open ambiguity (§4):** the one real battery
available (diffssd / language-by-speaker) has no factor-matched corpus in
that EER file — diffssd is dev-tier-for-thresholding only, never scored for
EER by `cross_test.py`. This association is `pending_input` for any factor
without a factor-matched scored corpus.

### C3 — Grouped-bootstrap CIs

> Verbatim: "grouped-bootstrap CIs."

This is a methodological requirement (how uncertainty must be computed), not
a numeric pass bar. It is already implemented in
`run_reliance_battery.py`'s `grouped_bootstrap_ci` behind `headline_bootstrap`
(resampling groups, not rows). The only remaining question is what counts as
a *usable* grouped bootstrap for gate purposes.

**`THRESHOLD TBD — decide before commit`**: proposed default —
`headline_bootstrap.status == "ok"`, `n_boot >= 1000`, `n_groups >= 8` (from
`prereg_candidate.n_groups`), and `grouping_degenerate == False`.
*Justification: reuses the already-recorded fields; 8 groups is the minimum
commonly cited for a bootstrap resampling distribution to be
non-degenerate; 1000 resamples matches the project's existing default and
observed real-battery n_boot.*

**Data source:** every battery's own `headline_bootstrap` and
`prereg_candidate` blocks — available today from the real Phase A fixtures.

### C4 — Intervention effects significantly exceeding equal-norm random controls, with task-direction removal as positive control

> Verbatim: "intervention effects significantly exceeding equal-norm random
> controls (with task-direction removal as positive control)."

`projection_removal_control` already computes exactly this shape per fold:
`true_effect`, 20 `random_effects`, and a boolean `exceeds_random =
true_effect > random_mean + 2*random_std`.

**`THRESHOLD TBD — decide before commit`**: proposed default — (a) the real
factor-projection's `exceeds_random` is `True` in a majority (**>=50%**) of
folds across the battery; AND (b) the positive control
(`task_direction_effect`) also shows `exceeds_random == True` wherever
estimable, confirming the control pipeline itself detects a known, real
effect (task removal) and is not silently inert. *Justification: the
existing 2-std bar is already a defensible "significantly exceeding" bar
(~97.7th percentile one-sided under a normal approximation) — reusing it
avoids inventing a second, parallel statistical convention; a simple fold
majority (rather than requiring all folds) tolerates ordinary fold-to-fold
noise without being a rubber stamp.*

**Data source:** `fold_results[*].effect.projection_removal_control` (real
data available today) and `..._control.task_direction_effect` (currently
`not_estimable` for the one real battery, same w-dim-mismatch reason as C2 —
`pending_input` until Phase B model-space embeddings exist, since
task-direction removal needs `w` in the same space as the embeddings).

### C5 — Rank stability

> Verbatim: "rank stability."

Already implemented: `rank_sensitivity` (headline metric across
`ranks_valid`) and `prereg_candidate.stable_rank_window`
(`summarize_prereg_candidate`'s existing rule: consecutive ranks whose
spread is within `0.25 * (abs(mean) + 1e-9)` of each other).

**`THRESHOLD TBD — decide before commit`**: proposed default — reuse the
existing 0.25-relative-spread rule unchanged, and require
`len(stable_rank_window) >= 2` including the headline rank (rank 1).
*Justification: this rule is already implemented and already exercised by
real Step 3 batteries; introducing a second, gate-specific stability
definition would let the same underlying curve pass or fail depending on
which document you read, which is exactly the kind of post-hoc-flexibility
pre-registration is meant to foreclose.*

**Data source:** `rank_sensitivity` + `prereg_candidate.stable_rank_window`,
available today from the real Phase A fixtures.

### C6 — Agreement across ≥2 subspace estimators

> Verbatim: "agreement across ≥2 subspace estimators."

The count (**2**: LDA-subspace and cross-fitted linear-probe) is explicit
and already exactly the estimator set Phase A computes
(`battery.estimators.{lda,probe}`). `prereg_candidate.estimators_agree_sign`
already exists as a boolean.

**`THRESHOLD TBD — decide before commit`**: proposed default — reuse
`estimators_agree_sign == True` unchanged as the pass condition.
*Justification: this field is already computed for exactly this purpose by
the existing Phase A code (`summarize_prereg_candidate`); no additional
numeric bar is needed on top of a boolean sign-agreement flag.*

**Data source:** `prereg_candidate.estimators_agree_sign`, available today.

### C7 — No collapse after controlling for clean EER / checkpoint quality / training corpus

> Verbatim: "no collapse after controlling for clean EER / checkpoint
> quality / training corpus."

**`THRESHOLD TBD — decide before commit`**: proposed default — a partial
correlation / residualized comparison: regress the per-checkpoint reliance
statistic on {clean EER (e.g. mean of inthewild/replaydf/ai4t EER),
checkpoint architecture (XLS-R vs WavLM once C1's second backbone exists),
training-corpus indicator} and check that the association from C2 survives
in **sign** on the residuals. *Justification and explicit limitation: with
n=3 checkpoints (soon n up to 6 once backbone #2 lands) this regression is
severely underpowered to "control for" anything in the usual statistical
sense — the proposed default is a sign-survival check, not a significance
test, and this criterion is the one most likely to need a qualitative
(descriptive table + reviewer judgment) treatment rather than an automated
pass/fail. Flagged in §4 as an open ambiguity for review, not resolved
silently here.*

**Data source:** reliance statistics from C2's source, EERs from
`experiments/e007/*_crosstest.json` (available today), architecture/training
corpus from checkpoint config metadata (available today, not yet threaded
into the gate consumer's checkpoint-quality regression — implementation
detail, not a missing input).

### C8 — Consistent effect direction across ≥3 independently seeded replicates

> Verbatim: "consistent effect direction across ≥3 independently seeded
> replicates."

The count (**3**) is explicit. What "effect" and "consistent" mean for a
retrained head is not specified.

**`THRESHOLD TBD — decide before commit`**: proposed default — train the
classifier head on cached embeddings with >=3 distinct seeds; for each seed
compute the same fold-level effect statistic used elsewhere in the battery
(`factor_separation_score` delta, or `projection_removal_control.true_effect`
sign); require **all** replicates to agree in sign (3/3, not a majority).
*Justification: "consistent" read plainly means unanimous, not majority, for
a quantity this cheap to replicate (minutes per seed per the roadmap); a
majority-only bar would be a weaker requirement than the roadmap's own
"consistent" wording supports.*

**Data source:** Task 3's seeded head-replicate machinery (this session,
tested on synthetic embeddings only). Real numbers are `pending_input` until
run against the real cached embeddings on the collaborator machine.

## 3. Three-outcome decision rule

Verbatim from the roadmap: *"strong success → Step 6 includes the reliance
regularizer; diagnostic-only → Step 6 becomes the analysis/toolkit paper
track (measurement release, factor-asymmetry findings); failure → pivot to
replay/media robustness (bar: beat W2V2-AASIST 18.2% / adaptive-RIR 11.0% on
the official ReplayDF protocol) on the repaired infrastructure."*

The **failure-pivot bar itself is an explicit number already given by the
roadmap** (beat 18.2% EER for W2V2-AASIST and 11.0% EER for adaptive-RIR, on
the official ReplayDF protocol) and is transcribed here unchanged — it is
not a `THRESHOLD TBD` item; it belongs to a downstream track (Step 5's
baseline set), not to computing C1–C8.

What the roadmap does not specify is how many of C1–C8 passing constitutes
each of the three outcomes.

**`THRESHOLD TBD — decide before commit`**: proposed default —

- **Strong success**: all 8 criteria pass (no `fail`, no `pending_input`
  outstanding).
- **Diagnostic-only**: no criterion shows a directional reversal (a
  criterion whose core sign requirement — C2, C4, C6, C8 — comes back
  negative/opposite of predicted), but at least one of C1/C3/C5/C7 fails or
  remains structurally weak (e.g. C7's underpowered n=3 regression).
- **Failure**: any of the sign-bearing criteria (C2, C4, C6, C8) shows a
  **consistent directional reversal** (not just a missed magnitude bar) —
  i.e. reliance is found to be unrelated to or negatively associated with
  factor degradation, or intervention/removal effects are indistinguishable
  from random controls, or subspace estimators disagree in sign, or seeded
  replicates disagree in sign.

*Justification: this maps the "three legitimate outcomes" framing (HV) onto
the 8 criteria in the only way that keeps "strong success" meaning
"everything the roadmap asked for came back clean," reserves "failure" for
an actual directional contradiction (not merely "one CI was a little wide"),
and treats every criterion classified below as `pending_input` as blocking
any of the three verdicts from being declared — the gate consumer emits an
overall classification only when no criterion is `pending_input` (see
`scripts/run_gate.py`).* This mapping is a proposal for review, not a
committed rule.

## 4. Open ambiguities for review (not resolved silently here)

1. **No criterion in the roadmap carries an explicit numeric pass bar** —
   every one of C1–C8 required a proposed default in this draft. This is the
   single largest ambiguity: review should confirm or replace each default
   in §2 deliberately, not let the code's defaults become the standard by
   default.
2. **C2/C7 factor-matching gap**: the one real Phase A battery available
   today (diffssd, language-by-speaker) has no EER-scored corpus that shares
   its factor — diffssd is dev-tier-for-thresholding only in
   `cross_test.py`/`reproduce_eval.py`, never itself EER-scored. Whether C2
   can be evaluated at all for this battery, or only for future
   factor/corpus pairs where a scored corpus shares the factor, needs a
   decision.
3. **C7's statistical power**: with 3 checkpoints (up to ~6 once a second
   backbone lands), "controlling for" covariates via regression is
   qualitative at best. Review should decide whether C7 is automated at all
   for this gate, or reported descriptively for human judgment.
4. **C1's "replication" bar** (sign agreement vs. CI overlap vs. magnitude
   tolerance) and **C8's unanimity-vs-majority** bar are both proposed here
   as the stricter reading of the roadmap's plain wording ("replication",
   "consistent") — review should confirm this is the intended strictness,
   not a stricter-than-intended one.
5. **Whether C3 (grouped-bootstrap CIs) is a pass/fail criterion at all**,
   versus a methodological precondition that gates whether the *other*
   criteria's numbers can be trusted. This draft treats it as the latter
   dressed as the former (a structural check) — review should confirm.

## 5. Sign-off

- [ ] Reviewed by user
- [ ] Reviewed by collaborator
- [ ] Every `THRESHOLD TBD` in §2 replaced with a committed value
- [ ] §4 ambiguities resolved and struck through or answered
- [ ] Committed to git before Step 4 is run
