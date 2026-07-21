# Step 4 Gate — Prereg Decision Memo

> Decision-ready input to reviewing and committing `docs/gate_prereg.md`.
> This file does **not** change that document — the humans own its
> resolution and its commit. Every recommendation below is a proposal for
> the reviewer to accept, adjust, or reject; none is authoritative.

**Blindness note (restated from `docs/gate_prereg.md` §0):** this memo was
written while the gate's decisive criteria (C1/C2/C4/C7 pending Phase B
model-space embeddings and a second backbone; C8 pending seeded
replicates on the real cache) still depend on data that does not exist
yet. Resolving their thresholds now is legitimate pre-registration written
blind, not tuning to a result. **No recommendation below was chosen to
make the accent battery (FSS ≈ 0.99) or the generator battery (FSS ≈ 0.89)
pass or fail** — those two numbers are mentioned only because the
`step4_gate_hardening_brief.md` that requested this memo already stated
them; nothing here reasons from their magnitude toward a threshold value.
Per Roadmap v3 §4: both are exploratory Phase A output, development-tier,
not decisive on their own.

## 1. Ranked attention list

Ordered by how much the choice actually moves `overall_classification`
(see `docs/gate_prereg.md` §3 / `scripts/run_gate.py`'s `classify_overall`,
which reads a sign-bearing FAIL on C2/C4/C6/C8 as "failure"). Read top to
bottom if time is short.

| # | Item | Load-bearing | Whose call |
|---|---|---|---|
| 1 | **§4 ambiguity 2 — C2/C7 factor↔corpus matching gap** | Critical — blocks C2/C7 from ever leaving `pending_input`/`not_estimable` for the only real batteries that exist or are imminent | **human-scientific** |
| 2 | **§4 ambiguity 1 — no criterion has a roadmap-given number** | Critical (meta) — the credibility of the whole document rests on this being resolved consciously | **human-scientific** |
| 3 | **C8 — unanimity (3/3) vs. majority across seeds** | High — sign-bearing; a 2-of-3 result passes under majority, fails under unanimity | **human-scientific** |
| 4 | **C4 — 2σ bar, fold-majority, and the main/control asymmetry** | High — sign-bearing | **human-scientific** |
| 5 | **C2 — sign-only at n=3, no p-value** | High — sign-bearing | **human-scientific** |
| 6 | **§4 ambiguity 3 — C7 automated pass/fail vs. descriptive** | Medium-high — decides whether C7 can ever push toward `diagnostic_only` | **human-scientific** |
| 7 | **C1 — replication bar (sign vs. sign+CI vs. magnitude)** | Medium — only bites once backbone #2 exists, but then directly sign-bearing | **human-scientific** |
| 8 | **C6 — reuse `estimators_agree_sign`** | High consequence, near-zero discretion (pure boolean reuse; the only real question is the new estimator-status guard, an implementation detail not a threshold) | **safe-default** |
| 9 | **§4 ambiguity 5 — is C3 pass/fail or a precondition** | Low-medium — interpretive, doesn't change today's PASS | **safe-default**, confirm only |
| 10 | **C3 — `n_boot≥1000`, `n_groups≥8`** | Low — not sign-bearing, generous bar, already passes on both real batteries | **safe-default** |
| 11 | **C5 — reuse the existing 0.25-spread rule** | Low — not sign-bearing, pure reuse of already-shipped code | **safe-default** |
| 12 | **§4 ambiguity 4 — C1/C8 strictness** | Same decision as rows 3 and 7, viewed from the ambiguity list | **human-scientific** |

## 2. Per-criterion detail

### C1 — Replication across ≥2 backbones

- **Roadmap wording:** "replication across ≥2 backbones."
- **Draft default:** sign agreement on the headline metric across both
  backbones, plus no disjoint-opposite `headline_bootstrap` CIs.
- **Recommendation:** keep the default as the pass/fail bar. **Add** (not
  gate on) a reported relative-magnitude ratio between the two backbones'
  headline means, purely for reviewer context — a replication that's
  "same sign but 10× smaller on backbone 2" is worth seeing even if it
  technically passes.
- **Justification:** sign agreement is the minimum defensible reading of
  "replication" and stays symmetric with C8's own bar; a magnitude bar on
  top would need its own threshold with no roadmap guidance, another TBD
  this memo would just be pushing downstream. Reporting magnitude without
  gating on it costs nothing and loses no information.
- **Load-bearing:** medium (inert until backbone #2 exists; sign-bearing
  once it does). **Whose call:** human-scientific — "what counts as
  replicated" is a real methodological position, not a mechanical reuse.

### C2 — Positive reliance↔matching-factor-degradation association

- **Roadmap wording:** "positive reliance↔matching-factor-degradation
  association across checkpoints."
- **Draft default:** sign of the correlation across checkpoints (n=3
  today), not a p-value.
- **Recommendation:** keep the default. Do not attempt a significance test
  at this n under any circumstances — report the correlation magnitude as
  evidence (already done, `numbers.per_battery[...].correlation`) but
  never as a gating number.
- **Justification:** a p-value from 3 points is decoration, not evidence;
  presenting one invites exactly the "statistically decisive" mislabeling
  Roadmap v3 §4 bans for single-seed-scale comparisons.
- **Load-bearing:** high (one of the four sign-bearing criteria).
  **Whose call:** human-scientific.

### C3 — Grouped-bootstrap CIs

- **Roadmap wording:** "grouped-bootstrap CIs."
- **Draft default:** `headline_bootstrap.status == "ok"`, `n_boot ≥ 1000`,
  `n_groups ≥ 8`, non-degenerate grouping.
- **Recommendation:** keep as-is.
- **Justification:** these are generous, already-met bars reused from
  existing code; the criterion is methodological (did we compute
  uncertainty the right way), not a discriminating scientific test, so
  there's little to gain from tightening it and no principled way to
  choose a different `n_groups` cutoff without more rationale than "some
  number felt right."
- **Load-bearing:** low (not sign-bearing). **Whose call:** safe-default.

### C4 — Intervention effects vs. equal-norm random controls

- **Roadmap wording:** "intervention effects significantly exceeding
  equal-norm random controls (with task-direction removal as positive
  control)."
- **Draft default:** main effect's `exceeds_random` (2σ bar) true in ≥50%
  of folds; positive control's own (fold-recomputed) `exceeds_random` true
  in **all** estimable folds.
- **Recommendation:** keep both bars, but make the asymmetry an explicit,
  stated design choice in the committed doc rather than an implicit one.
  The positive control is supposed to be an easy win — it's the actual
  task direction, not the factor under test — so holding it to unanimity
  while the real hypothesis only needs a majority is principled (a control
  that doesn't fire reliably even on a known-true effect indicates a
  broken control pipeline, not an ambiguous scientific result), but a
  reviewer who hasn't seen this reasoning would reasonably ask why the two
  halves of one criterion use different bars.
- **Justification:** the 2σ bar is already implemented
  (`projection_removal_control.exceeds_random`) and reused rather than
  inventing a second convention; ≥50% tolerates ordinary fold noise on the
  effect actually being tested without turning into a rubber stamp.
- **Load-bearing:** high (sign-bearing). **Whose call:** human-scientific.

### C5 — Rank stability

- **Roadmap wording:** "rank stability."
- **Draft default:** reuse the existing `stable_rank_window` rule (spread
  `< 0.25 × (|mean| + 1e-9)` across `ranks_valid`), require a window of
  ≥2 including rank 1.
- **Recommendation:** keep as-is.
- **Justification:** introducing a second, gate-specific stability rule on
  top of one Phase A already computes for the same purpose would let the
  same curve pass or fail depending on which document a reader consults —
  the opposite of what pre-registration is for.
- **Load-bearing:** low (not sign-bearing). **Whose call:** safe-default.

### C6 — Agreement across ≥2 subspace estimators

- **Roadmap wording:** "agreement across ≥2 subspace estimators."
- **Draft default:** reuse `prereg_candidate.estimators_agree_sign`.
- **Recommendation:** keep the reuse. Confirm the hardening added this
  round: an estimator whose own `status != "ok"` now excludes that battery
  from C6 instead of trusting `estimators_agree_sign` blindly — Phase A's
  own `summarize_prereg_candidate` reports `estimators_agree_sign=False`
  both when the estimators genuinely disagree AND when one of them never
  produced a result at all (a `nan` mean is treated as "doesn't match" by
  construction), so the boolean alone can't tell an operational failure
  from a real disagreement. This is an implementation fix, not a new
  threshold — nothing to decide here beyond confirming the fix is correct.
- **Load-bearing:** high consequence (sign-bearing) but essentially no
  discretion — there's no number to pick, only a boolean to trust (once
  the estimator-status guard is in place). **Whose call:** safe-default.

### C7 — No collapse after controlling for clean EER / checkpoint quality / training corpus

- **Roadmap wording:** "no collapse after controlling for clean EER /
  checkpoint quality / training corpus."
- **Draft default:** an automated sign-survival check (raw correlation
  sign vs. residualized-on-clean-EER correlation sign), reported as
  pass/fail.
- **Recommendation — a real change, not just accepting the draft:**
  **do not automate this as pass/fail at today's n (3 checkpoints, ~6 once
  backbone #2 lands).** Downgrade C7 to a **descriptive** report (the
  table of per-checkpoint reliance/clean-EER/target-EER, the raw and
  residualized signs, reported as evidence) until enough checkpoints exist
  for a regression to mean anything — a defensible lower bound is
  somewhere past 6-8 independent checkpoints, well beyond what Step 4 has
  available. Until then, C7 should read `pending_input` (or a new,
  explicitly non-sign-bearing status) rather than a number a reviewer
  might mistake for a real statistical control.
- **Justification:** a 2-covariate OLS fit on 3 points has zero residual
  degrees of freedom in the version that would also include an
  architecture indicator (§4 ambiguity 3's own concern) — reporting a
  sign from that as if it "controlled for" anything overstates precision
  in exactly the way Roadmap v3 §4's claim-language policy exists to
  prevent (nothing here should read as "statistically decisive").
- **Load-bearing:** medium-high (not one of the four canonical sign-bearing
  criteria, but it is what currently separates `diagnostic_only` from a
  cleaner read). **Whose call:** human-scientific — this is the memo's
  strongest disagreement with the draft's current default, and needs
  explicit sign-off, not silent adoption.

### C8 — Consistent effect direction across ≥3 seeded replicates

- **Roadmap wording:** "consistent effect direction across ≥3
  independently seeded replicates."
- **Draft default:** unanimous sign agreement (3/3), not majority.
- **Recommendation:** keep unanimity.
- **Justification:** "consistent" in plain English means unanimous, not
  "mostly agree"; replicates are cheap here (minutes per seed per the
  roadmap), so if 3/3 unanimity feels too strict in practice the answer is
  running more seeds, not loosening the bar retroactively — the latter is
  exactly the kind of post-hoc flexibility pre-registration exists to
  foreclose.
- **Load-bearing:** high (sign-bearing). **Whose call:** human-scientific.

## 3. §4 ambiguities not folded into a single criterion above

### Ambiguity 1 — no criterion carries a roadmap-given numeric threshold

Every one of C1–C8 needed a proposed default in the draft despite the
roadmap calling them "quantitative." **Recommendation:** treat this as
resolved by the act of reviewing §2 of `docs/gate_prereg.md` line by line
and this memo's per-criterion recommendations — there's no single number
to pick here, only the discipline of confirming each of the 8 rather than
letting the code's defaults become the standard through inertia.
**Load-bearing:** critical (meta). **Whose call:** human-scientific.

### Ambiguity 2 — the C2/C7 factor↔corpus matching gap

See the dedicated section below (§4 of this memo). **Load-bearing:**
critical. **Whose call:** human-scientific.

### Ambiguity 3 — C7's statistical power

Covered under C7 above: recommendation is to not automate pass/fail at
today's checkpoint count. **Load-bearing:** medium-high. **Whose call:**
human-scientific.

### Ambiguity 4 — C1's and C8's strictness

This is the same decision as rows 3 and 7 of the ranked list (C8's
unanimity-vs-majority, C1's sign-vs-sign+CI), restated from the ambiguity
list's own framing. Nothing new to add beyond the C1/C8 entries above.
**Load-bearing:** high. **Whose call:** human-scientific.

### Ambiguity 5 — is C3 a pass/fail criterion or a methodological precondition?

**Recommendation:** treat it as the latter — a gate on whether the
*other* criteria's numbers can be trusted, dressed in the same
pass/fail shape as the rest for implementation uniformity. This doesn't
change today's behavior (C3 already reports pass/fail either way); it's a
labeling/interpretation choice for how the committed doc frames C3 to a
paper reviewer, not a computational one. **Load-bearing:** low-medium.
**Whose call:** safe-default, confirm only.

## 4. The pivotal decision — C2/C7's factor↔corpus matching gap

**The problem, restated precisely:** C2 needs, per checkpoint, (a) a
reliance statistic on a factor and (b) that checkpoint's EER on a corpus
that varies the **same** factor. `cross_test.py`/`reproduce_eval.py` only
EER-score the development OOD diagnostic trio (ITW/ReplayDF/AI4T) —
DiffSSD, ReplayDF-as-dev, FakeOrReal, and ASVspoof5 are dev-tier-for-
thresholding only in that script, never themselves scored. So today's one
real battery (DiffSSD, language-by-speaker) and the imminent generator
battery/battries have no EER-scored corpus that shares their factor by
construction of the current pipeline, not by coincidence.

**Option A — evaluate only where a scored corpus already shares the
factor; mark the rest `not_applicable`.**
Cheapest, zero new infrastructure. Concretely: ITW/ReplayDF/AI4T's own
manifests carry `generator_id` (per the v3 manifest schema, audit
§4.6/§4.8) even though those corpora weren't run *as* generator-factor
batteries — so a **generator_id** battery on DiffSSD could, in principle,
still be checked against ITW's or ReplayDF's generator-tagged EER, because
the factor (generator identity) genuinely varies in both places, even
though the corpora differ. A **language**/accent battery has no such
overlap today (none of ITW/ReplayDF/AI4T carry meaningful language
variation in their current manifests) — C2/C7 stay `not_applicable` for
that factor until one does.
*Trade-off:* silently narrows which factors the gate can ever speak to;
"not_applicable" must be visibly distinct from "not yet run" in the
committed report so a reader doesn't mistake permanent inapplicability
for a temporary gap.

**Option B — extend `cross_test.py`'s scored `--corpora` list to include
the dev-tier corpora themselves (DiffSSD, FakeOrReal, ASVspoof5).**
Gives every existing battery a same-corpus EER. Requires re-running
`cross_test.py`/`reproduce_eval.py` with an expanded corpus list for all
checkpoints (cheap compute, but it's new committed numbers, a new
reproduction-gate tolerance check, and a real infrastructure change to a
script this brief and its predecessor were both explicit should stay
untouched during this work).
*Trade-off:* the most complete answer, but it's downstream work outside
this session's scope and re-opens the "is a dev corpus now being used as
if it were a scored eval corpus" framing question Roadmap v3 §4 already
drew a bright line around (dev trio = diagnostic, not a test set — scoring
DiffSSD the same way doesn't change its dev-tier status, but the
distinction needs to stay explicit in whatever report cites the number).

**Option C — a documented factor→scored-corpus rule using the eval trio,
codified once and reused (a middle ground: same idea as Option A's
generator-factor case, generalized and written down as policy rather than
discovered ad hoc per battery).**
Concretely: maintain a small, explicit, version-controlled mapping (the
`--factor-corpus-map` CLI input this round's hardening now supports as
`corpus:factor=eval_corpus`) reviewed once per factor as new batteries are
added, rather than re-litigating case by case.

**Recommendation: Option C, built on Option A's logic where a real overlap
exists, with Option B as a longer-term infrastructure item tracked
separately (not blocking Step 4).** This keeps the dev/eval-tier
boundary intact, costs no new compute or reproduction-gate churn, and
turns "is there a match" from an implicit per-battery judgment call into
an explicit, reviewable list. It is still the humans' call which specific
factor→corpus pairs belong in that list (this memo recommends the
mechanism, not the mapping's actual contents for language/accent, which
has no honest match today).

## 5. Not deciding here

- The actual resolution value for every `THRESHOLD TBD` in
  `docs/gate_prereg.md` §2 — this memo recommends, the user and
  collaborator commit.
- The actual `factor_corpus_map` entries (which corpus backs which
  factor) — the mechanism is recommended above; the specific pairs are a
  domain call.
- The git commit of the finalized `docs/gate_prereg.md` itself — this memo
  changes nothing in that file.
- Whether/when Step 4 actually runs — gated on Phase B embeddings, a
  second backbone, and seeded replicates existing, none of which this
  memo can or should accelerate.

**Blindness restated:** every recommendation above was reached from the
roadmap's own wording, the already-implemented code's existing
conventions, and general methodological reasoning about small-n
statistics — never from what the accent or generator battery's numbers
happened to be. Where this memo recommends *tightening* the draft (C7,
most notably), that recommendation would apply identically regardless of
which way either battery's result currently points.
