# Results

All numbers are Equal Error Rate (EER, lower is better) on the project's OOD
development corpora (In-the-Wild, ReplayDF, AI4T; see Disclosure below).
Evidence status is tracked because several older rows predate the current
artifact discipline.

## Disclosure and status of these results

The results in this file were produced before the project's external audit
and carry three qualifications that supersede any earlier framing.

First, the spoof_pos_weight decision chain: e007-A's `spoof_pos_weight: 1.0`
was selected on the basis of e005-C's AI4T/ITW results. e005-C was
subsequently found to have been run against the v1 manifest directory while
its sibling configs used manifests/v2 (fixed in the 2a repair branch), so
the evidential basis for that hyperparameter choice is weaker than the
original write-up implied. The setting is retained as-is pending the Step 5
baseline suite; it should be read as a historical choice, not a validated
one.

Second, evaluation-tier status: In-the-Wild, ReplayDF, and AI4T were used
iteratively during development of the experiments reported here, including
for decisions like the one above. They are therefore DEVELOPMENT corpora
for this project, not honest test sets, and no number in this file against
them constitutes an unbiased generalization estimate. Final-tier evaluation
(XMAD-Bench, ML-ITW, temporal holdout) is governed by a separate blindness
protocol and has not yet occurred.

Third, all numbers below predate the 2a correctness repairs (seeding,
sampler factorization, strict loading, EER computation, dependency
pinning). They are retained pending the hash-verified reproduction run;
if that run diverges, the divergence investigation supersedes this file's
numbers.

Review records governing this project's claims are committed under
docs/ (research audit, honest-verdict review, code-issue chain).

## Canonical OOD Table

| Run | Change | ITW | ReplayDF | AI4T | Evidence |
|---|---|---:|---:|---:|---|
| e003 | frozen linear baseline | 0.111 | 0.317 | 0.470 | number only; artifact absent - UNVERIFIABLE |
| e002-A | + aug, frozen | 0.127 | 0.296 | 0.417 | tracked log |
| e002-B | + consistency | 0.149 | 0.367 | 0.484 | summary only - UNVERIFIABLE |
| e004 | single-corpus shallow FT | 0.192 | 0.273 | 0.461 | tracked log |
| e005-A | frozen multi-corpus | 0.143 | 0.298 | 0.231 [CI 0.188-0.285] | tracked log |
| e005-C | + spoof pos-weight | 0.122 | 0.325 | 0.297 | tracked log |
| e006 | + XC SupCon, frozen | 0.135 | 0.328 | 0.260 | log absent - UNVERIFIABLE |
| e007-A | band FT, CE | 0.1805 | 0.3327 | 0.2565 | experiments/e007 JSON |
| e007-B | + projected SupCon | 0.1167 | 0.3276 | 0.2629 | experiments/e007 JSON |
| e007-C | XLS-R, transplanted band | 0.2009 | 0.4530 | 0.3435 | experiments/e007 JSON |

Single seed per condition; seeds are not yet plumbed. Comparisons are
provisional pending Roadmap Step 6, 5-seed replication.

## e007 Reading

e007-A's in-domain dev-corpus EER differs from the frozen multi-corpus
baselines' (not shown in this table); e007-A's ITW EER (0.1805) is higher than
the frozen multi-corpus baselines' ITW EER (e005-A 0.143, e005-C 0.122, e006
0.135). e007-B's ITW EER (0.1167) is lower than e007-A's (0.1805) and is the
lowest ITW EER among the three fine-tuned arms (e007-A 0.1805, e007-B 0.1167,
e007-C 0.2009); e007-B's ReplayDF EER (0.3276) is lower than e007-A's (0.3327),
and e007-B's AI4T EER (0.2629) is higher than e007-A's (0.2565). e007-C uses a
larger, multilingual XLS-R backbone under the transplanted WavLM layer band;
its ITW/ReplayDF/AI4T EER (0.2009/0.4530/0.3435) are higher than both
e007-A's (0.1805/0.3327/0.2565) and e007-B's (0.1167/0.3276/0.2629).

## Observation: Decodability and OOD Performance Are Decoupled

The corpus probe remains high across interventions, while OOD EER varies
across interventions (see Canonical OOD Table above). Corpus separability and
cross-corpus development-tier OOD performance are therefore decoupled: domain
information can remain highly decodable without being the dominant factor
controlling development-tier OOD performance.

## Domain-Reliance Score (Frozen Models)

The Domain-Reliance Score (DRS) measures how much of the spoof classifier's
weight vector lies in the corpus/domain subspace, i.e. whether the classifier
relies on domain directions, distinct from whether they are merely decodable.

| Model | DRS (probe-subspace) | Probe acc |
|-------|:--------------------:|:---------:|
| e002-A | 0.0137 | 0.962 |
| e002-B | 0.0041 | 0.960 |
| e004 | 0.0085 | 0.964 |
| e005-A | 0.0024 | 0.967 |
| e005-C | 0.0012 | 0.973 |
| e006 | 0.0026 | 0.972 |

With n=6 these values are interpreted as suggestive, not statistically
conclusive. See `analysis/drs/`.

## How To Read Each Experiment Record

Each `experiments/<name>/` folder contains the committed lightweight record when
available. Checkpoints remain outside git. For e007, `experiments/e007/` contains
the raw OOD JSONs; `CHECKPOINTS.md` records checkpoint hashes for provenance.
