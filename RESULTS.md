# Results

All numbers are Equal Error Rate (EER, lower is better) on strictly held-out
OOD corpora. Evidence status is tracked because several older rows predate the
current artifact discipline.

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

e007-A improved in-domain dev EER but regressed on ITW relative to the frozen
multi-corpus baselines. e007-B improved ITW substantially over e007-A and is the
strongest fine-tuned arm on ITW, but ReplayDF and AI4T do not show a universal
win. e007-C shows that a larger multilingual XLS-R backbone does not
automatically improve OOD generalization under the transplanted WavLM layer band.

## Central Finding: Decodability Is Not Generalization

The corpus probe remains high across interventions, while OOD EER changes
substantially. Corpus separability and cross-corpus generalization are therefore
decoupled: domain information can remain highly decodable without being the
dominant factor controlling OOD performance.

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
