# Results

All numbers are Equal Error Rate (EER, lower is better) on strictly held-out OOD corpora, plus the linear corpus-identity probe accuracy. Models were trained on a frozen WavLM-Large backbone with a balanced multi-corpus sampler (except e003/e002 single-corpus baselines).

## Frozen-backbone experiment ladder

| Model | Configuration | ITW ↓ | ReplayDF ↓ | AI4T ↓ | Probe | Verdict |
|-------|---------------|:-----:|:----------:|:------:|:-----:|---------|
| e003 | Frozen linear baseline | **0.111** | 0.317 | 0.470 | — | OOD floor |
| e002-A | Frozen + augmentation | 0.127 | 0.296 | 0.417 | 0.962 | Marginal; structure unchanged |
| e002-B | Augmentation + consistency loss | 0.149 | 0.367 | 0.484 | 0.960 | Regressed |
| e004 | Shallow fine-tuning (top-4) | 0.192 | 0.273 | 0.461 | 0.964 | Single-corpus FT entrenches shortcut |
| e005-A | Frozen multi-corpus (balanced) | 0.143 | 0.298 | **0.231** | 0.967 | AI4T breakthrough; CI-verified |
| e005-C | + spoof positive weight | 0.122 | 0.325 | 0.297 | 0.973 | ITW↑ / AI4T↓ trade-off |
| e006 | + cross-corpus contrastive | 0.135 | ~0.298 | ~0.231 | 0.972 | Contrastive dormant under freezing |

**Notes**
- e005-A's AI4T EER of **0.231** was validated by a clustered bootstrap (resampling whole source videos), 95% CI **[0.188, 0.285]** — the most robustly verified result.
- e006's contrastive loss stayed flat (~2.75) across all 16 epochs: on a frozen backbone the objective cannot reshape the geometry. It is expected to activate only under fine-tuning (e007).

## Central finding: decodability ≠ generalization

The corpus probe never leaves the **0.96–0.97** band and never decreases, across all six interventions — yet OOD EER varies enormously (AI4T 0.470 → 0.231; ITW 0.111 → 0.192). Corpus separability and cross-corpus generalization are **decoupled**: invariance is neither achieved by any method nor necessary for the gains observed.

## Domain-Reliance Score (frozen models)

The Domain-Reliance Score (DRS) measures how much of the spoof classifier's weight vector lies in the corpus/domain subspace — i.e. whether the classifier *relies* on domain directions, distinct from whether they are merely *decodable*.

| Model | DRS (probe-subspace) | Probe acc |
|-------|:--------------------:|:---------:|
| e002-A | 0.0137 | 0.962 |
| e002-B | 0.0041 | 0.960 |
| e004 | 0.0085 | 0.964 |
| e005-A | 0.0024 | 0.967 |
| e005-C | 0.0012 | 0.973 |
| e006 | 0.0026 | 0.972 |

**Reading:** DRS is **low and flat** (0.1–1.4% of classifier weight energy in the domain subspace) across all frozen models, while probe accuracy stays high. This *explains* the decoupling — the classifier barely uses the decodable corpus directions — and indicates that a reliance-control objective has little to act on in the frozen regime. The reliance method is therefore positioned for the **fine-tuning phase (e007)**, where the backbone can move and reliance may emerge. With n=6 these values are interpreted as suggestive, not statistically conclusive. See [`analysis/drs/`](analysis/drs/).

## How to read each experiment's record

Each `experiments/<name>/` folder contains:
- `run_config.json` — the exact resolved configuration used
- `crosstest.log` — per-corpus OOD EER, the Kwok bona-fide matrix, and the probe accuracy
- `train_summary.log` — per-epoch dev EER trajectory and early-stop point
