# AudioShield — Cross-Corpus Audio Deepfake Detection

A self-supervised audio deepfake detector built on a frozen **WavLM-Large** backbone, designed to generalize across corpora it was *not* trained on. Models are trained on multiple corpora and evaluated on three strictly held-out, real-world benchmarks: **In-the-Wild (ITW)**, **ReplayDF**, and **AI4T**.

**Central finding:** corpus separability and cross-corpus generalization are *decoupled*. A linear corpus-identity probe stays at ~0.96–0.97 across every intervention, yet out-of-distribution (OOD) error varies widely (AI4T 0.470 → 0.231). Reducing corpus separability is neither achieved nor required for the OOD gains we observe. See [`RESULTS.md`](RESULTS.md) and [`analysis/drs/`](analysis/drs/).

## Status

- ✅ Six controlled frozen-backbone experiments (e003 → e006)
- ✅ Domain-Reliance Score analysis (`analysis/drs/`) — reliance is low and flat in the frozen regime
- 🔜 Fine-tuning phase (**e007**), specified in [`docs/e007_finetuning_design.md`](docs/e007_finetuning_design.md), awaiting a ≥24 GB GPU

## Repository layout

| Path | Contents |
|------|----------|
| `src/audioshield/` | The package: SSL backbone, layer weighting, pooling, heads, losses, training loop, evaluation |
| `scripts/` | Entrypoints: training (`train_e002.py`), DRS analysis (`compute_drs.py`), manifest building |
| `configs/experiments/` | One YAML per experiment (e003–e006, e007_A/B/C) |
| `configs/models/`, `configs/datasets/` | Model architecture and dataset configs |
| `manifests/` | Per-corpus CSV manifests (utt_id, path, label, split); paths are relative to `data_root` |
| `experiments/` | Per-experiment **record**: config + cross-test result + training summary (no checkpoints) |
| `analysis/drs/` | Domain-Reliance Score analysis and interpretation |
| `analysis/logs/` | Cleaned training console logs |
| `docs/` | Design docs and progress reports (`.md`) |
| `legacy_biophys/` | Earlier BioPhys-HyperRADAR approach — **superseded, not maintained**, kept for provenance |

## Setup

```bash
pip install -e .                  # installs the `audioshield` package (uses pyproject.toml)
pip install -r requirements.txt
```

**Datasets are not in this repo.** Set `data_root` in each experiment config to your local dataset path. Training corpora: ASVspoof5, DiffSSD, FoR, VCTK. Held-out OOD: ITW, ReplayDF, AI4T.

## Reproducing the experiments

```bash
# Train a frozen experiment (e003–e006):
python -u scripts/train_e002.py \
    --exp-config configs/experiments/e006_xc_frozen_v1.yaml \
    --output-dir runs/e006

# Cross-test a checkpoint on the held-out OOD corpora:
python -u -m audioshield.evaluation.cross_test \
    --checkpoint runs/e006/best.pt --corpora inthewild replaydf ai4t
```

Each experiment's committed record (config, cross-test result, training summary) lives in `experiments/<name>/` — you can read the results without re-running anything.

## Domain-Reliance analysis

```bash
python -u scripts/compute_drs.py        # results + interpretation in analysis/drs/
```

The Domain-Reliance Score measures how much the spoof classifier's decision boundary aligns with the corpus/domain subspace — i.e. whether the classifier *uses* domain information, as distinct from whether domain information is merely *present* (probe accuracy). See [`analysis/drs/README.md`](analysis/drs/README.md).

## Fine-tuning phase (e007 — requires ≥24 GB GPU)

Three controlled arms, each changing one variable:

- **e007-A** — backbone adaptation (unfreeze top-K layers), cross-entropy only
- **e007-B** — A + cross-corpus class-conditional contrastive loss (**target model**)
- **e007-C** — B with a larger / multilingual backbone (XLS-R)

Design and rationale: [`docs/e007_finetuning_design.md`](docs/e007_finetuning_design.md).

## Results at a glance

Full table in [`RESULTS.md`](RESULTS.md). Headline:

| Model | Configuration | ITW ↓ | ReplayDF ↓ | AI4T ↓ | Probe |
|-------|---------------|:-----:|:----------:|:------:|:-----:|
| e003 | Frozen linear baseline | 0.111 | 0.317 | 0.470 | — |
| e005-A | Frozen multi-corpus (balanced) | 0.143 | 0.298 | **0.231** | 0.967 |
| e006 | + cross-corpus contrastive | 0.135 | ~0.298 | ~0.231 | 0.972 |

*EER (lower is better). e005-A's AI4T result carries a clustered 95% CI of [0.188, 0.285].*
