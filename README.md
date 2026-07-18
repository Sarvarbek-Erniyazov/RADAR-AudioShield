# AudioShield - Cross-Corpus Audio Deepfake Detection

A self-supervised audio deepfake detector built on SSL speech backbones,
evaluated on development-tier, real-world OOD corpora: In-the-Wild (ITW),
ReplayDF, and AI4T (see RESULTS.md's Disclosure section).

## Status

- Six controlled frozen-backbone experiments are recorded (e003 -> e006).
- Domain-Reliance Score analysis is recorded in `analysis/drs/`.
- Fine-tuning phase e007 is complete: WavLM weighted-band CE control, WavLM
  projected SupCon, and XLS-R transplanted-band arms were trained and
  cross-tested.

Single seed per condition; seeds are not yet plumbed. Comparisons are
provisional pending Roadmap Step 6, 5-seed replication.

## Repository Layout

| Path | Contents |
|------|----------|
| `src/audioshield/` | Package code: SSL backbone, layer weighting, pooling, heads, losses, training, evaluation |
| `scripts/` | Entrypoints for training, DRS analysis, manifest building |
| `configs/experiments/` | Experiment YAMLs, including e007_A/B/C |
| `configs/models/` | Model configs, including WavLM and XLS-R e007 configs |
| `manifests/` | Per-corpus CSV manifests with paths relative to `data_root` |
| `experiments/` | Lightweight experiment records; checkpoints stay out of git |
| `analysis/` | Audits, DRS outputs, cleaned logs |
| `legacy_biophys/` | Superseded earlier BioPhys-HyperRADAR approach, kept for provenance |

## Setup

```bash
pip install -r requirements.lock.local4060 \
  --extra-index-url https://download.pytorch.org/whl/cu121
pip install -e . --no-deps
```

The editable install is required even after installing the lockfile: the lockfile pins
third-party packages but cannot install this repository's local `audioshield` package.
See [`docs/repro_2a_harness_fixes.md`](docs/repro_2a_harness_fixes.md) for the pinned
reproduction runbook and its evidence record.

Datasets are not in this repo. Set `data_root` to the local dataset root.
Training corpora: ASVspoof5, DiffSSD, FakeOrReal, VCTK. Development-tier OOD
corpora: ITW, ReplayDF, AI4T.

## Reproducing A Cross-Test

```bash
python -u -m audioshield.evaluation.cross_test \
  --checkpoint runs/e007_B_fresh/best.pt \
  --data-root E:/AI_voice_detection \
  --manifest-dir manifests \
  --corpora inthewild replaydf ai4t \
  --bootstrap-reps 1000 \
  --out runs/e007_B_fresh_crosstest.json
```

## Fine-Tuning Phase e007

- e007-A: weighted-band backbone adaptation, cross-entropy only.
- e007-B: e007-A plus projected cross-corpus class-conditional SupCon.
- e007-C: e007-B with XLS-R backbone and transplanted WavLM layer band.

## Results At A Glance

Full table and evidence notes are in `RESULTS.md`.

| Run | Change | ITW | ReplayDF | AI4T | Evidence |
|---|---|---:|---:|---:|---|
| e003 | frozen linear baseline | 0.111 | 0.317 | 0.470 | number only; artifact absent - UNVERIFIABLE |
| e005-A | frozen multi-corpus | 0.143 | 0.298 | 0.231 [CI 0.188-0.285] | tracked log |
| e006 | + XC SupCon, frozen | 0.135 | 0.328 | 0.260 | log absent - UNVERIFIABLE |
| e007-A | band FT, CE | 0.1805 | 0.3327 | 0.2565 | experiments/e007 JSON |
| e007-B | + projected SupCon | 0.1167 | 0.3276 | 0.2629 | experiments/e007 JSON |
| e007-C | XLS-R, transplanted band | 0.2009 | 0.4530 | 0.3435 | experiments/e007 JSON |

EER is lower better. e007 JSONs include bootstrap CI metadata and full-corpus
guardrails.
