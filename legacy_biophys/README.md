# Legacy — BioPhys-HyperRADAR (superseded)

This folder preserves an **earlier research direction** that predates the current AudioShield approach. It is kept for provenance and is **not maintained**. The active project lives at the repository root (`src/audioshield/`, `scripts/`, `configs/`).

## What this was

The BioPhys-HyperRADAR line explored physiology-informed features and hyperbolic-geometry representations for deepfake detection, built around a `biophys_hyperradar` package. It was superseded by the current SSL-based approach (frozen WavLM-Large + multi-corpus training + cross-corpus contrastive learning), which proved substantially more effective on cross-corpus generalization.

## Contents

- `core/` — the original `biophys_hyperradar` library
- `implementation/` — a DiffSSD-specific build with its own README
- `experiments_biophys/` — per-corpus experiment scaffolding and small result summaries

Heavy artifacts (model checkpoints, per-utterance score dumps, datasets, run outputs) have been stripped; only code and lightweight records remain. Nothing in the active project imports from this folder.
