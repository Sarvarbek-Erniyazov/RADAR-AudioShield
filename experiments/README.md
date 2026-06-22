# Experiment Records

Each subfolder is a committed, lightweight record of one experiment — enough to understand what was run and what it produced, **without** the heavy checkpoints (which are gitignored). For full results and the cross-experiment story see [`../RESULTS.md`](../RESULTS.md).

Each folder contains:
- `run_config.json` — the exact resolved configuration
- `crosstest.log` — per-corpus OOD EER (ITW, ReplayDF, AI4T), Kwok bona-fide matrix, probe accuracy
- `train_summary.log` — per-epoch dev EER and the early-stop point

| Folder | What it tests | Headline |
|--------|---------------|----------|
| `e002A_augmentation` | Frozen + augmentation | ITW 0.127 |
| `e002B_consistency` | Augmentation + consistency loss | Regressed (ITW 0.149) |
| `e004_shallow_ft` | Shallow top-4 fine-tuning (single-corpus) | ITW 0.192 — entrenches shortcut |
| `e005A_multicorpus` | Frozen multi-corpus, balanced | AI4T 0.231 [CI .188–.285] |
| `e005C_posweight` | + spoof positive weight | ITW 0.122 / AI4T 0.297 trade-off |
| `e006_xc_contrastive` | + cross-corpus contrastive | ITW 0.135; contrastive dormant frozen |

*(e003 linear baseline — ITW 0.111 — predates this run directory; its numbers are in `RESULTS.md`.)*

To re-run any experiment, use its `run_config.json` with `scripts/train_e002.py` (see the root README).
