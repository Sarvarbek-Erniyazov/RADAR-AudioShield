#!/usr/bin/env bash
# ============================================================
# ASVspoof5 — BioPhys-HyperRADAR training (SSL unfrozen)
#
# Full model (hyperbolic prototypes + physiology + MoE) with the SAME
# multi-objective loss weights as DiffSSD/FakeOrReal, for fair comparison.
#
# The ONLY difference vs the frozen-SSL datasets is the learning rate:
# unfreezing WavLM requires a LOW lr (5e-6) to avoid catastrophic
# forgetting of the pretrained representation. With lr=2e-4 the SSL
# weights collapse and val_eer stays ~0.44; with lr=5e-6 the full model
# trains correctly (diagnostic: val_eer 0.44 -> 0.12 in one epoch).
#
# Run convert.sh FIRST.
# Usage:
#     bash experiments/01_asvspoof5/train.sh
# ============================================================
set -e

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$EXP_DIR/../.." && pwd)"
cd "$ROOT_DIR"

WAVLM="C:/Users/sharg/.cache/huggingface/hub/models--microsoft--wavlm-base-plus/snapshots/4c66d4806a428f2e922ccfa1a962776e232d487b"

PYTHONPATH="$EXP_DIR" python "$EXP_DIR/scripts/train.py" \
  --dataset-root datasets/01_ASVspoof5 \
  --output-dir "$EXP_DIR/runs/asvspoof5_wavlm_mild" \
  --ssl-model-name "$WAVLM" \
  --unfreeze-ssl \
  --epochs 100 \
  --batch-size 8 \
  --learning-rate 0.000005 \
  --weight-decay 0.0001 \
  --num-workers 8 \
  --prefetch-factor 2 \
  --duration-seconds 4.0 \
  --sample-rate 16000 \
  --balanced-sampler \
  --spoof-loss-weight 1.25 \
  --method-loss-weight 0.1 \
  --media-state-loss-weight 0.2 \
  --target-prototype-loss-weight 0.3 \
  --method-prototype-loss-weight 0.05 \
  --bona-fide-compactness-weight 0.1 \
  --energy-loss-weight 0.02 \
  --consistency-loss-weight 0.1 \
  --focal-gamma 1.0 \
  --spoof-pos-weight 1.0 \
  --early-stopping-patience 12 \
  2>&1 | tee "$EXP_DIR/runs/train_log.txt"
