#!/usr/bin/env bash
# ============================================================
# DiffSSD — BioPhys-HyperRADAR training
# Usage (from RADAR AudioShield root):
#     bash experiments/03_diffssd/train.sh
# ============================================================
set -e

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$EXP_DIR/../.." && pwd)"
cd "$ROOT_DIR"

WAVLM="C:/Users/sharg/.cache/huggingface/hub/models--microsoft--wavlm-base-plus/snapshots/4c66d4806a428f2e922ccfa1a962776e232d487b"

PYTHONPATH="$EXP_DIR" python "$EXP_DIR/scripts/train.py" \
  --dataset-root datasets/03_DiffSSD \
  --output-dir "$EXP_DIR/runs/diffssd_wavlm_mild" \
  --ssl-model-name "$WAVLM" \
  --epochs 100 \
  --batch-size 8 \
  --learning-rate 0.0002 \
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
