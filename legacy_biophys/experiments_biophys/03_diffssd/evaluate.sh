#!/usr/bin/env bash
# ============================================================
# DiffSSD — evaluate trained checkpoint on DiffSSD test split
# Usage (from RADAR AudioShield root):
#     bash experiments/03_diffssd/evaluate.sh
# ============================================================
set -e

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$EXP_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CKPT="$EXP_DIR/runs/diffssd_wavlm_mild/best.pt"

PYTHONPATH="$EXP_DIR" python "$EXP_DIR/scripts/evaluate.py" \
  --dataset-root datasets/03_DiffSSD \
  --checkpoint "$CKPT" \
  --split test --batch-size 8 --num-workers 8 \
  --output-json "$EXP_DIR/results/test_metrics.json" \
  --output-csv  "$EXP_DIR/results/test_records.csv"

PYTHONPATH="$EXP_DIR" python "$EXP_DIR/scripts/evaluate.py" \
  --dataset-root datasets/03_DiffSSD \
  --checkpoint "$CKPT" \
  --split test --batch-size 8 --num-workers 8 --score-mode fused \
  --output-json "$EXP_DIR/results/test_metrics_fused.json" \
  --output-csv  "$EXP_DIR/results/test_records_fused.csv"

echo "Done. Metrics written to experiments/03_diffssd/results/"
