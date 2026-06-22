#!/usr/bin/env bash
# ============================================================
# ASVspoof5 — evaluate checkpoint on the val split
# (ASVspoof5 eval set flac_E not downloaded; using dev/val)
# Usage (from RADAR AudioShield root):
#     bash experiments/01_asvspoof5/evaluate.sh
# ============================================================
set -e

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$EXP_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CKPT="$EXP_DIR/runs/asvspoof5_wavlm_mild/best.pt"

PYTHONPATH="$EXP_DIR" python "$EXP_DIR/scripts/evaluate.py" \
  --dataset-root datasets/01_ASVspoof5 \
  --checkpoint "$CKPT" \
  --split val --batch-size 8 --num-workers 8 \
  --output-json "$EXP_DIR/results/val_metrics.json" \
  --output-csv  "$EXP_DIR/results/val_records.csv"

echo "Done. Metrics written to experiments/01_asvspoof5/results/"
