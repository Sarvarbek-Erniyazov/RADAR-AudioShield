#!/usr/bin/env bash
# ============================================================
# LlamaPartialSpoof — evaluate on its own val split
# Usage (from RADAR AudioShield root):
#     bash experiments/00_llama_partial_spoof/evaluate.sh
# ============================================================
set -e

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$EXP_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CKPT="$EXP_DIR/runs/llama_wavlm_mild/best.pt"

PYTHONPATH="$EXP_DIR" python "$EXP_DIR/scripts/evaluate.py" \
  --dataset-root datasets/00_LlamaPartialSpoof \
  --checkpoint "$CKPT" \
  --split val --batch-size 8 --num-workers 8 \
  --output-json "$EXP_DIR/results/val_metrics.json" \
  --output-csv  "$EXP_DIR/results/val_records.csv"

echo "Done. Metrics: experiments/00_llama_partial_spoof/results/"
