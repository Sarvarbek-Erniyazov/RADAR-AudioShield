#!/usr/bin/env bash
# ============================================================
# ReplayDF — CROSS-DATASET test (test-only, replay-attack benchmark)
# No train split; evaluates a checkpoint trained on ANOTHER dataset.
#
# Usage (from RADAR AudioShield root):
#     bash experiments/04_replaydf/evaluate.sh <checkpoint> <tag>
# Example:
#     bash experiments/04_replaydf/evaluate.sh \
#       experiments/03_diffssd/runs/diffssd_wavlm_mild/best.pt diffssd
# ============================================================
set -e

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$EXP_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CKPT="${1:?Usage: evaluate.sh <checkpoint_path> <tag>}"
TAG="${2:?Usage: evaluate.sh <checkpoint_path> <tag>}"

PYTHONPATH="$EXP_DIR" python "$EXP_DIR/scripts/evaluate.py" \
  --dataset-root datasets/04_ReplayDF \
  --checkpoint "$CKPT" \
  --split test --batch-size 8 --num-workers 8 \
  --output-json "$EXP_DIR/results/replaydf_from_${TAG}_metrics.json" \
  --output-csv  "$EXP_DIR/results/replaydf_from_${TAG}_records.csv"

echo "Done. Metrics: experiments/04_replaydf/results/replaydf_from_${TAG}_metrics.json"
