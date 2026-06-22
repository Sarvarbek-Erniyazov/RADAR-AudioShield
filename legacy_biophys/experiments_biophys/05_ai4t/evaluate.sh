#!/usr/bin/env bash
# ============================================================
# AI4T — CROSS-DATASET test (test-only, ~279 clips, small benchmark)
# No train split; evaluates a checkpoint trained on ANOTHER dataset.
# Run build_manifest.py once before first use.
#
# Usage (from RADAR AudioShield root):
#     bash experiments/05_ai4t/evaluate.sh <checkpoint> <tag>
# Example:
#     bash experiments/05_ai4t/evaluate.sh \
#       experiments/03_diffssd/runs/diffssd_wavlm_mild/best.pt diffssd
# ============================================================
set -e

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$EXP_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CKPT="${1:?Usage: evaluate.sh <checkpoint_path> <tag>}"
TAG="${2:?Usage: evaluate.sh <checkpoint_path> <tag>}"

PYTHONPATH="$EXP_DIR" python "$EXP_DIR/scripts/evaluate.py" \
  --dataset-root datasets/05_AI4T/AI4T_dataset \
  --checkpoint "$CKPT" \
  --split test --batch-size 8 --num-workers 8 \
  --output-json "$EXP_DIR/results/ai4t_from_${TAG}_metrics.json" \
  --output-csv  "$EXP_DIR/results/ai4t_from_${TAG}_records.csv"

echo "Done. Metrics: experiments/05_ai4t/results/ai4t_from_${TAG}_metrics.json"
