#!/usr/bin/env bash
# ============================================================
# In-the-Wild — CROSS-DATASET test (test-only benchmark)
# In-the-Wild has no train split; it evaluates a checkpoint
# trained on ANOTHER dataset (DiffSSD, ASVspoof5, FakeOrReal...).
#
# Usage (from RADAR AudioShield root):
#     bash experiments/02_in_the_wild/evaluate.sh <checkpoint> <tag>
# Example:
#     bash experiments/02_in_the_wild/evaluate.sh \
#       experiments/03_diffssd/runs/diffssd_wavlm_mild/best.pt diffssd
# ============================================================
set -e

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$EXP_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CKPT="${1:?Usage: evaluate.sh <checkpoint_path> <tag>}"
TAG="${2:?Usage: evaluate.sh <checkpoint_path> <tag>}"

PYTHONPATH="$EXP_DIR" python "$EXP_DIR/scripts/evaluate.py" \
  --dataset-root datasets/02_In-the-Wild/release_in_the_wild \
  --checkpoint "$CKPT" \
  --split test --batch-size 8 --num-workers 8 \
  --output-json "$EXP_DIR/results/inthewild_from_${TAG}_metrics.json" \
  --output-csv  "$EXP_DIR/results/inthewild_from_${TAG}_records.csv"

echo "Done. Metrics: experiments/02_in_the_wild/results/inthewild_from_${TAG}_metrics.json"
