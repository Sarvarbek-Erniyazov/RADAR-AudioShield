#!/usr/bin/env bash
# ============================================================
# FakeOrReal — evaluate checkpoint (in-domain + cross-dataset)
# Usage (from RADAR AudioShield root):
#     bash experiments/07_fakeorreal/evaluate.sh
# ============================================================
set -e

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$EXP_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CKPT="$EXP_DIR/runs/fakeorreal_wavlm_mild/best.pt"

# 1. In-domain test (FakeOrReal)
PYTHONPATH="$EXP_DIR" python "$EXP_DIR/scripts/evaluate.py" \
  --dataset-root datasets/07_FakeOrReal/for-original \
  --checkpoint "$CKPT" \
  --split test --batch-size 8 --num-workers 8 \
  --output-json "$EXP_DIR/results/test_metrics.json" \
  --output-csv  "$EXP_DIR/results/test_records.csv"

# 2. Cross-dataset: In-the-Wild
PYTHONPATH="$EXP_DIR" python "$EXP_DIR/scripts/evaluate.py" \
  --dataset-root datasets/02_In-the-Wild/release_in_the_wild \
  --checkpoint "$CKPT" \
  --split test --batch-size 8 --num-workers 8 \
  --output-json "$EXP_DIR/results/inthewild_metrics.json" \
  --output-csv  "$EXP_DIR/results/inthewild_records.csv"

# 3. Cross-dataset: ReplayDF
PYTHONPATH="$EXP_DIR" python "$EXP_DIR/scripts/evaluate.py" \
  --dataset-root datasets/04_ReplayDF \
  --checkpoint "$CKPT" \
  --split test --batch-size 8 --num-workers 8 \
  --output-json "$EXP_DIR/results/replaydf_metrics.json" \
  --output-csv  "$EXP_DIR/results/replaydf_records.csv"

echo "Done. Metrics written to experiments/07_fakeorreal/results/"
