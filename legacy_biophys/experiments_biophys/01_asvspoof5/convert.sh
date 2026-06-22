#!/usr/bin/env bash
# ============================================================
# ASVspoof5 — build manifest from TSV (run ONCE before train)
# Usage (from RADAR AudioShield root):
#     bash experiments/01_asvspoof5/convert.sh
# ============================================================
set -e

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$EXP_DIR/../.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH="$EXP_DIR" python "$EXP_DIR/scripts/convert_asvspoof5.py" \
  --asv-root datasets/01_ASVspoof5

echo ""
echo "Verifying manifest..."
python -c "import pandas as pd; df=pd.read_csv('datasets/01_ASVspoof5/train_val_test_splits.csv'); print(df['method_name'].value_counts()); print(); print(df['set'].value_counts())"
