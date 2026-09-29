#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export USE_TF=0

CONFIG=configs/medical_objective_compare.yaml
WORK="${WORK:-/kaggle/working}"
BASE_DIR="${BASE_DIR:?Set BASE_DIR to the aligned BioClinical checkpoint directory}"
DATA_DIR="$WORK/gradient_data"
DEV_PACKS="$DATA_DIR/dev_packs.pt"
OUT_DIR="$WORK/gradient_compare"
NPROC="${NPROC:-2}"

python scripts/prepare_data.py --config "$CONFIG" --out-dir "$DATA_DIR"
python scripts/prepare_objective_dev.py --config "$CONFIG" --model-dir "$BASE_DIR" --out "$DEV_PACKS"

for arm in legacy loo pathwise; do
  echo "=== Training $arm from $BASE_DIR ==="
  torchrun --standalone --nproc_per_node="$NPROC" scripts/train_ddp.py \
    --config "$CONFIG" --model-dir "$BASE_DIR" \
    --train-items "$DATA_DIR/train_items.pt" --eval-packs "$DEV_PACKS" \
    --output-dir "$OUT_DIR/$arm" --gradient-estimator "$arm" --seed 20260923
  python scripts/evaluate.py --base-model-dir "$BASE_DIR" \
    --ft-model-dir "$OUT_DIR/$arm" --eval-packs "$DATA_DIR/eval_packs.pt" \
    --out "$OUT_DIR/test_$arm.json"
done

WINNER="$(python scripts/summarize_gradient_compare.py --results-dir "$OUT_DIR" \
  --out "$WORK/gradient_compare_stage1.json" --print-winner)"
echo "=== Training permutation + consistency with $WINNER estimator ==="
torchrun --standalone --nproc_per_node="$NPROC" scripts/train_ddp.py \
  --config "$CONFIG" --model-dir "$BASE_DIR" \
  --train-items "$DATA_DIR/train_items.pt" --eval-packs "$DEV_PACKS" \
  --output-dir "$OUT_DIR/permutation" --gradient-estimator "$WINNER" \
  --permute-options --consistency-weight 0.1 --seed 20260923
python scripts/evaluate.py --base-model-dir "$BASE_DIR" \
  --ft-model-dir "$OUT_DIR/permutation" --eval-packs "$DATA_DIR/eval_packs.pt" \
  --out "$OUT_DIR/test_permutation.json"

python scripts/audit_gradient_permutation.py --base-model-dir "$BASE_DIR" \
  --results-dir "$OUT_DIR" --dev-packs "$DEV_PACKS" \
  --out "$OUT_DIR/permutation_audit.json"
python scripts/summarize_gradient_compare.py --results-dir "$OUT_DIR" \
  --out "$WORK/gradient_comparison.json"
