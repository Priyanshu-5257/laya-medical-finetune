#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export USE_TF=0

CONFIG="${CONFIG:-configs/medical_objective_compare.yaml}"
WORK="${WORK:-/kaggle/working}"
BASE_DIR="${BASE_DIR:?Set BASE_DIR to the aligned BioClinical checkpoint directory}"
DATA_DIR="$WORK/medical_objective_data"
DEV_PACKS="$DATA_DIR/dev_packs.pt"
OUT_DIR="$WORK/medical_objective_compare"
NPROC="${NPROC:-2}"

python scripts/prepare_data.py --config "$CONFIG" --out-dir "$DATA_DIR"
python scripts/prepare_objective_dev.py --config "$CONFIG" --model-dir "$BASE_DIR" --out "$DEV_PACKS"

for arm in pure_rlcd rlcd_ce ce_only; do
  case "$arm" in
    pure_rlcd) RLCD=1; CE=0 ;;
    rlcd_ce) RLCD=1; CE=1 ;;
    ce_only) RLCD=0; CE=1 ;;
  esac
  echo "=== Training $arm from $BASE_DIR ==="
  torchrun --standalone --nproc_per_node="$NPROC" scripts/train_ddp.py \
    --config "$CONFIG" \
    --model-dir "$BASE_DIR" \
    --train-items "$DATA_DIR/train_items.pt" \
    --eval-packs "$DEV_PACKS" \
    --output-dir "$OUT_DIR/$arm" \
    --rlcd-weight "$RLCD" \
    --ce-weight "$CE" \
    --seed 20260923
  python scripts/evaluate.py \
    --base-model-dir "$BASE_DIR" \
    --ft-model-dir "$OUT_DIR/$arm" \
    --eval-packs "$DATA_DIR/eval_packs.pt" \
    --out "$OUT_DIR/test_$arm.json"
done

python scripts/summarize_objective_compare.py --results-dir "$OUT_DIR" --out "$WORK/medical_objective_comparison.json"
