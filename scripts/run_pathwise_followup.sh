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
DATA_DIR="$WORK/pathwise_followup_data"
DEV_PACKS="$DATA_DIR/dev_packs.pt"
OUT_DIR="$WORK/pathwise_followup"
NPROC="${NPROC:-2}"

python scripts/prepare_data.py --config "$CONFIG" --out-dir "$DATA_DIR"
python scripts/prepare_objective_dev.py --config "$CONFIG" --model-dir "$BASE_DIR" --out "$DEV_PACKS"

for seed in 20260923 20260924; do
  for intervention in baseline two_view; do
    arm="${intervention}_${seed}"
    echo "=== Training $arm, pathwise RLCD through epoch 2 of the 3-epoch schedule ==="
    extra=()
    if [[ "$intervention" == two_view ]]; then
      extra=(--permute-options --consistency-teacher-permuted --consistency-weight 0.5)
    fi
    torchrun --standalone --nproc_per_node="$NPROC" scripts/train_ddp.py \
      --config "$CONFIG" --model-dir "$BASE_DIR" \
      --train-items "$DATA_DIR/train_items.pt" --eval-packs "$DEV_PACKS" \
      --output-dir "$OUT_DIR/$arm" --gradient-estimator pathwise \
      --stop-after-epoch 2 --seed "$seed" "${extra[@]}"
    python scripts/evaluate.py --base-model-dir "$BASE_DIR" \
      --ft-model-dir "$OUT_DIR/$arm" --eval-packs "$DATA_DIR/eval_packs.pt" \
      --out "$OUT_DIR/test_$arm.json"
  done
done

python scripts/audit_pathwise_followup.py --base-model-dir "$BASE_DIR" \
  --results-dir "$OUT_DIR" --dev-packs "$DEV_PACKS" \
  --out "$OUT_DIR/permutation_audit.json"
python scripts/summarize_pathwise_followup.py --results-dir "$OUT_DIR" \
  --out "$WORK/pathwise_followup_summary.json"
