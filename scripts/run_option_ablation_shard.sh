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
SHARD="${SHARD:?Set SHARD to a or b}"
NPROC="${NPROC:-2}"
DATA_DIR="$WORK/option_ablation_data"
OUT_DIR="$WORK/option_ablation_$SHARD"

case "$SHARD" in
  a) seeds=(20260923 20260924) ;;
  b) seeds=(20260925 20260926) ;;
  *) echo "Unknown shard: $SHARD" >&2; exit 2 ;;
esac

python scripts/prepare_data.py --config "$CONFIG" --out-dir "$DATA_DIR"
python scripts/prepare_objective_dev.py --config "$CONFIG" --model-dir "$BASE_DIR" --out "$DATA_DIR/dev_packs.pt"
python scripts/record_ablation_environment.py --data-dir "$DATA_DIR" --base-dir "$BASE_DIR" --out "$OUT_DIR/environment.json"

for seed in "${seeds[@]}"; do
  for intervention in baseline permutation two_view; do
    arm="${intervention}_${seed}"
    echo "=== Training $arm ==="
    extra=()
    case "$intervention" in
      permutation) extra=(--permute-options) ;;
      two_view) extra=(--permute-options --consistency-teacher-permuted --consistency-weight 0.5) ;;
    esac
    torchrun --standalone --nproc_per_node="$NPROC" scripts/train_ddp.py \
      --config "$CONFIG" --model-dir "$BASE_DIR" \
      --train-items "$DATA_DIR/train_items.pt" --eval-packs "$DATA_DIR/dev_packs.pt" \
      --output-dir "$OUT_DIR/$arm" --gradient-estimator pathwise \
      --stop-after-epoch 2 --seed "$seed" "${extra[@]}"
    python scripts/evaluate.py --base-model-dir "$BASE_DIR" \
      --ft-model-dir "$OUT_DIR/$arm" --eval-packs "$DATA_DIR/eval_packs.pt" \
      --out "$OUT_DIR/test_$arm.json"
  done
done

python scripts/audit_option_ablation.py --base-model-dir "$BASE_DIR" \
  --results-dir "$OUT_DIR" --dev-packs "$DATA_DIR/dev_packs.pt" \
  --out "$OUT_DIR/permutation_audit.json"
python scripts/summarize_option_ablation.py --results-dir "$OUT_DIR" \
  --out "$WORK/option_ablation_${SHARD}_summary.json"
