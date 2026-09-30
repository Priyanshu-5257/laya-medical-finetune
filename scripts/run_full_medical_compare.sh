#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export USE_TF=0

WORK="${WORK:-/kaggle/working}"
SHARD="${SHARD:?Set SHARD to a or b}"
NPROC="${NPROC:-2}"
CONFIG=configs/medical_full_compare.yaml
DATA_DIR="$WORK/full_compare_data"
OUT_DIR="$WORK/full_compare_$SHARD"
mkdir -p "$OUT_DIR"

TEACHER_DIR="$(python -c 'from laya_medical import ensure_model_dir; print(ensure_model_dir("convaiinnovations/laya"))' | tail -1)"
python scripts/prepare_data.py --config "$CONFIG" --out-dir "$DATA_DIR"
python scripts/prepare_objective_dev.py --config "$CONFIG" --model-dir "$TEACHER_DIR" --out "$DATA_DIR/dev_packs.pt"
python scripts/record_ablation_environment.py --data-dir "$DATA_DIR" --base-dir "$TEACHER_DIR" --out "$OUT_DIR/environment.json"

case "$SHARD" in
  a)
    GENERIC_ALIGNED_DIR="${GENERIC_ALIGNED_DIR:?Set GENERIC_ALIGNED_DIR to the prior aligned checkpoint}"
    arms=(original_ce original_rlcd generic_aligned_ce)
    ;;
  b)
    BIO_DIR="$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("thomas-sounack/BioClinical-ModernBERT-large", revision="c684f470c2ee60531d8c47c32187dba746c84201", allow_patterns=["config.json","model.safetensors","tokenizer.json","tokenizer_config.json","special_tokens_map.json"]))' | tail -1)"
    python scripts/diagnose_bioclinical_swap.py --teacher-dir "$TEACHER_DIR" --bio-dir "$BIO_DIR" \
      --dev-packs "$DATA_DIR/dev_packs.pt" --out "$OUT_DIR/raw_swap_dev.json"
    ALIGN_DATA="$WORK/full_compare_alignment_data"
    python scripts/prepare_distill_data.py --config configs/distill_bioclinical_medical.yaml --out-dir "$ALIGN_DATA"
    MEDICAL_ALIGNED_DIR="$WORK/bioclinical_medical_aligned"
    torchrun --standalone --nproc_per_node="$NPROC" scripts/distill_ddp.py \
      --config configs/distill_bioclinical_medical.yaml \
      --teacher-dir "$TEACHER_DIR" --bio-dir "$BIO_DIR" \
      --train-items "$ALIGN_DATA/distill_items.pt" --output-dir "$MEDICAL_ALIGNED_DIR"
    python scripts/evaluate.py --base-model-dir "$TEACHER_DIR" \
      --ft-model-dir "$MEDICAL_ALIGNED_DIR" --eval-packs "$DATA_DIR/dev_packs.pt" \
      --out "$OUT_DIR/medical_alignment_dev.json"
    arms=(medical_aligned_ce medical_aligned_rlcd)
    ;;
  *) echo "Unknown shard: $SHARD" >&2; exit 2 ;;
esac

for arm in "${arms[@]}"; do
  case "$arm" in
    original_*) base="$TEACHER_DIR" ;;
    generic_aligned_*) base="$GENERIC_ALIGNED_DIR" ;;
    medical_aligned_*) base="$MEDICAL_ALIGNED_DIR" ;;
  esac
  objective=(--ce-weight 1 --rlcd-weight 0)
  if [[ "$arm" == *_rlcd ]]; then
    objective=(--ce-weight 0 --rlcd-weight 1)
  fi
  echo "=== $arm | $base ==="
  torchrun --standalone --nproc_per_node="$NPROC" scripts/train_ddp.py \
    --config "$CONFIG" --model-dir "$base" \
    --train-items "$DATA_DIR/train_items.pt" --eval-packs "$DATA_DIR/dev_packs.pt" \
    --output-dir "$OUT_DIR/$arm" --gradient-estimator pathwise \
    --seed 20260923 "${objective[@]}"
  python scripts/evaluate.py --base-model-dir "$base" \
    --ft-model-dir "$OUT_DIR/$arm" --eval-packs "$DATA_DIR/eval_packs.pt" \
    --out "$OUT_DIR/test_$arm.json"
done

python scripts/summarize_full_medical_compare.py --results-dir "$OUT_DIR" \
  --out "$WORK/full_compare_${SHARD}_summary.json"
