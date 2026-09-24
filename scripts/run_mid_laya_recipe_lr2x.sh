#!/usr/bin/env bash
# Mid-scale Laya recipe: ~50k Q × 5 epochs, LR ×5.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export USE_TF=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${CONFIG:-configs/mid_laya_recipe_lr2x.yaml}"
WORK="${WORK:-/kaggle/working}"
DATA_DIR="${DATA_DIR:-$WORK/data_mid_laya_lr2x}"
OUT_DIR="${OUT_DIR:-$WORK/laya_medical_mid_laya_lr2x}"

python - <<'PY'
import torch, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "n_gpu", torch.cuda.device_count())
PY

python - <<'PY'
from laya_medical.wandb_util import resolve_wandb_api_key
print("wandb_key_present:", bool(resolve_wandb_api_key()), flush=True)
PY

NPROC="${NPROC:-}"
if [[ -z "$NPROC" ]]; then
  NPROC="$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')"
fi
echo "Using nproc_per_node=$NPROC CONFIG=$CONFIG"

echo "=== prepare_data (force rebuild + soft teacher) ==="
rm -f "$DATA_DIR/train_items.pt" "$DATA_DIR/eval_packs.pt" "$DATA_DIR/data_meta.json"
python scripts/prepare_data.py --config "$CONFIG" --out-dir "$DATA_DIR"

MODEL_DIR="$(python - <<PY
import json
print(json.load(open("$DATA_DIR/data_meta.json"))["model_dir"])
PY
)"
echo "model_dir=$MODEL_DIR"

echo "=== torchrun train_ddp (Laya recipe mid, LR×2, 5 epochs) ==="
torchrun --standalone --nproc_per_node="$NPROC" scripts/train_ddp.py \
  --config "$CONFIG" \
  --model-dir "$MODEL_DIR" \
  --train-items "$DATA_DIR/train_items.pt" \
  --eval-packs "$DATA_DIR/eval_packs.pt" \
  --output-dir "$OUT_DIR"

echo "=== offline evaluate ==="
python scripts/evaluate.py \
  --base-model-dir "$MODEL_DIR" \
  --ft-model-dir "$OUT_DIR" \
  --eval-packs "$DATA_DIR/eval_packs.pt" \
  --out "$WORK/eval_report_mid_laya_lr2x.json"

python - <<PY
import json
from pathlib import Path
report = json.load(open("$WORK/eval_report_mid_laya_lr2x.json"))
meta = json.load(open("$DATA_DIR/data_meta.json"))
hist = {}
hp = Path("$OUT_DIR/train_history.json")
if hp.exists():
    hist = json.load(open(hp))
summary = {
  "ok": True,
  "recipe": "laya_ft_ce_soft_teacher_mid50k_lr2x_e5",
  "nproc": int("$NPROC"),
  "n_train": meta.get("n_train"),
  "soft_targets": meta.get("soft_targets"),
  "lr_encoder": 5.0e-5,
  "lr_head": 2.0e-4,
  "epochs": hist.get("epochs", 5),
  "seconds": hist.get("seconds"),
  "history": hist.get("history"),
  "generic_base_acc": report["base"]["generic"]["_all"]["accuracy"],
  "generic_ft_acc": report["finetuned"]["generic"]["_all"]["accuracy"],
  "medical_base_acc": report["base"]["medical"]["_all"]["accuracy"],
  "medical_ft_acc": report["finetuned"]["medical"]["_all"]["accuracy"],
  "generic_delta": report["deltas"]["generic"]["_all"]["accuracy_delta"],
  "medical_delta": report["deltas"]["medical"]["_all"]["accuracy_delta"],
  "vs_mid_laya_baseline": {"medical_delta": 0.035, "generic_delta": 0.0025, "epochs": 3, "lr_mult": 1},
}
Path("$WORK/summary_mid_laya_lr2x.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "DONE_MID_LAYA_LR5X"
