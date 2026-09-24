#!/usr/bin/env bash
# Smoke: pure RLCD + fixed MedMCQA labels (control vs Laya-recipe smoke).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export USE_TF=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${CONFIG:-configs/smoke_pure_rlcd.yaml}"
WORK="${WORK:-/kaggle/working}"
DATA_DIR="${DATA_DIR:-$WORK/data_pure_rlcd}"
OUT_DIR="${OUT_DIR:-$WORK/laya_medical_smoke_pure_rlcd}"

python - <<'PY'
import torch, platform, sys
print("python", sys.version.split()[0])
print("platform", platform.platform())
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
n = torch.cuda.device_count()
print("n_gpu", n)
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}", p.name, f"{p.total_memory/1e9:.1f}GB")
PY

NPROC="${NPROC:-}"
if [[ -z "$NPROC" ]]; then
  NPROC="$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')"
fi
echo "Using nproc_per_node=$NPROC CONFIG=$CONFIG"

echo "=== prepare_data (force rebuild; fixed MedMCQA 0-based labels) ==="
rm -f "$DATA_DIR/train_items.pt" "$DATA_DIR/eval_packs.pt" "$DATA_DIR/data_meta.json"
python scripts/prepare_data.py --config "$CONFIG" --out-dir "$DATA_DIR"

MODEL_DIR="$(python - <<PY
import json
print(json.load(open("$DATA_DIR/data_meta.json"))["model_dir"])
PY
)"
echo "model_dir=$MODEL_DIR"

echo "=== torchrun train_ddp (pure RLCD, CE=0) ==="
torchrun --standalone --nproc_per_node="$NPROC" scripts/train_ddp.py \
  --config "$CONFIG" \
  --model-dir "$MODEL_DIR" \
  --train-items "$DATA_DIR/train_items.pt" \
  --eval-packs "$DATA_DIR/eval_packs.pt" \
  --output-dir "$OUT_DIR"

echo "=== evaluate base vs finetuned ==="
python scripts/evaluate.py \
  --base-model-dir "$MODEL_DIR" \
  --ft-model-dir "$OUT_DIR" \
  --eval-packs "$DATA_DIR/eval_packs.pt" \
  --out "$WORK/eval_report_pure_rlcd.json"

python - <<PY
import json
from pathlib import Path
report = json.load(open("$WORK/eval_report_pure_rlcd.json"))
summary = {
  "ok": True,
  "recipe": "pure_rlcd_fixed_labels",
  "nproc": int("$NPROC"),
  "output_dir": "$OUT_DIR",
  "ce_weight": 0.0,
  "sph_weight": 0.5,
  "soft_targets": False,
  "medmcqa_cop": "ClassLabel_0_based_fixed",
  "generic_base_acc": report["base"]["generic"]["_all"]["accuracy"],
  "generic_ft_acc": report["finetuned"]["generic"]["_all"]["accuracy"],
  "medical_base_acc": report["base"]["medical"]["_all"]["accuracy"],
  "medical_ft_acc": report["finetuned"]["medical"]["_all"]["accuracy"],
  "generic_delta": report["deltas"]["generic"]["_all"]["accuracy_delta"],
  "medical_delta": report["deltas"]["medical"]["_all"]["accuracy_delta"],
  "vs_laya_recipe_smoke": {
    "laya_medical_delta": 0.00625,
    "laya_generic_delta": -0.00375,
  },
  "vs_old_buggy_label_smoke": {
    "medical_delta": 0.01375,
    "generic_delta": -0.005,
  },
}
Path("$WORK/summary_pure_rlcd.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "DONE_PURE_RLCD"
