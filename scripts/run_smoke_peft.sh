#!/usr/bin/env bash
# PEFT-middle smoke: same data/eval as run_smoke.sh, LoRA on middle encoder layers only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export USE_TF=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${CONFIG:-configs/smoke_peft_middle.yaml}"
WORK="${WORK:-/kaggle/working}"
DATA_DIR="${DATA_DIR:-$WORK/data}"
OUT_DIR="${OUT_DIR:-$WORK/laya_medical_smoke_peft}"
MODEL_ID="${MODEL_ID:-convaiinnovations/laya}"

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
echo "Using nproc_per_node=$NPROC"

echo "=== prepare_data ==="
python scripts/prepare_data.py --config "$CONFIG" --out-dir "$DATA_DIR"

MODEL_DIR="$(python - <<PY
import json
print(json.load(open("$DATA_DIR/data_meta.json"))["model_dir"])
PY
)"
echo "model_dir=$MODEL_DIR"

echo "=== torchrun train_ddp_peft (middle-layer LoRA) ==="
torchrun --standalone --nproc_per_node="$NPROC" scripts/train_ddp_peft.py \
  --config "$CONFIG" \
  --model-dir "$MODEL_DIR" \
  --train-items "$DATA_DIR/train_items.pt" \
  --output-dir "$OUT_DIR"

echo "=== evaluate base vs finetuned ==="
python scripts/evaluate.py \
  --base-model-dir "$MODEL_DIR" \
  --ft-model-dir "$OUT_DIR" \
  --eval-packs "$DATA_DIR/eval_packs.pt" \
  --out "$WORK/eval_report_peft.json"

python - <<PY
import json
from pathlib import Path
report = json.load(open("$WORK/eval_report_peft.json"))
summary = {
  "ok": True,
  "mode": "peft_middle_lora",
  "nproc": int("$NPROC"),
  "output_dir": "$OUT_DIR",
  "generic_base_acc": report["base"]["generic"]["_all"]["accuracy"],
  "generic_ft_acc": report["finetuned"]["generic"]["_all"]["accuracy"],
  "medical_base_acc": report["base"]["medical"]["_all"]["accuracy"],
  "medical_ft_acc": report["finetuned"]["medical"]["_all"]["accuracy"],
  "generic_delta": report["deltas"]["generic"]["_all"]["accuracy_delta"],
  "medical_delta": report["deltas"]["medical"]["_all"]["accuracy_delta"],
}
Path("$WORK/summary_peft.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "DONE_PEFT"
