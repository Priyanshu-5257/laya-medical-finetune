#!/usr/bin/env bash
# Full medical fine-tune: prepare data -> torchrun DDP (per-epoch dual eval + wandb) -> final report.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export USE_TF=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${CONFIG:-configs/full.yaml}"
WORK="${WORK:-/kaggle/working}"
DATA_DIR="${DATA_DIR:-$WORK/data}"
OUT_DIR="${OUT_DIR:-$WORK/laya_medical_full}"

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

# Resolve wandb key before torchrun (rank0 will also try Kaggle secrets).
python - <<'PY'
from laya_medical.wandb_util import resolve_wandb_api_key
key = resolve_wandb_api_key()
print("wandb_key_present:", bool(key), flush=True)
PY

NPROC="${NPROC:-}"
if [[ -z "$NPROC" ]]; then
  NPROC="$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')"
fi
echo "Using nproc_per_node=$NPROC"

echo "=== prepare_data (force rebuild train_items; do not reuse stale .pt) ==="
rm -f "$DATA_DIR/train_items.pt" "$DATA_DIR/eval_packs.pt" "$DATA_DIR/data_meta.json"
python scripts/prepare_data.py --config "$CONFIG" --out-dir "$DATA_DIR"

MODEL_DIR="$(python - <<PY
import json
print(json.load(open("$DATA_DIR/data_meta.json"))["model_dir"])
PY
)"
echo "model_dir=$MODEL_DIR"

echo "=== torchrun train_ddp (full FT + per-epoch eval + wandb) ==="
torchrun --standalone --nproc_per_node="$NPROC" scripts/train_ddp.py \
  --config "$CONFIG" \
  --model-dir "$MODEL_DIR" \
  --train-items "$DATA_DIR/train_items.pt" \
  --eval-packs "$DATA_DIR/eval_packs.pt" \
  --output-dir "$OUT_DIR"

echo "=== offline evaluate base vs finetuned (sanity) ==="
python scripts/evaluate.py \
  --base-model-dir "$MODEL_DIR" \
  --ft-model-dir "$OUT_DIR" \
  --eval-packs "$DATA_DIR/eval_packs.pt" \
  --out "$WORK/eval_report_full.json"

python - <<PY
import json
from pathlib import Path
report = json.load(open("$WORK/eval_report_full.json"))
hist = {}
hp = Path("$OUT_DIR/train_history.json")
if hp.exists():
    hist = json.load(open(hp))
summary = {
  "ok": True,
  "mode": "full_ft",
  "nproc": int("$NPROC"),
  "output_dir": "$OUT_DIR",
  "epochs": hist.get("epochs"),
  "n_train": hist.get("n_train"),
  "seconds": hist.get("seconds"),
  "history": hist.get("history"),
  "generic_base_acc": report["base"]["generic"]["_all"]["accuracy"],
  "generic_ft_acc": report["finetuned"]["generic"]["_all"]["accuracy"],
  "medical_base_acc": report["base"]["medical"]["_all"]["accuracy"],
  "medical_ft_acc": report["finetuned"]["medical"]["_all"]["accuracy"],
  "generic_delta": report["deltas"]["generic"]["_all"]["accuracy_delta"],
  "medical_delta": report["deltas"]["medical"]["_all"]["accuracy_delta"],
}
Path("$WORK/summary_full.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "DONE_FULL"
