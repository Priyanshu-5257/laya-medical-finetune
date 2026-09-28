#!/usr/bin/env bash
# BioClinical ModernBERT encoder + Laya-head distill, then a pure-RLCD medical smoke.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export USE_TF=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WORK="${WORK:-/kaggle/working}"
DISTILL_CFG="${DISTILL_CFG:-configs/distill_bioclinical.yaml}"
MED_CFG="${MED_CFG:-configs/smoke_pure_rlcd.yaml}"
DATA_DIR="${DATA_DIR:-$WORK/data_bioclinical_distill}"
MED_DATA="${MED_DATA:-$WORK/data_bioclinical_medical}"
DISTILL_OUT="${DISTILL_OUT:-$WORK/bioclinical_distill}"
MED_OUT="${MED_OUT:-$WORK/bioclinical_medical_smoke}"

python - <<'PY'
import torch, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "n_gpu", torch.cuda.device_count())
PY

NPROC="${NPROC:-}"
if [[ -z "$NPROC" ]]; then
  NPROC="$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')"
fi
echo "Using nproc_per_node=$NPROC"

echo "=== download BioClinical encoder (safetensors only) ==="
BIO_DIR="$(python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    "thomas-sounack/BioClinical-ModernBERT-large",
    allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
))
PY
)"
echo "bio_dir=$BIO_DIR"

echo "=== prepare typed-decisions distill set + dual eval ==="
rm -f "$DATA_DIR/distill_items.pt" "$DATA_DIR/distill_items_teacher.pt" "$DATA_DIR/eval_packs.pt" "$DATA_DIR/data_meta.json"
python scripts/prepare_distill_data.py --config "$DISTILL_CFG" --out-dir "$DATA_DIR"
TEACHER_DIR="$(python - <<PY
import json
print(json.load(open("$DATA_DIR/data_meta.json"))["teacher_dir"])
PY
)"
echo "teacher_dir=$TEACHER_DIR"

echo "=== distill (frozen BioClinical encoder, Laya soft targets) ==="
torchrun --standalone --nproc_per_node="$NPROC" scripts/distill_ddp.py \
  --config "$DISTILL_CFG" \
  --teacher-dir "$TEACHER_DIR" \
  --bio-dir "$BIO_DIR" \
  --train-items "$DATA_DIR/distill_items.pt" \
  --output-dir "$DISTILL_OUT"

echo "=== eval distilled student vs base Laya (before medical FT) ==="
python scripts/evaluate.py \
  --base-model-dir "$TEACHER_DIR" \
  --ft-model-dir "$DISTILL_OUT" \
  --eval-packs "$DATA_DIR/eval_packs.pt" \
  --out "$WORK/eval_report_distill.json"

python - <<PY
import json
from pathlib import Path
report = json.load(open("$WORK/eval_report_distill.json"))
summary = {
  "ok": True,
  "stage": "distill_bioclinical_head",
  "freeze_encoder": True,
  "generic_base_acc": report["base"]["generic"]["_all"]["accuracy"],
  "generic_ft_acc": report["finetuned"]["generic"]["_all"]["accuracy"],
  "medical_base_acc": report["base"]["medical"]["_all"]["accuracy"],
  "medical_ft_acc": report["finetuned"]["medical"]["_all"]["accuracy"],
  "generic_delta": report["deltas"]["generic"]["_all"]["accuracy_delta"],
  "medical_delta": report["deltas"]["medical"]["_all"]["accuracy_delta"],
  "pubmedqa_delta": report["deltas"]["medical"]["pubmedqa"]["accuracy_delta"],
  "medqa_delta": report["deltas"]["medical"]["medqa"]["accuracy_delta"],
}
Path("$WORK/summary_distill.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "=== medical pure-RLCD smoke from distilled student ==="
rm -f "$MED_DATA/train_items.pt" "$MED_DATA/eval_packs.pt" "$MED_DATA/data_meta.json"
python scripts/prepare_data.py --config "$MED_CFG" --out-dir "$MED_DATA"
torchrun --standalone --nproc_per_node="$NPROC" scripts/train_ddp.py \
  --config "$MED_CFG" \
  --model-dir "$DISTILL_OUT" \
  --train-items "$MED_DATA/train_items.pt" \
  --eval-packs "$MED_DATA/eval_packs.pt" \
  --output-dir "$MED_OUT"

echo "=== eval medical student vs base Laya ==="
python scripts/evaluate.py \
  --base-model-dir "$TEACHER_DIR" \
  --ft-model-dir "$MED_OUT" \
  --eval-packs "$MED_DATA/eval_packs.pt" \
  --out "$WORK/eval_report_bioclinical_medical.json"

python - <<PY
import json
from pathlib import Path
report = json.load(open("$WORK/eval_report_bioclinical_medical.json"))
distill = json.load(open("$WORK/summary_distill.json"))
hist = {}
hp = Path("$MED_OUT/train_history.json")
if hp.exists():
    hist = json.load(open(hp))
summary = {
  "ok": True,
  "stage": "bioclinical_distill_then_medical_smoke",
  "distill": distill,
  "seconds": hist.get("seconds"),
  "history": hist.get("history"),
  "generic_base_acc": report["base"]["generic"]["_all"]["accuracy"],
  "generic_ft_acc": report["finetuned"]["generic"]["_all"]["accuracy"],
  "medical_base_acc": report["base"]["medical"]["_all"]["accuracy"],
  "medical_ft_acc": report["finetuned"]["medical"]["_all"]["accuracy"],
  "generic_delta": report["deltas"]["generic"]["_all"]["accuracy_delta"],
  "medical_delta": report["deltas"]["medical"]["_all"]["accuracy_delta"],
  "pubmedqa_delta": report["deltas"]["medical"]["pubmedqa"]["accuracy_delta"],
  "medqa_delta": report["deltas"]["medical"]["medqa"]["accuracy_delta"],
  "vs_smoke_pure_rlcd_fixed": {"medical_delta": 0.01125, "generic_delta": 0.0025},
}
Path("$WORK/summary_bioclinical_medical.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "DONE_BIOCLINICAL"
