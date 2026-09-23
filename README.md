# Laya medical fine-tune (smoke → longer)

Public recipes to specialize [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) on medical decision tasks with **RLCD** (strictly proper scoring rules), using `torchrun` for multi-GPU (Kaggle T4×2).

## Smoke design (~1 hour)

| | |
|---|---|
| Base | `convaiinnovations/laya` (512 context) |
| Train | MedMCQA (~4k) + MedNLI (~1k), option-shuffled hard labels |
| Objective | Pure RLCD (`ce_weight=0`), REINFORCE + group-mean baseline (GRPO-style) |
| Eval A (generic) | AG News + DAIR Emotion (forgetting check) |
| Eval B (medical held-out) | PubMedQA labeled + MedQA USMLE (never in train) |

### Variants

| Script | What changes |
|---|---|
| `bash scripts/run_smoke.sh` | Full encoder + head fine-tune |
| `bash scripts/run_smoke_peft.sh` | PEFT LoRA on middle-third encoder layers only |
| `bash scripts/run_smoke_middle_ft.sh` | **Full-FT middle-third layers only** (no LoRA); early/late frozen |

## Quick start (Kaggle / any multi-GPU box)

```bash
pip install -r requirements.txt
bash scripts/run_smoke.sh
```

`run_smoke.sh` will:

1. `scripts/prepare_data.py` — build `train_items.pt` + dual `eval_packs.pt`
2. `torchrun --standalone --nproc_per_node=$NPROC scripts/train_ddp.py` — DDP RLCD
3. `scripts/evaluate.py` — base vs fine-tuned on generic + medical packs
4. write `/kaggle/working/summary.json` and `eval_report.json`

Override env vars as needed:

```bash
CONFIG=configs/smoke.yaml WORK=/kaggle/working NPROC=2 bash scripts/run_smoke.sh
```

## Layout

```
configs/smoke.yaml
laya_medical/          # shared tokenization / collate helpers
scripts/
  prepare_data.py
  train_ddp.py         # torchrun entrypoint
  evaluate.py
  run_smoke.sh
kaggle/laya-medical-smoke/   # thin kernel: clone this repo + run_smoke.sh
```

## Training notes

- Keep **RLCD** as the main loss; do not swap to pure CE (destroys calibration).
- RLCD with **one-hot MCQ labels** still pushes mass onto the gold option per example; it does **not** by itself invent epistemic uncertainty. Held-out temperature fit + NLL/Brier/ECE are what we use to judge calibration.
- MedMCQA `cop` is resolved **once** from dataset features (HF build is ClassLabel `0–3`); never dual 0-based/1-based per row.
- Advantages use a **per-question** group mean/std (not a batch-wide z-score).
- Calibration temperatures are fit on a **held-out** slice of the train pool.
- Pass gate before a longer run: medical accuracy up vs base, generic not collapsed.

## License

Apache-2.0 for this repo’s scripts. Upstream Laya weights follow their Apache-2.0 terms. Respect each dataset’s license (MedMCQA, MedNLI, PubMedQA, MedQA, AG News, Emotion).
