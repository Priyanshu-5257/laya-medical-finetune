# Laya medical fine-tune experiments

Public code: https://github.com/Priyanshu-5257/laya-medical-finetune

Goal: specialize `convaiinnovations/laya` (ModernBERT-large encoder + decision head, 512 context) on medical decisions without collapsing generic accuracy. Compute was Kaggle 2×T4. Training is Laya-style RLCD: Gaussian noise on logits, strictly proper reward (log + spherical, RPS on score items), REINFORCE with a group-mean baseline. Eval is always held out of the train mix: AG News + Emotion (generic) and PubMedQA + MedQA (medical), 400 items each. Base Laya on that split is generic **0.770** and medical **0.405** (PubMedQA 0.53, MedQA 0.28).

## Data and recipe notes

- Train sources were MedMCQA and MedNLI only. Medical eval sets were never in training.
- Early smokes used a dual 0-based/1-based `cop` map. On `openlifescienceai/medmcqa`, `cop` is a ClassLabel **0–3** (`a–d`). That dual map mislabeled about **71%** of rows. Fixed in `8b13a14` / `ce12556`. Numbers from runs before that fix are marked.
- Default RLCD knobs unless a row says otherwise: group size 4, σ 0.4→0.1, encoder LR `2.5e-5`, head LR `1.0e-4`, micro-batch 8, grad accum 4, AdamW, cosine to `1e-6`, fp16, `ce_weight=0`, `sph_weight=0.5`.
- Laya's own typed-decisions fine-tune differs: soft teacher targets, `ce_weight=1`, `sph_weight=0.75`, and the encoder is trained, not frozen.

## Smoke comparisons (~4–5k questions, 1 epoch)

| Run | Kernel | Medical Δ | Generic Δ | Notes |
|---|---|---:|---:|---|
| Full-FT pure RLCD | [laya-medical-smoke](https://www.kaggle.com/code/aivenger1st/laya-medical-smoke) | **+1.4pp** (0.419) | −0.5pp | Buggy MedMCQA labels. PubMedQA +3.3pp, MedQA flat |
| Middle-third LoRA | [laya-medical-smoke-peft](https://www.kaggle.com/code/aivenger1st/laya-medical-smoke-peft) | +0.1pp | 0.0pp | Layers 9–17/28, ~6.8% of encoder trainable |
| Middle-third full FT | [laya-medical-smoke-midft](https://www.kaggle.com/code/aivenger1st/laya-medical-smoke-midft) | −0.1pp | −0.1pp | Same layers, no LoRA |
| Laya recipe (CE + soft) | [laya-medical-smoke-laya-recipe](https://www.kaggle.com/code/aivenger1st/laya-medical-smoke-laya-recipe) | +0.6pp (0.411) | −0.4pp | Fixed labels. `ce=1`, `sph=0.75`, target = 0.5 one-hot + 0.5 base-Laya softmax |
| Pure RLCD, fixed labels | [laya-medical-smoke-pure-rlcd](https://www.kaggle.com/code/aivenger1st/laya-medical-smoke-pure-rlcd) | **+1.1pp** (0.416) | **+0.3pp** | Fair control vs the Laya-recipe smoke |

On the fair A/B, pure RLCD beat CE + soft teacher. Middle-layer-only updates did not move medical accuracy.

## Longer pure RLCD

Full MedMCQA train (~183k) + full MedNLI train (~11k), 5 epochs, ~9.9 h. [laya-medical-full](https://www.kaggle.com/code/aivenger1st/laya-medical-full). [WandB](https://wandb.ai/hbpkillerx/laya-medical/runs/olrnrxiw). This run is after the `cop` fix. About 192k train items after a 2k calibration holdout.

| Epoch | Medical acc | Medical Δ | Generic acc |
|---:|---:|---:|---:|
| 1 | 0.400 | −0.6pp | 0.774 |
| 2 | 0.455 | +4.9pp | 0.741 |
| **3** | **0.463** | **+5.6pp** | **0.775** |
| 4 | 0.446 | +4.0pp | 0.761 |
| 5 | 0.431 | +2.5pp | 0.768 |

Final calibrated checkpoint (epoch 5, choice temperature hit the 10.0 cap): medical **+2.8pp** (0.433), generic **−0.3pp** (0.768). PubMedQA +3.3pp, MedQA +2.3pp. Mean proper-score reward fell across epochs (about −1.60 → −2.1) while accuracy rose through epoch 3. That is expected: the logged reward is the raw proper score on noisy train samples, and log score punishes confident mistakes much harder than argmax accuracy measures them.

## Laya recipe at ~50k

45k MedMCQA + 5k MedNLI, CE×1, `sph=0.75`, soft teacher mix 0.5.

| Run | Kernel | Result |
|---|---|---|
| 3 epochs, 1× LR (~1.6 h) | [laya-medical-mid-laya-recipe](https://www.kaggle.com/code/aivenger1st/laya-medical-mid-laya-recipe) | Final medical **+3.5pp** (0.440), generic +0.3pp. PubMedQA +4.8pp, MedQA +2.3pp. Reward improved (−0.85 → −0.52) |
| 5 epochs, 5× LR | [laya-medical-mid-laya-lr5x](https://www.kaggle.com/code/aivenger1st/laya-medical-mid-laya-lr5x) | Diverged (`loss=nan` in epoch 1). Cancelled. Encoder `1.25e-4`, head `5e-4` is too high in fp16 with CE |
| 5 epochs, 2× LR (~2.6 h) | [laya-medical-mid-laya-lr2x](https://www.kaggle.com/code/aivenger1st/laya-medical-mid-laya-lr2x) | Reward improved every epoch (−0.89 → −0.40). Medical peaked at epoch 3 (**+3.9pp**) then fell. Final calibrated medical **+1.5pp**, generic −0.4pp. PubMedQA only +0.5pp, MedQA +2.5pp |

Same learning rate and stopping near epoch 3 beat raising the learning rate.

## BioClinical encoder swap

Hypothesis: Laya's ModernBERT backbone is a general encoder (2T mixed tokens, not a medical continued pretrain). Swap in `thomas-sounack/BioClinical-ModernBERT-large` (same hidden size 1024, same vocab and `[MASK]` ids), keep Laya's decision head, distill on Laya's public decision set, then do a medical RLCD smoke.

Run: [laya-medical-bioclinical](https://www.kaggle.com/code/aivenger1st/laya-medical-bioclinical). Encoder load covered **170/170** tensors. Distill was head-only (encoder frozen, 26.5M trainable head params) for 3 epochs on **6,000** `LocalLLaMA/typed-decisions` items, matching frozen Laya soft probabilities with cross-entropy. Distill CE only moved 1.13 → 1.01. Then a 1-epoch pure-RLCD smoke (4k MedMCQA + 1k MedNLI) with the encoder unfrozen.

| Stage | Generic | Medical | vs base Laya |
|---|---:|---:|---|
| After distill only | 0.179 | 0.349 | generic **−59.1pp**, medical **−5.6pp** |
| After medical smoke | 0.239 | 0.295 | generic **−53.1pp**, medical **−11.0pp** |

PubMedQA fell to 0.33 (−20pp) and MedQA to 0.26 (−2pp). The medical pretraining did not survive the swap in a form the Laya head could use. Three epochs of head-only distillation on 6k decisions was not enough to rebuild the decision interface, and the short medical fine-tune trained a broken student further down. This does **not** beat the plain Laya full fine-tune.

The in-loop medical-smoke deltas in `train_history` are relative to the distilled student, not to base Laya. The table above is the offline eval against base Laya.

## What the results support

- The starting Laya encoder is not a medical model. Full encoder RLCD on MedMCQA/MedNLI still helps: best held-out medical gain so far is **+5.6pp** at epoch 3 of the full pure-RLCD run, with generic accuracy held.
- Gains peak and then slip. Later epochs and a 2× learning rate improved the training reward while held-out medical accuracy got worse. Prefer the epoch-3 region over riding the reward curve.
- On the small fair smoke, pure RLCD beat Laya's CE + soft-teacher mix. At 50k×3 the Laya mix did reach +3.5pp, still short of the full pure-RLCD peak.
- Middle-only LoRA or middle-only full FT did not help.
- Dropping in BioClinical ModernBERT and distilling only the head failed the gate (medical already down before medical fine-tuning). A next attempt would need a much closer interface fit (more distillation, or a light encoder update) before spending a long medical run on that student.
