# Laya medical fine-tune experiments

Public code: https://github.com/Priyanshu-5257/laya-medical-finetune

Goal: specialize `convaiinnovations/laya` (ModernBERT-large encoder + decision head, 512 context) on medical decisions without collapsing generic accuracy. Compute was Kaggle 2×T4. Training is Laya-style RLCD: Gaussian noise on logits, strictly proper reward (log + spherical, RPS on score items), REINFORCE with a group-mean baseline. Eval is always held out of the train mix: AG News + Emotion (generic) and PubMedQA + MedQA (medical), 400 items each. Base Laya on that split is generic **0.770** and medical **0.405** (PubMedQA 0.53, MedQA 0.28).

## Experiments at a glance

Medical external is the combined 400 PubMedQA + 400 MedQA score; generic external is 400 AG News + 400 Emotion. Medical dev is 400 MedMCQA validation + 400 MedNLI dev and was introduced for the later controlled runs. Scores are percentages. Earlier runs used a different evaluation preparation, and some pre-audit MedQA labels were wrong, so compare scores within each experiment group rather than ranking every row together. The external medical pack was reused for exploration and is not a pristine final test.

| Approach | Short description | Medical dev | Medical external | Generic external |
|---|---|---:|---:|---:|
| Base Laya | Original general encoder and decision head | — | 40.5 | 77.0 |
| Small pure RLCD smoke, old labels | One epoch with the earlier MedMCQA label bug | — | 41.9 | ~76.5 |
| Small pure RLCD smoke, fixed labels | Full encoder and head, 5k questions, one epoch | — | 41.6 | ~77.3 |
| Middle-third LoRA | Train LoRA on encoder layers 9–17 only | — | ~40.6 | ~77.0 |
| Middle-third full FT | Train the same layers without LoRA | — | ~40.4 | ~76.9 |
| Small Laya recipe smoke | RLCD + CE, half teacher targets, one epoch | — | 41.1 | ~76.6 |
| Full-data pure RLCD, epoch 3 | ~194k medical questions; reported peak at epoch 3 of five | — | **46.3** | 77.5 |
| 50k Laya recipe | RLCD + CE + half teacher targets, three epochs | — | 44.0 | ~77.3 |
| 50k Laya recipe, 2× LR | Five epochs; final checkpoint after an epoch-3 peak | — | ~42.0 | ~76.6 |
| 50k Laya recipe, 5× LR | Training diverged in epoch 1 | — | — | — |
| BioClinical swap v1 | Frozen medical encoder; head-only distillation, then medical smoke | — | 29.5 | 23.9 |
| BioClinical alignment v2 | Train medical encoder and head on decisions plus generic tasks | — | 42.6 | 85.5 |
| Alignment v2 + medical smoke | One epoch of medical RLCD from the aligned checkpoint | — | 42.9 | 84.9 |
| Aligned checkpoint, corrected pack | Starting point for the later controlled runs | 28.0 | 42.3 | 85.5 |
| Pure RLCD objective | Same aligned checkpoint and 5k sample, three epochs | 31.4 | 45.3 | 85.6 |
| RLCD + CE objective | Add hard-label CE to the same medical run | 30.5 | 36.8 | 86.3 |
| CE-only objective | Hard-label CE without RLCD | 30.4 | 44.8 | 84.6 |
| Legacy normalized RLCD | Score-function gradient with batch-normalized advantage | 30.1 | 31.4 | 85.5 |
| Leave-one-out RLCD | Unnormalized leave-one-out score-function gradient | 30.4 | 44.9 | 86.1 |
| Pathwise RLCD | Differentiate the noisy proper-score reward directly | 34.6 | 45.0 | 85.3 |
| Pathwise + one shuffled view | Random student option order and consistency weight 0.1 | 32.4 | 43.8 | 86.5 |
| Pathwise control, seed 20260923 | Stop at epoch 2 of the three-epoch schedule | 32.6 | 45.6 | 85.6 |
| Pathwise + two views, seed 20260923 | Independent option orders; consistency weight 0.5 | **43.0** | **47.5** | 85.8 |
| Pathwise control, seed 20260924 | Same recipe with a second random seed | 31.9 | 34.5 | 85.4 |
| Pathwise + two views, seed 20260924 | Same stronger order intervention with second seed | 33.1 | 42.8 | 86.3 |

The two-view method improved both same-seed comparisons, but the medical gain varied sharply. The strongest single external score here is 47.5%; it comes from a repeatedly examined 800-item sample and should be treated as exploratory.

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

### BioClinical alignment v2 (28 September 2026)

[Private Kaggle run](https://www.kaggle.com/code/aivenger1st/laya-medical-bioclinical-v2). This corrected the distillation path: the BioClinical encoder received gradients (`detach_encoder=false`), with LR 5e-6 while the head used 1e-4; train examples combined 6k typed-decisions questions with 4k AG News and 4k Emotion questions from their **train** splits. Targets mixed the base Laya distribution with gold labels 50:50. Held-out typed-decisions test, generic test, PubMedQA, and MedQA were used for evaluation. Tokenizer identity and full encoder/head weight coverage were checked. The 3-epoch alignment took 2,839 seconds; CE fell 0.891 → 0.679. The checkpoint passed the pre-medical gate.

| Stage | Generic acc | Medical acc | Typed-decisions test acc | Generic NLL / ECE | Medical NLL / ECE |
|---|---:|---:|---:|---:|---:|
| Base Laya | 0.7713 | 0.4050 | 0.3635 | 0.805 / 0.111 | 1.526 / 0.246 |
| After alignment | **0.8550** | 0.4263 | 0.5290 | **0.456 / 0.076** | **1.180 / 0.103** |
| After 1-epoch medical RLCD smoke | 0.8488 | 0.4288 | — | 1.395 / 0.590 | 1.230 / 0.121 |

The medical smoke gained only **0.25pp** over the aligned checkpoint while generic accuracy fell **0.63pp**. More concerning, generic NLL rose from 0.456 to 1.395 and ECE from 0.076 to 0.590. MedQA accuracy stayed at 0.280 versus base Laya. PubMedQA accuracy rose from 0.530 to 0.578, but **0.578 is exactly the majority-class accuracy of this 400-item sample** (231 yes, 131 no, 38 maybe); MedQA's majority-class accuracy is 0.273. Aggregate accuracy alone therefore does not establish useful medical learning. The corrected swap is much better than v1 but does not beat the plain Laya full fine-tune's 0.463 medical peak. Keep the aligned checkpoint as the stronger v2 artifact and diagnose medical-stage calibration before a longer medical run. The 800-case medical evaluation is too small to establish a small gain reliably.

The Kaggle log also showed that fp16 gradient scaling skipped an optimizer step while the cosine scheduler advanced. The scheduler guards in `distill_ddp.py` and `train_ddp.py` were fixed after this run; the v2 results above used the earlier code.

RLCD implementation check: the noisy logits, proper-score reward, group baseline, detached action, and Gaussian log-probability gradient in `train_ddp.py` match Laya's published fine-tuning notebook. The medical smoke differed materially in its training recipe: hard one-hot targets, `ce_weight=0`, `sph_weight=0.5`, and one epoch at constant σ=0.4. Laya's notebook used soft targets, CE weight 1, spherical weight 0.75, and four epochs with σ decreasing toward 0.1. This smoke included 4,600 training examples after the calibration holdout, roughly 72 optimizer updates per rank. MedNLI contributed the intended 1,000 examples with all three labels represented; the label loader was not the failure mode in this run.

### Medical objective comparison (29 September 2026)

[Private Kaggle run](https://www.kaggle.com/code/aivenger1st/laya-medical-objective-compare), 2×T4. All arms started from the same aligned BioClinical v2 checkpoint and used the same 4k MedMCQA + 1k MedNLI training sample, 3 epochs, encoder LR 5e-6, head LR 1e-4, and seed. Only the medical loss changed: pure RLCD, RLCD + CE, or CE only. Selection used final-epoch accuracy on 400 MedMCQA validation and 400 MedNLI dev items, with a generic accuracy-drop limit of 3pp. The 400 PubMedQA and 400 MedQA items were reported separately; these test samples had already been examined in earlier experiments, so they are not a pristine final test.

| Medical objective | Medical dev acc | PubMedQA | MedQA | Medical test acc | Generic test acc |
|---|---:|---:|---:|---:|---:|
| Aligned checkpoint (before medical training) | 0.2788 | 0.5775 | 0.2675 | 0.4225 | 0.8550 |
| Pure RLCD | **0.3138** | **0.6050** | 0.3000 | **0.4525** | 0.8563 |
| RLCD + CE | 0.3050 | 0.4250 | **0.3100** | 0.3675 | **0.8625** |
| CE only | 0.3038 | **0.6050** | 0.2900 | 0.4475 | 0.8463 |

Pure RLCD won the prespecified dev accuracy rule and had the highest medical test accuracy. Its advantage over CE only was just 8/800 dev items and 4/800 test items, so this run does not establish a reliable superiority of RLCD. The simple majority baselines were 0.3025 on medical dev and 0.4225 on medical test; even the best arm was only 1.13pp and 3.00pp above them. MedNLI dev accuracy stayed below that sample's majority baseline in every arm. None of the models predicted PubMedQA's `maybe` class (38/400 cases). RLCD + CE became heavily biased toward `no` on PubMedQA, explaining its test collapse despite the highest MedQA score.

Probability quality also regressed: the aligned checkpoint had medical NLL 1.182 after correcting MedQA labels and generic NLL/ECE 0.456/0.076 on the test pack; pure RLCD ended at medical NLL 1.226 and generic NLL/ECE 1.022/0.467. CE only had the best medical test NLL of the three arms (1.218), but this was still worse than the aligned checkpoint. These data support pure RLCD as the exploratory accuracy choice, while the aligned checkpoint remains preferable when calibrated probabilities matter. The original [machine-readable comparison](results/medical_objective_comparison.json) retains the pre-audit scores; the corrected MedQA scores are in [corrected_medqa_scores.json](results/corrected_medqa_scores.json).

### Evaluation pipeline audit (29 September 2026)

[Private Kaggle audit](https://www.kaggle.com/code/aivenger1st/laya-medical-eval-audit) rebuilt all four 400-item medical packs from the source datasets. For MedMCQA validation, MedNLI dev, and PubMedQA, all 400 inputs and gold labels matched the saved packs and source rows. For MedQA, all inputs matched, but **3/400 saved gold labels were wrong**: the builder preferred answer text and accepted a substring match even though the dataset supplies `answer_idx`. The builder now uses `answer_idx`, with an exact answer-text fallback. The table above includes the corrected MedQA scores. The starting checkpoint's original *dev* baseline was also evaluated with dropout on; corrected eval-mode MedMCQA/MedNLI accuracy is 0.2725/0.2850 (0.2788 combined). The trained arms' per-epoch dev scores already used eval mode. `train_ddp.py` now switches to eval mode for its starting baseline as well.

No MedMCQA or MedNLI dev state was truncated, and all options remained distinct. State text was truncated in 14/400 PubMedQA and 3/400 MedQA items; no option text was truncated. The PubMedQA sample mixes **201 official test** and **199 development** items, confirmed against the [official test IDs](https://github.com/pubmedqa/pubmedqa/blob/master/data/test_ground_truth.json). It cannot be compared directly with published fixed-test scores.

Option order has a much larger effect than these rare truncations. When option token spans were reversed and predictions mapped back to their original answers, pure RLCD changed its selected answer on **47.8%** of MedMCQA, **63.0%** of MedNLI, **27.5%** of PubMedQA, and **44.5%** of MedQA items. MedMCQA accuracy changed from 0.3025 to 0.3450 under reversal on the *same questions*. This is evidence of a substantial presentation/position sensitivity; the audit report records the corresponding aligned-checkpoint rates and second permutation. No training occurred in this audit.

### RLCD gradient and option-order experiment (29 September 2026)

[Private Kaggle run](https://www.kaggle.com/code/aivenger1st/laya-medical-gradient-compare), 2×T4. Three arms train from the same aligned BioClinical checkpoint, on the same 4k MedMCQA + 1k MedNLI sample and three-epoch budget: the previous normalized score-function estimator, an unnormalized leave-one-out score-function estimator, and a pathwise gradient through the same noisy proper-score reward. With group size four, subtracting the group mean including the sampled reward scales the unnormalized expected score-function gradient by 3/4; the leave-one-out arm corrects this factor. Batch-wide advantage standardization changes the gradient scale as reward spread changes, while the pathwise arm directly differentiates the Gaussian-smoothed expected reward. These arms test estimator behavior rather than new medical data.

The best estimator was selected on the 800-item MedMCQA validation + MedNLI dev pack, subject to at most a three percentage point generic accuracy drop, with dev NLL as tie-breaker. A fourth arm started again from the same checkpoint and used the selected estimator with a fresh option permutation for each training item and a 0.1-weight teacher/student consistency KL. External PubMedQA and corrected MedQA scores were reported after selection. The option-order audit compared answer changes under reversal and rotation on the two dev tasks. [Machine-readable results](results/gradient_comparison.json) are tracked in the repo; the full Kaggle log remains in local ignored artifacts.

| Arm, final epoch | Medical dev | MedMCQA dev | MedNLI dev | PubMedQA | MedQA, corrected | Medical external | Generic external |
|---|---:|---:|---:|---:|---:|---:|---:|
| Aligned checkpoint | 0.2800 | 0.2725 | 0.2875 | 0.5775 | 0.2675 | 0.4225 | 0.8550 |
| Legacy normalized RLCD | 0.3013 | 0.2950 | 0.3075 | 0.3275 | 0.3000 | 0.3138 | 0.8550 |
| Leave-one-out RLCD | 0.3038 | 0.3000 | 0.3075 | **0.6050** | 0.2925 | 0.4488 | 0.8613 |
| **Pathwise RLCD** | **0.3463** | **0.3375** | **0.3550** | 0.6000 | **0.3000** | **0.4500** | 0.8525 |
| Pathwise + option permutation + consistency | 0.3238 | 0.3050 | 0.3425 | 0.5775 | 0.2975 | 0.4375 | **0.8650** |

Pathwise won the specified final-epoch dev rule by 4.25pp over leave-one-out (34 more correct of 800), with generic accuracy 0.25pp below the aligned checkpoint. Its best medical dev epoch was **0.3588 at epoch 2**, followed by 0.3463 at epoch 3. Leave-one-out also peaked at epoch 2 (0.3200) and fell to 0.3038. On the external medical sample, pathwise exceeded leave-one-out by just **1/800** and the previously run pure RLCD checkpoint by **−2/800**; this is no evidence of a meaningful transfer improvement. All three improved estimators stayed near the external sample's 0.4225 majority baseline. Pathwise medical external NLL was 1.214 versus 1.225 for leave-one-out and 1.233 for legacy, but the aligned checkpoint's NLL remained lower at about 1.182. The external sample has been examined in earlier experiments and is not a pristine final test.

The legacy estimator's pre-clip gradient norm rose from 5.6 to **74.1** across the σ=0.4→0.1 schedule; leave-one-out rose from 3.6 to 7.7, while pathwise stayed in the 9.0–14.3 range. This supports the predicted scale instability from reward standardization as σ shrinks. It does not by itself prove which change caused the accuracy difference, because estimator choice also changes update direction and magnitude. The legacy arm's PubMedQA accuracy collapsed to 0.3275 here, despite the previous pure RLCD run reaching 0.6050 under a different random stream, another sign that the 5k-example RLCD outcome is unstable.

Option permutation + consistency improved generic external accuracy to 0.8650 and generic NLL to 0.674 (pathwise without it: 0.8525 and 0.903), but reduced medical dev accuracy by 2.25pp and external medical accuracy by 1.25pp. It reduced the reverse-order semantic flip rate on MedMCQA from pathwise 47.8% to 38.3%, and on MedNLI from 84.5% to 60.0%. These rates are still high. In particular, pathwise MedNLI accuracy changed from about 35.8% in original order to 46.0% after rotation on the same 400 cases; the decision interface remains strongly order sensitive. The permutation arm helps stability but is not the accuracy winner at this budget and consistency weight.

### Pathwise epoch-2 and two-view follow-up (29 September 2026)

[Private Kaggle run](https://www.kaggle.com/code/aivenger1st/laya-medical-pathwise-followup), 2×T4. This follow-up fixes the stopping point at epoch 2 of the same three-epoch learning-rate and noise schedule (σ=0.4 then 0.25), because the preceding run peaked there. It compares pathwise RLCD with a two-view option-order arm at seeds 20260923 and 20260924. Each pair starts from the same aligned checkpoint and uses the same 5k medical sample, calibration holdout, dev packs, and external packs. The second seed varies model dropout, Gaussian reward noise, training order, and option shuffles. In the two-view arm, independent random teacher and student option orders are sampled for each item; the student receives pathwise RLCD plus a 0.5-weight teacher/student consistency KL after mapping probabilities back to semantic options. The control has pathwise RLCD alone. [Full result summary](results/pathwise_followup_summary.json).

| Seed | Arm | Medical dev | MedMCQA dev | MedNLI dev | PubMedQA | MedQA | Medical external | Generic external |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 20260923 | Pathwise control | 0.3263 | 0.3200 | 0.3325 | 0.6300 | 0.2825 | 0.4563 | 0.8563 |
| 20260923 | **Two-view** | **0.4300** | 0.3050 | **0.5550** | 0.6175 | **0.3325** | **0.4750** | **0.8575** |
| 20260924 | Pathwise control | 0.3188 | 0.2825 | 0.3550 | 0.4125 | 0.2775 | 0.3450 | 0.8538 |
| 20260924 | **Two-view** | **0.3313** | **0.3075** | 0.3550 | **0.5775** | 0.2775 | **0.4275** | **0.8625** |

Two-view beat its same-seed control on combined medical dev, external medical, generic external, and medical NLL in both seeds. The size of the medical dev gain varied from **+1.25pp** to **+10.38pp**; the first seed's gain came mainly from MedNLI (0.3325 → 0.5550), while the second seed's MedNLI score did not change. External medical gain varied from **+1.88pp** to **+8.25pp**, but the second seed's two-view external score of 0.4275 was only 0.5pp above that sample's 0.4225 majority baseline. The external pack has been examined repeatedly, so these differences are exploratory. Neither two-view run beat the aligned checkpoint's external medical NLL of 1.1824 (two-view: 1.1898 and 1.1988), although both substantially improved NLL over their pathwise controls.

Two-view lowered MedMCQA reverse-order semantic flips in both seeds (51.8% → 42.5% and 62.3% → 42.3%). For MedNLI, the first seed improved sharply (72.3% → 10.3%), but the second seed worsened under reversal (36.8% → 52.0%); its rotation flip rate did improve (70.0% → 47.3%). The method is promising but still seed sensitive and does not reliably solve presentation dependence. The pathwise-only 20260923 checkpoint also scored 0.3263 on medical dev here versus 0.3588 at epoch 2 of the preceding run despite nominally matching seed and schedule; this apparent reproduction gap should be investigated before treating a single-seed gain as settled.

### Published benchmark context

These are published scores on the named benchmarks, **not scores on our exact sampled items or with our Laya prompt**. They provide a scale for interpreting the gap, not a direct model ranking.

| Task | Our best (sample) | Biomedical encoder example | Larger model example |
|---|---:|---:|---:|
| PubMedQA | 63.0% (400) | BioLinkBERT-large 72.2% ([model card](https://huggingface.co/michiyasunaga/BioLinkBERT-large)) | Med-PaLM 2 81.8% ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11922739/)) |
| MedQA, 4 options | 33.3% (400, corrected labels) | BioLinkBERT-large 44.6% ([model card](https://huggingface.co/michiyasunaga/BioLinkBERT-large)) | MedGemma 27B Text 87.7%; Med-Gemini 91.1% ([Google Research](https://research.google/blog/medgemma-our-most-capable-open-models-for-health-ai-development/), [Med-Gemini report](https://research.google/blog/advancing-medical-ai-with-med-gemini/)) |
| MedMCQA validation | 33.8% (400) | PubMedBERT 40% without retrieval, 43% with PubMed retrieval ([dataset paper](https://proceedings.mlr.press/v174/pal22a/pal22a.pdf)) | Med-PaLM 2 72.3% ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11922739/)) |
| MedNLI dev | 55.5% (400) | Bio+Clinical BERT 82.7% on the **test** split ([paper](https://aclanthology.org/W19-1909/)) | — |

The PubMedQA comparison has a particularly important split mismatch: our builder samples 400 from all 1,000 `pqa_labeled` items, whereas the [benchmark's official preparation](https://github.com/pubmedqa/pubmedqa/blob/master/preprocess/split_dataset.py) reserves 500 items for test and uses the other 500 for train/development. Our model did not train on PubMedQA labels, but the published figures are still not evaluated on the same rows. MedNLI dev is another split mismatch; the Bio+Clinical BERT result uses test. Our medical training uses only 4k MedMCQA and 1k MedNLI items, while published task-specific encoders generally train on the full task data. The near-chance MedNLI result and MedQA result near the 25% four-choice chance level are stronger evidence of a weak medical decision interface than the aggregate medical accuracy alone.

## What the results support

- The starting Laya encoder is not a medical model. Full encoder RLCD on MedMCQA/MedNLI helped: the full-data run reported **46.3%** medical accuracy at epoch 3 with generic accuracy held. The later two-view run reached **47.5%** on the repeatedly examined external sample, but varied substantially between seeds.
- Gains peak and then slip. Later epochs and a 2× learning rate improved the training reward while held-out medical accuracy got worse. Prefer the epoch-3 region over riding the reward curve.
- On the small fair smoke, pure RLCD beat Laya's CE + soft-teacher mix. At 50k×3 the Laya mix did reach +3.5pp, still short of the full pure-RLCD peak.
- Middle-only LoRA or middle-only full FT did not help.
- Dropping in BioClinical ModernBERT and distilling only the head failed the gate. Updating the encoder during alignment recovered generic accuracy, but the later medical gains remained small and seed sensitive.
