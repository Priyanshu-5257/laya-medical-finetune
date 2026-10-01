# Full medical benchmark comparison

Evaluated 2026-10-01 on both Kaggle accounts, each using two T4 GPUs. Both runs completed. Evaluation code was cloned from Git at `9a24b68261b28ec24a31631460e07165f9ed3f63`. No additional training was performed.

## Our saved checkpoints

Accuracy (%). Every checkpoint used the same serialized evaluation pack; the two run manifests and pack SHA-256 match exactly. All expected examples were scored, with no dropped rows.

| Checkpoint | MedMCQA validation (4,183) | MedNLI test (1,422) | MedQA test, 4 options (1,273) | PubMedQA official test (500) |
|---|---:|---:|---:|---:|
| Original Laya | 28.69 | 69.27 | 27.73 | 51.80 |
| Original + CE | 31.72 | 83.19 | 27.81 | 54.80 |
| Original + pathwise RLCD | 31.87 | 82.98 | 28.75 | 55.00 |
| Generic alignment only | 32.32 | 33.33 | 27.73 | 55.20 |
| Generic alignment + CE | 34.28 | 84.46 | 31.97 | 56.80 |
| Medical alignment only | 29.69 | 84.11 | 30.01 | 55.20 |
| Medical alignment + CE | 35.33 | 86.85 | 31.03 | 55.40 |
| Medical alignment + pathwise RLCD | 34.59 | 87.06 | 30.71 | 55.40 |
| Constant majority label/position | 32.23 | 33.33 | 27.73 | 55.20 |

## Published reference results

These are selected published benchmarks, not a verified current SOTA list. Dashes mean no score recorded here. Encoder results use separate task-specific fine-tuning; their training and selection budgets differ from ours.

| Published model | MedMCQA validation | MedNLI test | MedQA test, 4 options | PubMedQA test | Source |
|---|---:|---:|---:|---:|---|
| Bio+Clinical BERT | — | 82.70 | — | — | [Table 2](https://aclanthology.org/W19-1909.pdf) |
| Bio-lm RoBERTa-large (ours-large) | — | 88.50 | — | — | [Tables 2 and 5](https://aclanthology.org/2020.clinicalnlp-1.17.pdf) |
| BioLinkBERT-base (110M) | — | — | 40.00 | 70.20 | [Tables 7 and 8](https://arxiv.org/html/2203.15827) |
| BioLinkBERT-large (340M) | — | — | 44.60 | 72.18 | [Tables 7 and 8](https://arxiv.org/html/2203.15827) |
| Med-PaLM 2 (best) | 72.30 | — | — | 81.80 | [Tables 1 and 4](https://arxiv.org/html/2305.09617) |

Med-PaLM 2 also reports 86.5% MedQA accuracy on 1,273 questions, but the exact four/five-option variant was not independently established here; that value is contextual and is excluded from the four-option column. Its best PubMedQA result uses self-consistency with 11 samples; its task maxima do not all represent one checkpoint/inference setup. [Primary paper](https://arxiv.org/html/2305.09617).

## What the full evaluation shows

- Medical alignment + CE is our strongest MedMCQA checkpoint (35.33%), improving 6.65 percentage points over original Laya. It reaches 86.85% MedNLI, a 17.58 point gain. MedMCQA’s majority-option baseline is 32.23%, so our best score exceeds it by only 3.10 points; original + CE and original + RLCD remain below that baseline.
- Medical alignment + pathwise RLCD reaches 87.06% MedNLI, versus 86.85% for CE: only three additional correct answers. CE is ahead by 0.74 points on MedMCQA and 0.31 on MedQA. One seed and aggregate counts do not establish a statistically reliable RLCD advantage.
- Generic alignment + CE is our strongest MedQA checkpoint (31.97%) and PubMedQA checkpoint (56.80%). There is no single winner across all tasks.
- PubMedQA remains a failure to learn the three-way task: medical CE and RLCD both predict yes/no/maybe = 499/1/0, yielding 55.40% against the 55.20% always-yes baseline. Generic + CE predicts 490/10/0. The accuracy increase does not demonstrate robust biomedical abstract reasoning.
- Published task-specific BioLinkBERT-large scores are 44.60% MedQA and 72.18% PubMedQA. Those are more useful near-size reference targets than the much larger LLM systems.

## Comparison limits

- All saved model arms evaluated without further training or test-time tuning; one training seed.
- MedMCQA uses all labeled validation examples, not hidden-label test; prior validation subsets informed development.
- MedNLI uses all 1422 test examples from a processed Hugging Face mirror; official dataset identity has not been independently authenticated.
- Earlier experiments inspected 400 MedQA test examples and a PubMedQA sample overlapping official test; these full scores are not fresh blind holdout results.
- Laya inference uses canonical option order, max_len=512, head_max_len=192; context truncation may affect long inputs.
- Published encoder scores use task-specific fine-tuning; our saved checkpoints are shared across medical tasks. LLM references also differ in scale, training data and inference budget.
- MedNLI train/test exact premise–hypothesis pair overlap was zero. This checks exact pair duplication only; it does not authenticate the mirror or rule out other contamination.
- PubMedQA test membership was matched to the dataset authors’ 500 official test IDs, with gold-label agreement asserted for every example. [Official test labels](https://github.com/pubmedqa/pubmedqa/blob/master/data/test_ground_truth.json).
- MedMCQA validation has been used during earlier model development. Its result is explicitly a validation score and should not be presented as an unseen official test result.

## Artifacts

- [Machine-readable full results](full_benchmark_comparison.json)
- [Published values and source notes](published_benchmark_references.json)
- [Kaggle run A](https://www.kaggle.com/code/aivenger1st/laya-full-benchmark-a)
- [Kaggle run B](https://www.kaggle.com/code/hbpkillerx/laya-full-benchmark-b)
