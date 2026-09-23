"""Build smoke train + dual eval packs for Laya medical fine-tuning."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from laya_medical import (
    ensure_model_dir,
    load_cfg,
    load_yaml,
    maybe_shuffle_choice,
    one_hot,
    tokenize_item,
)


def _subsample(rows: List[Any], n: int, seed: int) -> List[Any]:
    rng = random.Random(seed)
    rows = list(rows)
    if n <= 0 or n >= len(rows):
        rng.shuffle(rows)
        return rows
    return rng.sample(rows, n)


def build_medmcqa_items(tok, cfg: Dict, n: int, seed: int, shuffle_options: bool) -> List[Dict]:
    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    rows = _subsample(list(ds), n * 2, seed)  # oversample; some rows drop on empty options
    rng = random.Random(seed + 1)
    out: List[Dict] = []
    max_len, head = cfg["max_len"], cfg["head_max_len"]
    for row in rows:
        if len(out) >= n:
            break
        opts = {
            "A": str(row.get("opa", "") or ""),
            "B": str(row.get("opb", "") or ""),
            "C": str(row.get("opc", "") or ""),
            "D": str(row.get("opd", "") or ""),
        }
        keys = [k for k, v in opts.items() if v.strip()]
        if len(keys) < 2:
            continue
        cop = row.get("cop")
        try:
            cop_i = int(cop)
        except Exception:
            continue
        # HF medmcqa uses 0-based cop in some builds and 1-based in others.
        if cop_i in (1, 2, 3, 4) and {1: "A", 2: "B", 3: "C", 4: "D"}[cop_i] in keys:
            label_key = {1: "A", 2: "B", 3: "C", 4: "D"}[cop_i]
        elif cop_i in (0, 1, 2, 3) and {0: "A", 1: "B", 2: "C", 3: "D"}[cop_i] in keys:
            label_key = {0: "A", 1: "B", 2: "C", 3: "D"}[cop_i]
        else:
            continue
        keys, label_key = maybe_shuffle_choice(keys, label_key, rng=rng, enabled=shuffle_options)
        criteria = {k: opts[k] for k in keys}
        target = one_hot(len(keys), keys.index(label_key))
        state = {
            "subject": row.get("subject_name") or "",
            "topic": row.get("topic_name") or "",
            "question": row["question"],
        }
        it = tokenize_item(
            tok,
            state,
            "choice",
            "Select the correct medical answer.",
            criteria,
            target,
            max_len=max_len,
            head_max_len=head,
        )
        if it:
            it["meta"]["source"] = "medmcqa"
            out.append(it)
    return out


def _load_mednli(split: str):
    errors = []
    # araag2/MedNLI requires an explicit config (Hub error lists these names).
    for loader in (
        lambda: load_dataset("araag2/MedNLI", "processed", split=split),
        lambda: load_dataset("araag2/MedNLI", "source", split=split),
        lambda: load_dataset("araag2/MedNLI", "conversational", split=split),
        lambda: load_dataset("bigbio/med_nli", "med_nli_source", split=split, trust_remote_code=True),
    ):
        try:
            return loader()
        except Exception as e:
            errors.append(str(e))
    raise RuntimeError(
        "Could not load MedNLI. Tried araag2/MedNLI configs + bigbio/med_nli. Errors: "
        + " | ".join(errors)
    )


def build_mednli_items(tok, cfg: Dict, n: int, seed: int, shuffle_options: bool, split: str = "train") -> List[Dict]:
    ds = _load_mednli(split)
    rows = _subsample(list(ds), n * 2, seed)
    rng = random.Random(seed + 2)
    out: List[Dict] = []
    max_len, head = cfg["max_len"], cfg["head_max_len"]
    label_map = {
        "entailment": "entailment",
        "contradiction": "contradiction",
        "neutral": "neutral",
        0: "entailment",
        1: "contradiction",
        2: "neutral",
        "0": "entailment",
        "1": "contradiction",
        "2": "neutral",
    }
    for row in rows:
        if len(out) >= n:
            break
        gold = row.get("gold_label") or row.get("label") or row.get("labels")
        if isinstance(gold, list) and gold:
            gold = gold[0]
        gold = label_map.get(gold, gold)
        if gold not in ("entailment", "contradiction", "neutral"):
            continue
        keys = ["entailment", "contradiction", "neutral"]
        keys, gold = maybe_shuffle_choice(keys, gold, rng=rng, enabled=shuffle_options)
        criteria = {
            "entailment": "hypothesis follows from the premise",
            "contradiction": "hypothesis contradicts the premise",
            "neutral": "hypothesis is neither entailed nor contradicted",
        }
        criteria = {k: criteria[k] for k in keys}
        target = one_hot(len(keys), keys.index(gold))
        state = {
            "premise": row.get("sentence1") or row.get("premise") or row.get("premise_text") or row.get("sentence_1"),
            "hypothesis": row.get("sentence2") or row.get("hypothesis") or row.get("hypothesis_text") or row.get("sentence_2"),
        }
        if not state["premise"] or not state["hypothesis"]:
            continue
        it = tokenize_item(
            tok,
            state,
            "choice",
            "Clinical NLI: relation of hypothesis to premise?",
            criteria,
            target,
            max_len=max_len,
            head_max_len=head,
        )
        if it:
            it["meta"]["source"] = "mednli"
            out.append(it)
    return out


def build_ag_news_items(tok, cfg: Dict, n: int, seed: int) -> List[Dict]:
    ds = load_dataset("ag_news", split="test")
    rows = _subsample(list(ds), n, seed)
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    out = []
    max_len, head = cfg["max_len"], cfg["head_max_len"]
    for row in rows:
        lab = int(row["label"])
        criteria = {name: name for name in labels}
        target = one_hot(4, lab)
        it = tokenize_item(
            tok,
            {"text": row["text"]},
            "choice",
            "Which news topic is this article?",
            criteria,
            target,
            max_len=max_len,
            head_max_len=head,
        )
        if it:
            it["meta"]["source"] = "ag_news"
            out.append(it)
    return out


def build_emotion_items(tok, cfg: Dict, n: int, seed: int) -> List[Dict]:
    ds = load_dataset("dair-ai/emotion", split="test")
    rows = _subsample(list(ds), n, seed)
    names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    out = []
    max_len, head = cfg["max_len"], cfg["head_max_len"]
    for row in rows:
        lab = int(row["label"])
        if lab >= len(names):
            continue
        criteria = {name: name for name in names}
        target = one_hot(len(names), lab)
        it = tokenize_item(
            tok,
            {"text": row["text"]},
            "choice",
            "Which emotion does this text express?",
            criteria,
            target,
            max_len=max_len,
            head_max_len=head,
        )
        if it:
            it["meta"]["source"] = "emotion"
            out.append(it)
    return out


def build_pubmedqa_items(tok, cfg: Dict, n: int, seed: int) -> List[Dict]:
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    rows = _subsample(list(ds), n, seed)
    out = []
    max_len, head = cfg["max_len"], cfg["head_max_len"]
    for row in rows:
        ans = str(row.get("final_decision") or row.get("label") or "").lower().strip()
        if ans not in ("yes", "no", "maybe"):
            continue
        ctx = row.get("context")
        if isinstance(ctx, dict):
            contexts = ctx.get("contexts") or ctx.get("context") or []
            if isinstance(contexts, list):
                abstract = " ".join(contexts)
            else:
                abstract = str(contexts)
        elif isinstance(ctx, list):
            abstract = " ".join(map(str, ctx))
        else:
            abstract = str(ctx or "")
        keys = ["yes", "no", "maybe"]
        criteria = {
            "yes": "the abstract supports yes",
            "no": "the abstract supports no",
            "maybe": "the abstract is inconclusive",
        }
        target = one_hot(3, keys.index(ans))
        state = {"question": row["question"], "abstract": abstract}
        it = tokenize_item(
            tok,
            state,
            "choice",
            "Based on the abstract, answer the research question.",
            criteria,
            target,
            max_len=max_len,
            head_max_len=head,
        )
        if it:
            it["meta"]["source"] = "pubmedqa"
            out.append(it)
    return out


def build_medqa_items(tok, cfg: Dict, n: int, seed: int) -> List[Dict]:
    # Prefer a common HF mirror; fall back if schema differs.
    try:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    except Exception:
        ds = load_dataset("bigbio/med_qa", "med_qa_en_4options_bigbio_qa", split="test")
    rows = _subsample(list(ds), n, seed)
    out = []
    max_len, head = cfg["max_len"], cfg["head_max_len"]
    for row in rows:
        question = row.get("question") or row.get("question_text") or ""
        options = row.get("options") or row.get("choices") or {}
        answer = row.get("answer") or row.get("answer_idx") or row.get("answer_id")
        if isinstance(options, dict):
            criteria = {str(k): str(v) for k, v in options.items()}
            keys = list(criteria.keys())
        elif isinstance(options, list):
            # list of strings or dicts
            keys = []
            criteria = {}
            for i, opt in enumerate(options):
                if isinstance(opt, dict):
                    k = str(opt.get("key") or opt.get("label") or chr(ord("A") + i))
                    v = str(opt.get("value") or opt.get("text") or opt)
                else:
                    k = chr(ord("A") + i)
                    v = str(opt)
                keys.append(k)
                criteria[k] = v
        else:
            continue
        if len(keys) < 2:
            continue
        if isinstance(answer, int):
            label_key = keys[max(0, min(answer, len(keys) - 1))]
        else:
            answer_s = str(answer).strip()
            if answer_s in criteria:
                label_key = answer_s
            else:
                # match by option text
                label_key = None
                for k, v in criteria.items():
                    if answer_s.lower() in v.lower() or v.lower() in answer_s.lower():
                        label_key = k
                        break
                if label_key is None:
                    continue
        target = one_hot(len(keys), keys.index(label_key))
        it = tokenize_item(
            tok,
            {"question": question},
            "choice",
            "Select the correct USMLE answer.",
            criteria,
            target,
            max_len=max_len,
            head_max_len=head,
        )
        if it:
            it["meta"]["source"] = "medqa"
            out.append(it)
    return out


BUILDERS = {
    "medmcqa": build_medmcqa_items,
    "mednli": build_mednli_items,
    "ag_news": build_ag_news_items,
    "emotion": build_emotion_items,
    "pubmedqa": build_pubmedqa_items,
    "medqa": build_medqa_items,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--out-dir", default="/kaggle/working/data")
    ap.add_argument("--model-cache", default=None)
    args = ap.parse_args()

    conf = load_yaml(args.config)
    model_dir = ensure_model_dir(conf["model_id"], args.model_cache)
    base_cfg = load_cfg(model_dir)
    cfg = {
        "max_len": int(conf.get("max_len", base_cfg.get("max_len", 512))),
        "head_max_len": int(conf.get("head_max_len", base_cfg.get("head_max_len", 192))),
    }
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    train_conf = conf["train"]
    eval_conf = conf["eval"]
    seed = int(train_conf["seed"])
    shuffle = bool(train_conf.get("shuffle_options", True))

    os.makedirs(args.out_dir, exist_ok=True)

    print("Building MedMCQA train slice...")
    train_items = build_medmcqa_items(tok, cfg, int(train_conf["medmcqa_n"]), seed, shuffle)
    print(f"  medmcqa: {len(train_items)}")
    print("Building MedNLI train slice...")
    mednli = build_mednli_items(tok, cfg, int(train_conf["mednli_n"]), seed + 10, shuffle, split="train")
    print(f"  mednli: {len(mednli)}")
    train_items.extend(mednli)
    random.Random(seed).shuffle(train_items)
    train_path = os.path.join(args.out_dir, "train_items.pt")
    torch.save(train_items, train_path)
    print(f"Saved {len(train_items)} train items -> {train_path}")

    n_g = int(eval_conf["generic_n_per_task"])
    n_m = int(eval_conf["medical_n_per_task"])
    eval_packs: Dict[str, List[Dict]] = {"generic": {}, "medical": {}}

    print("Building generic eval packs...")
    for name in eval_conf["generic_tasks"]:
        if name == "ag_news":
            items = build_ag_news_items(tok, cfg, n_g, seed + 100)
        elif name == "emotion":
            items = build_emotion_items(tok, cfg, n_g, seed + 101)
        else:
            raise ValueError(name)
        eval_packs["generic"][name] = items
        print(f"  generic/{name}: {len(items)}")

    print("Building medical held-out eval packs (not used in train)...")
    for name in eval_conf["medical_tasks"]:
        if name == "pubmedqa":
            items = build_pubmedqa_items(tok, cfg, n_m, seed + 200)
        elif name == "medqa":
            items = build_medqa_items(tok, cfg, n_m, seed + 201)
        else:
            raise ValueError(name)
        eval_packs["medical"][name] = items
        print(f"  medical/{name}: {len(items)}")

    eval_path = os.path.join(args.out_dir, "eval_packs.pt")
    torch.save(eval_packs, eval_path)
    meta = {
        "model_id": conf["model_id"],
        "model_dir": model_dir,
        "cfg": cfg,
        "n_train": len(train_items),
        "train_sources": {"medmcqa": int(train_conf["medmcqa_n"]), "mednli": int(train_conf["mednli_n"])},
        "eval": {
            "generic": {k: len(v) for k, v in eval_packs["generic"].items()},
            "medical": {k: len(v) for k, v in eval_packs["medical"].items()},
        },
    }
    with open(os.path.join(args.out_dir, "data_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Wrote", eval_path)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
