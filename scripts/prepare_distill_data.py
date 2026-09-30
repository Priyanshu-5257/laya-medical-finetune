"""Typed-decisions items for Laya distillation, plus the usual dual eval packs."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import importlib.util

from laya_medical import ensure_model_dir, load_cfg, load_yaml, tokenize_item

_spec = importlib.util.spec_from_file_location("prepare_data", _ROOT / "scripts" / "prepare_data.py")
_prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prep)
build_ag_news_items = _prep.build_ag_news_items
build_emotion_items = _prep.build_emotion_items
build_medqa_items = _prep.build_medqa_items
build_pubmedqa_items = _prep.build_pubmedqa_items


def build_typed_items(tok, cfg: dict, split: str = "train") -> list:
    ds = load_dataset("LocalLLaMA/typed-decisions", "all", split=split)
    out = []
    max_len, head = cfg["max_len"], cfg["head_max_len"]
    for row in ds:
        state = json.loads(row["state"]) if isinstance(row["state"], str) else row["state"]
        questions = json.loads(row["questions"]) if isinstance(row["questions"], str) else row["questions"]
        gold = json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"]
        for qid, q in questions.items():
            if qid not in gold:
                continue
            g = gold[qid]
            probs = g.get("probabilities") if isinstance(g, dict) else None
            if not isinstance(probs, dict):
                continue
            t = q["type"]
            crit = q.get("criteria", {})
            if t == "choice":
                keys = list(crit.keys())
                target = [float(probs.get(k, 0.0)) for k in keys]
            elif t == "noul":
                target = [
                    float(probs.get("false", 0.5)),
                    float(probs.get("true", 0.5)),
                ]
            elif t == "score":
                n_levels = len(crit) if isinstance(crit, list) else 4
                target = [float(probs.get(str(i), 0.0)) for i in range(n_levels)]
            else:
                continue
            it = tokenize_item(
                tok,
                state,
                t,
                q.get("instructions") or "",
                crit,
                target,
                max_len=max_len,
                head_max_len=head,
            )
            if it:
                it["meta"]["source"] = "typed-decisions"
                it["meta"]["qid"] = qid
                out.append(it)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/distill_bioclinical.yaml")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    conf = load_yaml(args.config)
    laya_dir = ensure_model_dir(conf["teacher_id"])
    base_cfg = load_cfg(laya_dir)
    cfg = {
        "max_len": int(conf.get("max_len", base_cfg.get("max_len", 512))),
        "head_max_len": int(conf.get("head_max_len", base_cfg.get("head_max_len", 192))),
    }
    tok = AutoTokenizer.from_pretrained(os.path.join(laya_dir, "tokenizer"))
    seed = int(conf.get("seed", 20260923))
    print("Building alignment train items...", flush=True)
    typed_items = build_typed_items(tok, cfg, split="train")
    n_generic = int(conf.get("generic_train_n_per_task", 4000))
    train_items = typed_items + build_ag_news_items(tok, cfg, n_generic, seed=seed, split="train")
    train_items += build_emotion_items(tok, cfg, n_generic, seed=seed + 1, split="train")
    medical_conf = conf.get("medical_alignment") or {}
    medical_counts = {}
    if medical_conf.get("enabled", False):
        n_nli = int(medical_conf.get("mednli_n", 0))
        n_mcqa = int(medical_conf.get("medmcqa_n", 0))
        nli_items = _prep.build_mednli_items(tok, cfg, n_nli, seed + 10, True, split="train")
        mcqa_items = _prep.build_medmcqa_items(tok, cfg, n_mcqa, seed, True, split="train")
        for item in nli_items:
            item["meta"]["teacher_mix"] = float(medical_conf.get("mednli_teacher_mix", 0.5))
        for item in mcqa_items:
            item["meta"]["teacher_mix"] = 0.0
        medical_counts = {"mednli": len(nli_items), "medmcqa": len(mcqa_items)}
        train_items.extend(nli_items)
        train_items.extend(mcqa_items)
    random.Random(seed).shuffle(train_items)
    print(f"  typed-decisions: {len(typed_items)}, medical: {medical_counts}, total: {len(train_items)}", flush=True)
    eval_conf = conf["eval"]
    packs = {"generic": {}, "medical": {}, "alignment": {}}
    packs["generic"]["ag_news"] = build_ag_news_items(tok, cfg, int(eval_conf["generic_n_per_task"]), seed + 100)
    packs["generic"]["emotion"] = build_emotion_items(tok, cfg, int(eval_conf["generic_n_per_task"]), seed + 101)
    packs["medical"]["pubmedqa"] = build_pubmedqa_items(tok, cfg, int(eval_conf["medical_n_per_task"]), seed + 200)
    packs["medical"]["medqa"] = build_medqa_items(tok, cfg, int(eval_conf["medical_n_per_task"]), seed + 201)
    packs["alignment"]["typed_decisions"] = build_typed_items(tok, cfg, split="test")
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(train_items, os.path.join(args.out_dir, "distill_items.pt"))
    torch.save(packs, os.path.join(args.out_dir, "eval_packs.pt"))
    meta = {
        "teacher_id": conf["teacher_id"],
        "teacher_dir": laya_dir,
        "n_distill": len(train_items),
        "medical_alignment": medical_counts,
        "eval": {
            "generic": {k: len(v) for k, v in packs["generic"].items()},
            "medical": {k: len(v) for k, v in packs["medical"].items()},
        },
    }
    with open(os.path.join(args.out_dir, "data_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
