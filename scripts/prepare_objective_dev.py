"""Build validation packs for the medical objective comparison."""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from laya_medical import load_yaml

spec = importlib.util.spec_from_file_location("prepare_data", ROOT / "scripts" / "prepare_data.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    conf = load_yaml(args.config)
    cfg = {"max_len": int(conf["max_len"]), "head_max_len": int(conf["head_max_len"])}
    seed = int(conf["train"]["seed"])
    tok = AutoTokenizer.from_pretrained(os.path.join(args.model_dir, "tokenizer"))
    packs = {
        "generic": {
            "ag_news": prep.build_ag_news_items(tok, cfg, 400, seed + 100),
            "emotion": prep.build_emotion_items(tok, cfg, 400, seed + 101),
        },
        "medical": {
            "medmcqa_val": prep.build_medmcqa_items(tok, cfg, 400, seed + 300, True, split="validation"),
            "mednli_dev": prep.build_mednli_items(tok, cfg, 400, seed + 301, True, split="dev"),
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(packs, args.out)
    print({domain: {name: len(items) for name, items in tasks.items()} for domain, tasks in packs.items()})


if __name__ == "__main__":
    main()
