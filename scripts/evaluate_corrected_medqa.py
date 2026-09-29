"""Re-evaluate all objective arms after using MedQA's answer_idx as gold."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from laya_medical import load_yaml
from laya_medical.eval_metrics import eval_items
from evaluate import load_agent_model

spec = importlib.util.spec_from_file_location("prepare_data", ROOT / "scripts" / "prepare_data.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--config", default="configs/medical_objective_compare.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    conf = load_yaml(args.config)
    cfg = {"max_len": int(conf["max_len"]), "head_max_len": int(conf["head_max_len"])}
    seed = int(conf["train"]["seed"])
    tok = AutoTokenizer.from_pretrained(os.path.join(args.base_model_dir, "tokenizer"))
    items = prep.build_medqa_items(tok, cfg, 400, seed + 201)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {"n": len(items), "models": {}}
    for name, model_dir in (("aligned", args.base_model_dir),
                            *((arm, os.path.join(args.results_dir, arm))
                              for arm in ("pure_rlcd", "rlcd_ce", "ce_only"))):
        print("Evaluating", name, flush=True)
        model, model_tok, _ = load_agent_model(model_dir, device)
        out["models"][name] = eval_items(model, model_tok, items, device)
        print(name, out["models"][name]["accuracy"], flush=True)
        del model
        torch.cuda.empty_cache()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("Wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
