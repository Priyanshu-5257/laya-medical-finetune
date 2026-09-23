"""Evaluate base vs fine-tuned Laya on generic + held-out medical packs."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from laya.common import build_model
from laya_medical.eval_metrics import eval_packs


def load_agent_model(model_dir: str, device: torch.device):
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    temps = cfg.get("temperature", [1.0, 1.0, 1.0])
    if isinstance(temps, list):
        model.temperature.copy_(torch.tensor(temps, dtype=torch.float32))
    model.to(device).eval()
    return model, tok, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model-dir", required=True)
    ap.add_argument("--ft-model-dir", required=True)
    ap.add_argument("--eval-packs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    packs = torch.load(args.eval_packs, weights_only=False)

    print("Evaluating BASE checkpoint...", flush=True)
    base_model, base_tok, _ = load_agent_model(args.base_model_dir, device)
    base_metrics = eval_packs(base_model, base_tok, packs, device)
    del base_model
    torch.cuda.empty_cache()

    print("Evaluating FINE-TUNED checkpoint...", flush=True)
    ft_model, ft_tok, _ = load_agent_model(args.ft_model_dir, device)
    ft_metrics = eval_packs(ft_model, ft_tok, packs, device)

    report = {
        "base": base_metrics,
        "finetuned": ft_metrics,
        "deltas": {},
    }
    for domain in packs:
        report["deltas"][domain] = {}
        for task in list(packs[domain]) + ["_all"]:
            b = base_metrics[domain][task]["accuracy"]
            f = ft_metrics[domain][task]["accuracy"]
            report["deltas"][domain][task] = {
                "accuracy_base": b,
                "accuracy_ft": f,
                "accuracy_delta": (f - b) if (b == b and f == f) else None,
            }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    print("Wrote", args.out, flush=True)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    main()
