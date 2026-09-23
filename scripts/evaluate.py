"""Evaluate base vs fine-tuned Laya on generic + held-out medical packs."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from laya.common import build_model, ece_score
from laya_medical import collate_train_batch


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


@torch.no_grad()
def eval_items(model, tok, items: List[Dict[str, Any]], device: torch.device, batch_size: int = 8) -> Dict[str, float]:
    if not items:
        return {"n": 0, "accuracy": float("nan"), "ece": float("nan"), "nll": float("nan"), "brier": float("nan")}
    correct = []
    confs = []
    nlls = []
    briers = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        batch = collate_train_batch(chunk, tok.pad_token_id)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits, _ = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )
        logits = logits.float()
        mask = batch["marker_mask"].to(device)
        qtype = batch["qtype"].to(device)
        # apply per-type temperature
        temps = model.temperature.to(device)[qtype].unsqueeze(-1).clamp(min=0.1)
        probs = torch.softmax(logits.masked_fill(~mask, -1e4) / temps, -1)
        labels = batch["label"].to(device)
        target = batch["target"].to(device)
        pred = probs.argmax(-1)
        for j in range(len(chunk)):
            k = int(mask[j].sum().item())
            p = probs[j, :k].cpu().numpy()
            y = int(labels[j].item())
            t = target[j, :k].cpu().numpy()
            ok = float(y == int(pred[j].item()))
            correct.append(ok)
            confs.append(float(p.max()))
            nlls.append(float(-(np.log(max(p[y], 1e-12)))))
            briers.append(float(((p - t) ** 2).sum()))
    return {
        "n": len(correct),
        "accuracy": float(np.mean(correct)),
        "ece": float(ece_score(np.asarray(confs), np.asarray(correct))),
        "nll": float(np.mean(nlls)),
        "brier": float(np.mean(briers)),
    }


def eval_packs(model, tok, packs: Dict[str, Dict[str, List]], device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for domain, tasks in packs.items():
        out[domain] = {}
        domain_items = []
        for task, items in tasks.items():
            m = eval_items(model, tok, items, device)
            out[domain][task] = m
            domain_items.extend(items)
        out[domain]["_all"] = eval_items(model, tok, domain_items, device)
    return out


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
