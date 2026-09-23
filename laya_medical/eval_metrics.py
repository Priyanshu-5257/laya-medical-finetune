"""Shared dual-eval metrics for train-time and offline evaluation."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from laya.common import ece_score


@torch.no_grad()
def eval_items(model, tok, items: List[Dict[str, Any]], device: torch.device, batch_size: int = 8) -> Dict[str, float]:
    # Local import avoids circular import with laya_medical.__init__.
    from laya_medical import collate_train_batch

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


def flatten_eval_for_log(metrics: Dict[str, Any], prefix: str = "eval") -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for domain, tasks in metrics.items():
        for task, m in tasks.items():
            for k, v in m.items():
                if k == "n":
                    continue
                flat[f"{prefix}/{domain}/{task}/{k}"] = float(v)
    return flat
