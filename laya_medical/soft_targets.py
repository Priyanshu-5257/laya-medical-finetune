"""Soft teacher targets for Laya-style FT (hard-label datasets → soft probs)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from laya.common import build_model
from laya_medical import collate_train_batch


@torch.no_grad()
def apply_teacher_soft_targets(
    items: List[Dict[str, Any]],
    model_dir: str,
    *,
    mix: float = 0.5,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Replace item['target'] with mix of hard one-hot and base-Laya softmax.

    mix=0 → keep hard labels; mix=1 → pure teacher probs (Laya typed-decisions style
    when gold soft labels exist; here teacher = frozen base checkpoint).
    item['label'] stays the hard gold index for metrics.
    """
    mix = float(mix)
    if mix <= 0.0 or not items:
        return {"enabled": False, "mix": mix, "n": len(items)}

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(f"{model_dir}/rl_agent_config.json") as f:
        import json

        cfg = json.load(f)
    tok = AutoTokenizer.from_pretrained(f"{model_dir}/tokenizer")
    model = build_model(cfg, encoder_dir=f"{model_dir}/encoder")
    weights = load_file(f"{model_dir}/model.safetensors")
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()

    n_changed = 0
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
        probs = torch.softmax(logits.masked_fill(~mask, -1e4), -1)
        hard = batch["target"].to(device)
        soft = (1.0 - mix) * hard + mix * probs
        # renorm over valid options
        soft = soft * mask
        soft = soft / soft.sum(-1, keepdim=True).clamp_min(1e-8)
        for j, it in enumerate(chunk):
            k = len(it["markers"])
            it["target"] = soft[j, :k].cpu().tolist()
            it["meta"] = dict(it.get("meta") or {})
            it["meta"]["soft_target_mix"] = mix
            n_changed += 1

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"enabled": True, "mix": mix, "n": n_changed, "device": str(device)}
