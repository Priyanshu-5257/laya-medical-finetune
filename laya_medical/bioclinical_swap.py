"""Build a Laya decision model whose encoder is BioClinical ModernBERT-large."""
from __future__ import annotations

import json
import os
from typing import Any, Dict

from safetensors.torch import load_file

from laya.common import build_model


def _map_encoder_key(key: str) -> str:
    if key.startswith("encoder."):
        return key
    if key.startswith("model."):
        return "encoder." + key[len("model.") :]
    return "encoder." + key


def load_bioclinical_student(laya_dir: str, bio_dir: str):
    """Laya decision head + BioClinical ModernBERT-large encoder weights.

    Tokenizer and head come from Laya. Encoder tensors come from BioClinical.
    Architectures match (ModernBERT-large, hidden 1024, vocab 50368).
    """
    with open(os.path.join(laya_dir, "rl_agent_config.json")) as f:
        cfg: Dict[str, Any] = json.load(f)
    cfg["encoder"] = "thomas-sounack/BioClinical-ModernBERT-large"
    cfg["encoder_swap"] = "bioclinical_modernbert_large"
    model = build_model(cfg, encoder_dir=bio_dir)
    laya_sd = load_file(os.path.join(laya_dir, "model.safetensors"))
    head_sd = {k: v for k, v in laya_sd.items() if not k.startswith("encoder.")}
    missing_head, unexpected_head = model.load_state_dict(head_sd, strict=False)
    bio_sd = load_file(os.path.join(bio_dir, "model.safetensors"))
    model_sd = model.state_dict()
    mapped = {}
    unmatched = []
    for k, v in bio_sd.items():
        dest = _map_encoder_key(k)
        if dest in model_sd and tuple(model_sd[dest].shape) == tuple(v.shape):
            mapped[dest] = v
        else:
            unmatched.append(k)
    missing_enc, unexpected_enc = model.load_state_dict(mapped, strict=False)
    enc_names = [k for k in model_sd if k.startswith("encoder.")]
    covered = sum(1 for k in enc_names if k in mapped)
    info = {
        "encoder_params_covered": covered,
        "encoder_params_total": len(enc_names),
        "bio_unmatched": len(unmatched),
        "head_tensors": len(head_sd),
        "missing_head": [k for k in missing_head if not k.startswith("encoder.")][:12],
        "unexpected_head": list(unexpected_head)[:8],
        "unexpected_enc": list(unexpected_enc)[:8],
    }
    if covered < max(1, int(0.95 * len(enc_names))):
        raise RuntimeError(f"BioClinical encoder load incomplete: {info}")
    return model, cfg, info


def save_agent_dir(model, cfg: Dict[str, Any], tok, out_dir: str, extra: Dict[str, Any] | None = None) -> None:
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    sd = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}
    # keep encoder master weights in fp16 to match other checkpoints; head stays fp32-safe via half
    sd = {k: (v.half() if v.dtype.is_floating_point else v) for k, v in sd.items()}
    save_file(sd, os.path.join(out_dir, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(out_dir, "encoder"))
    tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    out_cfg = dict(cfg)
    if extra:
        out_cfg.update(extra)
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(out_cfg, f, indent=2)
