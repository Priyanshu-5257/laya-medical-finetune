"""PEFT helpers: LoRA on middle encoder layers only; freeze the rest."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch.nn as nn


def middle_layer_range(n_layers: int, start_frac: float = 1.0 / 3.0, end_frac: float = 2.0 / 3.0) -> Tuple[int, int]:
    start = int(n_layers * start_frac)
    end = int(n_layers * end_frac)
    if end <= start:
        end = min(n_layers, start + max(1, n_layers // 4))
    return start, end


def get_encoder_layers(encoder: nn.Module) -> Tuple[nn.ModuleList, str]:
    if hasattr(encoder, "layers") and isinstance(encoder.layers, nn.ModuleList):
        return encoder.layers, "layers"
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        return encoder.encoder.layer, "encoder.layer"
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layers"):
        return encoder.encoder.layers, "encoder.layers"
    raise AttributeError(f"Cannot locate transformer layers on {type(encoder)}")


def apply_middle_lora(
    encoder: nn.Module,
    *,
    start_frac: float = 1.0 / 3.0,
    end_frac: float = 2.0 / 3.0,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Sequence[str] | None = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Wrap encoder with PEFT LoRA on the middle third of layers only.

    Early/late layer base weights stay frozen (no adapters). Embeddings stay frozen.
    """
    from peft import LoraConfig, get_peft_model

    layers, pattern = get_encoder_layers(encoder)
    n = len(layers)
    mid_start, mid_end = middle_layer_range(n, start_frac, end_frac)
    middle = list(range(mid_start, mid_end))

    # Freeze entire encoder first; LoRA injects trainable adapters on selected layers.
    for p in encoder.parameters():
        p.requires_grad = False

    targets = list(target_modules) if target_modules else ["Wqkv", "Wo", "Wi"]
    # layers_pattern leaf name expected by peft (e.g. "layers")
    leaf = pattern.split(".")[-1]
    lora_cfg = LoraConfig(
        r=int(r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        bias="none",
        target_modules=targets,
        layers_to_transform=middle,
        layers_pattern=leaf,
    )
    peft_encoder = get_peft_model(encoder, lora_cfg)

    trainable = sum(p.numel() for p in peft_encoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_encoder.parameters())
    meta = {
        "n_layers": n,
        "middle_start": mid_start,
        "middle_end": mid_end,
        "middle_indices": middle,
        "layers_pattern": pattern,
        "target_modules": targets,
        "lora_r": int(r),
        "lora_alpha": int(lora_alpha),
        "trainable_encoder_params": int(trainable),
        "total_encoder_params": int(total),
        "trainable_encoder_pct": float(100.0 * trainable / max(1, total)),
    }
    return peft_encoder, meta


def merge_encoder_if_peft(encoder: nn.Module) -> nn.Module:
    if hasattr(encoder, "merge_and_unload"):
        return encoder.merge_and_unload()
    return encoder


def count_trainable(model: nn.Module) -> Dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": int(trainable), "total": int(total)}
