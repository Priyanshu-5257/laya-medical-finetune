"""PEFT-style middle-layer training: LoRA if possible, else freeze early/late."""
from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

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


def apply_middle_layer_freeze(
    encoder: nn.Module,
    *,
    start_frac: float = 1.0 / 3.0,
    end_frac: float = 2.0 / 3.0,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Freeze embeddings + early/late layers; full-FT only the middle block."""
    layers, pattern = get_encoder_layers(encoder)
    n = len(layers)
    mid_start, mid_end = middle_layer_range(n, start_frac, end_frac)
    middle = list(range(mid_start, mid_end))

    for p in encoder.parameters():
        p.requires_grad = False
    for i in middle:
        for p in layers[i].parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in encoder.parameters())
    meta = {
        "mode": "middle_layer_full_ft",
        "n_layers": n,
        "middle_start": mid_start,
        "middle_end": mid_end,
        "middle_indices": middle,
        "layers_pattern": pattern,
        "trainable_encoder_params": int(trainable),
        "total_encoder_params": int(total),
        "trainable_encoder_pct": float(100.0 * trainable / max(1, total)),
    }
    return encoder, meta


def apply_middle_lora(
    encoder: nn.Module,
    *,
    start_frac: float = 1.0 / 3.0,
    end_frac: float = 2.0 / 3.0,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Sequence[str] | None = None,
    allow_freeze_fallback: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Wrap encoder with PEFT LoRA on the middle third of layers only.

    Falls back to freezing early/late + full-FT middle layers if peft/torchao
    is incompatible (common on some Kaggle images).
    """
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as e:
        if allow_freeze_fallback:
            meta_fb = {"lora_error": f"peft import failed: {e}"}
            enc, meta = apply_middle_layer_freeze(encoder, start_frac=start_frac, end_frac=end_frac)
            meta.update(meta_fb)
            return enc, meta
        raise

    layers, pattern = get_encoder_layers(encoder)
    n = len(layers)
    mid_start, mid_end = middle_layer_range(n, start_frac, end_frac)
    middle = list(range(mid_start, mid_end))

    for p in encoder.parameters():
        p.requires_grad = False

    targets = list(target_modules) if target_modules else ["Wqkv", "Wo", "Wi"]
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
    try:
        peft_encoder = get_peft_model(encoder, lora_cfg)
    except Exception as e:
        if allow_freeze_fallback:
            # Restore a clean encoder path: reload requires_grad via freeze helper.
            # get_peft_model may have partially mutated modules; freeze helper re-sets flags.
            enc, meta = apply_middle_layer_freeze(encoder, start_frac=start_frac, end_frac=end_frac)
            meta["lora_error"] = str(e)
            meta["fallback"] = "middle_layer_freeze"
            return enc, meta
        raise

    trainable = sum(p.numel() for p in peft_encoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_encoder.parameters())
    meta = {
        "mode": "middle_lora",
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


def apply_middle_tuning(encoder: nn.Module, peft_conf: Dict[str, Any] | None = None):
    """Dispatch: mode=lora | full_ft (middle-layer full fine-tune, no LoRA)."""
    peft_conf = peft_conf or {}
    start_frac = float(peft_conf.get("start_frac", 1 / 3))
    end_frac = float(peft_conf.get("end_frac", 2 / 3))
    mode = str(peft_conf.get("mode", "full_ft")).lower()
    if mode in ("lora", "middle_lora", "peft_lora"):
        return apply_middle_lora(
            encoder,
            start_frac=start_frac,
            end_frac=end_frac,
            r=int(peft_conf.get("r", 16)),
            lora_alpha=int(peft_conf.get("lora_alpha", 32)),
            lora_dropout=float(peft_conf.get("lora_dropout", 0.05)),
            target_modules=peft_conf.get("target_modules"),
            allow_freeze_fallback=bool(peft_conf.get("allow_freeze_fallback", True)),
        )
    return apply_middle_layer_freeze(encoder, start_frac=start_frac, end_frac=end_frac)


def merge_encoder_if_peft(encoder: nn.Module) -> nn.Module:
    if hasattr(encoder, "merge_and_unload"):
        return encoder.merge_and_unload()
    return encoder


def count_trainable(model: nn.Module) -> Dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": int(trainable), "total": int(total)}
