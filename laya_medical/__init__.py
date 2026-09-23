"""Shared helpers for Laya medical fine-tuning."""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import yaml
from laya.common import QTYPES, build_sequence, render_options
from laya.agent import _fix_tokenizer_config


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_model_dir(model_id: str, cache_dir: Optional[str] = None) -> str:
    from huggingface_hub import snapshot_download

    kw = {}
    if cache_dir:
        kw["cache_dir"] = cache_dir
    model_dir = snapshot_download(model_id, **kw)
    _fix_tokenizer_config(model_dir)
    return model_dir


def load_cfg(model_dir: str) -> Dict[str, Any]:
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        return json.load(f)


def one_hot(n: int, idx: int) -> List[float]:
    t = [0.0] * n
    t[int(idx)] = 1.0
    return t


def maybe_shuffle_choice(
    keys: List[str],
    label_key: str,
    *,
    rng: random.Random,
    enabled: bool,
) -> Tuple[List[str], str]:
    if not enabled or len(keys) <= 1:
        return keys, label_key
    keys = list(keys)
    rng.shuffle(keys)
    return keys, label_key


def tokenize_item(
    tok,
    state: Any,
    qtype: str,
    instructions: str,
    criteria: Any,
    target: Sequence[float],
    *,
    max_len: int,
    head_max_len: int,
) -> Optional[Dict[str, Any]]:
    q = {"t": qtype, "ins": instructions, "crit": criteria}
    opts = render_options(q)
    k = len(opts)
    if k < 1 or len(target) != k:
        return None
    seq, markers = build_sequence(tok, state, q, max_len, head_max_len)
    if len(markers) != k:
        return None
    s = float(sum(target))
    target = [float(v) / s for v in target] if s > 0 else [1.0 / k] * k
    return {
        "ids": seq,
        "markers": markers,
        "qtype": QTYPES[qtype],
        "target": target,
        "label": int(max(range(k), key=lambda i: target[i])),
        "meta": {"qtype_name": qtype, "n_options": k},
    }


def collate_train_batch(items: List[Dict[str, Any]], pad_id: int) -> Dict[str, torch.Tensor]:
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


def fit_one_temp(sel: List[Tuple[Any, Any]]) -> float:
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())
