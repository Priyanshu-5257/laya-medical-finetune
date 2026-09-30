"""Distill frozen Laya soft distributions into a BioClinical-encoder student."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from laya.common import build_model
from laya_medical import collate_train_batch, load_yaml
from laya_medical.bioclinical_swap import load_bioclinical_student, save_agent_dir


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/distill_bioclinical.yaml")
    ap.add_argument("--teacher-dir", required=True)
    ap.add_argument("--bio-dir", required=True)
    ap.add_argument("--train-items", required=True)
    ap.add_argument("--output-dir", required=True)
    return ap.parse_args()


@torch.no_grad()
def teacher_targets(teacher, tok, items, device, batch_size=16, teacher_mix=0.5):
    teacher.eval()
    for it in items:
        it["gold_target"] = list(it["target"])
    eligible = [it for it in items if float(it.get("meta", {}).get("teacher_mix", teacher_mix)) > 0]
    for i in range(0, len(eligible), batch_size):
        chunk = eligible[i : i + batch_size]
        batch = collate_train_batch(chunk, tok.pad_token_id)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits, _ = teacher(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )
        logits = logits.float()
        mask = batch["marker_mask"].to(device)
        probs = torch.softmax(logits.masked_fill(~mask, -1e4), -1)
        for j, it in enumerate(chunk):
            k = len(it["markers"])
            mix = float(it.get("meta", {}).get("teacher_mix", teacher_mix))
            if not 0.0 <= mix <= 1.0:
                raise ValueError(f"teacher_mix must be in [0,1], got {mix}")
            gold = torch.tensor(it["gold_target"], device=device)
            it["target"] = ((1 - mix) * gold + mix * probs[j, :k]).cpu().tolist()


def main():
    args = parse_args()
    conf = load_yaml(args.config)
    tconf = conf["train"]

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    tok = AutoTokenizer.from_pretrained(os.path.join(args.teacher_dir, "tokenizer"))
    items = torch.load(args.train_items, weights_only=False)

    if rank == 0:
        print("Loading frozen Laya teacher and labeling distill items...", flush=True)
        with open(os.path.join(args.teacher_dir, "rl_agent_config.json")) as f:
            tcfg = json.load(f)
        teacher = build_model(tcfg, encoder_dir=os.path.join(args.teacher_dir, "encoder"))
        teacher.load_state_dict(load_file(os.path.join(args.teacher_dir, "model.safetensors")), strict=True)
        teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        teacher_targets(teacher, tok, items, device, teacher_mix=float(tconf.get("teacher_mix", 0.5)))
        del teacher
        torch.cuda.empty_cache()
        labeled = os.path.join(os.path.dirname(args.train_items), "distill_items_teacher.pt")
        torch.save(items, labeled)
        print(f"Wrote teacher soft targets -> {labeled} n={len(items)}", flush=True)
    dist.barrier()
    items = torch.load(os.path.join(os.path.dirname(args.train_items), "distill_items_teacher.pt"), weights_only=False)

    if rank == 0:
        print("Loading BioClinical student (Laya head, frozen medical encoder)...", flush=True)
    model, cfg, swap_info = load_bioclinical_student(args.teacher_dir, args.bio_dir)
    if rank == 0:
        print("swap:", json.dumps(swap_info), flush=True)
    freeze = bool(conf.get("freeze_encoder", True))
    if freeze:
        for n, p in model.named_parameters():
            if n.startswith("encoder."):
                p.requires_grad = False
    model.to(device)
    model.train()
    if freeze:
        model.encoder.eval()
    else:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    encoder_params = [p for n, p in ddp.named_parameters() if n.startswith("module.encoder.") and p.requires_grad]
    head_params = [p for n, p in ddp.named_parameters() if not n.startswith("module.encoder.") and p.requires_grad]
    train_params = encoder_params + head_params
    n_train_params = sum(p.numel() for p in train_params)
    if rank == 0:
        enc_train = sum(p.numel() for n, p in ddp.named_parameters() if n.startswith("module.encoder.") and p.requires_grad)
        print(f"trainable head params={n_train_params} encoder_trainable={enc_train} freeze_encoder={freeze}", flush=True)

    my_items = items[rank::world]
    epochs = int(tconf["epochs"])
    micro = int(tconf["micro_batch"])
    accum = int(tconf["grad_accum"])
    lr = float(tconf["lr_head"])
    opt = torch.optim.AdamW(
        [{"params": encoder_params, "lr": float(tconf.get("lr_encoder", 5e-6))},
         {"params": head_params, "lr": lr}],
        weight_decay=float(tconf.get("weight_decay", 0.01)),
    )
    updates = max(1, (len(my_items) // max(1, micro * accum)) * epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=updates, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    log_every = int(tconf.get("log_every", 25))

    t0 = time.time()
    history = []
    step = 0
    for epoch in range(epochs):
        random.seed(42 + epoch + rank)
        random.shuffle(my_items)
        if freeze:
            ddp.module.encoder.eval()
        loss_sum, n_batches = 0.0, 0
        opt.zero_grad(set_to_none=True)
        accum_i = 0
        for b in range(0, len(my_items), micro):
            chunk = my_items[b : b + micro]
            if not chunk:
                continue
            batch = collate_train_batch(chunk, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, act = ddp(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                    detach_encoder=freeze,
                )
            mask = batch["marker_mask"].to(device)
            target = batch["target"].to(device)
            logp = torch.log_softmax(logits.float().masked_fill(~mask, -1e4), -1)
            loss = -(target * logp).sum(-1).mean() / accum + 0.0 * act.sum()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite distill loss at epoch {epoch+1} step {n_batches+1}")
            scaler.scale(loss).backward()
            accum_i += 1
            if accum_i % accum == 0 or (b + micro) >= len(my_items):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(train_params, 1.0)
                scale_before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            loss_sum += float(loss.item() * accum)
            n_batches += 1
            if rank == 0 and n_batches % log_every == 0:
                print(
                    f"  distill epoch {epoch+1}/{epochs} step {n_batches} "
                    f"loss={loss.item()*accum:.4f} lr={sched.get_last_lr()[0]:.2e}",
                    flush=True,
                )
        avg = loss_sum / max(1, n_batches)
        if rank == 0:
            print(f"=== distill epoch {epoch+1}/{epochs} avg_ce={avg:.4f} t={time.time()-t0:.1f}s ===", flush=True)
            history.append({"epoch": epoch + 1, "avg_ce": avg})
        dist.barrier()

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        save_agent_dir(
            model,
            cfg,
            tok,
            args.output_dir,
            extra={
                "distill": {
                    "freeze_encoder": freeze,
                    "teacher": conf["teacher_id"],
                    "encoder": conf["encoder_id"],
                    "epochs": epochs,
                    "seconds": time.time() - t0,
                    "history": history,
                    "swap": swap_info,
                }
            },
        )
        print(f"Saved distilled student to {args.output_dir}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    main()
