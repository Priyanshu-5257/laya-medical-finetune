"""DDP RLCD fine-tune for Laya. Launch with torchrun --nproc_per_node=N."""
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
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from laya.common import build_model, proper_reward
from laya_medical import collate_train_batch, fit_one_temp, load_yaml


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--train-items", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--ce-weight", type=float, default=None)
    ap.add_argument("--group-size", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    conf = load_yaml(args.config)
    tconf = conf["train"]

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    with open(os.path.join(args.model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["gradient_checkpointing"] = True
    cfg["max_tokens_per_batch"] = int(tconf.get("max_tokens_per_batch", 4096))
    cfg["max_len"] = int(conf.get("max_len", cfg.get("max_len", 512)))
    cfg["head_max_len"] = int(conf.get("head_max_len", cfg.get("head_max_len", 192)))

    tok = AutoTokenizer.from_pretrained(os.path.join(args.model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(args.model_dir, "encoder"))
    weights = load_file(os.path.join(args.model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    model.encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.head_checkpointing = True
    model.to(device)
    model.train()
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    all_items = torch.load(args.train_items, weights_only=False)
    CALIB_MAX = int(tconf.get("calib_max", 400))
    order = list(range(len(all_items)))
    random.Random(20260922).shuffle(order)
    n_calib = min(CALIB_MAX, max(1, len(all_items) // 10))
    calib_items = [all_items[i] for i in sorted(order[:n_calib])]
    train_items = [all_items[i] for i in sorted(order[n_calib:])]
    my_items = train_items[rank::world_size]

    EPOCHS = int(args.epochs if args.epochs is not None else tconf["epochs"])
    MICRO_BATCH = int(tconf["micro_batch"])
    GRAD_ACCUM = int(tconf["grad_accum"])
    GROUP_SIZE = int(args.group_size if args.group_size is not None else tconf["group_size"])
    LR_ENCODER = float(tconf["lr_encoder"])
    LR_HEAD = float(tconf["lr_head"])
    SIGMA_START = float(tconf["sigma_start"])
    SIGMA_END = float(tconf["sigma_end"])
    CE_WEIGHT = float(args.ce_weight if args.ce_weight is not None else tconf.get("ce_weight", 0.0))
    W_SPH = float(tconf.get("sph_weight", 0.5))
    W_RPS = float(tconf.get("rps_weight", 1.0))

    enc_params = [p for n, p in ddp_model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in ddp_model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": LR_ENCODER}, {"params": head_params, "lr": LR_HEAD}],
        weight_decay=0.01,
    )
    total_updates = max(1, (len(my_items) // max(1, MICRO_BATCH * GRAD_ACCUM)) * EPOCHS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    if rank == 0:
        print(
            f"DDP world={world_size} | train={len(train_items)} "
            f"(calib holdout={len(calib_items)}) | per_rank={len(my_items)} | "
            f"epochs={EPOCHS} | G={GROUP_SIZE} | ce_weight={CE_WEIGHT} | "
            f"max_len={cfg['max_len']}",
            flush=True,
        )

    t0 = time.time()
    history = []
    for epoch in range(EPOCHS):
        random.seed(42 + epoch + rank)
        random.shuffle(my_items)
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        accum_step = 0
        progress = epoch / max(1, EPOCHS - 1)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * progress
        reward_sum = 0.0

        for b_idx in range(0, len(my_items), MICRO_BATCH):
            chunk = my_items[b_idx : b_idx + MICRO_BATCH]
            if not chunk:
                continue
            batch = collate_train_batch(chunk, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, act = ddp_model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float().clamp(min=1.0)
            target = batch["target"].to(device)

            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

            with torch.no_grad():
                r = proper_reward(
                    q,
                    target.unsqueeze(0),
                    batch["qtype"].to(device),
                    mask,
                    w_sph=W_SPH,
                    w_rps=W_RPS,
                )
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + CE_WEIGHT * loss_ce) / GRAD_ACCUM + 0.0 * act.sum()

            scaler.scale(loss).backward()
            accum_step += 1
            if accum_step % GRAD_ACCUM == 0 or (b_idx + MICRO_BATCH) >= len(my_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(loss.item() * GRAD_ACCUM)
            reward_sum += float(r.mean().item())
            n_batches += 1
            if rank == 0 and (n_batches % 25) == 0:
                print(
                    f"  epoch {epoch+1}/{EPOCHS} step {n_batches} "
                    f"loss={loss.item()*GRAD_ACCUM:.4f} reward={r.mean().item():.3f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )

        avg_loss = epoch_loss / max(1, n_batches)
        avg_r = reward_sum / max(1, n_batches)
        if rank == 0:
            print(
                f"=== epoch {epoch+1}/{EPOCHS} done in {time.time()-t0:.1f}s "
                f"| avg_loss={avg_loss:.4f} | avg_reward={avg_r:.3f} ===",
                flush=True,
            )
            history.append({"epoch": epoch + 1, "avg_loss": avg_loss, "avg_reward": avg_r})
            ckpt_dir = os.path.join(args.output_dir, "checkpoint_latest")
            os.makedirs(ckpt_dir, exist_ok=True)
            sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
            save_file(sd, os.path.join(ckpt_dir, "model.safetensors"))
            model.encoder.config.save_pretrained(os.path.join(ckpt_dir, "encoder"))
            tok.save_pretrained(os.path.join(ckpt_dir, "tokenizer"))
            with open(os.path.join(ckpt_dir, "checkpoint_meta.json"), "w") as f:
                json.dump({"epoch": epoch + 1, "total_epochs": EPOCHS, "avg_loss": avg_loss}, f, indent=2)

        dist.barrier()

    if rank == 0:
        print("Fitting held-out calibration temperatures...", flush=True)
        del optimizer, scaler, scheduler
        torch.cuda.empty_cache()
        model.eval()
        calib_preds = []
        with torch.no_grad():
            for c_idx in range(0, len(calib_items), 16):
                c_chunk = calib_items[c_idx : c_idx + 16]
                cb = collate_train_batch(c_chunk, tok.pad_token_id)
                with torch.autocast("cuda", dtype=torch.float16):
                    l_sub, _ = model(
                        cb["input_ids"].to(device),
                        cb["attention_mask"].to(device),
                        cb["marker_pos"].to(device),
                        cb["marker_mask"].to(device),
                        cb["qtype"].to(device),
                    )
                l_np = l_sub.float().cpu().numpy()
                for i, it in enumerate(c_chunk):
                    k = len(it["markers"])
                    calib_preds.append((it["qtype"], l_np[i, :k], it["target"]))
        fitted_temps = [1.2, 1.2, 1.2]
        try:
            for qt in range(3):
                sel = [(z, t) for q_type, z, t in calib_preds if q_type == qt]
                if sel:
                    fitted_temps[qt] = fit_one_temp(sel)
            print("Fitted temperatures (choice, score, noul):", [round(t, 3) for t in fitted_temps])
        except Exception as e:
            print("Temperature fit fallback:", e)

        os.makedirs(args.output_dir, exist_ok=True)
        sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
        save_file(sd, os.path.join(args.output_dir, "model.safetensors"))
        model.encoder.config.save_pretrained(os.path.join(args.output_dir, "encoder"))
        tok.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
        cfg["fine_tuned"] = True
        cfg["model_name"] = "laya-medical-smoke"
        cfg["temperature"] = fitted_temps
        cfg.pop("temperature_by_options", None)
        with open(os.path.join(args.output_dir, "rl_agent_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        with open(os.path.join(args.output_dir, "train_history.json"), "w") as f:
            json.dump(
                {
                    "history": history,
                    "seconds": time.time() - t0,
                    "n_train": len(train_items),
                    "n_calib": len(calib_items),
                    "world_size": world_size,
                    "ce_weight": CE_WEIGHT,
                    "group_size": GROUP_SIZE,
                    "epochs": EPOCHS,
                },
                f,
                indent=2,
            )
        print(f"Saved checkpoint to {args.output_dir}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    # Allow `python scripts/train_ddp.py` and `torchrun scripts/train_ddp.py`
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    main()
