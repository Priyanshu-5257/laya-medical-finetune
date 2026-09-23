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
from laya_medical.eval_metrics import eval_packs, flatten_eval_for_log
from laya_medical.wandb_util import maybe_init_wandb, wandb_finish, wandb_log


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--train-items", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--eval-packs", default=None, help="Optional dual-eval packs for per-epoch eval")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--ce-weight", type=float, default=None)
    ap.add_argument("--group-size", type=int, default=None)
    return ap.parse_args()


def _run_dual_eval(model, tok, packs, device, base_metrics=None):
    model.eval()
    with torch.no_grad():
        ft_metrics = eval_packs(model, tok, packs, device)
    model.train()
    out = {"finetuned": ft_metrics}
    if base_metrics is not None:
        deltas = {}
        for domain in packs:
            deltas[domain] = {}
            for task in list(packs[domain]) + ["_all"]:
                b = base_metrics[domain][task]["accuracy"]
                f = ft_metrics[domain][task]["accuracy"]
                deltas[domain][task] = {
                    "accuracy_base": b,
                    "accuracy_ft": f,
                    "accuracy_delta": (f - b) if (b == b and f == f) else None,
                }
        out["base"] = base_metrics
        out["deltas"] = deltas
    return out


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
    LOG_EVERY = int(tconf.get("log_every", 25))
    EVAL_EVERY_EPOCH = bool(tconf.get("eval_every_epoch", True))

    eval_packs_path = args.eval_packs
    packs = None
    if eval_packs_path and os.path.isfile(eval_packs_path):
        packs = torch.load(eval_packs_path, weights_only=False)

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

    wb = None
    base_metrics = None
    global_step = 0
    if rank == 0:
        print(
            f"DDP world={world_size} | train={len(train_items)} "
            f"(calib holdout={len(calib_items)}) | per_rank={len(my_items)} | "
            f"epochs={EPOCHS} | G={GROUP_SIZE} | ce_weight={CE_WEIGHT} | "
            f"max_len={cfg['max_len']} | eval_every_epoch={EVAL_EVERY_EPOCH and packs is not None}",
            flush=True,
        )
        wb = maybe_init_wandb(
            conf,
            {
                "model_id": conf.get("model_id"),
                "epochs": EPOCHS,
                "medmcqa_n": tconf.get("medmcqa_n"),
                "mednli_n": tconf.get("mednli_n"),
                "lr_encoder": LR_ENCODER,
                "lr_head": LR_HEAD,
                "micro_batch": MICRO_BATCH,
                "grad_accum": GRAD_ACCUM,
                "group_size": GROUP_SIZE,
                "ce_weight": CE_WEIGHT,
                "max_len": cfg["max_len"],
                "world_size": world_size,
                "n_train": len(train_items),
            },
        )
        if packs is not None and EVAL_EVERY_EPOCH:
            print("Evaluating BASE checkpoint (once)...", flush=True)
            base_metrics = eval_packs(model, tok, packs, device)
            wandb_log(
                wb,
                {
                    **flatten_eval_for_log(base_metrics, "base"),
                    "epoch": 0,
                },
                step=0,
            )
            print(
                f"  base generic={base_metrics['generic']['_all']['accuracy']:.4f} "
                f"medical={base_metrics['medical']['_all']['accuracy']:.4f}",
                flush=True,
            )

    dist.barrier()

    t0 = time.time()
    history = []
    epoch_evals = []
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
                global_step += 1

            epoch_loss += float(loss.item() * GRAD_ACCUM)
            reward_sum += float(r.mean().item())
            n_batches += 1
            if rank == 0 and (n_batches % LOG_EVERY) == 0:
                step_loss = loss.item() * GRAD_ACCUM
                step_r = r.mean().item()
                lr0 = scheduler.get_last_lr()[0]
                print(
                    f"  epoch {epoch+1}/{EPOCHS} step {n_batches} "
                    f"loss={step_loss:.4f} reward={step_r:.3f} "
                    f"lr={lr0:.2e}",
                    flush=True,
                )
                wandb_log(
                    wb,
                    {
                        "train/loss": step_loss,
                        "train/reward": step_r,
                        "train/lr_encoder": lr0,
                        "train/sigma": sigma,
                        "epoch": epoch + 1,
                    },
                    step=global_step,
                )

        avg_loss = epoch_loss / max(1, n_batches)
        avg_r = reward_sum / max(1, n_batches)
        if rank == 0:
            print(
                f"=== epoch {epoch+1}/{EPOCHS} done in {time.time()-t0:.1f}s "
                f"| avg_loss={avg_loss:.4f} | avg_reward={avg_r:.3f} ===",
                flush=True,
            )
            hist_row = {"epoch": epoch + 1, "avg_loss": avg_loss, "avg_reward": avg_r}
            ckpt_dir = os.path.join(args.output_dir, "checkpoint_latest")
            os.makedirs(ckpt_dir, exist_ok=True)
            sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
            save_file(sd, os.path.join(ckpt_dir, "model.safetensors"))
            model.encoder.config.save_pretrained(os.path.join(ckpt_dir, "encoder"))
            tok.save_pretrained(os.path.join(ckpt_dir, "tokenizer"))
            with open(os.path.join(ckpt_dir, "checkpoint_meta.json"), "w") as f:
                json.dump({"epoch": epoch + 1, "total_epochs": EPOCHS, "avg_loss": avg_loss}, f, indent=2)

            if packs is not None and EVAL_EVERY_EPOCH:
                print(f"=== dual eval after epoch {epoch+1} ===", flush=True)
                report = _run_dual_eval(model, tok, packs, device, base_metrics=base_metrics)
                ft = report["finetuned"]
                g = ft["generic"]["_all"]["accuracy"]
                m = ft["medical"]["_all"]["accuracy"]
                gd = report.get("deltas", {}).get("generic", {}).get("_all", {}).get("accuracy_delta")
                md = report.get("deltas", {}).get("medical", {}).get("_all", {}).get("accuracy_delta")
                print(
                    f"  epoch {epoch+1} eval generic={g:.4f} (Δ={gd}) "
                    f"medical={m:.4f} (Δ={md})",
                    flush=True,
                )
                hist_row["generic_acc"] = g
                hist_row["medical_acc"] = m
                hist_row["generic_delta"] = gd
                hist_row["medical_delta"] = md
                epoch_evals.append({"epoch": epoch + 1, "report": report})
                eval_path = os.path.join(args.output_dir, f"eval_epoch_{epoch+1}.json")
                os.makedirs(args.output_dir, exist_ok=True)
                with open(eval_path, "w") as f:
                    json.dump(report, f, indent=2)
                log_payload = {
                    **flatten_eval_for_log(ft, "eval"),
                    "eval/generic_acc": g,
                    "eval/medical_acc": m,
                    "train/epoch_loss": avg_loss,
                    "train/epoch_reward": avg_r,
                    "epoch": epoch + 1,
                }
                if gd is not None:
                    log_payload["eval/generic_delta"] = gd
                if md is not None:
                    log_payload["eval/medical_delta"] = md
                wandb_log(wb, log_payload, step=global_step)
            else:
                wandb_log(
                    wb,
                    {
                        "train/epoch_loss": avg_loss,
                        "train/epoch_reward": avg_r,
                        "epoch": epoch + 1,
                    },
                    step=global_step,
                )

            history.append(hist_row)

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
        cfg["model_name"] = "laya-medical-full"
        cfg["temperature"] = fitted_temps
        cfg.pop("temperature_by_options", None)
        with open(os.path.join(args.output_dir, "rl_agent_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        with open(os.path.join(args.output_dir, "train_history.json"), "w") as f:
            json.dump(
                {
                    "history": history,
                    "epoch_evals_present": bool(epoch_evals),
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
        # final calibrated dual eval
        if packs is not None:
            model.temperature.copy_(torch.tensor(fitted_temps, dtype=torch.float32))
            print("=== final dual eval (calibrated temps) ===", flush=True)
            final_report = _run_dual_eval(model, tok, packs, device, base_metrics=base_metrics)
            with open(os.path.join(args.output_dir, "eval_final.json"), "w") as f:
                json.dump(final_report, f, indent=2)
            ft = final_report["finetuned"]
            wandb_log(
                wb,
                {
                    **flatten_eval_for_log(ft, "final"),
                    "final/generic_acc": ft["generic"]["_all"]["accuracy"],
                    "final/medical_acc": ft["medical"]["_all"]["accuracy"],
                    "final/generic_delta": final_report.get("deltas", {})
                    .get("generic", {})
                    .get("_all", {})
                    .get("accuracy_delta"),
                    "final/medical_delta": final_report.get("deltas", {})
                    .get("medical", {})
                    .get("_all", {})
                    .get("accuracy_delta"),
                    "temps/choice": fitted_temps[0],
                    "temps/score": fitted_temps[1],
                    "temps/noul": fitted_temps[2],
                },
                step=global_step + 1,
            )
            print(json.dumps({
                "generic": ft["generic"]["_all"]["accuracy"],
                "medical": ft["medical"]["_all"]["accuracy"],
                "deltas": final_report.get("deltas"),
            }, indent=2), flush=True)
        print(f"Saved checkpoint to {args.output_dir}", flush=True)
        wandb_finish(wb)

    dist.destroy_process_group()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    main()
