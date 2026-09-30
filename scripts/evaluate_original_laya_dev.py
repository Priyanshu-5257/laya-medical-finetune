"""Evaluate untouched Laya on the fixed MedMCQA/MedNLI objective dev packs."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from laya_medical import ensure_model_dir
from laya_medical.eval_metrics import eval_packs
from evaluate import load_agent_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medical_objective_compare.yaml")
    parser.add_argument("--work", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    model_dir = ensure_model_dir("convaiinnovations/laya")
    dev_path = args.work / "original_laya_dev_packs.pt"
    subprocess.run([sys.executable, "scripts/prepare_objective_dev.py", "--config", args.config,
                    "--model-dir", model_dir, "--out", str(dev_path)], check=True)
    packs = torch.load(dev_path, weights_only=False)
    model, tok, _ = load_agent_model(model_dir, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    scores = eval_packs(model, tok, packs, next(model.parameters()).device)
    result = {
        "model": "convaiinnovations/laya",
        "checkpoint": "untouched original Laya",
        "dev_pack_sha256": hashlib.sha256(dev_path.read_bytes()).hexdigest(),
        "scores": scores,
    }
    out = args.work / "original_laya_dev_scores.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
