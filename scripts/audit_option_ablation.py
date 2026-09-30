"""Measure option order sensitivity for all models in one ablation shard."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from audit_eval_pipeline import permutation_audit
from evaluate import load_agent_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--dev-packs", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    packs = torch.load(args.dev_packs, weights_only=False)["medical"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    directories = [p for p in args.results_dir.iterdir() if p.is_dir() and (p / "eval_final.json").exists()]
    assert len(directories) == 6, f"Expected 6 completed arms; found {len(directories)}"
    result = {}
    for arm, model_dir in [("aligned", args.base_model_dir), *[(p.name, str(p)) for p in sorted(directories)]]:
        model, tok, _ = load_agent_model(model_dir, device)
        result[arm] = {}
        for task, items in packs.items():
            result[arm][task] = permutation_audit(model, tok, items, device)
            print(arm, task, result[arm][task], flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
