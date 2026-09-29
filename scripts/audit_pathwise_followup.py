"""Measure option-order sensitivity for the pathwise follow-up arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from audit_eval_pipeline import permutation_audit
from evaluate import load_agent_model


ARMS = ("aligned", "baseline_20260923", "two_view_20260923",
        "baseline_20260924", "two_view_20260924")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model-dir", required=True)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--dev-packs", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    packs = torch.load(args.dev_packs, weights_only=False)["medical"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {}
    for arm in ARMS:
        model_dir = args.base_model_dir if arm == "aligned" else str(args.results_dir / arm)
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
