"""Report the fixed epoch-2 pathwise replication and two-view intervention."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ARMS = ("baseline_20260923", "two_view_20260923",
        "baseline_20260924", "two_view_20260924")


def compact(packs):
    return {domain: {task: {key: row[key] for key in ("n", "accuracy", "nll", "brier", "ece")}
                     for task, row in tasks.items()}
            for domain, tasks in packs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    audit = json.loads((args.results_dir / "permutation_audit.json").read_text())
    rows = {}
    for arm in ARMS:
        directory = args.results_dir / arm
        dev = json.loads((directory / "eval_final.json").read_text())
        test = json.loads((args.results_dir / f"test_{arm}.json").read_text())
        history = json.loads((directory / "train_history.json").read_text())
        assert history["stop_after_epoch"] == 2 and history["epochs"] == 3
        rows[arm] = {
            "seed": int(arm.rsplit("_", 1)[1]),
            "two_view": arm.startswith("two_view"),
            "train_history": history["history"],
            "dev": compact(dev["finetuned"]),
            "test": compact(test["finetuned"]),
            "order_audit": audit[arm],
        }
    result = {
        "design": "Same aligned checkpoint and fixed train/dev/external packs; pathwise RLCD; three-epoch sigma and LR schedule stopped after epoch 2; seeds vary dropout, noise, training order, and permutations; two-view uses independent random teacher/student option orders and consistency KL weight 0.5.",
        "base_dev": compact(dev["base"]),
        "base_test": compact(test["base"]),
        "base_order_audit": audit["aligned"],
        "arms": rows,
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
