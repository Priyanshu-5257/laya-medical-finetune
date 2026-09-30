"""Collect comparable dev, external, generic, and option-order scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def compact(packs):
    return {domain: {task: {key: row[key] for key in ("n", "accuracy", "nll", "brier", "ece")}
                     for task, row in tasks.items()}
            for domain, tasks in packs.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads((args.results_dir / "permutation_audit.json").read_text())
    environment = json.loads((args.results_dir / "environment.json").read_text())
    arms = {}
    for directory in sorted(args.results_dir.iterdir()):
        if not directory.is_dir() or not (directory / "eval_final.json").exists():
            continue
        arm = directory.name
        dev = json.loads((directory / "eval_final.json").read_text())
        test = json.loads((args.results_dir / f"test_{arm}.json").read_text())
        history = json.loads((directory / "train_history.json").read_text())
        assert history["stop_after_epoch"] == 2 and history["epochs"] == 3
        arms[arm] = {
            "seed": int(arm.rsplit("_", 1)[1]),
            "intervention": arm.rsplit("_", 1)[0],
            "train_history": history["history"],
            "dev": compact(dev["finetuned"]),
            "test": compact(test["finetuned"]),
            "order_audit": audit[arm],
        }
    assert len(arms) == 6, f"Expected 6 arms; found {len(arms)}"
    result = {
        "design": "Paired three-arm pathwise RLCD ablation; fixed checkpoint and data; 2 epochs of the 3-epoch schedule; baseline, random option permutation only, and independent teacher/student permutation with detached KL weight 0.5.",
        "environment": environment,
        "base_dev": compact(dev["base"]),
        "base_test": compact(test["base"]),
        "base_order_audit": audit["aligned"],
        "arms": arms,
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
