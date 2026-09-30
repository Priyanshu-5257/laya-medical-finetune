"""Summarize all completed arms of one full medical comparison shard."""
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
    rows = {}
    for directory in sorted(args.results_dir.iterdir()):
        if not directory.is_dir() or not (directory / "eval_final.json").exists():
            continue
        arm = directory.name
        dev = json.loads((directory / "eval_final.json").read_text())
        test = json.loads((args.results_dir / f"test_{arm}.json").read_text())
        history = json.loads((directory / "train_history.json").read_text())
        rows[arm] = {
            "dev": compact(dev["finetuned"]),
            "test": compact(test["finetuned"]),
            "starting_dev": compact(dev["base"]),
            "starting_test": compact(test["base"]),
            "train_history": history,
        }
    expected = 3 if args.results_dir.name.endswith("_a") else 2
    assert len(rows) == expected, f"Expected {expected} completed arms, found {len(rows)}"
    out = {"environment": json.loads((args.results_dir / "environment.json").read_text()), "arms": rows}
    raw = args.results_dir / "raw_swap_dev.json"
    if raw.exists():
        out["raw_swap_dev"] = compact(json.loads(raw.read_text())["scores"])
        aligned = json.loads((args.results_dir / "medical_alignment_dev.json").read_text())
        out["medical_alignment_dev"] = compact(aligned["finetuned"])
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
