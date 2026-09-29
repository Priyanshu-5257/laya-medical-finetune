"""Summarize fixed-budget medical objective arms using validation for selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text())


def brief(metrics):
    return {key: metrics[key] for key in ("n", "accuracy", "nll", "brier", "ece", "majority_accuracy")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = {}
    for arm in ("pure_rlcd", "rlcd_ce", "ce_only"):
        dev = load(args.results_dir / arm / "eval_final.json")
        test = load(args.results_dir / f"test_{arm}.json")
        hist = load(args.results_dir / arm / "train_history.json")
        med_dev = dev["finetuned"]["medical"]
        generic_base = dev["base"]["generic"]["_all"]["accuracy"]
        generic_ft = dev["finetuned"]["generic"]["_all"]["accuracy"]
        rows[arm] = {
            "epochs": hist["epochs"],
            "rlcd_weight": hist["rlcd_weight"],
            "ce_weight": hist["ce_weight"],
            "dev_medical": {k: brief(v) for k, v in med_dev.items()},
            "dev_generic": {k: brief(v) for k, v in dev["finetuned"]["generic"].items()},
            "test_medical": {k: brief(v) for k, v in test["finetuned"]["medical"].items()},
            "test_generic": {k: brief(v) for k, v in test["finetuned"]["generic"].items()},
            "dev_medical_base": {k: brief(v) for k, v in dev["base"]["medical"].items()},
            "test_medical_base": {k: brief(v) for k, v in test["base"]["medical"].items()},
            "generic_drop": generic_ft - generic_base,
            "eligible": generic_ft >= generic_base - 0.03,
            "checkpoint_dir": str(args.results_dir / arm),
        }
    eligible = [arm for arm, row in rows.items() if row["eligible"]]
    winner = max(eligible, key=lambda a: (
        rows[a]["dev_medical"]["_all"]["accuracy"],
        -rows[a]["dev_medical"]["_all"]["nll"],
    )) if eligible else None
    summary = {
        "selection_rule": "Highest combined MedMCQA-validation + MedNLI-dev accuracy at epoch 3, with generic accuracy drop <= 3 percentage points; dev NLL breaks ties. External test packs are reported, not used for selection.",
        "winner": winner,
        "arms": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
