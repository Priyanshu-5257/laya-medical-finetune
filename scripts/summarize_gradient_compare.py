"""Select an RLCD gradient estimator on validation, and report all arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def brief(metrics):
    return {key: metrics[key] for key in ("n", "accuracy", "nll", "brier", "ece")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--print-winner", action="store_true")
    args = ap.parse_args()
    rows = {}
    for arm in ("legacy", "loo", "pathwise", "permutation"):
        directory = args.results_dir / arm
        if not (directory / "eval_final.json").exists():
            continue
        dev = json.loads((directory / "eval_final.json").read_text())
        hist = json.loads((directory / "train_history.json").read_text())
        test_path = args.results_dir / f"test_{arm}.json"
        test = json.loads(test_path.read_text()) if test_path.exists() else None
        base_generic = dev["base"]["generic"]["_all"]["accuracy"]
        ft_generic = dev["finetuned"]["generic"]["_all"]["accuracy"]
        rows[arm] = {
            "gradient_estimator": hist["gradient_estimator"],
            "permutation": hist["permute_options"],
            "consistency_weight": hist["consistency_weight"],
            "epoch_history": hist["history"],
            "dev_medical": {k: brief(v) for k, v in dev["finetuned"]["medical"].items()},
            "dev_generic": {k: brief(v) for k, v in dev["finetuned"]["generic"].items()},
            "test_medical": {k: brief(v) for k, v in test["finetuned"]["medical"].items()} if test else None,
            "test_generic": {k: brief(v) for k, v in test["finetuned"]["generic"].items()} if test else None,
            "generic_delta": ft_generic - base_generic,
            "eligible": ft_generic >= base_generic - 0.03,
        }
    eligible = [arm for arm in ("legacy", "loo", "pathwise") if rows[arm]["eligible"]]
    winner = max(eligible, key=lambda arm: (
        rows[arm]["dev_medical"]["_all"]["accuracy"],
        -rows[arm]["dev_medical"]["_all"]["nll"],
    )) if eligible else max(("legacy", "loo", "pathwise"), key=lambda arm: (
        rows[arm]["dev_medical"]["_all"]["accuracy"],
        rows[arm]["generic_delta"],
    ))
    summary = {
        "selection_rule": "Final epoch combined MedMCQA validation + MedNLI dev accuracy, subject to generic accuracy drop <= 3 percentage points; dev NLL breaks ties. If none meet retention threshold, choose medical accuracy then generic delta. External test is not used for selection.",
        "winner_estimator": winner,
        "arms": rows,
    }
    audit_path = args.results_dir / "permutation_audit.json"
    if audit_path.exists():
        summary["permutation_audit"] = json.loads(audit_path.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(winner if args.print_winner else json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
