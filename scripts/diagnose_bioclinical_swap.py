"""Score the encoder-swapped Laya head before any student training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from laya_medical.bioclinical_swap import load_bioclinical_student
from laya_medical.eval_metrics import eval_packs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", required=True)
    parser.add_argument("--bio-dir", required=True)
    parser.add_argument("--dev-packs", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model, _, swap = load_bioclinical_student(args.teacher_dir, args.bio_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"{args.teacher_dir}/tokenizer")
    packs = torch.load(args.dev_packs, weights_only=False)
    report = {"swap": swap, "scores": eval_packs(model, tok, packs, device)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
