"""Create a Git-pinned Kaggle notebook for untouched Laya dev evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def cell(source: str, cell_id: str) -> dict:
    return {"cell_type": "code", "id": cell_id, "execution_count": None,
            "metadata": {}, "outputs": [], "source": source.splitlines(True)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    assert len(args.commit) == 40 and all(c in "0123456789abcdef" for c in args.commit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = "laya-original-medical-dev"
    nb = {
        "cells": [
            cell("%pip -q install 'laya>=0.3.7' 'transformers>=4.48.0' 'datasets>=3.0.0' safetensors huggingface_hub accelerate scipy pyarrow pandas PyYAML wandb\n", "deps"),
            cell(
                "import pathlib, os, subprocess\n"
                "root = pathlib.Path('/kaggle/working/repo')\n"
                "subprocess.run(['git', 'clone', 'https://github.com/Priyanshu-5257/laya-medical-finetune.git', str(root)], check=True)\n"
                f"commit = {args.commit!r}\n"
                "subprocess.run(['git', 'checkout', '--detach', commit], cwd=root, check=True)\n"
                "assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() == commit\n"
                "env = os.environ.copy()\n"
                "env.update({'PYTHONPATH': str(root), 'USE_TF': '0'})\n"
                "subprocess.run(['python', 'scripts/evaluate_original_laya_dev.py'], cwd=root, env=env, check=True)\n"
                "print(pathlib.Path('/kaggle/working/original_laya_dev_scores.json').read_text())\n",
                "evaluate",
            ),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (args.out_dir / f"{name}.ipynb").write_text(json.dumps(nb))
    (args.out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"aivenger1st/{name}", "title": name,
        "code_file": f"{name}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }, indent=2))
    print(args.out_dir)


if __name__ == "__main__":
    main()
