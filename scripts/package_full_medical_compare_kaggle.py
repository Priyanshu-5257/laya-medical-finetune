"""Create Git-pinned notebooks for the paired Kaggle medical comparison."""
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
    parser.add_argument("--shard", choices=("a", "b"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    assert len(args.commit) == 40 and all(c in "0123456789abcdef" for c in args.commit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = f"laya-full-medical-compare-{args.shard}"
    owner = "aivenger1st" if args.shard == "a" else "hbpkillerx"
    source = ["aivenger1st/laya-medical-bioclinical-v2"] if args.shard == "a" else []
    setup = (
        "candidates = list(pathlib.Path('/kaggle/input').rglob('bioclinical_distill/rl_agent_config.json'))\n"
        "assert len(candidates) == 1, candidates\n"
        "env['GENERIC_ALIGNED_DIR'] = str(candidates[0].parent)\n"
    ) if args.shard == "a" else ""
    nb = {
        "cells": [
            cell("import torch\nassert torch.cuda.device_count() == 2, torch.cuda.device_count()\nprint([torch.cuda.get_device_name(i) for i in range(2)])\n", "gpu-check"),
            cell("%pip -q install 'laya>=0.3.7' 'transformers>=4.48.0' 'datasets>=3.0.0' safetensors huggingface_hub accelerate scipy pyarrow pandas PyYAML wandb\n", "deps"),
            cell(
                "import os, pathlib, subprocess\n"
                "root = pathlib.Path('/kaggle/working/repo')\n"
                "subprocess.run(['git', 'clone', 'https://github.com/Priyanshu-5257/laya-medical-finetune.git', str(root)], check=True)\n"
                f"commit = {args.commit!r}\n"
                "subprocess.run(['git', 'checkout', '--detach', commit], cwd=root, check=True)\n"
                "assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() == commit\n"
                "env = os.environ.copy()\n"
                f"env.update({{'WORK': '/kaggle/working', 'SHARD': {args.shard!r}, 'PYTHONPATH': str(root), 'USE_TF': '0'}})\n"
                f"{setup}"
                "print('Commit:', commit, flush=True)\n"
                "subprocess.run(['bash', 'scripts/run_full_medical_compare.sh'], cwd=root, env=env, check=True)\n"
                f"print(pathlib.Path('/kaggle/working/full_compare_{args.shard}_summary.json').read_text())\n",
                "compare",
            ),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (args.out_dir / f"{name}.ipynb").write_text(json.dumps(nb))
    (args.out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{owner}/{name}", "title": name,
        "code_file": f"{name}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
        "competition_sources": [], "kernel_sources": source, "model_sources": [],
    }, indent=2))
    print(args.out_dir)


if __name__ == "__main__":
    main()
