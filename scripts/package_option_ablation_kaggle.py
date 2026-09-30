"""Create a Git-pinned Kaggle notebook for one ablation shard."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO = "https://github.com/Priyanshu-5257/laya-medical-finetune.git"
ALIGNED_KERNEL = "aivenger1st/laya-medical-bioclinical-v2"
ALIGNED_DATASET = "hbpkillerx/laya-bioclinical-aligned"


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
    name = f"laya-option-ablation-{args.shard}"
    owner = "aivenger1st" if args.shard == "a" else "hbpkillerx"
    source_key = "kernel_sources" if args.shard == "a" else "dataset_sources"
    source = ALIGNED_KERNEL if args.shard == "a" else ALIGNED_DATASET
    nb = {
        "cells": [
            cell("import torch\nprint('GPUs:', torch.cuda.device_count())\nassert torch.cuda.device_count() == 2\n", "gpu-check"),
            cell("%pip -q install 'laya>=0.3.7' 'transformers>=4.48.0' 'datasets>=3.0.0' safetensors huggingface_hub accelerate scipy pyarrow pandas PyYAML wandb\n", "deps"),
            cell(
                "import pathlib, os, subprocess\n"
                f"repo = {REPO!r}\ncommit = {args.commit!r}\n"
                "root = pathlib.Path('/kaggle/working/repo')\n"
                "subprocess.run(['git', 'clone', repo, str(root)], check=True)\n"
                "subprocess.run(['git', 'checkout', '--detach', commit], cwd=root, check=True)\n"
                "actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()\n"
                "assert actual == commit, (actual, commit)\n"
                "candidates = list(pathlib.Path('/kaggle/input').rglob('bioclinical_distill/rl_agent_config.json'))\n"
                "print('Aligned checkpoint candidates:', candidates, flush=True)\n"
                "assert len(candidates) == 1, 'Expected exactly one aligned BioClinical checkpoint'\n"
                "base_dir = candidates[0].parent\n"
                "print('Commit:', actual, 'checkpoint:', base_dir, flush=True)\n",
                "project",
            ),
            cell(
                "env = os.environ.copy()\n"
                f"env.update({{'WORK': '/kaggle/working', 'BASE_DIR': str(base_dir), 'SHARD': {args.shard!r}, 'PYTHONPATH': str(root), 'USE_TF': '0'}})\n"
                "subprocess.run(['bash', 'scripts/run_option_ablation_shard.sh'], cwd=root, env=env, check=True)\n"
                f"print(pathlib.Path('/kaggle/working/option_ablation_{args.shard}_summary.json').read_text())\n",
                "ablation",
            ),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (args.out_dir / f"{name}.ipynb").write_text(json.dumps(nb))
    metadata = {
        "id": f"{owner}/{name}", "title": name,
        "code_file": f"{name}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    metadata[source_key] = [source]
    (args.out_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    print(args.out_dir)


if __name__ == "__main__":
    main()
