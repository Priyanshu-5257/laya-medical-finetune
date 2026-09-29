"""Package the medical objective comparison as a self-contained Kaggle notebook."""
from __future__ import annotations

import base64
import io
import json
import sys
import zipfile
from pathlib import Path


root = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/laya-medical-objective-compare")
out.mkdir(parents=True, exist_ok=True)
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for folder, suffixes in (("scripts", (".py", ".sh")), ("laya_medical", (".py",)), ("configs", (".yaml",))):
        for path in sorted((root / folder).iterdir()):
            if path.suffix in suffixes:
                archive.write(path, path.relative_to(root))
payload = base64.b64encode(buf.getvalue()).decode("ascii")


def cell(source: str, cell_id: str):
    return {"cell_type": "code", "id": cell_id, "execution_count": None,
            "metadata": {}, "outputs": [], "source": source.splitlines(True)}


nb = {
    "cells": [
        cell("import torch\nprint('GPUs:', torch.cuda.device_count())\nassert torch.cuda.device_count() == 2\n", "gpu-check"),
        cell("%pip -q install 'laya>=0.3.7' 'transformers>=4.48.0' 'datasets>=3.0.0' safetensors huggingface_hub accelerate scipy pyarrow pandas PyYAML wandb\n", "deps"),
        cell(
            "import base64, io, zipfile, pathlib, os, subprocess\n"
            "root = pathlib.Path('/kaggle/working/repo')\nroot.mkdir(exist_ok=True)\n"
            f"payload = '{payload}'\n"
            "zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))).extractall(root)\n"
            "candidates = list(pathlib.Path('/kaggle/input').rglob('bioclinical_distill/rl_agent_config.json'))\n"
            "print('Aligned checkpoint candidates:', candidates, flush=True)\n"
            "assert len(candidates) == 1, 'Expected one BioClinical v2 aligned checkpoint from kernel_sources'\n"
            "base_dir = candidates[0].parent\n"
            "print('Using aligned checkpoint:', base_dir, flush=True)\n",
            "project",
        ),
        cell(
            "env = os.environ.copy()\n"
            "env.update({'WORK': '/kaggle/working', 'BASE_DIR': str(base_dir), 'PYTHONPATH': str(root), 'USE_TF': '0'})\n"
            "subprocess.run(['bash', 'scripts/run_medical_objective_compare.sh'], cwd=root, env=env, check=True)\n"
            "print(pathlib.Path('/kaggle/working/medical_objective_comparison.json').read_text())\n",
            "compare",
        ),
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
(out / "laya-medical-objective-compare.ipynb").write_text(json.dumps(nb))
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": "aivenger1st/laya-medical-objective-compare",
    "title": "laya-medical-objective-compare",
    "code_file": "laya-medical-objective-compare.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [], "competition_sources": [],
    "kernel_sources": ["aivenger1st/laya-medical-bioclinical-v2"], "model_sources": [],
}, indent=2))
print(out)
