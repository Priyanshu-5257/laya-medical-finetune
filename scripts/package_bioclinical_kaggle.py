"""Package the current working tree into a self-contained private Kaggle notebook."""
from __future__ import annotations

import base64
import io
import json
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/laya-bioclinical-v2")
OUT.mkdir(parents=True, exist_ok=True)

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for folder, suffixes in (("scripts", (".py", ".sh")), ("laya_medical", (".py",)), ("configs", (".yaml",))):
        for path in sorted((ROOT / folder).iterdir()):
            if path.suffix in suffixes:
                archive.write(path, path.relative_to(ROOT))
payload = base64.b64encode(buf.getvalue()).decode("ascii")


def cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


notebook = {
    "cells": [
        cell("import torch\nprint('GPUs:', torch.cuda.device_count())\nassert torch.cuda.device_count() == 2, 'This experiment requires Kaggle T4 x2'\n"),
        cell("%pip -q install 'laya>=0.3.7' 'transformers>=4.48.0' 'datasets>=3.0.0' safetensors huggingface_hub accelerate scipy pyarrow pandas tabulate PyYAML peft wandb\n"),
        cell(
            "import base64, io, zipfile, pathlib, os, subprocess\n"
            "root = pathlib.Path('/kaggle/working/repo')\nroot.mkdir(exist_ok=True)\n"
            f"payload = '{payload}'\n"
            "zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))).extractall(root)\n"
            "print('Packaged project files:', len(list(root.rglob('*'))))\n"
        ),
        cell(
            "env = os.environ.copy()\n"
            "env.update({'WORK': '/kaggle/working', 'PYTHONPATH': str(root), 'USE_TF': '0'})\n"
            "try:\n"
            "    subprocess.run(['bash', 'scripts/run_bioclinical_try.sh'], cwd=root, env=env, check=True)\n"
            "finally:\n"
            "    for name in ('summary_distill.json', 'distill_gate.json', 'summary_bioclinical_medical.json'):\n"
            "        p = pathlib.Path('/kaggle/working') / name\n"
            "        print(name, p.read_text() if p.exists() else 'MISSING', flush=True)\n"
        ),
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
(OUT / "laya-medical-bioclinical-v2.ipynb").write_text(json.dumps(notebook))
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "aivenger1st/laya-medical-bioclinical-v2",
    "title": "laya-medical-bioclinical-v2",
    "code_file": "laya-medical-bioclinical-v2.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [], "kernel_sources": [], "competition_sources": [], "model_sources": [],
}, indent=2))
print(OUT)
