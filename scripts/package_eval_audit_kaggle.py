"""Package the evaluation audit as a self-contained private Kaggle notebook."""
from __future__ import annotations

import base64
import io
import json
import sys
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/laya-medical-eval-audit")
out.mkdir(parents=True, exist_ok=True)
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for folder, suffixes in (("scripts", (".py", ".sh")), ("laya_medical", (".py",)),
                             ("configs", (".yaml",))):
        for path in sorted((root / folder).iterdir()):
            if path.suffix in suffixes:
                archive.write(path, path.relative_to(root))
payload = base64.b64encode(buf.getvalue()).decode("ascii")


def cell(source, cell_id):
    return {"cell_type": "code", "id": cell_id, "execution_count": None,
            "metadata": {}, "outputs": [], "source": source.splitlines(True)}


nb = {
    "cells": [
        cell("import torch\nprint('GPUs:', torch.cuda.device_count())\nassert torch.cuda.device_count() == 2\n", "gpu-check"),
        cell("%pip -q install 'laya>=0.3.7' 'transformers>=4.48.0' 'datasets>=3.0.0' safetensors huggingface_hub accelerate scipy pyarrow pandas PyYAML\n", "deps"),
        cell(
            "import base64, io, zipfile, pathlib, os, subprocess\n"
            "root = pathlib.Path('/kaggle/working/repo')\nroot.mkdir(exist_ok=True)\n"
            f"payload = '{payload}'\n"
            "zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))).extractall(root)\n"
            "input_root = pathlib.Path('/kaggle/input')\n"
            "base = list(input_root.rglob('bioclinical_distill/rl_agent_config.json'))\n"
            "fine = list(input_root.rglob('medical_objective_compare/pure_rlcd/rl_agent_config.json'))\n"
            "dev = list(input_root.rglob('medical_objective_data/dev_packs.pt'))\n"
            "test = list(input_root.rglob('medical_objective_data/eval_packs.pt'))\n"
            "print('Inputs:', base, fine, dev, test, flush=True)\n"
            "assert all(len(x) == 1 for x in (base, fine, dev, test))\n"
            "args = ['python', 'scripts/audit_eval_pipeline.py', '--base-model-dir', str(base[0].parent), "
            "'--ft-model-dir', str(fine[0].parent), '--dev-packs', str(dev[0]), "
            "'--test-packs', str(test[0]), '--out', '/kaggle/working/eval_pipeline_audit.json']\n"
            "env = os.environ.copy()\nenv.update({'PYTHONPATH': str(root), 'USE_TF': '0'})\n"
            "subprocess.run(args, cwd=root, env=env, check=True)\n"
            "subprocess.run(['python', 'scripts/evaluate_corrected_medqa.py', "
            "'--base-model-dir', str(base[0].parent), '--results-dir', str(fine[0].parent.parent), "
            "'--out', '/kaggle/working/corrected_medqa_scores.json'], cwd=root, env=env, check=True)\n"
            "print(pathlib.Path('/kaggle/working/eval_pipeline_audit.json').read_text())\n",
            "audit",
        ),
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4, "nbformat_minor": 5,
}
(out / "laya-medical-eval-audit.ipynb").write_text(json.dumps(nb))
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": "aivenger1st/laya-medical-eval-audit", "title": "laya-medical-eval-audit",
    "code_file": "laya-medical-eval-audit.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [],
    "kernel_sources": ["aivenger1st/laya-medical-bioclinical-v2",
                       "aivenger1st/laya-medical-objective-compare"],
    "model_sources": [],
}, indent=2))
print(out)
