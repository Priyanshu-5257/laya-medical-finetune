"""Generate thin Git-pinned full-test evaluation notebooks."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--shard", choices=("a", "b"), required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    assert len(args.commit) == 40 and all(c in "0123456789abcdef" for c in args.commit)
    owner = "aivenger1st" if args.shard == "a" else "hbpkillerx"
    name = f"laya-full-benchmark-{args.shard}"
    sources = [f"{owner}/laya-full-medical-compare-{args.shard}"]
    if args.shard == "a":
        sources.append("aivenger1st/laya-medical-bioclinical-v2")
    source = (
        "import os, pathlib, subprocess, torch\nassert torch.cuda.device_count() == 2\n"
        "root = pathlib.Path('/kaggle/working/repo')\n"
        "subprocess.run(['git','clone','https://github.com/Priyanshu-5257/laya-medical-finetune.git',str(root)],check=True)\n"
        f"commit = {args.commit!r}\n"
        "subprocess.run(['git','checkout','--detach',commit],cwd=root,check=True)\n"
        "assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==commit\n"
        "env=os.environ.copy()\nenv.update({'PYTHONPATH':str(root),'USE_TF':'0'})\n"
        f"subprocess.run(['python','scripts/evaluate_full_benchmarks.py','--shard',{args.shard!r}],cwd=root,env=env,check=True)\n"
    )
    def cell(text):
        return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":text.splitlines(True)}
    notebook = {"cells":[cell("%pip -q install 'laya==0.3.22' 'transformers==5.0.0' 'datasets==5.0.0' safetensors huggingface_hub accelerate scipy pyarrow pandas PyYAML wandb\n"),cell(source)],
                "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"nbformat":4,"nbformat_minor":5}
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/f"{name}.ipynb").write_text(json.dumps(notebook))
    (args.out_dir/"kernel-metadata.json").write_text(json.dumps({"id":f"{owner}/{name}","title":name,"code_file":f"{name}.ipynb",
        "language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":True,"enable_internet":True,
        "machine_shape":"NvidiaTeslaT4","kernel_sources":sources,"dataset_sources":[],"competition_sources":[],"model_sources":[]},indent=2))


if __name__ == "__main__":
    main()
