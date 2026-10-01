"""Full canonical-order medical benchmark evaluation, parallel across two GPUs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

import prepare_data as prep
from evaluate import load_agent_model
from laya_medical import ensure_model_dir, tokenize_item
from laya_medical.eval_metrics import eval_items


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def build_packs(tok, work):
    cfg = {"max_len": 512, "head_max_len": 192}
    packs = {
        "medmcqa_validation": prep.build_medmcqa_items(tok, cfg, 0, 20260923, False, split="validation"),
        "mednli_test": prep.build_mednli_items(tok, cfg, 0, 20260923, False, split="test"),
        "medqa_test": prep.build_medqa_items(tok, cfg, 0, 20260923),
    }
    truth_url = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json"
    truth = fetch_json(truth_url)
    rows = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    pubmed = []
    ids = []
    for row in rows:
        pmid = str(row["pubid"])
        if pmid not in truth:
            continue
        gold = row["final_decision"].lower().strip()
        assert gold == truth[pmid], pmid
        item = tokenize_item(tok, {"question": row["question"], "abstract": " ".join(row["context"]["contexts"])},
                             "choice", "Based on the abstract, answer the research question.",
                             {"yes": "the abstract supports yes", "no": "the abstract supports no", "maybe": "the abstract is inconclusive"},
                             prep.one_hot(3, ["yes", "no", "maybe"].index(gold)), **cfg)
        assert item is not None, pmid
        pubmed.append(item)
        ids.append(pmid)
    assert set(ids) == set(truth) and len(pubmed) == 500
    packs["pubmedqa_test"] = pubmed
    expected_mcqa = len(load_dataset("openlifescienceai/medmcqa", split="validation"))
    assert len(packs["medmcqa_validation"]) == expected_mcqa
    assert len(packs["mednli_test"]) == 1422
    assert len(packs["medqa_test"]) == 1273
    nli_train, nli_test = prep._load_mednli("train"), prep._load_mednli("test")
    def pairs(ds):
        return {(row["Premise"], row["Hypothesis"]) for row in ds}
    overlap = len(pairs(nli_train) & pairs(nli_test))
    assert overlap == 0, f"MedNLI train/test exact pair overlap: {overlap}"
    manifest = {
        "counts": {name: len(items) for name, items in packs.items()},
        "option_order": "canonical source order; no option shuffling",
        "max_len": 512, "head_max_len": 192,
        "pubmedqa_test_ids": sorted(ids), "pubmedqa_truth_url": truth_url,
        "mednli_source": "araag2/MedNLI processed mirror; official-size test, provenance not independently authenticated",
        "mednli_train_test_exact_pair_overlap": overlap,
        "medmcqa_split": "full labeled validation; official test labels are unavailable",
    }
    path = work / "benchmark_packs.pt"
    torch.save(packs, path)
    manifest["pack_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (work / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2))
    return path, manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", choices=("a", "b"))
    ap.add_argument("--work", type=Path, default=Path("/kaggle/working"))
    ap.add_argument("--model-dir")
    ap.add_argument("--packs")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.model_dir:
        model, tok, _ = load_agent_model(args.model_dir, torch.device("cuda"))
        packs = torch.load(args.packs, weights_only=False)
        scores = {}
        for name, items in packs.items():
            scores[name] = eval_items(model, tok, items, torch.device("cuda"))
            print(name, scores[name], flush=True)
            Path(args.out).write_text(json.dumps(scores, indent=2))
        return
    args.work.mkdir(parents=True, exist_ok=True)
    original = ensure_model_dir("convaiinnovations/laya")
    tok = AutoTokenizer.from_pretrained(f"{original}/tokenizer")
    path, manifest = build_packs(tok, args.work)
    patterns = ([("original_ce", "full_compare_a/original_ce"), ("original_rlcd", "full_compare_a/original_rlcd"),
                 ("generic_aligned_ce", "full_compare_a/generic_aligned_ce"), ("generic_aligned", "bioclinical_distill")]
                if args.shard == "a" else
                [("medical_aligned", "bioclinical_medical_aligned"), ("medical_aligned_ce", "full_compare_b/medical_aligned_ce"),
                 ("medical_aligned_rlcd", "full_compare_b/medical_aligned_rlcd")])
    models = {"original_laya": original} if args.shard == "a" else {}
    for name, pattern in patterns:
        candidates = list(Path("/kaggle/input").rglob(f"{pattern}/rl_agent_config.json"))
        assert len(candidates) == 1, (name, candidates)
        models[name] = str(candidates[0].parent)
    def gpu_queue(gpu, entries):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        for name, directory in entries:
            subprocess.run([sys.executable, __file__, "--model-dir", directory, "--packs", str(path),
                            "--out", str(args.work / f"benchmark_{name}.json")], env=env, check=True)
    entries = list(models.items())
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(gpu_queue, gpu, entries[gpu::2]) for gpu in range(2)]
        for future in futures:
            future.result()
    result = {"manifest": manifest, "models": {name: json.loads((args.work / f"benchmark_{name}.json").read_text()) for name in models}}
    (args.work / f"full_benchmark_{args.shard}_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
