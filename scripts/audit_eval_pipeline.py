"""Audit source labels, truncation, split membership, and option-order sensitivity."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from laya.common import build_sequence, render_options
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from laya_medical import collate_train_batch, load_yaml, tokenize_item
from evaluate import load_agent_model

spec = importlib.util.spec_from_file_location("prepare_data", ROOT / "scripts" / "prepare_data.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def source_gold(task, row):
    if task == "medmcqa_val":
        return ("A", "B", "C", "D")[int(row["cop"])]
    if task == "mednli_dev":
        return str(row["Label"]).lower().strip()
    if task == "pubmedqa":
        return str(row["final_decision"]).lower().strip()
    if task == "medqa":
        return str(row["answer_idx"]).strip()
    raise ValueError(task)


def source_key(task, row):
    if task == "mednli_dev":
        return (str(row["Premise"]), str(row["Hypothesis"]))
    return str(row["question"])


def state_key(task, state):
    if task == "mednli_dev":
        return (str(state["premise"]), str(state["hypothesis"]))
    return str(state["question"])


def audit_builder(task, tok, cfg, seed, reference):
    if task == "medmcqa_val":
        ds = load_dataset("openlifescienceai/medmcqa", split="validation")
        build = lambda: prep.build_medmcqa_items(tok, cfg, 400, seed, True, split="validation")
    elif task == "mednli_dev":
        ds = prep._load_mednli("dev")
        build = lambda: prep.build_mednli_items(tok, cfg, 400, seed, True, split="dev")
    elif task == "pubmedqa":
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        build = lambda: prep.build_pubmedqa_items(tok, cfg, 400, seed)
    elif task == "medqa":
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
        build = lambda: prep.build_medqa_items(tok, cfg, 400, seed)
    else:
        raise ValueError(task)

    lookup = {}
    duplicated_keys = set()
    for row in ds:
        key = source_key(task, row)
        if key in lookup:
            duplicated_keys.add(key)
        lookup[key] = row
    measures = []
    old_tokenize = prep.tokenize_item

    def measured_tokenize(tok_arg, state, qtype, instructions, criteria, target, *, max_len, head_max_len):
        item = tokenize_item(tok_arg, state, qtype, instructions, criteria, target,
                             max_len=max_len, head_max_len=head_max_len)
        if item is None:
            return None
        q = {"t": qtype, "ins": instructions, "crit": criteria}
        full_ids, full_markers = build_sequence(tok_arg, state, q, max_len=100000,
                                                 head_max_len=head_max_len)
        keys = list(criteria)
        gold_index = max(range(len(target)), key=lambda i: target[i])
        row = lookup.get(state_key(task, state))
        expected = source_gold(task, row) if row is not None else None
        end = full_ids.index(tok_arg.sep_token_id, full_markers[-1])
        option_spans = [full_ids[full_markers[i]:full_markers[i+1] if i+1 < len(keys) else end]
                        for i in range(len(keys))]
        option_lengths = [len(tok_arg(" " + option.replace(tok_arg.mask_token, " "),
                                      add_special_tokens=False)["input_ids"])
                          for option in render_options(q)]
        option_tokens_lost = [max(0, raw - (len(span)-1))
                              for raw, span in zip(option_lengths, option_spans)]
        measured = {
            "source_found": row is not None,
            "source_key_unique": state_key(task, state) not in duplicated_keys,
            "gold_matches_source": expected == keys[gold_index] if expected is not None else False,
            "tokens_full": len(full_ids),
            "tokens_kept": len(item["ids"]),
            "tokens_lost": len(full_ids) - len(item["ids"]),
            "all_options_present": len(item["markers"]) == len(keys),
            "options_distinct": len({tuple(span) for span in option_spans}) == len(keys),
            "options_token_truncated": any(option_tokens_lost),
            "gold_option_truncated": option_tokens_lost[gold_index] > 0,
            "option_tokens_lost": sum(option_tokens_lost),
        }
        if task == "pubmedqa" and row is not None:
            measured["pubid"] = str(row["pubid"])
        measures.append(measured)
        return item

    try:
        prep.tokenize_item = measured_tokenize
        rebuilt = build()
    finally:
        prep.tokenize_item = old_tokenize
    assert len(rebuilt) == len(measures) == len(reference), (task, len(rebuilt), len(reference))
    exact = sum(a["ids"] == b["ids"] and a["target"] == b["target"]
                and a["markers"] == b["markers"] for a, b in zip(rebuilt, reference))
    input_matches = sum(a["ids"] == b["ids"] and a["markers"] == b["markers"]
                        for a, b in zip(rebuilt, reference))
    target_matches = sum(a["target"] == b["target"] for a, b in zip(rebuilt, reference))
    report = {
        "n": len(measures), "exact_pack_matches": exact,
        "input_pack_matches": input_matches, "target_pack_matches": target_matches,
        "source_found": sum(x["source_found"] for x in measures),
        "source_key_unique": sum(x["source_key_unique"] for x in measures),
        "gold_matches_source": sum(x["gold_matches_source"] for x in measures),
        "state_truncated": sum(x["tokens_lost"] > 0 for x in measures),
        "tokens_lost_total": sum(x["tokens_lost"] for x in measures),
        "tokens_lost_max": max(x["tokens_lost"] for x in measures),
        "all_options_present": sum(x["all_options_present"] for x in measures),
        "options_distinct": sum(x["options_distinct"] for x in measures),
        "options_token_truncated": sum(x["options_token_truncated"] for x in measures),
        "gold_option_truncated": sum(x["gold_option_truncated"] for x in measures),
        "option_tokens_lost_total": sum(x["option_tokens_lost"] for x in measures),
        "full_length_p50": sorted(x["tokens_full"] for x in measures)[len(measures)//2],
        "full_length_p95": sorted(x["tokens_full"] for x in measures)[int(len(measures)*.95)],
    }
    if task == "pubmedqa":
        url = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json"
        with urllib.request.urlopen(url, timeout=30) as response:
            official_test = json.load(response)
        ids = {x["pubid"] for x in measures}
        report["official_test_overlap"] = len(ids & set(official_test))
        report["official_development_overlap"] = len(ids - set(official_test))
        report["official_test_size"] = len(official_test)
    return report, rebuilt


def permute_item(item, order, sep_id):
    ids = item["ids"]
    marks = item["markers"]
    end = ids.index(sep_id, marks[-1])
    spans = [ids[marks[i]:marks[i+1] if i+1 < len(marks) else end] for i in range(len(marks))]
    new_ids = list(ids[:marks[0]])
    new_marks = []
    for i in order:
        new_marks.append(len(new_ids))
        new_ids.extend(spans[i])
    new_ids.extend(ids[end:])
    out = dict(item)
    out["ids"] = new_ids
    out["markers"] = new_marks
    out["target"] = [item["target"][i] for i in order]
    out["label"] = order.index(item["label"])
    return out


@torch.no_grad()
def predict(model, tok, items, device, batch_size=8):
    preds = []
    for start in range(0, len(items), batch_size):
        batch = collate_train_batch(items[start:start+batch_size], tok.pad_token_id)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                              batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                              batch["qtype"].to(device))
        preds.extend(logits.masked_fill(~batch["marker_mask"].to(device), -1e4).argmax(-1).cpu().tolist())
    return preds


def permutation_audit(model, tok, items, device):
    originals = predict(model, tok, items, device)
    orders = []
    variants = []
    for permutation in ("reverse", "rotate"):
        local_orders = []
        for item in items:
            k = len(item["markers"])
            order = list(reversed(range(k))) if permutation == "reverse" else list(range(1, k)) + [0]
            local_orders.append(order)
            variants.append(permute_item(item, order, tok.sep_token_id))
        orders.append(local_orders)
    varied = predict(model, tok, variants, device)
    n = len(items)
    original_accuracy = sum(p == item["label"] for p, item in zip(originals, items)) / n
    results = {"n": n, "original_accuracy": original_accuracy,
               "original_pred_positions": dict(Counter(originals))}
    for j, name in enumerate(("reverse", "rotate")):
        mapped = [orders[j][i][varied[j*n+i]] for i in range(n)]
        results[name] = {
            "accuracy": sum(p == item["label"] for p, item in zip(mapped, items)) / n,
            "semantic_flip_rate": sum(a != b for a, b in zip(originals, mapped)) / n,
            "pred_positions": dict(Counter(varied[j*n:(j+1)*n])),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--ft-model-dir", required=True)
    parser.add_argument("--dev-packs", required=True)
    parser.add_argument("--test-packs", required=True)
    parser.add_argument("--config", default="configs/medical_objective_compare.yaml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    conf = load_yaml(args.config)
    cfg = {"max_len": int(conf["max_len"]), "head_max_len": int(conf["head_max_len"])}
    seed = int(conf["train"]["seed"])
    dev = torch.load(args.dev_packs, weights_only=False)
    test = torch.load(args.test_packs, weights_only=False)
    packs = {**dev["medical"], **test["medical"]}
    tok = AutoTokenizer.from_pretrained(os.path.join(args.base_model_dir, "tokenizer"))
    seeds = {"medmcqa_val": seed+300, "mednli_dev": seed+301,
             "pubmedqa": seed+200, "medqa": seed+201}
    report = {"config": cfg, "data": {}, "permutation": {}}
    for task, items in packs.items():
        print("Auditing source and input:", task, flush=True)
        report["data"][task], corrected = audit_builder(task, tok, cfg, seeds[task], items)
        if task == "medqa":
            packs[task] = corrected
        print(json.dumps(report["data"][task]), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model_name, model_dir in (("aligned", args.base_model_dir), ("pure_rlcd", args.ft_model_dir)):
        print("Auditing option order:", model_name, flush=True)
        model, model_tok, _ = load_agent_model(model_dir, device)
        report["permutation"][model_name] = {}
        for task, items in packs.items():
            report["permutation"][model_name][task] = permutation_audit(model, model_tok, items, device)
            print(task, json.dumps(report["permutation"][model_name][task]), flush=True)
        del model
        torch.cuda.empty_cache()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("Wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
