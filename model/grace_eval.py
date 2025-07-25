import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.multiprocessing as mp
from tree_sitter import Language, Parser
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

import custom_generate
from eval_metric import compute_metric_stmt
from eval_utils import compute_mean_logp
from eval_utils import (
    postprocess_code_lines,
    extract_identifiers,
    cal_edit_sim,
    remove_comments
)

from functools import partial


parser = None

COMMENT_SYMBOL = {
    "python": "#",
    "java": "//",
    "csharp": "//",
    "typescript": "//"
}


def compute_id_match(pred_ids, target_ids):
    pred_ids = list(set(pred_ids))
    target_ids = list(set(target_ids))
    tp = 0
    fp = 0
    fn = 0
    for pid in pred_ids:
        if pid in target_ids:
            tp += 1
        else:
            fp += 1
    for tid in target_ids:
        if tid not in pred_ids:
            fn += 1
    return tp, fp, fn

def compute_code_prf1(pred_lines, gt_lines):
    pred_set = set(pred_lines)
    gt_set = set(gt_lines)
    tp = len(pred_set.intersection(gt_set))
    fp = len(pred_set.difference(gt_set))
    fn = len(gt_set.difference(pred_set))
    return tp, fp, fn

def compute_edit_sim(samples):
    refs, hyps = [], []
    for s in samples:
        refs.append(s["target"])
        hyps.append(s["pred"])
    return cal_edit_sim(refs, hyps)


def process_examples(lang, args):
    sample, ex = args
    global parser

    prediction = postprocess_code_lines(ex["prompt"], sample["pred"], parser, lang)
    prediction = remove_comments(prediction)
    target = ex["groundtruth"]
    target = remove_comments(target)

    pred_lines = [l.strip() for l in prediction.split("\n") if l.strip()]
    gt_lines = [l.strip() for l in target.split("\n") if l.strip()]
    em_label = int(pred_lines == gt_lines)

    pred_ids = extract_identifiers(prediction, lang)
    target_ids = extract_identifiers(target, lang)

    trunc_s = {
        "task_id": sample["task_id"],
        "pred": prediction,
        "target": target,
        "pred_ids": pred_ids,
        "target_ids": target_ids
    }
    return trunc_s, em_label


def compute_metric_stmt(args):
    examples = {}
    with open(os.path.join(args.output_dir, "prediction.jsonl"), "r") as f_pred:
        samples = []
        for l in f_pred.readlines():
            l_pred = json.loads(l)
            samples.append(l_pred)
            examples[l_pred["task_id"]] = {
                "prompt": l_pred["code_context"],
                "groundtruth": l_pred["groundtruth"]
            }


    assert len(samples) == len(examples), f"{len(samples)} != {len(examples)}"

    global parser
    ts_lang = "c_sharp" if args.language == "csharp" else args.language
    language = Language(args.ts_lib, ts_lang)
    parser = Parser()
    parser.set_language(language)

    truncated_samples = []
    em_labels = []

    print("post-processing samples ...")
    pool = mp.Pool(16)
    worker = partial(process_examples, args.language)

    with tqdm(total=len(samples)) as pbar:
        for output in pool.imap_unordered(worker, zip(samples, [examples[s["task_id"]] for s in samples])):
            trunc_s, em_label = output
            em_labels.append(em_label)
            truncated_samples.append(trunc_s)
            pbar.update()

    exact_match = 0
    with open(os.path.join(args.output_dir, "prediction_truncated.jsonl"), 'w', encoding="utf-8") as pt, \
            open(f"{args.output_dir}/exact_match_idx.jsonl", 'w') as em:
        for trunc_s, em_label in zip(truncated_samples, em_labels):
            pt.write(json.dumps(trunc_s) + "\n")
            if em_label == 1:
                em.write(f'{trunc_s["task_id"]}\n')
                exact_match += 1

    ### Score calculation

    id_em = []
    edit_similarities = []
    detailed_results = []

    for idx, trunc_s in enumerate(truncated_samples):
        identifier_em = int(trunc_s["pred_ids"] == trunc_s["target_ids"])
        es = cal_edit_sim([trunc_s["target"]], [trunc_s["pred"]])
        code_tp, code_fp, code_fn = compute_code_prf1(trunc_s["pred"], trunc_s["target"])
        code_precision = code_tp / (code_tp + code_fp) if (code_tp + code_fp) > 0 else 0
        code_recall = code_tp / (code_tp + code_fn) if (code_tp + code_fn) > 0 else 0
        code_f1 = 2 * code_precision * code_recall / (code_precision + code_recall) if (code_precision + code_recall) > 0 else 0
        
        id_tp, id_fp, id_fn = compute_id_match(trunc_s["pred_ids"], trunc_s["target_ids"])
        id_em.append(identifier_em)
        edit_similarities.append(es)
        sorted_pred_ids = sorted(trunc_s["pred_ids"])
        sorted_target_ids = sorted(trunc_s["target_ids"])
        id_es = cal_edit_sim([" ".join(sorted_target_ids)], [" ".join(sorted_pred_ids)])

        detailed_results.append({
            "task_id": trunc_s["task_id"],
            "em": em_labels[idx],
            "es": es,
            "code_recall": code_recall,
            "code_f1": code_f1,
            "id_em": identifier_em,
            "id_es": id_es,
            "id_precision": id_tp / (id_tp + id_fp) if (id_tp + id_fp) != 0 else 0,
            "id_recall": id_tp / (id_tp + id_fn) if (id_tp + id_fn) != 0 else 0,
            "id_f1": 2 * id_tp / (2 * id_tp + id_fp + id_fn) if (2 * id_tp + id_fp + id_fn) != 0 else 0,
        })

    em_ratio = round(exact_match / len(samples) * 100, 2)
    edit_sim = round(sum(edit_similarities) / len(edit_similarities), 2)
    code_recall = round(sum(detailed_results[idx]['code_recall'] for idx in range(len(detailed_results))) / len(detailed_results) * 100,2)
    code_f1 = round(sum(detailed_results[idx]['code_f1'] for idx in range(len(detailed_results))) / len(detailed_results) * 100, 2)

    id_em_ratio = round(sum(detailed_results[idx]['id_em'] for idx in range(len(detailed_results))) / len(detailed_results) * 100, 2)
    id_es_ratio = round(sum(detailed_results[idx]['id_es'] for idx in range(len(detailed_results))) / len(detailed_results), 2)
    id_precision = round(sum(detailed_results[idx]['id_precision'] for idx in range(len(detailed_results))) / len(detailed_results) * 100, 2)
    id_recall = round(sum(detailed_results[idx]['id_recall'] for idx in range(len(detailed_results))) / len(detailed_results) * 100,2)
    id_f1 = round(sum(detailed_results[idx]['id_f1'] for idx in range(len(detailed_results))) / len(detailed_results) * 100, 2)

    print(
        f"Code Matching: "
        f"EM {em_ratio:.2f}, "
        f"ES {edit_sim:.2f}, "
        f"Recall {code_recall}, "
        f"F1 {code_f1}"
    )

    print(
        f"ID matching: "
        f"EM {id_em_ratio}, "
        f"ES {id_es_ratio}, "
        f"Recall {id_recall}, "
        f"F1 {id_f1}"
    )

    with open(os.path.join(args.output_dir, "detailed_results.json"), 'w') as f:
        for dr in detailed_results:
            f.write(json.dumps(dr) + "\n")

    # write the results to a file
    print(f'writing results to {os.path.join(args.output_dir, "results.json")}')
    with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
        res = {
            "em": em_ratio,
            "es": edit_sim,
            "code_recall": code_recall,
            "code_f1": code_f1,

            "id_em": id_em_ratio,
            "id_es": id_es_ratio,
            "id_recall": id_recall,
            "id_f1": id_f1,
            "total": len(truncated_samples)
        }
        f.write(json.dumps(res, indent=2))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model inference args
    parser.add_argument("--language", type=str,default="python", help="language name")
    parser.add_argument("--model_name_or_path", default=None, type=str, help="Pre-trained Model Path")
    parser.add_argument(
        "--model_type",
        type=str,
        default="codelm",
        choices=["codelm", "codelm_cfc"],
        help="Model type to be loaded"
    )
    parser.add_argument("--prompt_file", type=str, default=None, help="file with a list of prompts")
    parser.add_argument("--gen_length", type=int, default=50, help="max length of generated token sequence")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="max length of prompt")
    parser.add_argument(
        "--cfc_seq_length",
        type=int,
        default=512,
        help="For model_type=codelm_cfc: Text sequence length corresponding to the retrieved nodes"
    )
    parser.add_argument(
        "--min_cfc_score",
        type=float,
        default=float('-inf'),
        help="For model_type=codelm_cfc: min score of a chunk to be considered as CFC chunk"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for code completion")
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling"
    )
    parser.add_argument("--output_dir", type=str, help="output directory to save predictions")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="The parameter for repetition penalty.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=1,
        help="The number of processes to use for the preprocessing."
    )
    parser.add_argument(
        "--overwrite_cache",
        type=bool,
        default=False,
        help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--dtype", type=str, default='bf16')
    parser.add_argument("--do_sample", action="store_true", help="whether we do sampling or greedy/beam-search")
    parser.add_argument("--num_beams", type=int, default=1, help="num of beam for beam-search")
    # compute metric args
    parser.add_argument(
        "--ts_lib",
        type=str,
        default="build/python-lang-parser.so",
        help="tree-sitter lib for tokenize code"
    )
    # only compute metric
    # parser.add_argument("--only_compute_metric", action="store_true", help="only compute metric")
    parser.add_argument("--only_compute_metric", default=True, type=bool, help="only compute metric")
    parser.add_argument("--task", type=str, default="line_completion", help="task name")
    args = parser.parse_args()
    args.prompt_file = f"dataset/dataset_crosscodeeval/data/{args.language}/{args.task}.jsonl"
    args.output_dir = f"results/crosscodeeval/{args.language}/grace/"

    set_seed(args.seed, device_specific=False)

    if args.num_return_sequences > 1:
        assert args.do_sample, "sampling must be set to True when num_return_sequences > 1"

    accelerator = Accelerator()

    if accelerator.is_main_process:
        compute_metric_stmt(args)
