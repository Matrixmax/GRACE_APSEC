"""Train the structural encoder f_theta and the alpha gate with InfoNCE.

Paper (Methodology): trained on development repositories that are project-disjoint from the test set,
split into fitting and held-out tuning repositories; positives contain the definition / callee /
type / dependency-path evidence used by the reference completion; negatives are same-language
unrelated regions plus hard negatives from the semantic candidate pool; InfoNCE loss; early stopping
on Recall@K_c on the tuning repositories; test repositories never used.

Implementation details:

* Training queries are ``grace_tasks`` JSONL rows (fields: task_id, repo_name, file_path, prefix,
  groundtruth, optional repo_dir, optional ``positive_units``). Each query is processed with the same
  leakage masking as at test time.
* Positives: if ``positive_units`` (unit ids such as ``func:pkg/a.py:f``) are given they are used as
  is. Otherwise :func:`heuristic_positive_units` resolves identifiers of the reference line against
  function/class units of the masked graph (identifier matching).
* Score used for training and evaluation is the retrieval score
  ``alpha * cos(v_q, v_c) + (1 - alpha) * cos(h_Gq, h_Gc)``; logits are ``score / temperature``.
  With several positives, the loss is ``-log(sum_pos exp / sum_all exp)``.
* Random negatives: units of the same repository that are not positives and are not in the
  semantic top-K_c pool. Hard negatives: non-positive units from the semantic top-K_c pool.
* Recall@K_c (tuning): fraction of tuning queries with at least one positive among the top-K_c
  units ranked by the score above over all units of the task index.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .config import GraceConfig, load_config, resolve_device
from .data import CompletionTask, assert_project_disjoint, load_tasks, read_repo_list
from .encoder import (EncoderSpec, StructuralRetrievalModel, checkpoint_fingerprint, collate, graph_to_arrays,
                      save_checkpoint, spec_summary)
from .graph_features import TextEncoder, node_features
from .pipeline import GracePipeline, PreparedTask
from .subgraphs import unit_graph

LOGGER = logging.getLogger(__name__)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def heuristic_positive_units(prepared: PreparedTask, groundtruth: str) -> List[int]:
    """Indices of units whose definition name occurs as an identifier in the reference line."""
    names = set(_IDENT.findall(groundtruth or ""))
    return [i for i, u in enumerate(prepared.units) if u.name and u.name in names]


@dataclass
class TrainingExample:
    task_id: str
    repo_name: str
    v_q: np.ndarray
    query_item: tuple
    unit_items: List[tuple]
    unit_sem: np.ndarray
    positives: List[int]
    hard_pool: List[int]
    positive_source: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_example(pipe: GracePipeline, task: CompletionTask, row_positive_units: Optional[Sequence[str]],
                  k_c: int, lappe_dim: int) -> Optional[TrainingExample]:
    prepared = pipe.prepare(task)
    if not prepared.units:
        return None
    if row_positive_units:
        wanted = set(row_positive_units)
        positives = [i for i, u in enumerate(prepared.units) if u.unit_id in wanted]
        source = "explicit"
    else:
        positives = heuristic_positive_units(prepared, task.groundtruth)
        source = "heuristic_identifier_match"
    if not positives:
        return None
    enc = pipe.text_encoder
    v_q = enc.encode([prepared.local.text], truncation_side="left")[0]
    unit_sem = enc.encode([u.text for u in prepared.units])
    sims = unit_sem @ v_q
    pool = [int(i) for i in np.argsort(-sims, kind="stable")[:k_c]]
    qx = node_features(prepared.query_graph, enc, nodes=prepared.query_nodes, lappe_dim=lappe_dim)
    q_item = (qx,) + graph_to_arrays(prepared.query_graph, prepared.query_nodes)
    unit_items = []
    for u in prepared.units:
        g = unit_graph(prepared.graph, u)
        x = node_features(g, enc, nodes=u.nodes, lappe_dim=lappe_dim)
        unit_items.append((x,) + graph_to_arrays(g, u.nodes))
    return TrainingExample(task.task_id, task.repo_name, v_q, q_item, unit_items, unit_sem, positives,
                           [i for i in pool if i not in positives], source)


def _scores(model: StructuralRetrievalModel, ex: TrainingExample, cand: Sequence[int], device: str) -> torch.Tensor:
    items = [ex.query_item] + [ex.unit_items[i] for i in cand]
    _, hg = model.encode(collate(items).to(device))
    hg = torch.nn.functional.normalize(hg, dim=-1)
    v_q = torch.as_tensor(ex.v_q, dtype=torch.float32, device=device)
    alpha = model.alpha(v_q.unsqueeze(0), hg[:1]).squeeze(0)
    sem = torch.as_tensor(ex.unit_sem[list(cand)] @ ex.v_q, dtype=torch.float32, device=device)
    struct = hg[1:] @ hg[0]
    return alpha * sem + (1 - alpha) * struct


def sample_candidates(ex: TrainingExample, n_random: int, n_hard: int, rng: random.Random) -> List[int]:
    hard = [i for i in ex.hard_pool][:n_hard]
    others = [i for i in range(len(ex.unit_items)) if i not in ex.positives and i not in ex.hard_pool]
    rnd = rng.sample(others, min(n_random, len(others)))
    return list(ex.positives) + hard + rnd


def info_nce(scores: torch.Tensor, n_pos: int, temperature: float) -> torch.Tensor:
    logits = scores / temperature
    return -(torch.logsumexp(logits[:n_pos], 0) - torch.logsumexp(logits, 0))


def recall_at_k(model: StructuralRetrievalModel, examples: Sequence[TrainingExample], k: int, device: str) -> float:
    if not examples:
        return float("nan")
    model.eval()
    hits = 0
    with torch.no_grad():
        for ex in examples:
            scores = _scores(model, ex, list(range(len(ex.unit_items))), device).cpu().numpy()
            top = set(int(i) for i in np.argsort(-scores, kind="stable")[:k])
            hits += int(bool(top & set(ex.positives)))
    return hits / len(examples)


def _optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError("optimizer must be 'adam' or 'adamw'")


TRAINING_KEYS = ["infonce_temperature", "optimizer", "learning_rate", "weight_decay", "batch_size",
                 "max_epochs", "early_stopping_patience", "num_random_negatives", "num_hard_negatives",
                 "random_seed"]


def train(cfg: GraceConfig, fit_examples: List[TrainingExample], tune_examples: List[TrainingExample],
          out_dir: str, device: str = "cpu", max_steps: Optional[int] = None, metadata: Optional[dict] = None) -> dict:
    hp = cfg.require_all(TRAINING_KEYS)
    k_c = int(cfg.require("candidate_pool_size_Kc"))
    set_seed(int(hp["random_seed"]))
    rng = random.Random(int(hp["random_seed"]))
    spec = EncoderSpec.from_config(cfg)
    model = StructuralRetrievalModel(spec).to(device)
    opt = _optimizer(hp["optimizer"], model.parameters(), float(hp["learning_rate"]), float(hp["weight_decay"]))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_path, last_path = out / "best.pt", out / "last.pt"
    history = []
    best, bad_epochs, step = -1.0, 0, 0
    meta = dict(metadata or {})
    meta.update({"config": cfg.snapshot(), "spec": spec_summary(spec)})
    stop = False
    for epoch in range(int(hp["max_epochs"])):
        model.train()
        order = list(range(len(fit_examples)))
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), int(hp["batch_size"])):
            batch = [fit_examples[i] for i in order[start:start + int(hp["batch_size"])]]
            opt.zero_grad()
            loss = torch.zeros((), device=device)
            for ex in batch:
                cand = sample_candidates(ex, int(hp["num_random_negatives"]), int(hp["num_hard_negatives"]), rng)
                loss = loss + info_nce(_scores(model, ex, cand, device), len(ex.positives),
                                       float(hp["infonce_temperature"]))
            loss = loss / max(len(batch), 1)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            step += 1
            if max_steps is not None and step >= max_steps:
                stop = True
                break
        recall = recall_at_k(model, tune_examples, k_c, device)
        record = {"epoch": epoch, "steps": step, "train_loss": float(np.mean(losses)) if losses else None,
                  f"tune_recall_at_{k_c}": recall}
        history.append(record)
        LOGGER.info("epoch %d: %s", epoch, record)
        save_checkpoint(model, str(last_path), {**meta, "epoch": epoch, "steps": step})
        improved = not np.isnan(recall) and recall > best
        if improved or not best_path.exists():
            best = recall if not np.isnan(recall) else best
            bad_epochs = 0
            save_checkpoint(model, str(best_path), {**meta, "epoch": epoch, "steps": step})
        else:
            bad_epochs += 1
            if bad_epochs >= int(hp["early_stopping_patience"]):
                LOGGER.info("early stopping after epoch %d", epoch)
                break
        if stop:
            break
    summary = {"history": history, "best_checkpoint": str(best_path),
               "best_checkpoint_sha256_16": checkpoint_fingerprint(str(best_path)),
               "num_fit_examples": len(fit_examples), "num_tune_examples": len(tune_examples)}
    with open(out / "training_log.json", "w", encoding="utf-8") as handle:
        json.dump({**summary, "metadata": meta}, handle, indent=2, ensure_ascii=False, default=str)
    return summary


def load_rows(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_examples(pipe: GracePipeline, rows: List[dict], language: str, repos_root: Optional[str], k_c: int,
                     lappe_dim: int) -> List[TrainingExample]:
    examples, skipped = [], 0
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
        for row in rows:
            tmp.write(json.dumps(row) + "\n")
        tmp_path = tmp.name
    tasks = load_tasks("grace_tasks", tmp_path, language, repos_root)
    Path(tmp_path).unlink(missing_ok=True)
    for task, row in zip(tasks, rows):
        ex = build_example(pipe, task, row.get("positive_units"), k_c, lappe_dim)
        if ex is None:
            skipped += 1
        else:
            examples.append(ex)
    LOGGER.info("prepared %d training examples (%d skipped: no units or no positives)", len(examples), skipped)
    return examples


def main(argv: Optional[Sequence[str]] = None, text_encoder: Optional[TextEncoder] = None) -> dict:
    parser = argparse.ArgumentParser(description="Train the GRACE structural encoder (InfoNCE)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[], help="override key=value")
    parser.add_argument("--language", required=True, choices=["python", "java"])
    parser.add_argument("--train-queries", required=True, help="grace_tasks JSONL for development repositories")
    parser.add_argument("--repos-root", default=None)
    parser.add_argument("--fit-repos", required=True, help="file listing fitting repositories")
    parser.add_argument("--tune-repos", required=True, help="file listing tuning repositories")
    parser.add_argument("--test-repos", required=True, nargs="+",
                        help="repository lists or benchmark files whose repositories must be excluded")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=None, help="stop after N optimizer steps (smoke tests)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config, args.set)
    # Fail before the (expensive) example preparation if a training value is unset.
    cfg.require_all(TRAINING_KEYS)
    EncoderSpec.from_config(cfg)
    device = resolve_device(cfg.get("device"))
    fit_repos, tune_repos = read_repo_list(args.fit_repos), read_repo_list(args.tune_repos)
    test_repos: List[str] = []
    for path in args.test_repos:
        test_repos.extend(repo_names_from_file(path))
    assert_project_disjoint({"fit": fit_repos, "tune": tune_repos, "test": test_repos})

    rows = load_rows(args.train_queries)
    allowed_fit, allowed_tune = set(fit_repos), set(tune_repos)
    unknown = sorted({r["repo_name"] for r in rows} - allowed_fit - allowed_tune)
    if unknown:
        raise ValueError(f"Training queries reference repositories outside the fit/tune lists: {unknown[:5]}")
    assert_project_disjoint({"train_queries": {r["repo_name"] for r in rows}, "test": test_repos})

    if text_encoder is None:
        from .graph_features import build_text_encoder

        text_encoder = build_text_encoder(cfg, device)
    pipe = GracePipeline(cfg, text_encoder, model=None, device=device, index_backend="exact")
    k_c, lappe = int(cfg.require("candidate_pool_size_Kc")), int(cfg.require("laplacian_pe_dim"))
    fit_rows = [r for r in rows if r["repo_name"] in allowed_fit]
    tune_rows = [r for r in rows if r["repo_name"] in allowed_tune]
    fit = prepare_examples(pipe, fit_rows, args.language, args.repos_root, k_c, lappe)
    tune = prepare_examples(pipe, tune_rows, args.language, args.repos_root, k_c, lappe)
    if not fit:
        raise ValueError("No usable fitting examples (no positives found)")
    sources = sorted({ex.positive_source for ex in fit + tune})
    return train(cfg, fit, tune, args.output_dir, device=device, max_steps=args.max_steps,
                 metadata={"fit_repos": fit_repos, "tune_repos": tune_repos, "positive_sources": sources,
                           "text_encoder": text_encoder.name, "language": args.language})


def repo_names_from_file(path: str) -> List[str]:
    """Read repository names from a plain list or from a benchmark JSONL file."""
    if not path.endswith(".jsonl"):
        return read_repo_list(path)
    names = set()
    for row in load_rows(path):
        meta = row.get("metadata", {})
        if "repository" in meta:
            names.add(meta["repository"])
        elif "fpath_tuple" in meta:
            names.add(meta["fpath_tuple"][0])
        elif "repo_name" in row:
            names.add(row["repo_name"])
    return sorted(names)


if __name__ == "__main__":
    main()
