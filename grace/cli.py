"""Command line entry points.

    python -m grace.cli build-graphs --dataset ... --language ... --data-file ... --repos-root ... --out ...
    python -m grace.cli index        ... --checkpoint best.pt --index-dir ...
    python -m grace.cli retrieve     ... --checkpoint best.pt --output retrieval.jsonl
    python -m grace.cli generate     --retrieval retrieval.jsonl --model gpt-4o-mini --output-dir results/...

Training: ``python -m grace.train_encoder`` (see README). This module never computes metrics; use
model/grace_eval.py on the generated prediction.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .config import GraceConfig, load_config, resolve_device
from .data import canonical_dataset, default_data_file, load_tasks, write_jsonl
from .encoder import checkpoint_fingerprint, load_checkpoint
from .graph_builder import build_repo_graph, write_graphml
from .leakage import make_masked_snapshot, read_repo_sources
from .llm import LLMRequest, OpenAICompatibleClient, build_messages, decoding_params, parse_completion, utc_now
from .pipeline import GracePipeline

LOGGER = logging.getLogger("grace")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="YAML config (default: configs/paper_reproduction.yaml)")
    p.add_argument("--set", action="append", default=[], help="override, e.g. --set gamma=0.2 (repeatable)")


def _task_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", required=True, help="crosscodeeval | repoeval_updated | grace_tasks")
    p.add_argument("--language", required=True, choices=["python", "java"])
    p.add_argument("--data-file", default=None, help="benchmark JSONL; default derived from --dataset-root")
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--repos-root", required=True, help="directory with one repository snapshot per repo")
    p.add_argument("--limit", type=int, default=None, help="process only the first N tasks")


def _load_tasks(args):
    data_file = args.data_file or default_data_file(args.dataset, args.dataset_root or ".", args.language)
    tasks = load_tasks(args.dataset, data_file, args.language, args.repos_root)
    return tasks[: args.limit] if args.limit else tasks


def _pipeline(cfg: GraceConfig, args, need_model: bool = True, text_encoder=None) -> GracePipeline:
    device = resolve_device(cfg.get("device"))
    model, ckpt_id = None, None
    if need_model:
        model, _meta = load_checkpoint(args.checkpoint, map_location=device)
        ckpt_id = checkpoint_fingerprint(args.checkpoint)
    if text_encoder is None:
        from .graph_features import build_text_encoder

        text_encoder = build_text_encoder(cfg, device)
    return GracePipeline(cfg, text_encoder, model, device=device, index_backend=args.index_backend,
                         checkpoint_id=ckpt_id)


def cmd_build_graphs(args, cfg: GraceConfig) -> int:
    out = Path(args.out)
    sources = {}
    count = 0
    for task in _load_tasks(args):
        key = (task.repo_dir, task.language)
        if key not in sources:
            sources = {key: read_repo_sources(task.repo_dir, task.language)}
        snapshot = make_masked_snapshot(task, sources[key])
        graph = build_repo_graph(snapshot.files, task.language)
        target = out / task.safe_key / "repo_graph.graphml"
        target.parent.mkdir(parents=True, exist_ok=True)
        write_graphml(graph, str(target))
        count += 1
    LOGGER.info("wrote %d masked task graphs to %s", count, out)
    return 0


def cmd_index(args, cfg: GraceConfig, text_encoder=None) -> int:
    pipe = _pipeline(cfg, args, text_encoder=text_encoder)
    for task in _load_tasks(args):
        prepared = pipe.prepare(task, graph_cache_dir=args.graph_dir)
        index = pipe.build_index(prepared)
        index.save(str(Path(args.index_dir) / task.safe_key),
                   metadata={"task_id": task.task_id, "checkpoint": pipe.checkpoint_id,
                             "text_encoder": pipe.text_encoder.name, "config_name": cfg.name})
    return 0


def cmd_retrieve(args, cfg: GraceConfig, text_encoder=None) -> int:
    pipe = _pipeline(cfg, args, text_encoder=text_encoder)
    rows = []
    for task in _load_tasks(args):
        rows.append(pipe.run_task(task, graph_cache_dir=args.graph_dir, index_dir=args.index_dir))
    write_jsonl(args.output, rows)
    with open(Path(args.output).with_suffix(".config.json"), "w", encoding="utf-8") as handle:
        json.dump(cfg.snapshot(), handle, indent=2, ensure_ascii=False, default=str)
    LOGGER.info("wrote %d retrieval records to %s", len(rows), args.output)
    return 0


def cmd_generate(args, cfg: GraceConfig, transport=None) -> int:
    model = cfg.resolve_model(args.model)
    params = decoding_params(cfg)
    system_prompt = cfg.require("system_prompt")
    policy = cfg.require("parse_failure_policy")
    on_error = cfg.require("on_api_error")
    client = OpenAICompatibleClient(model["base_url"], model["api_key_env"],
                                    max_retries=int(cfg.require("llm_max_retries")),
                                    backoff_seconds=float(cfg.require("llm_retry_backoff_seconds")),
                                    timeout_seconds=float(cfg.require("llm_timeout_seconds")),
                                    transport=transport)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.retrieval, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    preds, failures = [], []
    for row in rows:
        request = LLMRequest(model_id=model["model_id"], messages=build_messages(row["prompt"], system_prompt),
                             params=params)
        started = utc_now()
        response = client.complete(request)
        record = {
            "task_id": row["task_id"],
            "code_context": row["code_context"],
            "groundtruth": row["groundtruth"],
            "provider": model["provider"],
            "base_url": model["base_url"],
            "model": model["name"],
            "model_id": model["model_id"],
            "request_params": params,
            "system_prompt": system_prompt,
            "request_started_utc": started,
            "request_finished_utc": utc_now(),
            "attempts": response.attempts,
            "response_meta": response.raw_meta,
            "config_name": cfg.name,
        }
        if not response.ok:
            record["api_error"] = response.error
            failures.append(record)
            if on_error == "skip":
                continue
            if on_error != "record_empty":
                raise ValueError("on_api_error must be 'skip' or 'record_empty'")
            record.update({"pred": "", "pred_raw": None, "parse_status": "api_error"})
        else:
            parsed = parse_completion(response.text, policy)
            record.update({
                "pred": parsed.completed_code,
                "pred_raw": response.text,
                "parse_status": parsed.parse_status,
                "parse_error": parsed.parse_error,
                "explanation": parsed.explanation,
                "confidence_score": parsed.confidence_score,
                "referenced_nodes": parsed.referenced_nodes,
            })
        preds.append(record)
    write_jsonl(str(out_dir / "prediction.jsonl"), preds)
    write_jsonl(str(out_dir / "failures.jsonl"), failures)
    LOGGER.info("wrote %d predictions (%d API failures) to %s", len(preds), len(failures), out_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grace")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-graphs", help="leakage-masked multi-level graph per task")
    _common(p)
    _task_args(p)
    p.add_argument("--out", required=True)

    for name in ("index", "retrieve"):
        p = sub.add_parser(name)
        _common(p)
        _task_args(p)
        p.add_argument("--checkpoint", required=True, help="structural encoder checkpoint (train_encoder.py)")
        p.add_argument("--graph-dir", default=None, help="cache of masked graphs (build-graphs output)")
        p.add_argument("--index-dir", default=None, required=name == "index", help="per-task index cache")
        p.add_argument("--index-backend", default="hnsw", choices=["hnsw", "exact"])
        if name == "retrieve":
            p.add_argument("--output", required=True)

    p = sub.add_parser("generate", help="query the LLM with prompts from a retrieval file")
    _common(p)
    p.add_argument("--retrieval", required=True)
    p.add_argument("--model", required=True, help="model key or alias from llm.models")
    p.add_argument("--output-dir", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None, text_encoder=None, transport=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.set)
    if args.command == "build-graphs":
        return cmd_build_graphs(args, cfg)
    if args.command == "index":
        return cmd_index(args, cfg, text_encoder=text_encoder)
    if args.command == "retrieve":
        return cmd_retrieve(args, cfg, text_encoder=text_encoder)
    if args.command == "generate":
        return cmd_generate(args, cfg, transport=transport)
    return 1


if __name__ == "__main__":
    sys.exit(main())
