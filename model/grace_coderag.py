"""Compatibility entry point for the former ``model/grace_coderag.py`` script.

The original script is archived in ``legacy/grace_coderag_original_release.py`` (it imports ``dgl``).

This script keeps the old command line shape (``--model --dataset --language``) and old default
directories, and runs the pipeline in ``grace/``:

    grace.cli retrieve  ->  <dataset-root>/grace_search/line_search_results_<lang>_<model>.jsonl
    grace.cli generate  ->  results/<dataset>/<language>/grace/prediction.jsonl

Differences from the archived script:

* ``--checkpoint`` (trained structural encoder, see ``python -m grace.train_encoder``) is required.
* ``--k`` and ``--alpha`` default to ``None``: the config values are used (paper: K_f=3, learned gate).
  Passing ``--k`` overrides ``fused_subgraphs_Kf``; passing ``--alpha`` switches to the explicit
  fixed-alpha ablation (``alpha_mode=fixed``).
* ``--fusion-type`` only accepts ``attention`` (the fusion described in the paper).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace import cli  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GRACE pipeline (compatibility wrapper around grace.cli)")
    parser.add_argument("--model", required=True, help="model key or alias from llm.models (e.g. gpt-4o-mini)")
    parser.add_argument("--dataset", default="crosscodeeval",
                        choices=["crosscodeeval", "repoeval_updated", "repoeval-updated"])
    parser.add_argument("--language", default="python", choices=["python", "java"])
    parser.add_argument("--checkpoint", required=True, help="structural encoder checkpoint (best.pt)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[], help="config override key=value (repeatable)")
    parser.add_argument("--dataset-root", default=None, help="default: dataset/<dataset>")
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--repos-root", default=None, help="default: <dataset-root>/repos")
    parser.add_argument("--results-root", default=None, help="default: results/<dataset>/<language>/grace")
    parser.add_argument("--index-backend", default="hnsw", choices=["hnsw", "exact"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fusion-type", default="attention", choices=["attention"])
    parser.add_argument("--k", type=int, default=None, help="override K_f (paper config: 3)")
    parser.add_argument("--alpha", type=float, default=None, help="fixed-alpha ablation (default: learned gate)")
    parser.add_argument("--device", default=None, help="override implementation.device (auto|cpu|cuda|cuda:N)")
    return parser


def translate(args: argparse.Namespace) -> List[List[str]]:
    """Return the ``grace.cli`` argument lists for the retrieve and generate stages."""
    dataset = "repoeval_updated" if args.dataset == "repoeval-updated" else args.dataset
    dataset_root = Path(args.dataset_root or Path("dataset") / dataset)
    repos_root = Path(args.repos_root or dataset_root / "repos")
    results_root = Path(args.results_root or Path("results") / dataset / args.language / "grace")
    retrieval = dataset_root / "grace_search" / f"line_search_results_{args.language}_{args.model}.jsonl"

    overrides: List[str] = list(args.set)
    if args.k is not None:
        overrides.append(f"fused_subgraphs_Kf={args.k}")
    if args.alpha is not None:
        overrides += ["alpha_mode=fixed", f"fixed_alpha={args.alpha}"]
    if args.device is not None:
        overrides.append(f"device={args.device}")
    common: List[str] = []
    if args.config:
        common += ["--config", args.config]
    for item in overrides:
        common += ["--set", item]

    retrieve = ["retrieve", *common, "--dataset", dataset, "--language", args.language,
                "--dataset-root", str(dataset_root), "--repos-root", str(repos_root),
                "--checkpoint", args.checkpoint, "--graph-dir", str(dataset_root / "graphs"),
                "--index-dir", str(dataset_root / "index"), "--index-backend", args.index_backend,
                "--output", str(retrieval)]
    if args.data_file:
        retrieve += ["--data-file", args.data_file]
    if args.limit:
        retrieve += ["--limit", str(args.limit)]
    generate = ["generate", *common, "--retrieval", str(retrieval), "--model", args.model,
                "--output-dir", str(results_root)]
    return [retrieve, generate]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    for stage in translate(args):
        code = cli.main(stage)
        if code:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
