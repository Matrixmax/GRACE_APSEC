"""End-to-end GRACE pipeline for one completion task.

prepare (mask -> multi-level graph -> G_q -> retrieval units) -> index (CodeT5p + GAT)
-> hybrid retrieval (gate alpha, MMR, K_c, K_f) -> fusion (gamma, theta, align) -> serialization
-> token-budgeted prompt.  Generation is separate (see cli.py ``generate``).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .config import GraceConfig
from .data import CompletionTask
from .encoder import StructuralRetrievalModel, graph_to_arrays
from .fusion import RetrievedSubgraph, fuse_graphs
from .graph_builder import build_query_graph, build_repo_graph, read_graphml, write_graphml
from .graph_features import TextEncoder, node_features
from .index import TaskIndex, build_task_index
from .leakage import MaskedSnapshot, make_masked_snapshot, read_repo_sources
from .prompt import (LocalContext, RetrievedBlock, TokenCounter, build_retrieved_context,
                     build_token_counter, render_prompt, snippet_header, truncate_local_context)
from .retrieve import RetrievalResult, hybrid_retrieve
from .subgraphs import RetrievalUnit, extract_units

LOGGER = logging.getLogger(__name__)


@dataclass
class PreparedTask:
    task: CompletionTask
    snapshot: MaskedSnapshot
    graph: nx.MultiDiGraph
    local: LocalContext
    window: Tuple[int, int]
    query_graph: nx.MultiDiGraph
    query_nodes: List[str]
    units: List[RetrievalUnit]


@dataclass
class QueryEncoding:
    v_q: np.ndarray
    node_emb: np.ndarray
    h_gq: np.ndarray
    alpha: float
    alpha_mode: str


class RepoSourceCache:
    """Small LRU cache of repository sources so tasks of the same repository reuse file reads."""

    def __init__(self, capacity: int = 4):
        self.capacity = capacity
        self.data: "OrderedDict[Tuple[str, str], Dict[str, str]]" = OrderedDict()

    def get(self, repo_dir: Optional[str], language: str) -> Dict[str, str]:
        key = (str(repo_dir), language)
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        files = read_repo_sources(repo_dir, language)
        self.data[key] = files
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)
        return files


def local_window(snapshot: MaskedSnapshot, local: LocalContext) -> Tuple[int, int]:
    last = max(snapshot.visible_line_count, 1)
    first = max(last - max(local.kept_lines, 1) + 1, 1)
    return first, last


class GracePipeline:
    def __init__(self, cfg: GraceConfig, text_encoder: TextEncoder, model: Optional[StructuralRetrievalModel],
                 device: str = "cpu", index_backend: str = "hnsw", token_counter: Optional[TokenCounter] = None,
                 checkpoint_id: Optional[str] = None):
        self.cfg = cfg
        self.text_encoder = text_encoder
        self.model = model
        self.device = device
        self.index_backend = index_backend
        self.checkpoint_id = checkpoint_id
        self.sources = RepoSourceCache()
        self._counter = token_counter

    # -- configuration helpers ----------------------------------------------------------------
    @property
    def counter(self) -> TokenCounter:
        if self._counter is None:
            self._counter = build_token_counter(self.cfg.require("prompt_tokenizer"))
        return self._counter

    def hnsw_params(self) -> Dict[str, int]:
        if self.index_backend != "hnsw":
            return {}
        return {
            "M": int(self.cfg.require("hnsw_M")),
            "ef_construction": int(self.cfg.require("hnsw_ef_construction")),
            "ef_search": int(self.cfg.require("hnsw_ef_search")),
        }

    def _check_budgets(self) -> Tuple[int, int]:
        total = int(self.cfg.require("context_tokens"))
        local = int(self.cfg.require("local_context_tokens"))
        retrieved = int(self.cfg.require("retrieved_context_tokens"))
        if local + retrieved > total:
            raise ValueError("local_context_tokens + retrieved_context_tokens exceeds context_tokens")
        return local, retrieved

    # -- stage 1: masking, graph, query graph, units ------------------------------------------
    def prepare(self, task: CompletionTask, graph_cache_dir: Optional[str] = None) -> PreparedTask:
        files = self.sources.get(task.repo_dir, task.language)
        snapshot = make_masked_snapshot(task, files)
        graph = None
        cache_file = Path(graph_cache_dir) / task.safe_key / "repo_graph.graphml" if graph_cache_dir else None
        if cache_file is not None and cache_file.exists():
            graph = read_graphml(str(cache_file))
        if graph is None:
            graph = build_repo_graph(snapshot.files, task.language)
            if cache_file is not None:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                write_graphml(graph, str(cache_file))
        local_budget, _ = self._check_budgets()
        local = truncate_local_context(snapshot.visible_text, self.counter, local_budget)
        window = local_window(snapshot, local)
        query_graph = build_query_graph(graph, task.file_path, *window)
        units = extract_units(graph, snapshot.files, int(self.cfg.require("subgraph_hops")),
                              current_file=task.file_path, query_window=window)
        return PreparedTask(task, snapshot, graph, local, window, query_graph, sorted(query_graph.nodes()), units)

    # -- stage 2: index -----------------------------------------------------------------------
    def _require_model(self) -> StructuralRetrievalModel:
        if self.model is None:
            raise RuntimeError("A trained structural encoder checkpoint is required (see train_encoder.py)")
        return self.model

    def build_index(self, prepared: PreparedTask) -> TaskIndex:
        return build_task_index(prepared.graph, prepared.units, self.text_encoder, self._require_model(),
                                int(self.cfg.require("laplacian_pe_dim")), backend=self.index_backend,
                                hnsw_params=self.hnsw_params(), device=self.device)

    # -- stage 3: query encoding and retrieval ------------------------------------------------
    def encode_query(self, prepared: PreparedTask) -> QueryEncoding:
        model = self._require_model()
        v_q = self.text_encoder.encode([prepared.local.text], truncation_side="left")[0]
        x = node_features(prepared.query_graph, self.text_encoder, nodes=prepared.query_nodes,
                          lappe_dim=int(self.cfg.require("laplacian_pe_dim")))
        ei, et = graph_to_arrays(prepared.query_graph, prepared.query_nodes)
        node_emb, graph_emb = model.encode_numpy([(x, ei, et)], device=self.device)
        h_gq = graph_emb[0]
        mode = self.cfg.require("alpha_mode")
        if mode == "learned_gate":
            alpha = model.alpha_numpy(v_q, h_gq, device=self.device)
        elif mode == "fixed":
            alpha = float(self.cfg.require("fixed_alpha"))  # explicit ablation only
        else:
            raise ValueError("alpha_mode must be 'learned_gate' or 'fixed'")
        return QueryEncoding(v_q=v_q, node_emb=node_emb[0], h_gq=h_gq, alpha=alpha, alpha_mode=mode)

    def retrieve(self, index: TaskIndex, query: QueryEncoding) -> RetrievalResult:
        return hybrid_retrieve(index, query.v_q, query.h_gq, query.alpha, query.alpha_mode,
                               k_c=int(self.cfg.require("candidate_pool_size_Kc")),
                               k_f=int(self.cfg.require("fused_subgraphs_Kf")),
                               mmr_lambda=float(self.cfg.require("mmr_lambda")))

    # -- stage 4: fusion and prompt -----------------------------------------------------------
    def fuse(self, prepared: PreparedTask, index: TaskIndex, query: QueryEncoding, result: RetrievalResult):
        retrieved = [
            RetrievedSubgraph(unit_id=c.unit_id, score=c.score, graph=index.unit_graphs[c.index],
                              nodes=list(index.units[c.index].nodes), node_emb=index.node_emb[c.index])
            for c in result.selected
        ]
        return fuse_graphs(prepared.query_graph, prepared.query_nodes, query.node_emb, retrieved,
                           gamma=float(self.cfg.require("gamma")), theta=float(self.cfg.require("theta")),
                           type_rule=str(self.cfg.require("type_compatibility")))

    def build_prompt(self, prepared: PreparedTask, index: TaskIndex, result: RetrievalResult, fusion) -> dict:
        _, retrieved_budget = self._check_budgets()
        blocks = []
        for rank, cand in enumerate(result.selected):
            unit = index.units[cand.index]
            blocks.append(RetrievedBlock(snippet_header(prepared.task.language, rank, unit), unit.text, unit.nodes))
        ctx = build_retrieved_context(fusion.graph, prepared.query_nodes, blocks, self.counter, retrieved_budget)
        prompt = render_prompt(
            {
                "repo_name": prepared.task.repo_name,
                "current_file_path": prepared.task.file_path,
                "code_before_cursor": prepared.local.text,
                "code_context": ctx.code_context,
                "graph_context": ctx.graph_context,
            }
        )
        return {
            "prompt": prompt,
            "code_context_block": ctx.code_context,
            "graph_context": ctx.graph_context,
            "graph_id_map": {str(k): v for k, v in ctx.id_map.items()},
            "prompt_stats": {
                "tokenizer": self.counter.name,
                "local_tokens": prepared.local.tokens,
                "local_budget": int(self.cfg.require("local_context_tokens")),
                "local_lines_kept": prepared.local.kept_lines,
                "local_lines_dropped": prepared.local.dropped_lines,
                "local_truncated": prepared.local.truncated,
                "retrieved_tokens": ctx.tokens,
                "retrieved_budget": retrieved_budget,
                **{f"retrieved_{k}": v for k, v in ctx.stats.items() if k != "tokens"},
            },
        }

    # -- full task ----------------------------------------------------------------------------
    def run_task(self, task: CompletionTask, graph_cache_dir: Optional[str] = None,
                 index_dir: Optional[str] = None) -> dict:
        prepared = self.prepare(task, graph_cache_dir=graph_cache_dir)
        index = None
        task_index_dir = Path(index_dir) / task.safe_key if index_dir else None
        if task_index_dir is not None and (task_index_dir / "meta.json").exists():
            if index_meta(task_index_dir).get("code_stripped"):
                raise RuntimeError(f"{task_index_dir} was packaged without source code; run "
                                   "scripts/package_indexes.py hydrate against the local repositories first")
            index = TaskIndex.load(str(task_index_dir))
            if self.checkpoint_id and index_meta(task_index_dir).get("checkpoint") not in (None, self.checkpoint_id):
                LOGGER.warning("Index %s was built with another checkpoint; rebuilding", task_index_dir)
                index = None
        if index is None:
            index = self.build_index(prepared)
            if task_index_dir is not None:
                index.save(str(task_index_dir), metadata={"task_id": task.task_id, "checkpoint": self.checkpoint_id,
                                                          "text_encoder": self.text_encoder.name})
        query = self.encode_query(prepared)
        result = self.retrieve(index, query)
        fusion = self.fuse(prepared, index, query, result)
        prompt_info = self.build_prompt(prepared, index, result, fusion)
        snippets = []
        for rank, cand in enumerate(result.selected):
            unit = index.units[cand.index]
            snippets.append({"rank": rank, "unit_id": unit.unit_id, "file_path": unit.file_path,
                             "start_line": unit.start_line, "end_line": unit.end_line, "code": unit.text,
                             "score": cand.score})
        return {
            "task_id": task.task_id,
            "dataset": task.dataset,
            "language": task.language,
            "repo_name": task.repo_name,
            "file_path": task.file_path,
            # code_context/groundtruth keep the field names read by model/grace_eval.py.
            "code_context": task.prefix,
            "groundtruth": task.groundtruth,
            "local_context": prepared.local.text,
            "cursor_located": prepared.snapshot.cursor_located,
            "query_window": list(prepared.window),
            "query_graph_size": [prepared.query_graph.number_of_nodes(), prepared.query_graph.number_of_edges()],
            "num_index_units": len(index),
            "retrieval": result.to_dict(),
            "retrieved_snippets": snippets,
            "fusion": fusion.stats(),
            **prompt_info,
            "config_name": self.cfg.name,
            "text_encoder": self.text_encoder.name,
            "structural_checkpoint": self.checkpoint_id,
        }


def index_meta(directory: Path) -> dict:
    import json

    try:
        with open(Path(directory) / "meta.json", "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}
