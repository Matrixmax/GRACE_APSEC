"""Structural encoder f_theta (L-layer GAT) and the learnable balance gate.

Implemented in plain PyTorch (no PyG/DGL dependency).

Edge-type handling is configurable (``gat_edge_type_mode``):

* ``homogeneous`` - edge relations are ignored; standard GAT attention (Velickovic et al.).
* ``relation_attention_bias`` - a learned scalar per (relation, head) is added to the attention logit
  of every edge with that relation, so typed edges can be weighted differently.

Self-loops are always added (relation ``self``). ``gat_add_reverse_edges`` adds the reverse of every
edge with the same relation so that information flows in both directions.

Graph embedding h_G is mean- or max-pooling over node embeddings (``graph_pooling``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_builder import RELATIONS, relation_id
from .graph_features import CODE_EMBEDDING_DIM, NODE_FEATURE_DIM

EDGE_TYPE_MODES = ("homogeneous", "relation_attention_bias")
POOLING_MODES = ("mean", "max")
CHECKPOINT_FORMAT = "grace-structural-encoder-v1"


@dataclass
class EncoderSpec:
    in_dim: int
    hidden_dim: int
    layers: int
    heads: int
    dropout: float
    edge_type_mode: str
    add_reverse_edges: bool
    pooling: str
    query_dim: int = CODE_EMBEDDING_DIM
    num_relations: int = len(RELATIONS)

    def validate(self) -> None:
        if self.in_dim != NODE_FEATURE_DIM:
            raise ValueError(f"in_dim must be {NODE_FEATURE_DIM} (256 CodeT5p + 5 LapPE), got {self.in_dim}")
        if self.layers < 1 or self.heads < 1 or self.hidden_dim < 1:
            raise ValueError("gat_layers, gat_heads and gat_hidden_dim must be positive")
        if self.hidden_dim % self.heads != 0:
            raise ValueError("gat_hidden_dim must be divisible by gat_heads")
        if self.edge_type_mode not in EDGE_TYPE_MODES:
            raise ValueError(f"gat_edge_type_mode must be one of {EDGE_TYPE_MODES}")
        if self.pooling not in POOLING_MODES:
            raise ValueError(f"graph_pooling must be one of {POOLING_MODES}")

    @classmethod
    def from_config(cls, cfg) -> "EncoderSpec":
        values = cfg.require_all(
            [
                "gat_hidden_dim",
                "gat_layers",
                "gat_heads",
                "gat_dropout",
                "gat_edge_type_mode",
                "gat_add_reverse_edges",
                "graph_pooling",
            ]
        )
        spec = cls(
            in_dim=cfg.require("code_embedding_dim") + cfg.require("laplacian_pe_dim"),
            hidden_dim=int(values["gat_hidden_dim"]),
            layers=int(values["gat_layers"]),
            heads=int(values["gat_heads"]),
            dropout=float(values["gat_dropout"]),
            edge_type_mode=str(values["gat_edge_type_mode"]),
            add_reverse_edges=bool(values["gat_add_reverse_edges"]),
            pooling=str(values["graph_pooling"]),
            query_dim=int(cfg.require("code_embedding_dim")),
        )
        spec.validate()
        return spec


# --------------------------------------------------------------------------------------------------
# Graph tensors
# --------------------------------------------------------------------------------------------------

@dataclass
class GraphTensors:
    x: torch.Tensor  # (N, in_dim)
    edge_index: torch.Tensor  # (2, E) long, src -> dst
    edge_type: torch.Tensor  # (E,) long
    batch: torch.Tensor  # (N,) long graph id
    num_graphs: int

    def to(self, device) -> "GraphTensors":
        return GraphTensors(self.x.to(device), self.edge_index.to(device), self.edge_type.to(device),
                            self.batch.to(device), self.num_graphs)


def graph_to_arrays(graph: nx.Graph, nodes: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(edge_index (2,E), edge_type (E,))`` for ``graph`` in ``nodes`` order."""
    index = {n: i for i, n in enumerate(nodes)}
    src, dst, rel = [], [], []
    if graph.is_multigraph():
        edges = graph.edges(keys=True, data=True)
        items = ((u, v, d.get("relation", k)) for u, v, k, d in edges)
    else:
        items = ((u, v, d.get("relation", "other")) for u, v, d in graph.edges(data=True))
    for u, v, r in items:
        if u in index and v in index and u != v:
            src.append(index[u])
            dst.append(index[v])
            rel.append(relation_id(str(r)))
    return np.array([src, dst], dtype=np.int64).reshape(2, -1), np.array(rel, dtype=np.int64)


def collate(items: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> GraphTensors:
    """Batch ``(features, edge_index, edge_type)`` triples into one disjoint graph."""
    xs, eis, ets, batch = [], [], [], []
    offset = 0
    for gid, (x, ei, et) in enumerate(items):
        n = x.shape[0]
        if n == 0:
            # Empty graphs get one zero placeholder node so that pooling stays defined.
            x = np.zeros((1, x.shape[1] if x.ndim == 2 else NODE_FEATURE_DIM), dtype=np.float32)
            ei = np.zeros((2, 0), dtype=np.int64)
            et = np.zeros((0,), dtype=np.int64)
            n = 1
        xs.append(torch.as_tensor(x, dtype=torch.float32))
        eis.append(torch.as_tensor(ei, dtype=torch.long) + offset)
        ets.append(torch.as_tensor(et, dtype=torch.long))
        batch.append(torch.full((n,), gid, dtype=torch.long))
        offset += n
    return GraphTensors(
        x=torch.cat(xs, 0),
        edge_index=torch.cat(eis, 1) if eis else torch.zeros((2, 0), dtype=torch.long),
        edge_type=torch.cat(ets, 0) if ets else torch.zeros((0,), dtype=torch.long),
        batch=torch.cat(batch, 0),
        num_graphs=len(items),
    )


# --------------------------------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------------------------------

def _segment_softmax(logits: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Softmax of ``logits`` (E, H) within groups given by ``index`` (E,)."""
    heads = logits.shape[1]
    seg_max = torch.full((num_segments, heads), float("-inf"), device=logits.device, dtype=logits.dtype)
    seg_max = seg_max.scatter_reduce(0, index.unsqueeze(1).expand(-1, heads), logits, reduce="amax", include_self=True)
    exp = torch.exp(logits - seg_max[index])
    denom = torch.zeros((num_segments, heads), device=logits.device, dtype=logits.dtype).index_add_(0, index, exp)
    return exp / denom[index].clamp_min(1e-16)


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int, dropout: float, edge_type_mode: str,
                 num_relations: int, concat: bool = True):
        super().__init__()
        self.heads = heads
        self.concat = concat
        self.head_dim = out_dim // heads if concat else out_dim
        self.lin = nn.Linear(in_dim, self.heads * self.head_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(heads * self.head_dim if concat else self.head_dim))
        self.edge_type_mode = edge_type_mode
        self.rel_bias = nn.Parameter(torch.zeros(num_relations, heads)) if edge_type_mode == "relation_attention_bias" else None
        self.dropout = dropout
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        h = self.lin(x).view(n, self.heads, self.head_dim)
        src, dst = edge_index[0], edge_index[1]
        logits = (h[src] * self.att_src).sum(-1) + (h[dst] * self.att_dst).sum(-1)
        if self.rel_bias is not None:
            logits = logits + self.rel_bias[edge_type]
        logits = F.leaky_relu(logits, 0.2)
        att = _segment_softmax(logits, dst, n)
        att = F.dropout(att, p=self.dropout, training=self.training)
        out = torch.zeros_like(h).index_add_(0, dst, h[src] * att.unsqueeze(-1))
        out = out.reshape(n, -1) if self.concat else out.mean(dim=1)
        return out + self.bias


class GATEncoder(nn.Module):
    """f_theta: node features (N, 261) -> node embeddings (N, d) and pooled graph embeddings (B, d)."""

    def __init__(self, spec: EncoderSpec):
        super().__init__()
        spec.validate()
        self.spec = spec
        self.layers = nn.ModuleList()
        dim = spec.in_dim
        for i in range(spec.layers):
            last = i == spec.layers - 1
            self.layers.append(
                GATLayer(dim, spec.hidden_dim, spec.heads, spec.dropout, spec.edge_type_mode, spec.num_relations,
                         concat=not last)
            )
            dim = spec.hidden_dim
        self.self_rel = relation_id("self")

    def _prepare_edges(self, g: GraphTensors) -> Tuple[torch.Tensor, torch.Tensor]:
        ei, et = g.edge_index, g.edge_type
        if self.spec.add_reverse_edges and ei.numel():
            ei = torch.cat([ei, ei.flip(0)], dim=1)
            et = torch.cat([et, et], dim=0)
        n = g.x.shape[0]
        loops = torch.arange(n, device=g.x.device)
        ei = torch.cat([ei, torch.stack([loops, loops])], dim=1)
        et = torch.cat([et, torch.full((n,), self.self_rel, dtype=torch.long, device=g.x.device)], dim=0)
        return ei, et

    def forward(self, g: GraphTensors) -> Tuple[torch.Tensor, torch.Tensor]:
        if g.x.shape[1] != self.spec.in_dim:
            raise ValueError(f"node features have dim {g.x.shape[1]}, expected {self.spec.in_dim}")
        ei, et = self._prepare_edges(g)
        h = g.x
        for i, layer in enumerate(self.layers):
            h = F.dropout(h, p=self.spec.dropout, training=self.training)
            h = layer(h, ei, et)
            if i < len(self.layers) - 1:
                h = F.elu(h)
        return h, self.pool(h, g.batch, g.num_graphs)

    def pool(self, h: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        d = h.shape[1]
        if self.spec.pooling == "mean":
            total = torch.zeros((num_graphs, d), device=h.device, dtype=h.dtype).index_add_(0, batch, h)
            count = torch.zeros((num_graphs,), device=h.device, dtype=h.dtype).index_add_(
                0, batch, torch.ones_like(batch, dtype=h.dtype))
            return total / count.clamp_min(1.0).unsqueeze(1)
        out = torch.full((num_graphs, d), float("-inf"), device=h.device, dtype=h.dtype)
        out = out.scatter_reduce(0, batch.unsqueeze(1).expand(-1, d), h, reduce="amax", include_self=True)
        return torch.where(torch.isinf(out), torch.zeros_like(out), out)


class AlphaGate(nn.Module):
    """alpha = sigmoid(W_alpha [v_q ; h_Gq] + b_alpha), always in [0, 1]."""

    def __init__(self, query_dim: int, graph_dim: int):
        super().__init__()
        self.linear = nn.Linear(query_dim + graph_dim, 1)

    def forward(self, v_q: torch.Tensor, h_gq: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(torch.cat([v_q, h_gq], dim=-1))).squeeze(-1)


class StructuralRetrievalModel(nn.Module):
    """GAT encoder + alpha gate, trained jointly (see train_encoder.py)."""

    def __init__(self, spec: EncoderSpec):
        super().__init__()
        self.spec = spec
        self.encoder = GATEncoder(spec)
        self.gate = AlphaGate(spec.query_dim, spec.hidden_dim)

    def encode(self, g: GraphTensors) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(g)

    def alpha(self, v_q: torch.Tensor, h_gq: torch.Tensor) -> torch.Tensor:
        return self.gate(v_q, h_gq)

    def encode_numpy(self, items: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]], device: str = "cpu",
                     batch_graphs: int = 64) -> Tuple[List[np.ndarray], np.ndarray]:
        """Inference helper: per-graph node embeddings and stacked graph embeddings."""
        self.eval()
        node_out: List[np.ndarray] = []
        graph_out: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(items), batch_graphs):
                chunk = items[start:start + batch_graphs]
                g = collate(chunk).to(device)
                h, hg = self.encode(g)
                h = h.cpu().numpy()
                batch = g.batch.cpu().numpy()
                for gid, (x, _, _) in enumerate(chunk):
                    rows = h[batch == gid]
                    node_out.append(rows[: x.shape[0]] if x.shape[0] else rows[:0])
                graph_out.append(hg.cpu().numpy())
        stacked = np.vstack(graph_out) if graph_out else np.zeros((0, self.spec.hidden_dim), np.float32)
        return node_out, stacked.astype(np.float32)

    def alpha_numpy(self, v_q: np.ndarray, h_gq: np.ndarray, device: str = "cpu") -> float:
        self.eval()
        with torch.no_grad():
            a = self.alpha(torch.as_tensor(v_q, dtype=torch.float32, device=device).unsqueeze(0),
                           torch.as_tensor(h_gq, dtype=torch.float32, device=device).unsqueeze(0))
        return float(a.item())


# --------------------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------------------

def save_checkpoint(model: StructuralRetrievalModel, path: str, metadata: Optional[Dict] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "spec": asdict(model.spec),
            "relations": list(RELATIONS),
            "state_dict": model.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )


def load_checkpoint(path: str, map_location: str = "cpu") -> Tuple[StructuralRetrievalModel, Dict]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a GRACE structural encoder checkpoint")
    if list(payload.get("relations", [])) != list(RELATIONS):
        raise ValueError("Checkpoint relation vocabulary differs from this code version")
    model = StructuralRetrievalModel(EncoderSpec(**payload["spec"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model.to(map_location), payload.get("metadata", {})


def checkpoint_fingerprint(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def spec_summary(spec: EncoderSpec) -> str:
    return json.dumps(asdict(spec), sort_keys=True)
