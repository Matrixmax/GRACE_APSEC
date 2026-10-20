"""Code graph fusion (paper Eqs. (7)-(12)).

* alpha_i^F = softmax over the top-K_f retrieval scores.
* beta_{v,u}^i = softmax_u (h_v . h_u / sqrt(d)) over the nodes u of retrieved subgraph i.
* h'_v = h_v + sum_i 1[alpha_i^F > gamma] alpha_i^F sum_u beta_{v,u}^i h_u.
* E_align = {(v, align, u) | alpha_i^F > gamma, beta_{v,u}^i > theta, TypeCompatible(v, u)}.
* G_f = G_q  U  retrieved subgraphs  U  E_align; original nodes and relations are kept. Nodes that
  appear in several subgraphs (same source span) are merged by identity.

TypeCompatible rules (``type_compatibility``):

* ``same_node_type``: v and u have the same node type (file/class/function/statement/variable).
* ``same_node_type_or_reference``: same node type, or v refers to u by name (v.refs contains u.name),
  or v is a variable whose inferred type name equals u.name.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import networkx as nx
import numpy as np

TYPE_RULES = ("same_node_type", "same_node_type_or_reference")


def type_compatible(v: dict, u: dict, rule: str) -> bool:
    if rule not in TYPE_RULES:
        raise ValueError(f"type_compatibility must be one of {TYPE_RULES}")
    if v.get("type") and v.get("type") == u.get("type"):
        return True
    if rule == "same_node_type_or_reference":
        name = u.get("name")
        if name and name in str(v.get("refs", "")).split():
            return True
        if name and v.get("type_name") == name:
            return True
    return False


@dataclass
class RetrievedSubgraph:
    unit_id: str
    score: float
    graph: nx.MultiDiGraph
    nodes: List[str]
    node_emb: np.ndarray  # (len(nodes), d)


@dataclass
class FusionResult:
    graph: nx.MultiDiGraph
    graph_weights: List[float]
    active: List[bool]
    align_edges: List[dict] = field(default_factory=list)
    updated_query_emb: Dict[str, np.ndarray] = field(default_factory=dict)

    def stats(self) -> dict:
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "graph_weights": [round(w, 6) for w in self.graph_weights],
            "active_subgraphs": self.active,
            "num_align_edges": len(self.align_edges),
        }


def _softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def fuse_graphs(query_graph: nx.MultiDiGraph, query_nodes: Sequence[str], query_emb: np.ndarray,
                retrieved: Sequence[RetrievedSubgraph], gamma: float, theta: float,
                type_rule: str) -> FusionResult:
    fused = nx.MultiDiGraph()
    for nid, data in query_graph.nodes(data=True):
        fused.add_node(nid, **data, origin="query")
    fused.add_edges_from(query_graph.edges(keys=True, data=True))

    weights = _softmax([r.score for r in retrieved])
    active = [w > gamma for w in weights]
    query_emb = np.asarray(query_emb, dtype=np.float32)
    updated = {nid: query_emb[i].copy() for i, nid in enumerate(query_nodes)} if len(query_nodes) else {}
    align: List[dict] = []

    for rank, (sub, weight, is_active) in enumerate(zip(retrieved, weights, active)):
        for nid, data in sub.graph.nodes(data=True):
            if nid in fused:
                fused.nodes[nid].setdefault("origin", f"retrieved:{rank}")
            else:
                fused.add_node(nid, **data, origin=f"retrieved:{rank}")
        for u, v, k, d in sub.graph.edges(keys=True, data=True):
            if not fused.has_edge(u, v, key=k):
                fused.add_edge(u, v, key=k, **d)
        if not is_active or not len(query_nodes) or not len(sub.nodes):
            continue
        h_u = np.asarray(sub.node_emb, dtype=np.float32)
        d = h_u.shape[1]
        logits = (query_emb @ h_u.T) / math.sqrt(d)
        logits = logits - logits.max(axis=1, keepdims=True)
        beta = np.exp(logits)
        beta = beta / beta.sum(axis=1, keepdims=True)
        message = beta @ h_u
        for qi, v in enumerate(query_nodes):
            updated[v] = updated[v] + weight * message[qi]
            for ui in np.nonzero(beta[qi] > theta)[0]:
                u = sub.nodes[int(ui)]
                if u == v:
                    continue
                if not type_compatible(query_graph.nodes[v], sub.graph.nodes[u], type_rule):
                    continue
                if not fused.has_edge(v, u, key="align"):
                    fused.add_edge(v, u, key="align", relation="align", weight=float(beta[qi, ui]),
                                   subgraph_rank=rank)
                    align.append({"source": v, "target": u, "beta": float(beta[qi, ui]), "subgraph_rank": rank,
                                  "graph_weight": weight})
    return FusionResult(graph=fused, graph_weights=weights, active=active, align_edges=align,
                        updated_query_emb=updated)
