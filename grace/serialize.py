"""Fused-graph serialization in the paper format.

Records:
    NODE(<int id>, <type>, "<code>")
    EDGE(<int source>, <relation>, <int target>)

Native relations keep their labels; virtual query-evidence edges are always ``align``. Code is a
JSON string literal (newlines escaped). Integer ids are assigned in emission order and the mapping to
original node ids is returned so that ``referenced_nodes`` in model output can be traced back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx


@dataclass
class GraphRecord:
    kind: str  # NODE | EDGE
    text: str
    group: int  # -1 = query graph, i = retrieved rank i
    needs: Tuple[str, ...] = ()  # original node ids that must be emitted first (edges only)
    node: Optional[str] = None


@dataclass
class Serializer:
    graph: nx.MultiDiGraph
    ids: Dict[str, int] = field(default_factory=dict)

    def node_id(self, nid: str) -> int:
        if nid not in self.ids:
            self.ids[nid] = len(self.ids)
        return self.ids[nid]

    def node_record(self, nid: str) -> str:
        data = self.graph.nodes[nid]
        code = str(data.get("code") or data.get("name") or "")
        return f"NODE({self.node_id(nid)}, {data.get('type', 'unknown')}, {json.dumps(code, ensure_ascii=False)})"

    def edge_record(self, u: str, v: str, relation: str) -> str:
        return f"EDGE({self.node_id(u)}, {relation}, {self.node_id(v)})"

    def id_map(self) -> Dict[int, str]:
        return {i: n for n, i in self.ids.items()}


def edge_relation(key, data: dict) -> str:
    return str(data.get("relation", key))


def serialize_graph(graph: nx.MultiDiGraph, node_order: Optional[Sequence[str]] = None) -> Tuple[str, Dict[int, str]]:
    """Serialize the whole graph without a budget (nodes first, then edges)."""
    ser = Serializer(graph)
    order = list(node_order) if node_order is not None else list(graph.nodes())
    lines = [ser.node_record(n) for n in order]
    for u, v, k, d in graph.edges(keys=True, data=True):
        lines.append(ser.edge_record(u, v, edge_relation(k, d)))
    return "\n".join(lines), ser.id_map()


def ordered_groups(graph: nx.MultiDiGraph, query_nodes: Sequence[str], retrieved_nodes: Sequence[Sequence[str]]):
    """Yield ``(group, node_ids, edges)`` in priority order for budgeted serialization.

    For each retrieved rank i: its new nodes, then align edges into it (with their query endpoints),
    then its native edges. The remaining query-graph nodes and edges come last.
    """
    query_set = set(query_nodes)
    seen = set()
    used_edges = set()
    groups = []
    all_edges = sorted(
        ((u, v, edge_relation(k, d), d) for u, v, k, d in graph.edges(keys=True, data=True)),
        key=lambda e: (str(e[0]), str(e[1]), e[2]),
    )
    for rank, nodes in enumerate(retrieved_nodes):
        node_set = set(nodes)
        align, native = [], []
        for u, v, rel, d in all_edges:
            if (u, v, rel) in used_edges:
                continue
            if rel == "align":
                if v in node_set and d.get("subgraph_rank", rank) == rank:
                    align.append((u, v, rel))
                    used_edges.add((u, v, rel))
            elif u in node_set and v in node_set:
                native.append((u, v, rel))
                used_edges.add((u, v, rel))
        new_nodes = [n for n in nodes if n not in seen]
        seen.update(new_nodes)
        groups.append((rank, new_nodes, align + native))
    rest_nodes = [n for n in query_nodes if n not in seen]
    rest_edges = []
    for u, v, rel, d in all_edges:
        if (u, v, rel) not in used_edges and rel != "align" and u in query_set and v in query_set:
            rest_edges.append((u, v, rel))
            used_edges.add((u, v, rel))
    groups.append((-1, rest_nodes, rest_edges))
    return groups
