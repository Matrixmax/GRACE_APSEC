"""Retrieval units: definition-centred subgraphs.

The paper retrieves "snippets or subgraphs". A retrieval unit is one function/class definition:

* body: the centre plus nodes reachable through ``ast_root``/``ast_child``/``defines`` (for a class
  also its declared member headers via ``declares``, without their bodies);
* context: ``subgraph_hops`` hops in either direction over semantic relations (calls, inherits,
  type_ref, imports, dep_call, type_flow, declares) starting from body nodes; neighbours are added
  without their own bodies;
* edges: all edges of the repository graph induced by the unit's node set (original relations kept);
* text: for a function, its source lines; for a class, its header line(s), statements directly in
  the class body and the headers of its members (member bodies are separate units). The text is
  used by the semantic path and in the prompt.

Units centred in the current file whose span overlaps the local query window are not indexed: they
duplicate the query graph G_q (the current file is already cut at the cursor, so they cannot leak).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import networkx as nx

from .graph_builder import SEMANTIC_RELATIONS

_BODY_DOWN = {"ast_root", "ast_child", "defines"}


@dataclass
class RetrievalUnit:
    unit_id: str
    centre: str
    kind: str
    name: str
    file_path: str
    start_line: int
    end_line: int
    text: str
    nodes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "unit_id": self.unit_id,
            "centre": self.centre,
            "kind": self.kind,
            "name": self.name,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "text": self.text,
            "nodes": list(self.nodes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RetrievalUnit":
        return cls(**data)


def _relations(graph: nx.MultiDiGraph, u, v) -> List[str]:
    data = graph.get_edge_data(u, v) or {}
    return [d.get("relation", k) for k, d in data.items()]


def _body(graph: nx.MultiDiGraph, centre: str) -> List[str]:
    is_class = graph.nodes[centre].get("type") == "class"
    seen = {centre}
    order = [centre]
    stack = [centre]
    while stack:
        node = stack.pop()
        for succ in graph.successors(node):
            if succ in seen:
                continue
            rels = set(_relations(graph, node, succ))
            if rels & _BODY_DOWN:
                seen.add(succ)
                order.append(succ)
                stack.append(succ)
            elif is_class and node == centre and "declares" in rels:
                seen.add(succ)  # member header only
                order.append(succ)
    return order


def _expand(graph: nx.MultiDiGraph, seeds: Sequence[str], hops: int) -> List[str]:
    seen = set(seeds)
    frontier = list(seeds)
    added: List[str] = []
    for _ in range(max(int(hops), 0)):
        nxt = []
        for node in frontier:
            neighbours = [(s, _relations(graph, node, s)) for s in graph.successors(node)]
            neighbours += [(p, _relations(graph, p, node)) for p in graph.predecessors(node)]
            for other, rels in neighbours:
                if other in seen or not (set(rels) & SEMANTIC_RELATIONS):
                    continue
                seen.add(other)
                added.append(other)
                nxt.append(other)
        frontier = nxt
    return added


def extract_units(
    graph: nx.MultiDiGraph,
    files: Dict[str, str],
    hops: int,
    current_file: Optional[str] = None,
    query_window: Optional[tuple] = None,
) -> List[RetrievalUnit]:
    units: List[RetrievalUnit] = []
    lines_cache: Dict[str, List[str]] = {}
    for nid in sorted(graph.nodes()):
        data = graph.nodes[nid]
        if data.get("type") not in ("function", "class"):
            continue
        rel = data.get("file_path")
        start = int(data.get("start_line", 0) or 0)
        end = int(data.get("end_line", start) or start)
        if current_file is not None and rel == current_file and query_window is not None:
            lo, hi = query_window
            if end >= lo and start <= hi:
                continue
        body = _body(graph, nid)
        nodes = body + _expand(graph, body, hops)
        if rel not in lines_cache:
            lines_cache[rel] = files.get(rel, "").split("\n")
        src_lines = lines_cache[rel]
        if data.get("type") == "class":
            # Class text = lines not covered by member definitions, plus each member's header.
            covered = set()
            headers = {}
            for succ in graph.successors(nid):
                sd = graph.nodes[succ]
                if sd.get("type") in ("function", "class") and "declares" in _relations(graph, nid, succ):
                    s0 = int(sd.get("start_line", 0) or 0)
                    s1 = int(sd.get("end_line", s0) or s0)
                    covered.update(range(s0, s1 + 1))
                    headers[s0] = str(sd.get("code", ""))
            out_lines = []
            for ln in range(start, end + 1):
                if ln in headers:
                    indent = src_lines[ln - 1][: len(src_lines[ln - 1]) - len(src_lines[ln - 1].lstrip())] if ln - 1 < len(src_lines) else ""
                    out_lines.append(indent + headers[ln] + (" ..." if headers[ln].rstrip().endswith(":") else ""))
                elif ln not in covered and 0 < ln <= len(src_lines):
                    out_lines.append(src_lines[ln - 1])
            text = "\n".join(out_lines)
        else:
            text = "\n".join(src_lines[max(start - 1, 0):end])
        text = text or str(data.get("code", ""))
        units.append(
            RetrievalUnit(
                unit_id=nid,
                centre=nid,
                kind=str(data.get("type")),
                name=str(data.get("name", "")),
                file_path=str(rel),
                start_line=start,
                end_line=end,
                text=text,
                nodes=nodes,
            )
        )
    return units


def unit_graph(graph: nx.MultiDiGraph, unit: RetrievalUnit) -> nx.MultiDiGraph:
    return graph.subgraph(unit.nodes).copy()
