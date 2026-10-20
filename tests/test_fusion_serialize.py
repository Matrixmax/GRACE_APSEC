"""Acceptance 7 and 8: align edges only under (alpha_F > gamma, beta > theta, TypeCompatible);
serialization emits NODE(...)/EDGE(...) records with original relations and ``align``."""

import re

import networkx as nx
import numpy as np
import pytest

from grace.fusion import RetrievedSubgraph, fuse_graphs, type_compatible
from grace.serialize import serialize_graph

RULE = "same_node_type_or_reference"


def query():
    g = nx.MultiDiGraph()
    g.add_node("q0", type="statement", code="big = scale(x, 2)", refs="scale x")
    g.add_node("q1", type="variable", code="circle", name="circle", type_name="Circle")
    g.add_edge("q0", "q1", key="defines", relation="defines")
    emb = np.array([[10, 0, 0, 0], [0, 0, 0, 10]], np.float32)
    return g, ["q0", "q1"], emb


def subgraph(prefix, score):
    g = nx.MultiDiGraph()
    g.add_node(f"{prefix}0", type="function", name="scale", code="def scale(value, factor):")
    g.add_node(f"{prefix}1", type="statement", code="return value * factor")
    g.add_edge(f"{prefix}0", f"{prefix}1", key="ast_root", relation="ast_root")
    g.add_edge(f"{prefix}1", f"{prefix}0", key="calls", relation="calls")
    emb = np.array([[10, 0, 0, 0], [0, 10, 0, 0]], np.float32)
    return RetrievedSubgraph(unit_id=f"{prefix}0", score=score, graph=g, nodes=[f"{prefix}0", f"{prefix}1"],
                             node_emb=emb)


def align_pairs(result):
    return {(a["source"], a["target"]) for a in result.align_edges}


def test_align_edge_when_all_conditions_hold():
    g, nodes, emb = query()
    res = fuse_graphs(g, nodes, emb, [subgraph("a", 1.0)], gamma=0.0, theta=0.4, type_rule=RULE)
    # q0 attends to a0 (beta ~ 1 > theta) and refers to it by name -> align edge
    # q1 has beta = 0.5 > theta for both nodes but is not type compatible -> no edge
    assert align_pairs(res) == {("q0", "a0")}
    assert res.graph.has_edge("q0", "a0", key="align")
    assert res.graph.edges["q0", "a0", "align"]["relation"] == "align"
    assert res.align_edges[0]["beta"] > 0.4


def test_no_align_when_graph_weight_not_above_gamma():
    g, nodes, emb = query()
    res = fuse_graphs(g, nodes, emb, [subgraph("a", 1.0)], gamma=1.0, theta=0.4, type_rule=RULE)
    assert res.active == [False]
    assert res.align_edges == []
    # inactive subgraphs do not update query node embeddings
    np.testing.assert_allclose(res.updated_query_emb["q0"], emb[0])


def test_gamma_gates_individual_subgraphs():
    g, nodes, emb = query()
    subs = [subgraph("a", 2.0), subgraph("b", 0.0)]  # softmax weights ~ [0.88, 0.12]
    res = fuse_graphs(g, nodes, emb, subs, gamma=0.5, theta=0.4, type_rule=RULE)
    assert res.graph_weights == pytest.approx([np.exp(2) / (np.exp(2) + 1), 1 / (np.exp(2) + 1)])
    assert res.active == [True, False]
    assert align_pairs(res) == {("q0", "a0")}
    # the active subgraph contributes alpha_F * sum_u beta h_u to h'_v
    assert res.updated_query_emb["q0"][0] > emb[0][0]


def test_no_align_when_beta_not_above_theta():
    g, nodes, emb = query()
    res = fuse_graphs(g, nodes, emb, [subgraph("a", 1.0)], gamma=0.0, theta=1.0, type_rule=RULE)
    assert res.align_edges == []


def test_no_align_when_types_incompatible():
    g, nodes, emb = query()
    res = fuse_graphs(g, nodes, emb, [subgraph("a", 1.0)], gamma=0.0, theta=0.4, type_rule="same_node_type")
    assert res.align_edges == []


def test_type_compatible_rules():
    assert type_compatible({"type": "function"}, {"type": "function"}, "same_node_type")
    assert not type_compatible({"type": "statement", "refs": "scale"}, {"type": "function", "name": "scale"},
                               "same_node_type")
    assert type_compatible({"type": "statement", "refs": "scale"}, {"type": "function", "name": "scale"}, RULE)
    assert type_compatible({"type": "variable", "type_name": "Circle"}, {"type": "class", "name": "Circle"}, RULE)
    with pytest.raises(ValueError):
        type_compatible({}, {}, "anything")


def test_fused_graph_keeps_original_nodes_and_edges():
    g, nodes, emb = query()
    res = fuse_graphs(g, nodes, emb, [subgraph("a", 1.0)], gamma=0.0, theta=0.4, type_rule=RULE)
    relations = {d["relation"] for _, _, d in res.graph.edges(data=True)}
    assert {"defines", "ast_root", "calls", "align"} <= relations
    assert set(res.graph.nodes()) == {"q0", "q1", "a0", "a1"}


def test_empty_retrieval_returns_query_graph():
    g, nodes, emb = query()
    res = fuse_graphs(g, nodes, emb, [], gamma=0.0, theta=0.4, type_rule=RULE)
    assert set(res.graph.nodes()) == set(g.nodes())
    assert res.align_edges == [] and res.graph_weights == []
    empty = fuse_graphs(nx.MultiDiGraph(), [], np.zeros((0, 4), np.float32), [subgraph("a", 1.0)], 0.0, 0.4, RULE)
    assert empty.align_edges == []


NODE_RE = re.compile(r'^NODE\((\d+), (\w+), (".*")\)$')
EDGE_RE = re.compile(r"^EDGE\((\d+), (\w+), (\d+)\)$")


def test_serialization_format():
    g, nodes, emb = query()
    res = fuse_graphs(g, nodes, emb, [subgraph("a", 1.0)], gamma=0.0, theta=0.4, type_rule=RULE)
    text, id_map = serialize_graph(res.graph)
    lines = text.split("\n")
    node_lines = [ln for ln in lines if ln.startswith("NODE(")]
    edge_lines = [ln for ln in lines if ln.startswith("EDGE(")]
    assert len(node_lines) == 4 and len(edge_lines) == 4
    assert all(NODE_RE.match(ln) for ln in node_lines)
    assert all(EDGE_RE.match(ln) for ln in edge_lines)
    relations = {EDGE_RE.match(ln).group(2) for ln in edge_lines}
    assert relations == {"defines", "ast_root", "calls", "align"}
    align = [EDGE_RE.match(ln) for ln in edge_lines if ", align, " in ln][0]
    assert id_map[int(align.group(1))] == "q0" and id_map[int(align.group(3))] == "a0"
    assert 'NODE(' in text and '"def scale(value, factor):"' in text


def test_serialization_escapes_multiline_code():
    g = nx.MultiDiGraph()
    g.add_node("x", type="function", code='def f():\n    return "a"')
    text, _ = serialize_graph(g)
    assert text == 'NODE(0, function, "def f():\\n    return \\"a\\"")'


def test_legacy_module_exposes_paper_format():
    from model.graph_text_encoders import encode_code_graph, encode_code_graph_paper_format

    g = nx.DiGraph()
    g.add_node(0, type="function", code="def f(): pass")
    g.add_node(1, type="statement", code="f()")
    g.add_edge(1, 0, type="calls")
    assert encode_code_graph_paper_format(g).split("\n") == [
        'NODE(0, function, "def f(): pass")', 'NODE(1, statement, "f()")', "EDGE(1, calls, 0)"]
    assert "Node descriptions" in encode_code_graph(g)  # legacy format still available
