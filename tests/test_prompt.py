"""Acceptance 9: local and retrieved context stay within the configured token budgets."""

import networkx as nx
import pytest

from grace.prompt import (RegexTokenCounter, RetrievedBlock, build_retrieved_context, build_token_counter,
                          render_prompt, truncate_local_context)

COUNTER = RegexTokenCounter()


def long_prefix(n=200):
    return "\n".join(f"value_{i} = compute(value_{i - 1}, {i})" for i in range(n)) + "\n"


@pytest.mark.parametrize("budget", [1, 7, 50, 1024])
def test_local_context_budget_keeps_most_recent_lines(budget):
    prefix = long_prefix()
    local = truncate_local_context(prefix, COUNTER, budget)
    assert COUNTER.count(local.text) <= budget
    assert local.tokens == COUNTER.count(local.text)
    assert prefix.endswith(local.text)  # oldest lines are dropped first
    if COUNTER.count(prefix) > budget:
        assert local.truncated and local.dropped_lines > 0


def test_local_context_untouched_when_it_fits():
    local = truncate_local_context("x = 1\n", COUNTER, 1024)
    assert local.text == "x = 1\n" and not local.truncated


def fused_graph(n_units=4, nodes_per_unit=6):
    g = nx.MultiDiGraph()
    g.add_node("q", type="statement", code="total = helper(a, b)")
    blocks = []
    for u in range(n_units):
        ids = [f"u{u}_{k}" for k in range(nodes_per_unit)]
        for k, nid in enumerate(ids):
            g.add_node(nid, type="statement", code=f"line_{u}_{k} = op(line_{u}_{k - 1}) + {k}")
            if k:
                g.add_edge(ids[k - 1], nid, key="control_flow", relation="control_flow")
        g.add_edge("q", ids[0], key="align", relation="align", subgraph_rank=u)
        code = "\n".join(g.nodes[n]["code"] for n in ids)
        blocks.append(RetrievedBlock(header=f"# [{u + 1}] mod{u}.py:1-{nodes_per_unit}", code=code, node_ids=ids))
    return g, blocks


@pytest.mark.parametrize("budget", [0, 10, 60, 150, 400, 1024])
def test_retrieved_context_budget(budget):
    g, blocks = fused_graph()
    ctx = build_retrieved_context(g, ["q"], blocks, COUNTER, budget)
    assert ctx.tokens <= budget
    assert COUNTER.count(ctx.code_context) + COUNTER.count(ctx.graph_context) <= budget
    for line in filter(None, ctx.graph_context.split("\n")):
        assert line.startswith("NODE(") or line.startswith("EDGE(")


def test_retrieved_context_order_and_align_endpoints():
    g, blocks = fused_graph(n_units=2)
    ctx = build_retrieved_context(g, ["q"], blocks, COUNTER, 10_000)
    assert ctx.code_context.index("[1]") < ctx.code_context.index("[2]")  # rerank order
    lines = ctx.graph_context.split("\n")
    align = [ln for ln in lines if ", align, " in ln]
    assert len(align) == 2
    # every edge refers to already emitted nodes
    emitted = set()
    for ln in lines:
        if ln.startswith("NODE("):
            emitted.add(int(ln[5:].split(",")[0]))
        else:
            src, _, dst = ln[5:-1].split(", ")
            assert int(src) in emitted and int(dst) in emitted
    assert ctx.stats["snippets"] == 2 and ctx.stats["records_skipped"] == 0


def test_render_prompt_single_pass():
    prompt = render_prompt({"repo_name": "r", "current_file_path": "a.py",
                            "code_before_cursor": "s = '{code_context}'", "code_context": "CTX",
                            "graph_context": "NODE(0, function, \"f\")"})
    assert "s = '{code_context}'" in prompt  # inserted code is not substituted again
    assert "CTX" in prompt and "NODE(0, function" in prompt
    assert "{repo_name}" not in prompt


def test_token_counter_spec():
    assert build_token_counter("regex").name == "regex-approximation"
    with pytest.raises(ValueError):
        build_token_counter("whitespace")
