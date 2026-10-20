"""Acceptance 9, 10, 11: Python and Java tiny repositories go through graph construction, indexing,
retrieval, fusion, serialization and prompting, within budget and without target leakage.

The structural model is untrained and the text encoder is a hashing stand-in, so only structure,
format and invariants are checked (no retrieval quality is measured).
"""

import re

import pytest

from conftest import JAVA_FORBIDDEN, PY_FORBIDDEN, java_task, python_task
from grace.config import MissingParameter
from grace.graph_builder import RELATIONS
from grace.leakage import audit_leakage, hidden_lines, make_masked_snapshot, read_repo_sources
from grace.pipeline import GracePipeline
from grace.prompt import RegexTokenCounter

CASES = {
    "python": (python_task, "py_repo", PY_FORBIDDEN),
    "java": (java_task, "java_repo", JAVA_FORBIDDEN),
}


def run(request, language, dev_cfg, text_encoder, model, tmp_path, backend="hnsw"):
    make_task, repo_fixture, forbidden = CASES[language]
    repo = request.getfixturevalue(repo_fixture)
    task = make_task(repo)
    pipe = GracePipeline(dev_cfg, text_encoder, model, index_backend=backend, checkpoint_id="test")
    out = pipe.run_task(task, graph_cache_dir=str(tmp_path / "graphs"), index_dir=str(tmp_path / "index"))
    return pipe, task, out, forbidden


@pytest.mark.parametrize("language", ["python", "java"])
def test_end_to_end(request, language, dev_cfg, text_encoder, model, tmp_path):
    pipe, task, out, _ = run(request, language, dev_cfg, text_encoder, model, tmp_path)
    assert out["cursor_located"]
    assert out["num_index_units"] >= 3
    q_nodes, _ = out["query_graph_size"]
    assert q_nodes > 0
    retrieval = out["retrieval"]
    assert 0.0 <= retrieval["alpha"] <= 1.0 and retrieval["alpha_mode"] == "learned_gate"
    assert len(retrieval["pool"]) <= dev_cfg.require("candidate_pool_size_Kc")
    assert len(retrieval["selected"]) == min(dev_cfg.require("fused_subgraphs_Kf"), len(retrieval["pool"]))
    assert len(out["retrieved_snippets"]) == len(retrieval["selected"])
    # paper serialization in the prompt
    graph_lines = out["graph_context"].split("\n")
    assert any(ln.startswith("NODE(") for ln in graph_lines)
    assert any(ln.startswith("EDGE(") for ln in graph_lines)
    rels = {m.group(1) for m in re.finditer(r"^EDGE\(\d+, (\w+), \d+\)$", out["graph_context"], re.M)}
    assert rels <= set(RELATIONS)
    assert out["graph_context"] in out["prompt"] and out["local_context"] in out["prompt"]
    # budgets
    counter = RegexTokenCounter()
    stats = out["prompt_stats"]
    assert counter.count(out["local_context"]) <= stats["local_budget"] == 1024
    assert stats["retrieved_tokens"] <= stats["retrieved_budget"] == 1024
    assert counter.count(out["code_context_block"]) + counter.count(out["graph_context"]) <= 1024
    # outputs stay compatible with model/grace_eval.py
    assert out["code_context"] == task.prefix and out["groundtruth"] == task.groundtruth
    assert (tmp_path / "graphs" / task.safe_key / "repo_graph.graphml").exists()
    assert (tmp_path / "index" / task.safe_key / "meta.json").exists()


@pytest.mark.parametrize("language", ["python", "java"])
def test_tight_budget_is_respected(request, language, dev_cfg, text_encoder, model, tmp_path):
    dev_cfg.set("local_context_tokens", 20)
    dev_cfg.set("retrieved_context_tokens", 40)
    dev_cfg.set("context_tokens", 60)
    _, _, out, _ = run(request, language, dev_cfg, text_encoder, model, tmp_path, backend="exact")
    counter = RegexTokenCounter()
    assert counter.count(out["local_context"]) <= 20
    assert counter.count(out["code_context_block"]) + counter.count(out["graph_context"]) <= 40
    assert out["prompt_stats"]["local_truncated"]


def test_budget_split_must_fit_total(request, dev_cfg, text_encoder, model, tmp_path):
    dev_cfg.set("context_tokens", 100)
    with pytest.raises(ValueError, match="exceeds"):
        run(request, "python", dev_cfg, text_encoder, model, tmp_path)


@pytest.mark.parametrize("language", ["python", "java"])
def test_no_target_leakage(request, language, dev_cfg, text_encoder, model, tmp_path):
    pipe, task, out, forbidden = run(request, language, dev_cfg, text_encoder, model, tmp_path)
    # target line and distinctive post-cursor lines (bare braces such as "}" occur everywhere)
    forbidden = forbidden + [ln for ln in hidden_lines(task) if len(ln) > 3]
    prepared = pipe.prepare(task, graph_cache_dir=str(tmp_path / "graphs"))
    graph_texts = [str(v) for _, d in prepared.graph.nodes(data=True) for v in d.values()]
    graph_texts += [str(n) for n in prepared.graph.nodes()]
    index = pipe.build_index(prepared)
    unit_texts = [u.text for u in index.units]
    unit_node_texts = [str(v) for g in index.unit_graphs for _, d in g.nodes(data=True) for v in d.values()]
    hits = audit_leakage({
        "graph": graph_texts,
        "index_units": unit_texts,
        "index_unit_nodes": unit_node_texts,
        "snippets": [s["code"] for s in out["retrieved_snippets"]],
        "code_context_block": [out["code_context_block"]],
        "graph_context": [out["graph_context"]],
        "local_context": [out["local_context"]],
        "prompt": [out["prompt"]],
    }, forbidden)
    assert hits == []
    # the masked snapshot is exactly the visible prefix of the current file
    assert prepared.snapshot.files[task.file_path] == task.prefix


def test_masking_falls_back_to_prompt_when_cursor_ambiguous(py_repo):
    task = python_task(py_repo)
    files = read_repo_sources(str(py_repo), "python")
    files["main.py"] = "something else entirely\n" + files["main.py"].replace(task.prefix, "")
    snap = make_masked_snapshot(task, files)
    assert not snap.cursor_located
    assert snap.files["main.py"] == task.prefix
    assert files["pkg/utils.py"] == snap.files["pkg/utils.py"]


def test_index_is_reused_from_disk(request, dev_cfg, text_encoder, model, tmp_path):
    _, _, first, _ = run(request, "python", dev_cfg, text_encoder, model, tmp_path)
    _, _, second, _ = run(request, "python", dev_cfg, text_encoder, model, tmp_path)
    assert first["retrieval"]["selected"] == second["retrieval"]["selected"]
    a, b = first["retrieval"]["pool"], second["retrieval"]["pool"]
    assert [c["unit_id"] for c in a] == [c["unit_id"] for c in b]
    assert [c["score"] for c in a] == pytest.approx([c["score"] for c in b], abs=1e-6)
    assert first["prompt"] == second["prompt"]


def test_fixed_alpha_is_an_explicit_ablation(request, dev_cfg, text_encoder, model, tmp_path):
    dev_cfg.set("alpha_mode", "fixed")
    with pytest.raises(MissingParameter, match="fixed_alpha"):
        run(request, "python", dev_cfg, text_encoder, model, tmp_path, backend="exact")
    dev_cfg.set("fixed_alpha", 0.3)
    _, _, out, _ = run(request, "python", dev_cfg, text_encoder, model, tmp_path, backend="exact")
    assert out["retrieval"]["alpha"] == 0.3 and out["retrieval"]["alpha_mode"] == "fixed"


def test_pipeline_requires_trained_model(py_repo, dev_cfg, text_encoder):
    pipe = GracePipeline(dev_cfg, text_encoder, model=None, index_backend="exact")
    with pytest.raises(RuntimeError, match="checkpoint"):
        pipe.run_task(python_task(py_repo))


def test_graph_has_multi_level_relations(py_repo, java_repo):
    from grace.graph_builder import build_repo_graph

    for repo, lang in ((py_repo, "python"), (java_repo, "java")):
        g = build_repo_graph(read_repo_sources(str(repo), lang), lang)
        rels = {d.get("relation") for _, _, d in g.edges(data=True)}
        # repository, module and function levels
        assert {"contains", "declares", "calls", "inherits", "ast_root", "control_flow"} <= rels, (lang, rels)
        types = {d.get("type") for _, d in g.nodes(data=True)}
        assert {"file", "class", "function", "statement"} <= types
