"""CLI stages (build-graphs, index, retrieve, generate) with the dev config, benchmark loaders, the
compatibility wrapper, and index packaging. No model download and no API call."""

import json
import sys
from pathlib import Path

import pytest

from conftest import (JAVA_FORBIDDEN, PY_FORBIDDEN, PY_PREFIX, PY_TARGET, crosscodeeval_row, java_task,
                      python_task, write_jsonl)
from grace import cli
from grace.config import DEV_CONFIG
from grace.data import default_data_file, load_tasks
from grace.encoder import save_checkpoint

ROOT = Path(__file__).resolve().parent.parent


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.fixture
def bench(tmp_path, py_repo, java_repo, model):
    py = write_jsonl(tmp_path / "cceval_py.jsonl", [crosscodeeval_row(python_task(py_repo, repo_name="o/pyrepo"))])
    jv = write_jsonl(tmp_path / "cceval_java.jsonl",
                     [crosscodeeval_row(java_task(java_repo, repo_name="o/javarepo"))])
    ckpt = tmp_path / "ckpt" / "best.pt"
    save_checkpoint(model, str(ckpt))
    return {"python": py, "java": jv, "ckpt": ckpt, "repos": py_repo.parent}


@pytest.mark.parametrize("language", ["python", "java"])
def test_cli_full_flow(tmp_path, bench, text_encoder, language):
    common = ["--config", str(DEV_CONFIG), "--dataset", "crosscodeeval", "--language", language,
              "--data-file", str(bench[language]), "--repos-root", str(bench["repos"])]
    graphs, index = tmp_path / "graphs", tmp_path / "index"
    assert cli.main(["build-graphs", *common, "--out", str(graphs)]) == 0
    graph_files = list(graphs.glob("*/repo_graph.graphml"))
    assert len(graph_files) == 1
    text = graph_files[0].read_text()
    forbidden = PY_FORBIDDEN if language == "python" else JAVA_FORBIDDEN
    assert not any(f in text for f in forbidden)

    model_args = ["--checkpoint", str(bench["ckpt"]), "--graph-dir", str(graphs), "--index-dir", str(index)]
    assert cli.main(["index", *common, *model_args], text_encoder=text_encoder) == 0
    assert len(list(index.glob("*/meta.json"))) == 1

    retrieval = tmp_path / "retrieval.jsonl"
    assert cli.main(["retrieve", *common, *model_args, "--output", str(retrieval)], text_encoder=text_encoder) == 0
    rows = read_jsonl(retrieval)
    assert len(rows) == 1 and "NODE(" in rows[0]["prompt"]
    snapshot = json.loads(retrieval.with_suffix(".config.json").read_text())
    assert snapshot["config_name"] == "dev_smoke_test"

    def transport(request):
        return {"text": json.dumps({"completed_code": "pass", "explanation": "mock"})}

    out = tmp_path / "results"
    assert cli.main(["generate", "--config", str(DEV_CONFIG), "--retrieval", str(retrieval), "--model",
                     "mock-model", "--output-dir", str(out)], transport=transport) == 0
    preds = read_jsonl(out / "prediction.jsonl")
    assert preds[0]["pred"] == "pass" and preds[0]["task_id"] == rows[0]["task_id"]


def test_repoeval_updated_layouts(tmp_path, py_repo):
    a = {"prompt": PY_PREFIX, "metadata": {"task_id": "pyrepo/0", "ground_truth": PY_TARGET,
                                           "fpath_tuple": ["pyrepo", "main.py"]}}
    b = {"repo_name": "pyrepo", "file_path": "main.py", "all_code": PY_PREFIX, "next_line": PY_TARGET}
    path = write_jsonl(tmp_path / "line_level.python.test.jsonl", [a, b])
    for name in ("repoeval_updated", "repoeval-updated"):
        tasks = load_tasks(name, str(path), "python", str(py_repo.parent))
        assert [t.dataset for t in tasks] == ["repoeval_updated"] * 2
        assert all(t.file_path == "main.py" and t.groundtruth == PY_TARGET for t in tasks)
        assert all(t.repo_dir == str(py_repo) and t.prefix == PY_PREFIX for t in tasks)
    with pytest.raises(ValueError, match="layout"):
        write_jsonl(tmp_path / "bad.jsonl", [{"x": 1}])
        load_tasks("repoeval_updated", str(tmp_path / "bad.jsonl"), "python")


def test_default_data_files():
    assert default_data_file("crosscodeeval", "d", "java") == "d/rawdata/java/line_completion.jsonl"
    assert default_data_file("repoeval-updated", "d", "java") == "d/rawdata/java/line_level.java.test.jsonl"
    with pytest.raises(ValueError):
        default_data_file("unknown", "d", "python")


def test_compat_wrapper_translates_legacy_arguments():
    sys.path.insert(0, str(ROOT / "model"))
    try:
        import importlib

        wrapper = importlib.import_module("model.grace_coderag")
    finally:
        sys.path.remove(str(ROOT / "model"))
    args = wrapper.build_parser().parse_args(["--model", "gpt40-mini", "--dataset", "repoeval-updated",
                                              "--language", "java", "--checkpoint", "c.pt", "--k", "5",
                                              "--alpha", "0.5", "--device", "cpu"])
    retrieve, generate = wrapper.translate(args)
    assert retrieve[0] == "retrieve" and generate[0] == "generate"
    joined = " ".join(retrieve)
    assert "--dataset repoeval_updated" in joined and "--language java" in joined
    assert "fused_subgraphs_Kf=5" in joined and "alpha_mode=fixed" in joined and "fixed_alpha=0.5" in joined
    assert "device=cpu" in joined
    assert "--model gpt40-mini" in " ".join(generate)
    # without --k/--alpha the paper settings from the config are used
    plain = wrapper.translate(wrapper.build_parser().parse_args(["--model", "m", "--checkpoint", "c.pt"]))[0]
    assert "--set" not in plain


def test_package_strip_unpack_hydrate(tmp_path, bench, text_encoder, py_repo):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import package_indexes as pkg
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    common = ["--config", str(DEV_CONFIG), "--dataset", "crosscodeeval", "--language", "python",
              "--data-file", str(bench["python"]), "--repos-root", str(bench["repos"])]
    index = tmp_path / "index"
    cli.main(["index", *common, "--checkpoint", str(bench["ckpt"]), "--index-dir", str(index)],
             text_encoder=text_encoder)
    task_dir = next(index.iterdir())
    original_units = (task_dir / "units.jsonl").read_text()

    archive = tmp_path / "idx.tar.gz"
    manifest = pkg.pack(str(index), str(archive), strip=True)
    assert manifest["code_stripped"] and manifest["num_tasks"] == 1
    restored = tmp_path / "restored"
    pkg.unpack(str(archive), str(restored))
    rdir = restored / task_dir.name
    units = [json.loads(line) for line in (rdir / "units.jsonl").read_text().splitlines()]
    assert all(u["text"] == "" for u in units)
    assert "return value * factor" not in (rdir / "unit_nodes.graphml").read_text()

    # a stripped index is refused by the pipeline until hydrated
    retrieval = tmp_path / "r.jsonl"
    with pytest.raises(RuntimeError, match="hydrate"):
        cli.main(["retrieve", *common, "--checkpoint", str(bench["ckpt"]), "--index-dir", str(restored),
                  "--output", str(retrieval)], text_encoder=text_encoder)

    from grace.config import load_config

    tasks = load_tasks("crosscodeeval", str(bench["python"]), "python", str(bench["repos"]))
    assert pkg.hydrate(str(restored), tasks, load_config(str(DEV_CONFIG))) == 1
    assert (rdir / "units.jsonl").read_text() == original_units
    assert cli.main(["retrieve", *common, "--checkpoint", str(bench["ckpt"]), "--index-dir", str(restored),
                     "--output", str(retrieval)], text_encoder=text_encoder) == 0


def test_legacy_graph_builder_uses_masking(tmp_path, py_repo):
    pytest.importorskip("staticfg")
    pytest.importorskip("mypy")
    pytest.importorskip("astroid")
    pytest.importorskip("matplotlib")
    sys.path.insert(0, str(ROOT / "data_process"))
    try:
        import multilevel_graph_builder as legacy
    finally:
        sys.path.remove(str(ROOT / "data_process"))
    data = write_jsonl(tmp_path / "cc.jsonl", [crosscodeeval_row(python_task(py_repo, repo_name="o/pyrepo"))])
    legacy.main("crosscodeeval", "python", data_file=str(data), repos_root=str(py_repo.parent),
                graphs_dir=str(tmp_path / "graphs"))
    out = tmp_path / "graphs" / "0_pyrepo" / "repo_multi_graph.graphml"
    assert out.exists()
    assert not any(f in out.read_text() for f in PY_FORBIDDEN)
