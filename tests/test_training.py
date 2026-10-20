"""Acceptance 3 and 4: one InfoNCE step on synthetic data saves a checkpoint; fitting, tuning and test
repositories must be project-disjoint and test repositories never enter training."""

import json
import math

import pytest
import torch

from conftest import PY_PREFIX, crosscodeeval_row, make_python_repo, python_task, write_jsonl
from grace.config import DEFAULT_CONFIG, DEV_CONFIG, MissingParameter
from grace.data import ProjectOverlapError, assert_project_disjoint
from grace.encoder import load_checkpoint
from grace.train_encoder import info_nce, main as train_main, sample_candidates


def setup_dev_repos(tmp_path):
    root = tmp_path / "dev_repos"
    make_python_repo(root / "fitrepo")
    make_python_repo(root / "tunerepo")
    rows = [
        {"task_id": "fit/0", "repo_name": "example/fitrepo", "file_path": "main.py", "prefix": PY_PREFIX,
         "groundtruth": "    doubled = scale(big, 2)"},
        {"task_id": "fit/1", "repo_name": "example/fitrepo", "file_path": "main.py", "prefix": PY_PREFIX,
         "groundtruth": "    name = describe(circle)", "positive_units": ["func:pkg/utils.py:describe"]},
        {"task_id": "tune/0", "repo_name": "example/tunerepo", "file_path": "main.py", "prefix": PY_PREFIX,
         "groundtruth": "    doubled = scale(big, 2)"},
    ]
    queries = write_jsonl(tmp_path / "train_queries.jsonl", rows)
    (tmp_path / "fit.txt").write_text("example/fitrepo\n")
    (tmp_path / "tune.txt").write_text("# tuning repositories\nexample/tunerepo\n")
    (tmp_path / "test.txt").write_text("example/pyrepo\n")
    return root, queries


def argv(tmp_path, root, queries, config=DEV_CONFIG, test=None, extra=()):
    return ["--config", str(config), "--language", "python", "--train-queries", str(queries),
            "--repos-root", str(root), "--fit-repos", str(tmp_path / "fit.txt"),
            "--tune-repos", str(tmp_path / "tune.txt"), "--test-repos", *(test or [str(tmp_path / "test.txt")]),
            "--output-dir", str(tmp_path / "run"), *extra]


def test_one_training_step_saves_checkpoint(tmp_path, text_encoder):
    root, queries = setup_dev_repos(tmp_path)
    summary = train_main(argv(tmp_path, root, queries, extra=["--max-steps", "1"]), text_encoder=text_encoder)
    run = tmp_path / "run"
    assert (run / "best.pt").exists() and (run / "last.pt").exists()
    log = json.loads((run / "training_log.json").read_text())
    assert log["history"][0]["steps"] == 1
    assert "tune_recall_at_10" in log["history"][0]
    assert summary["num_fit_examples"] == 2 and summary["num_tune_examples"] == 1
    assert set(log["metadata"]["positive_sources"]) == {"explicit", "heuristic_identifier_match"}
    assert log["metadata"]["config"]["config_name"] == "dev_smoke_test"
    model, meta = load_checkpoint(str(run / "best.pt"))
    assert meta["fit_repos"] == ["example/fitrepo"] and meta["tune_repos"] == ["example/tunerepo"]
    assert model.spec.in_dim == 261


def test_training_step_changes_parameters(tmp_path, text_encoder):
    root, queries = setup_dev_repos(tmp_path)
    torch.manual_seed(0)
    train_main(argv(tmp_path, root, queries, extra=["--max-steps", "1"]), text_encoder=text_encoder)
    from grace.encoder import EncoderSpec, StructuralRetrievalModel
    from grace.config import load_config
    from grace.train_encoder import set_seed

    cfg = load_config(str(DEV_CONFIG))
    set_seed(int(cfg.require("random_seed")))
    initial = StructuralRetrievalModel(EncoderSpec.from_config(cfg)).state_dict()
    trained, _ = load_checkpoint(str(tmp_path / "run" / "last.pt"))
    changed = [k for k, v in trained.state_dict().items() if not torch.allclose(v, initial[k])]
    assert changed, "one optimizer step should update the encoder/gate"


def test_test_repository_in_fit_list_is_rejected(tmp_path, text_encoder):
    root, queries = setup_dev_repos(tmp_path)
    (tmp_path / "test.txt").write_text("example/pyrepo\nexample/fitrepo\n")
    with pytest.raises(ProjectOverlapError):
        train_main(argv(tmp_path, root, queries), text_encoder=text_encoder)
    assert not (tmp_path / "run").exists()


def test_test_repositories_read_from_benchmark_file(tmp_path, text_encoder):
    root, queries = setup_dev_repos(tmp_path)
    task = python_task(root / "tunerepo", repo_name="other-owner/tunerepo")  # same project basename
    bench = write_jsonl(tmp_path / "bench.jsonl", [crosscodeeval_row(task)])
    with pytest.raises(ProjectOverlapError):
        train_main(argv(tmp_path, root, queries, test=[str(bench)]), text_encoder=text_encoder)


def test_fit_and_tune_must_be_disjoint(tmp_path, text_encoder):
    root, queries = setup_dev_repos(tmp_path)
    (tmp_path / "tune.txt").write_text("example/tunerepo\nexample/fitrepo\n")
    with pytest.raises(ProjectOverlapError):
        train_main(argv(tmp_path, root, queries), text_encoder=text_encoder)


def test_queries_outside_development_lists_are_rejected(tmp_path, text_encoder):
    root, queries = setup_dev_repos(tmp_path)
    with open(queries, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"task_id": "x", "repo_name": "example/pyrepo", "file_path": "main.py",
                                 "prefix": PY_PREFIX, "groundtruth": "x"}) + "\n")
    with pytest.raises(ValueError, match="outside the fit/tune lists"):
        train_main(argv(tmp_path, root, queries), text_encoder=text_encoder)


def test_training_requires_hyperparameters(tmp_path, text_encoder):
    root, queries = setup_dev_repos(tmp_path)
    with pytest.raises(MissingParameter):
        train_main(argv(tmp_path, root, queries, config=DEFAULT_CONFIG,
                        extra=["--set", "infonce_temperature=null"]), text_encoder=text_encoder)


def test_assert_project_disjoint_identity_keys():
    assert_project_disjoint({"fit": ["a/x"], "tune": ["a/y"], "test": ["b/z"]})
    with pytest.raises(ProjectOverlapError):
        assert_project_disjoint({"fit": ["a/x"], "test": ["b/X.git"]})


def test_info_nce_value():
    scores = torch.tensor([2.0, 0.0, 0.0])
    expected = -math.log(math.exp(2) / (math.exp(2) + 2))
    assert info_nce(scores, 1, 1.0).item() == pytest.approx(expected, rel=1e-6)
    # several positives: -log(sum_pos / sum_all)
    expected2 = -math.log((math.exp(2) + 1) / (math.exp(2) + 2))
    assert info_nce(scores, 2, 1.0).item() == pytest.approx(expected2, rel=1e-6)


def test_candidate_sampling_uses_positives_hard_and_random():
    import random

    class Ex:
        positives = [0]
        hard_pool = [1, 2, 3]
        unit_items = [None] * 10

    cand = sample_candidates(Ex, n_random=2, n_hard=2, rng=random.Random(0))
    assert cand[:3] == [0, 1, 2]
    assert len(cand) == 5 and len(set(cand)) == 5
    assert all(c not in (0, 1, 2, 3) for c in cand[3:])
