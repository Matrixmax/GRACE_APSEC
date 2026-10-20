"""Configuration sections, overrides, and null values that stop a stage."""

import logging

import pytest

from grace.config import DEFAULT_CONFIG, MissingParameter, load_config
from grace.encoder import EncoderSpec
from grace.llm import decoding_params

PAPER_VALUES = {
    "context_tokens": 2048,
    "retrieved_context_tokens": 1024,
    "local_context_tokens": 1024,
    "max_new_tokens": 100,
    "candidate_pool_size_Kc": 10,
    "fused_subgraphs_Kf": 3,
    "theta": 0.4,
    "hnsw_M": 32,
    "hnsw_ef_search": 256,
    "code_encoder": "Salesforce/codet5p-110m-embedding",
    "code_embedding_dim": 256,
    "laplacian_pe_dim": 5,
}

METHOD_HYPERPARAMETERS = ["hnsw_ef_construction", "code_encoder_max_length", "gamma",
                    "gat_layers", "gat_heads", "gat_hidden_dim", "gat_dropout", "gat_edge_type_mode",
                    "gat_add_reverse_edges", "graph_pooling", "subgraph_hops", "type_compatibility",
                    "infonce_temperature", "optimizer", "learning_rate", "weight_decay", "batch_size",
                    "max_epochs", "early_stopping_patience", "num_random_negatives", "num_hard_negatives",
                    "train_queries_per_repo", "mmr_lambda", "random_seed", "prompt_tokenizer",
                    "llm_temperature", "llm_top_p", "llm_seed", "system_prompt"]


def test_paper_config_values_and_sections():
    cfg = load_config(str(DEFAULT_CONFIG))
    assert cfg.is_paper_config
    for key, value in PAPER_VALUES.items():
        assert cfg.get(key) == value, key
        assert cfg.provenance[key] == "paper_reported"
    for key in METHOD_HYPERPARAMETERS:
        assert cfg.get(key) is not None, key
        assert cfg.provenance[key] == "method_hyperparameters", key
    assert cfg.get("gamma") == 0.5
    assert cfg.get("hnsw_ef_construction") == 400


def test_paper_config_loads_without_warning(caplog):
    with caplog.at_level(logging.WARNING):
        cfg = load_config(str(DEFAULT_CONFIG))
    assert caplog.text == ""
    snap = cfg.snapshot()
    assert snap["config_name"] == "paper_reproduction"
    assert snap["values"]["theta"] == 0.4
    assert snap["provenance"]["theta"] == "paper_reported"
    assert snap["provenance"]["gat_layers"] == "method_hyperparameters"
    assert snap["overrides"] == {}


@pytest.mark.parametrize("key", METHOD_HYPERPARAMETERS)
def test_null_parameter_raises(key):
    cfg = load_config(str(DEFAULT_CONFIG), [f"{key}=null"])
    with pytest.raises(MissingParameter, match=key):
        cfg.require(key)


def test_stages_refuse_to_run_when_a_value_is_unset():
    cfg = load_config(str(DEFAULT_CONFIG), ["gat_heads=null", "llm_temperature=null", "gamma=null"])
    with pytest.raises(MissingParameter):
        EncoderSpec.from_config(cfg)
    with pytest.raises(MissingParameter):
        decoding_params(cfg)
    with pytest.raises(MissingParameter):
        cfg.require_all(["theta", "gamma"])


def test_stages_accept_paper_config_defaults():
    cfg = load_config(str(DEFAULT_CONFIG))
    spec = EncoderSpec.from_config(cfg)
    assert (spec.layers, spec.heads, spec.hidden_dim) == (2, 4, 256)
    assert decoding_params(cfg) == {"max_tokens": 100, "temperature": 0.0}
    assert cfg.require_all(["theta", "gamma"]) == {"theta": 0.4, "gamma": 0.5}


def test_override_supplies_value():
    cfg = load_config(str(DEFAULT_CONFIG), ["gamma=0.25", "llm_temperature=omit"])
    assert cfg.require("gamma") == 0.25
    assert cfg.require("llm_temperature") == "omit"
    assert cfg.provenance["gamma"] == "method_hyperparameters"
    assert cfg.overrides == {"gamma": 0.25, "llm_temperature": "omit"}


def test_overriding_paper_value_warns(caplog):
    cfg = load_config(str(DEFAULT_CONFIG))
    with caplog.at_level(logging.WARNING):
        cfg.set("theta", 0.5)
    assert "Overriding paper value theta" in caplog.text


def test_dev_config_is_separate(dev_cfg):
    assert not dev_cfg.is_paper_config
    assert dev_cfg.name == "dev_smoke_test"
    # inherits the paper-reported values unchanged
    for key, value in PAPER_VALUES.items():
        assert dev_cfg.get(key) == value
    # test overrides replace the inherited values and are tagged as such
    assert dev_cfg.get("gat_hidden_dim") == 16
    assert dev_cfg.provenance["gat_hidden_dim"] == "test_overrides"
    assert dev_cfg.provenance["gamma"] == "test_overrides"


def test_duplicate_key_across_flat_sections_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("paper_reported:\n  theta: 0.4\nmethod_hyperparameters:\n  theta: 0.3\n")
    with pytest.raises(ValueError, match="theta"):
        load_config(str(path))


def test_model_resolution():
    cfg = load_config(str(DEFAULT_CONFIG))
    # CLI alias resolves to the canonical entry
    spec = cfg.resolve_model("gpt40-mini")
    assert spec["name"] == "gpt-4o-mini" and spec["model_id"] == "gpt-4o-mini"
    spec = cfg.resolve_model("deepseek-v3")
    assert spec["provider"] == "deepseek" and spec["api_key_env"] == "DEEPSEEK_API_KEY"
    assert spec["model_id"] == "deepseek-chat"
    spec = cfg.resolve_model("qwen2.5-coder-14b-instruct")
    assert spec["provider"] == "dashscope" and spec["api_key_env"] == "DASHSCOPE_API_KEY"
    with pytest.raises(KeyError):
        cfg.resolve_model("not-a-model")


def test_model_without_model_id_raises():
    cfg = load_config(str(DEFAULT_CONFIG))
    cfg.llm["models"]["deepseek-v3"]["model_id"] = None
    with pytest.raises(MissingParameter, match="deepseek-v3"):
        cfg.resolve_model("deepseek-v3")
