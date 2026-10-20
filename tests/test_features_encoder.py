"""Acceptance 1, 2, 5: 261-d node features, GAT shapes + checkpoints, alpha gate in [0, 1]."""

import dataclasses

import networkx as nx
import numpy as np
import pytest
import torch

from grace.encoder import (EncoderSpec, StructuralRetrievalModel, collate, graph_to_arrays, load_checkpoint,
                           save_checkpoint)
from grace.graph_features import TextEncoder, laplacian_pe, node_features


def small_graph(n=5, relation="calls"):
    g = nx.MultiDiGraph()
    for i in range(n):
        g.add_node(f"n{i}", type="statement", code=f"x{i} = f(x{i - 1})")
    for i in range(n - 1):
        g.add_edge(f"n{i}", f"n{i + 1}", key=relation, relation=relation)
    return g


def item(g, encoder):
    nodes = sorted(g.nodes())
    x = node_features(g, encoder, nodes=nodes)
    return (x,) + graph_to_arrays(g, nodes)


def test_node_features_are_261_dim(text_encoder):
    g = small_graph()
    x = node_features(g, text_encoder)
    assert x.shape == (5, 261)
    assert x.dtype == np.float32
    # first 256 columns are the code embedding, last 5 the Laplacian PE
    np.testing.assert_allclose(x[:, :256], text_encoder.encode([g.nodes[n]["code"] for n in g.nodes()]))
    np.testing.assert_allclose(x[:, 256:], laplacian_pe(g, 5, list(g.nodes())), atol=1e-6)


def test_node_features_reject_wrong_dimensions(text_encoder):
    class WrongDim(TextEncoder):
        name, dim = "wrong", 768

        def encode(self, texts, truncation_side="right"):
            return np.zeros((len(texts), 768), np.float32)

    with pytest.raises(ValueError, match="256"):
        node_features(small_graph(), WrongDim())
    with pytest.raises(ValueError):
        node_features(small_graph(), text_encoder, lappe_dim=8)


@pytest.mark.parametrize("graph", [
    nx.MultiDiGraph(),
    nx.MultiDiGraph([("a", "a")]),
    nx.MultiDiGraph([("a", "b"), ("c", "d")]),  # disconnected
    nx.path_graph(12, create_using=nx.MultiDiGraph),
])
def test_laplacian_pe_edge_cases(graph):
    pe = laplacian_pe(graph, 5)
    assert pe.shape == (graph.number_of_nodes(), 5)
    assert np.isfinite(pe).all()


def test_laplacian_pe_sign_is_deterministic():
    g = nx.path_graph(8, create_using=nx.MultiDiGraph)
    np.testing.assert_allclose(laplacian_pe(g, 5), laplacian_pe(g.copy(), 5))


@pytest.mark.parametrize("mode", ["homogeneous", "relation_attention_bias"])
@pytest.mark.parametrize("pooling", ["mean", "max"])
@pytest.mark.parametrize("reverse", [True, False])
def test_gat_shapes(dev_cfg, text_encoder, mode, pooling, reverse):
    spec = dataclasses.replace(EncoderSpec.from_config(dev_cfg), edge_type_mode=mode, pooling=pooling,
                               add_reverse_edges=reverse)
    torch.manual_seed(0)
    model = StructuralRetrievalModel(spec)
    items = [item(small_graph(3), text_encoder), item(small_graph(6, "ast_child"), text_encoder)]
    h, hg = model.encode(collate(items))
    assert h.shape == (9, spec.hidden_dim)
    assert hg.shape == (2, spec.hidden_dim)
    assert torch.isfinite(h).all() and torch.isfinite(hg).all()
    nodes, graphs = model.encode_numpy(items)
    assert [n.shape for n in nodes] == [(3, spec.hidden_dim), (6, spec.hidden_dim)]
    # batching does not change a graph's embedding
    _, single = model.encode_numpy(items[:1])
    np.testing.assert_allclose(single[0], graphs[0], atol=1e-5)


def test_gat_handles_single_node_and_empty_graph(model, text_encoder):
    g = nx.MultiDiGraph()
    g.add_node("only", type="function", code="def f(): pass")
    empty = (np.zeros((0, 261), np.float32), np.zeros((2, 0), np.int64), np.zeros((0,), np.int64))
    nodes, graphs = model.encode_numpy([item(g, text_encoder), empty])
    assert nodes[0].shape == (1, model.spec.hidden_dim)
    assert nodes[1].shape[0] == 0
    assert graphs.shape == (2, model.spec.hidden_dim)
    assert np.isfinite(graphs).all()


def test_encoder_spec_validates_input_dim(dev_cfg):
    spec = dataclasses.replace(EncoderSpec.from_config(dev_cfg), in_dim=768)
    with pytest.raises(ValueError, match="261"):
        spec.validate()


def test_checkpoint_round_trip(tmp_path, model, text_encoder):
    items = [item(small_graph(4), text_encoder)]
    _, before = model.encode_numpy(items)
    path = tmp_path / "ckpt" / "model.pt"
    save_checkpoint(model, str(path), {"note": "unit-test"})
    loaded, meta = load_checkpoint(str(path))
    _, after = loaded.encode_numpy(items)
    np.testing.assert_allclose(before, after, atol=1e-6)
    assert meta == {"note": "unit-test"}
    assert loaded.spec == model.spec


def test_checkpoint_rejects_foreign_files(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"state_dict": {}}, path)
    with pytest.raises(ValueError):
        load_checkpoint(str(path))


def test_alpha_gate_is_bounded(model):
    torch.manual_seed(1)
    v_q = torch.randn(64, 256) * 100.0
    h_g = torch.randn(64, model.spec.hidden_dim) * 100.0
    alpha = model.alpha(v_q, h_g)
    assert alpha.shape == (64,)
    assert torch.all(alpha >= 0.0) and torch.all(alpha <= 1.0)
    a = model.alpha_numpy(np.ones(256, np.float32), np.ones(model.spec.hidden_dim, np.float32))
    assert 0.0 <= a <= 1.0


def test_alpha_gate_matches_formula(model):
    v_q = torch.randn(1, 256)
    h_g = torch.randn(1, model.spec.hidden_dim)
    lin = model.gate.linear
    expected = torch.sigmoid(lin.weight @ torch.cat([v_q, h_g], dim=-1).T + lin.bias).squeeze()
    assert torch.allclose(model.alpha(v_q, h_g).squeeze(), expected, atol=1e-6)


def test_alpha_gate_receives_gradients(model):
    v_q = torch.randn(2, 256)
    h_g = torch.randn(2, model.spec.hidden_dim)
    model.alpha(v_q, h_g).sum().backward()
    assert model.gate.linear.weight.grad is not None


class _MockTokenizer:
    truncation_side = "right"

    def __call__(self, batch, **kwargs):
        self.last_side = self.truncation_side
        return {"input_ids": torch.ones(len(batch), 3, dtype=torch.long),
                "attention_mask": torch.ones(len(batch), 3, dtype=torch.long)}


class _MockEmbeddingModel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, input_ids, attention_mask):
        return torch.arange(input_ids.shape[0] * self.dim, dtype=torch.float32).reshape(-1, self.dim) + 1.0


def test_codet5p_wrapper_returns_normalised_256d_without_download():
    from grace.graph_features import CodeT5pEncoder

    tok = _MockTokenizer()
    enc = CodeT5pEncoder("local-mock", tokenizer=tok, model=_MockEmbeddingModel(256), batch_size=2)
    out = enc.encode(["a", "b", "c"], truncation_side="left")
    assert out.shape == (3, 256)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)
    assert tok.last_side == "left" and tok.truncation_side == "right"  # restored afterwards


def test_codet5p_wrapper_rejects_hidden_state_width():
    from grace.graph_features import CodeT5pEncoder

    enc = CodeT5pEncoder("local-mock", tokenizer=_MockTokenizer(), model=_MockEmbeddingModel(768))
    with pytest.raises(ValueError, match="256"):
        enc.encode(["a"])
