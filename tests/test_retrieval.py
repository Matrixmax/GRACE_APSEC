"""Acceptance 6: semantic + structural candidates are merged, MMR keeps K_c, the top K_f are selected."""

import networkx as nx
import numpy as np
import pytest

from grace.index import TaskIndex
from grace.retrieve import hybrid_retrieve
from grace.subgraphs import RetrievalUnit

D = 4


def unit(i):
    return RetrievalUnit(unit_id=f"u{i}", centre=f"u{i}", kind="function", name=f"f{i}", file_path="a.py",
                         start_line=i, end_line=i, text=f"def f{i}(): pass", nodes=[f"u{i}"])


def e(i, dim):
    v = np.zeros(dim, np.float32)
    v[i] = 1.0
    return v


def make_index(backend="exact"):
    sem = np.stack([
        e(0, 256),                                   # u0: semantic match
        e(1, 256),                                   # u1: no semantic match
        e(0, 256),                                   # u2: duplicate of u0
        0.1 * e(0, 256) + np.sqrt(0.99) * e(5, 256),  # u3: weak semantic match
        e(6, 256),                                   # u4: unrelated
    ])
    struct = np.stack([
        0.2 * e(0, D) + np.sqrt(0.96) * e(1, D),
        e(0, D),                                     # u1: structural match
        0.2 * e(0, D) + np.sqrt(0.96) * e(1, D),
        e(2, D),
        e(3, D),
    ])
    units = [unit(i) for i in range(5)]
    graphs = []
    for u in units:
        g = nx.MultiDiGraph()
        g.add_node(u.unit_id, type="function", code=u.text)
        graphs.append(g)
    params = {"M": 32, "ef_construction": 400, "ef_search": 256} if backend == "hnsw" else {}
    return TaskIndex(units=units, sem=sem, struct=struct, node_emb=[s[None, :] for s in struct],
                     unit_graphs=graphs, backend=backend, hnsw_params=params)


Q_SEM, Q_STRUCT = e(0, 256), e(0, D)


def test_union_of_both_paths():
    res = hybrid_retrieve(make_index(), Q_SEM, Q_STRUCT, alpha=0.5, alpha_mode="learned_gate", k_c=1, k_f=1,
                          mmr_lambda=0.5)
    by_id = {c.unit_id: c for c in res.merged}
    assert set(by_id) == {"u0", "u1"}
    assert by_id["u0"].from_semantic and not by_id["u0"].from_structural
    assert by_id["u1"].from_structural and not by_id["u1"].from_semantic


@pytest.mark.parametrize("backend", ["exact", "hnsw"])
def test_scores_mmr_kc_kf(backend):
    alpha = 0.5
    res = hybrid_retrieve(make_index(backend), Q_SEM, Q_STRUCT, alpha=alpha, alpha_mode="learned_gate", k_c=3,
                          k_f=2, mmr_lambda=0.5)
    for c in res.merged:
        assert c.score == pytest.approx(alpha * c.sem_sim + (1 - alpha) * c.struct_sim, abs=1e-6)
    pool_ids = [c.unit_id for c in res.pool]
    assert len(res.pool) == 3
    assert "u2" not in pool_ids          # redundant duplicate of u0 removed by MMR
    assert set(pool_ids) == {"u0", "u1", "u3"}
    assert [c.unit_id for c in res.selected] == ["u0", "u1"]  # top-K_f by score
    assert res.selected == res.pool[:2]
    scores = [c.score for c in res.pool]
    assert scores == sorted(scores, reverse=True)


def test_alpha_controls_balance():
    sem_only = hybrid_retrieve(make_index(), Q_SEM, Q_STRUCT, 1.0, "fixed", k_c=5, k_f=1, mmr_lambda=1.0)
    struct_only = hybrid_retrieve(make_index(), Q_SEM, Q_STRUCT, 0.0, "fixed", k_c=5, k_f=1, mmr_lambda=1.0)
    assert sem_only.selected[0].unit_id in {"u0", "u2"}
    assert struct_only.selected[0].unit_id == "u1"


def test_pool_never_exceeds_kc_and_selection_never_exceeds_kf():
    for k_c in range(1, 6):
        for k_f in range(1, 6):
            res = hybrid_retrieve(make_index(), Q_SEM, Q_STRUCT, 0.3, "learned_gate", k_c=k_c, k_f=k_f,
                                  mmr_lambda=0.7)
            assert len(res.pool) <= k_c
            assert len(res.selected) == min(k_f, len(res.pool))


def test_invalid_alpha_and_empty_index():
    with pytest.raises(ValueError):
        hybrid_retrieve(make_index(), Q_SEM, Q_STRUCT, 1.5, "fixed", 3, 2, 0.5)
    empty = TaskIndex(units=[], sem=np.zeros((0, 256), np.float32), struct=np.zeros((0, D), np.float32),
                      node_emb=[], unit_graphs=[])
    res = hybrid_retrieve(empty, Q_SEM, Q_STRUCT, 0.5, "learned_gate", 3, 2, 0.5)
    assert res.merged == [] and res.pool == [] and res.selected == []


def test_index_save_load_round_trip(tmp_path):
    idx = make_index("hnsw")
    idx.save(str(tmp_path / "task"), metadata={"task_id": "t"})
    assert (tmp_path / "task" / "sem_hnsw.bin").exists()
    loaded = TaskIndex.load(str(tmp_path / "task"))
    assert [u.unit_id for u in loaded.units] == [u.unit_id for u in idx.units]
    np.testing.assert_allclose(loaded.sem, idx.sem)
    a = hybrid_retrieve(idx, Q_SEM, Q_STRUCT, 0.5, "learned_gate", 3, 2, 0.5)
    b = hybrid_retrieve(loaded, Q_SEM, Q_STRUCT, 0.5, "learned_gate", 3, 2, 0.5)
    assert [c.unit_id for c in a.pool] == [c.unit_id for c in b.pool]
