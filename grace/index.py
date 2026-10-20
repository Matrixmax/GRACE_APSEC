"""Per-task retrieval index (semantic + structural) with HNSW or exact search.

An index is built for each completion task from its leakage-masked repository snapshot, because the
current file differs per task. Text embeddings are cached by content hash and reused across tasks of
the same repository, so unchanged files are encoded only once.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from .encoder import StructuralRetrievalModel, graph_to_arrays
from .graph_builder import read_graphml, write_graphml
from .graph_features import CODE_EMBEDDING_DIM, TextEncoder, node_features
from .subgraphs import RetrievalUnit, unit_graph

LOGGER = logging.getLogger(__name__)
INDEX_FORMAT = "grace-task-index-v1"


def _normalise(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        return mat / max(float(np.linalg.norm(mat)), 1e-12)
    return mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)


class VectorIndex:
    """Cosine kNN over a fixed matrix. ``backend`` is ``hnsw`` (hnswlib) or ``exact``."""

    def __init__(self, vectors: np.ndarray, backend: str = "hnsw", M: Optional[int] = None,
                 ef_construction: Optional[int] = None, ef_search: Optional[int] = None,
                 saved_path: Optional[str] = None):
        self.vectors = _normalise(vectors) if len(vectors) else np.zeros((0, vectors.shape[1] if vectors.ndim == 2 else 1), np.float32)
        self.backend = backend
        self.M, self.ef_construction, self.ef_search = M, ef_construction, ef_search
        self._hnsw = None
        if backend == "hnsw" and len(self.vectors):
            import hnswlib

            if None in (M, ef_construction, ef_search):
                raise ValueError("HNSW needs M, ef_construction and ef_search")
            self._hnsw = hnswlib.Index(space="cosine", dim=self.vectors.shape[1])
            if saved_path and Path(saved_path).exists():
                # Reuse a distributed/prebuilt graph instead of rebuilding it.
                self._hnsw.load_index(saved_path, max_elements=len(self.vectors))
                if self._hnsw.get_current_count() != len(self.vectors):
                    raise ValueError(f"{saved_path} does not match the stored vectors")
            else:
                self._hnsw.init_index(max_elements=len(self.vectors), ef_construction=int(ef_construction),
                                      M=int(M))
                self._hnsw.add_items(self.vectors, np.arange(len(self.vectors)))
            self._hnsw.set_ef(int(ef_search))
        elif backend not in ("hnsw", "exact"):
            raise ValueError(f"unknown index backend {backend}")

    def __len__(self) -> int:
        return len(self.vectors)

    def search(self, query: np.ndarray, k: int) -> List[int]:
        k = min(int(k), len(self.vectors))
        if k <= 0:
            return []
        query = _normalise(query)
        if self._hnsw is not None:
            self._hnsw.set_ef(max(int(self.ef_search), k))
            labels, _ = self._hnsw.knn_query(query.reshape(1, -1), k=k)
            return [int(i) for i in labels[0]]
        sims = self.vectors @ query
        order = np.argsort(-sims, kind="stable")[:k]
        return [int(i) for i in order]

    def save(self, path: str) -> None:
        if self._hnsw is not None:
            self._hnsw.save_index(path)


@dataclass
class TaskIndex:
    units: List[RetrievalUnit]
    sem: np.ndarray  # (n, 256) normalised CodeT5p embeddings of unit text
    struct: np.ndarray  # (n, d) normalised GAT graph embeddings
    node_emb: List[np.ndarray]  # per unit (n_i, d) GAT node embeddings in unit.nodes order
    unit_graphs: List[nx.MultiDiGraph]
    backend: str = "hnsw"
    hnsw_params: Dict[str, int] = field(default_factory=dict)
    sem_index: Optional[VectorIndex] = None
    struct_index: Optional[VectorIndex] = None
    load_dir: Optional[str] = None  # directory with saved HNSW graphs (set by load())

    def __post_init__(self):
        self.sem = _normalise(self.sem) if len(self.sem) else np.zeros((0, CODE_EMBEDDING_DIM), np.float32)
        self.struct = _normalise(self.struct) if len(self.struct) else self.struct.astype(np.float32)
        self.build_search()

    def build_search(self) -> None:
        params = dict(self.hnsw_params or {})
        if not len(self.units):
            self.sem_index = self.struct_index = None
            return
        saved = Path(self.load_dir) if self.load_dir else None
        self.sem_index = VectorIndex(self.sem, self.backend, **params,
                                     saved_path=str(saved / "sem_hnsw.bin") if saved else None)
        self.struct_index = VectorIndex(self.struct, self.backend, **params,
                                        saved_path=str(saved / "struct_hnsw.bin") if saved else None)

    def __len__(self) -> int:
        return len(self.units)

    # -- persistence --------------------------------------------------------------------------
    def save(self, directory: str, metadata: Optional[dict] = None) -> None:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "sem.npy", self.sem)
        np.save(out / "struct.npy", self.struct)
        np.savez_compressed(out / "node_emb.npz", *self.node_emb)
        with open(out / "units.jsonl", "w", encoding="utf-8") as handle:
            for unit in self.units:
                handle.write(json.dumps(unit.to_dict(), ensure_ascii=False) + "\n")
        merged = nx.MultiDiGraph()
        for g in self.unit_graphs:
            merged.add_nodes_from(g.nodes(data=True))
            merged.add_edges_from(g.edges(keys=True, data=True))
        write_graphml(merged, str(out / "unit_nodes.graphml"))
        if self.sem_index is not None:
            self.sem_index.save(str(out / "sem_hnsw.bin"))
            self.struct_index.save(str(out / "struct_hnsw.bin"))
        meta = {"format": INDEX_FORMAT, "backend": self.backend, "hnsw_params": self.hnsw_params}
        meta.update(metadata or {})
        with open(out / "meta.json", "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, directory: str) -> "TaskIndex":
        src = Path(directory)
        with open(src / "meta.json", "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if meta.get("format") != INDEX_FORMAT:
            raise ValueError(f"{directory} is not a GRACE task index")
        with open(src / "units.jsonl", "r", encoding="utf-8") as handle:
            units = [RetrievalUnit.from_dict(json.loads(line)) for line in handle if line.strip()]
        arrays = np.load(src / "node_emb.npz")
        node_emb = [arrays[f"arr_{i}"] for i in range(len(units))]
        merged = read_graphml(str(src / "unit_nodes.graphml"))
        graphs = [merged.subgraph(u.nodes).copy() for u in units]
        return cls(
            units=units,
            sem=np.load(src / "sem.npy"),
            struct=np.load(src / "struct.npy"),
            node_emb=node_emb,
            unit_graphs=graphs,
            backend=meta.get("backend", "hnsw"),
            hnsw_params=meta.get("hnsw_params", {}),
            load_dir=str(src),
        )


def graph_items(graphs: Sequence[nx.MultiDiGraph], node_orders: Sequence[Sequence[str]], encoder: TextEncoder,
                lappe_dim: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    items = []
    for g, order in zip(graphs, node_orders):
        x = node_features(g, encoder, nodes=order, lappe_dim=lappe_dim)
        ei, et = graph_to_arrays(g, order)
        items.append((x, ei, et))
    return items


def build_task_index(
    graph: nx.MultiDiGraph,
    units: List[RetrievalUnit],
    text_encoder: TextEncoder,
    model: StructuralRetrievalModel,
    lappe_dim: int,
    backend: str = "hnsw",
    hnsw_params: Optional[Dict[str, int]] = None,
    device: str = "cpu",
) -> TaskIndex:
    graphs = [unit_graph(graph, u) for u in units]
    orders = [list(u.nodes) for u in units]
    sem = text_encoder.encode([u.text for u in units]) if units else np.zeros((0, CODE_EMBEDDING_DIM), np.float32)
    if units:
        items = graph_items(graphs, orders, text_encoder, lappe_dim)
        node_emb, struct = model.encode_numpy(items, device=device)
    else:
        node_emb, struct = [], np.zeros((0, model.spec.hidden_dim), np.float32)
    return TaskIndex(units=units, sem=sem, struct=struct, node_emb=node_emb, unit_graphs=graphs,
                     backend=backend, hnsw_params=hnsw_params or {})
