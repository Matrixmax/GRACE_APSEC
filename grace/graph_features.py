"""Node features x_v = [CodeT5p(code); LapPE] (256 + 5 = 261 dimensions).

* Code embeddings: ``Salesforce/codet5p-110m-embedding`` (256-d, L2-normalised). The checkpoint is
  read from the configuration; a local path also works.
* Laplacian positional encoding: eigenvectors of L = I - D^{-1/2} A D^{-1/2} for the first ``k``
  non-trivial eigenvalues, computed with numpy/scipy (no DGL dependency). The graph is treated as
  undirected and simple. Missing dimensions (small graphs) are zero-padded. Eigenvector signs are
  fixed deterministically (largest-magnitude entry positive).
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Dict, List, Optional, Sequence

import networkx as nx
import numpy as np

LOGGER = logging.getLogger(__name__)

CODE_EMBEDDING_DIM = 256
LAPPE_DIM = 5
NODE_FEATURE_DIM = CODE_EMBEDDING_DIM + LAPPE_DIM
_ZERO_EIG_TOL = 1e-8
_DENSE_LIMIT = 3000


# --------------------------------------------------------------------------------------------------
# Laplacian positional encoding
# --------------------------------------------------------------------------------------------------

def laplacian_pe(graph: nx.Graph, k: int = LAPPE_DIM, nodes: Optional[Sequence] = None) -> np.ndarray:
    """Return an ``(n, k)`` float32 matrix of Laplacian eigenvector features in ``nodes`` order.

    Eigenvalues numerically equal to zero (one per connected component, including isolated nodes)
    are skipped, which handles disconnected graphs.
    """
    order = list(nodes) if nodes is not None else list(graph.nodes())
    n = len(order)
    out = np.zeros((n, k), dtype=np.float32)
    if n < 2 or k <= 0:
        return out
    index = {node: i for i, node in enumerate(order)}
    rows, cols = [], []
    for u, v in graph.edges():
        if u == v or u not in index or v not in index:
            continue
        rows += [index[u], index[v]]
        cols += [index[v], index[u]]
    if not rows:
        return out
    adj = np.zeros((n, n), dtype=np.float64) if n <= _DENSE_LIMIT else None
    if adj is not None:
        adj[rows, cols] = 1.0
        deg = adj.sum(axis=1)
        inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
        lap = np.eye(n) - (inv_sqrt[:, None] * adj * inv_sqrt[None, :])
        evals, evecs = np.linalg.eigh(lap)
    else:  # pragma: no cover - large graphs only
        import scipy.sparse as sp
        import scipy.sparse.linalg as sla

        a = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
        a.data[:] = 1.0
        deg = np.asarray(a.sum(axis=1)).ravel()
        inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
        d = sp.diags(inv_sqrt)
        lap = sp.eye(n) - d @ a @ d
        n_comp = nx.number_connected_components(graph.to_undirected(as_view=True))
        want = min(n - 1, k + n_comp + 1)
        evals, evecs = sla.eigsh(lap, k=want, which="SM")
    order_idx = np.argsort(evals)
    evals, evecs = evals[order_idx], evecs[:, order_idx]
    nontrivial = [i for i, val in enumerate(evals) if val > _ZERO_EIG_TOL][:k]
    for j, col in enumerate(nontrivial):
        vec = evecs[:, col]
        pivot = int(np.argmax(np.abs(vec)))
        if vec[pivot] < 0:
            vec = -vec
        out[:, j] = vec.astype(np.float32)
    return out


# --------------------------------------------------------------------------------------------------
# Text encoders
# --------------------------------------------------------------------------------------------------

class TextEncoder:
    """Interface: ``encode(texts) -> (n, dim)`` float32, rows L2-normalised."""

    dim: int = CODE_EMBEDDING_DIM
    name: str = "abstract"

    def encode(self, texts: Sequence[str], truncation_side: str = "right") -> np.ndarray:
        raise NotImplementedError


class CodeT5pEncoder(TextEncoder):
    """Wrapper around the CodeT5+ 110M embedding model (256-d normalised output)."""

    def __init__(self, checkpoint: str, device: str = "cpu", max_length: int = 512, batch_size: int = 32,
                 tokenizer=None, model=None):
        self.name = checkpoint
        self.device = device
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        if tokenizer is None or model is None:
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
            model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
        self.tokenizer = tokenizer
        self.model = model.to(device) if hasattr(model, "to") else model
        if hasattr(self.model, "eval"):
            self.model.eval()

    def encode(self, texts: Sequence[str], truncation_side: str = "right") -> np.ndarray:
        import torch

        texts = [t if t else " " for t in texts]
        chunks: List[np.ndarray] = []
        previous_side = getattr(self.tokenizer, "truncation_side", "right")
        try:
            if hasattr(self.tokenizer, "truncation_side"):
                self.tokenizer.truncation_side = truncation_side
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start:start + self.batch_size])
                enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length,
                                     return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
                with torch.no_grad():
                    out = self.model(**enc)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                emb = out.detach().float().cpu().numpy()
                if emb.ndim != 2 or emb.shape[1] != CODE_EMBEDDING_DIM:
                    raise ValueError(
                        f"Code encoder returned shape {tuple(emb.shape)}; expected (batch, {CODE_EMBEDDING_DIM}). "
                        "Use the CodeT5p *embedding* model, not the encoder hidden states."
                    )
                chunks.append(emb)
        finally:
            if hasattr(self.tokenizer, "truncation_side"):
                self.tokenizer.truncation_side = previous_side
        emb = np.vstack(chunks).astype(np.float32) if chunks else np.zeros((0, CODE_EMBEDDING_DIM), np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.maximum(norms, 1e-12)


class HashingTextEncoder(TextEncoder):
    """Deterministic bag-of-token hashing encoder used by the unit tests in place of CodeT5p."""

    name = "test-only-hashing-encoder"
    _TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")

    def __init__(self, dim: int = CODE_EMBEDDING_DIM):
        self.dim = dim

    def encode(self, texts: Sequence[str], truncation_side: str = "right") -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in self._TOKEN.findall(text or ""):
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 64) % 2 == 0 else -1.0
            norm = np.linalg.norm(out[i])
            out[i] = out[i] / norm if norm > 0 else out[i]
            if norm == 0:
                out[i, 0] = 1.0
        return out


class CachedEncoder(TextEncoder):
    """Memoises embeddings by text hash; optionally persisted to an ``.npz`` file."""

    def __init__(self, base: TextEncoder):
        self.base = base
        self.dim = base.dim
        self.name = base.name
        self.cache: Dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(text: str, side: str) -> str:
        return hashlib.sha1(f"{side}\0{text}".encode("utf-8")).hexdigest()

    def encode(self, texts: Sequence[str], truncation_side: str = "right") -> np.ndarray:
        keys = [self._key(t or "", truncation_side) for t in texts]
        todo = {}
        for key, text in zip(keys, texts):
            if key not in self.cache and key not in todo:
                todo[key] = text or ""
        self.misses += len(todo)
        self.hits += len(keys) - len(todo)
        if todo:
            emb = self.base.encode(list(todo.values()), truncation_side=truncation_side)
            for key, vec in zip(todo.keys(), emb):
                self.cache[key] = vec
        if not keys:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.cache[k] for k in keys]).astype(np.float32)

    def save(self, path: str) -> None:
        if self.cache:
            np.savez_compressed(path, keys=np.array(list(self.cache.keys())), values=np.stack(list(self.cache.values())))

    def load(self, path: str) -> None:
        data = np.load(path, allow_pickle=False)
        for key, vec in zip(data["keys"], data["values"]):
            self.cache[str(key)] = vec.astype(np.float32)


def build_text_encoder(cfg, device: str) -> TextEncoder:
    checkpoint = cfg.require("code_encoder")
    encoder = CodeT5pEncoder(
        checkpoint,
        device=device,
        max_length=cfg.require("code_encoder_max_length"),
        batch_size=cfg.require("embedding_batch_size"),
    )
    expected = cfg.require("code_embedding_dim")
    if expected != CODE_EMBEDDING_DIM:
        raise ValueError(f"code_embedding_dim={expected}, but this implementation expects {CODE_EMBEDDING_DIM}")
    return CachedEncoder(encoder)


# --------------------------------------------------------------------------------------------------
# Feature assembly
# --------------------------------------------------------------------------------------------------

def node_text(data: dict) -> str:
    return str(data.get("code") or data.get("name") or f"<{data.get('type', 'node')}>")


def node_features(graph: nx.Graph, encoder: TextEncoder, nodes: Optional[Sequence] = None,
                  lappe_dim: int = LAPPE_DIM) -> np.ndarray:
    """Return ``(n, 261)`` features ``[CodeT5p(code); LapPE]`` in ``nodes`` order."""
    order = list(nodes) if nodes is not None else list(graph.nodes())
    if lappe_dim != LAPPE_DIM:
        raise ValueError(f"laplacian_pe_dim={lappe_dim}, but this implementation expects {LAPPE_DIM}")
    if not order:
        return np.zeros((0, NODE_FEATURE_DIM), dtype=np.float32)
    text_emb = encoder.encode([node_text(graph.nodes[n]) for n in order])
    if text_emb.shape != (len(order), CODE_EMBEDDING_DIM):
        raise ValueError(f"code embedding shape {text_emb.shape}, expected ({len(order)}, {CODE_EMBEDDING_DIM})")
    pe = laplacian_pe(graph, k=lappe_dim, nodes=order)
    assert pe.shape == (len(order), LAPPE_DIM)
    feats = np.concatenate([text_emb, pe], axis=1).astype(np.float32)
    assert feats.shape[1] == NODE_FEATURE_DIM
    return feats
