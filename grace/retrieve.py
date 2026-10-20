"""Hybrid graph retriever: semantic + structural candidates, gated scoring, MMR.

Score(c) = alpha * SemSim(q, c) + (1 - alpha) * StructSim(G_q, G_c), both cosine similarities.

1. Each path retrieves up to ``K_c`` candidates (semantic: CodeT5p of the local context against unit
   text embeddings; structural: pooled GAT embedding of G_q against unit graph embeddings).
2. The union is scored with exact cosines and the gate value ``alpha``.
3. MMR keeps ``K_c`` candidates: argmax  lambda * Score(c) - (1 - lambda) * max_{s in S} sim(c, s),
   with sim(c, s) = alpha * cos_sem(c, s) + (1 - alpha) * cos_struct(c, s).
4. The pool is reranked by Score and the top ``K_f`` units are fused.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List

import numpy as np

from .index import TaskIndex, _normalise


@dataclass
class Candidate:
    index: int
    unit_id: str
    sem_sim: float
    struct_sim: float
    score: float
    from_semantic: bool
    from_structural: bool
    mmr_order: int = -1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalResult:
    alpha: float
    alpha_mode: str
    merged: List[Candidate]  # union of both paths, sorted by score
    pool: List[Candidate]  # after MMR (<= K_c), sorted by score
    selected: List[Candidate]  # top-K_f of pool

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "alpha_mode": self.alpha_mode,
            "num_merged": len(self.merged),
            "pool": [c.to_dict() for c in self.pool],
            "selected": [c.unit_id for c in self.selected],
        }


def mmr_select(cands: List[Candidate], index: TaskIndex, alpha: float, k: int, lam: float) -> List[Candidate]:
    remaining = list(cands)
    chosen: List[Candidate] = []
    while remaining and len(chosen) < k:
        best, best_val = None, -np.inf
        for cand in remaining:
            if chosen:
                ids = [c.index for c in chosen]
                red_sem = index.sem[ids] @ index.sem[cand.index]
                red_struct = index.struct[ids] @ index.struct[cand.index]
                redundancy = float(np.max(alpha * red_sem + (1 - alpha) * red_struct))
            else:
                redundancy = 0.0
            value = lam * cand.score - (1 - lam) * redundancy
            if value > best_val + 1e-12:
                best, best_val = cand, value
        best.mmr_order = len(chosen)
        chosen.append(best)
        remaining.remove(best)
    return chosen


def hybrid_retrieve(index: TaskIndex, v_q: np.ndarray, h_gq: np.ndarray, alpha: float, alpha_mode: str,
                    k_c: int, k_f: int, mmr_lambda: float) -> RetrievalResult:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0,1], got {alpha}")
    if len(index) == 0:
        return RetrievalResult(alpha, alpha_mode, [], [], [])
    v_q = _normalise(v_q)
    h_gq = _normalise(h_gq)
    sem_ids = index.sem_index.search(v_q, k_c)
    struct_ids = index.struct_index.search(h_gq, k_c)
    merged = []
    for i in dict.fromkeys(sem_ids + struct_ids):
        sem = float(index.sem[i] @ v_q)
        struct = float(index.struct[i] @ h_gq)
        merged.append(
            Candidate(
                index=i,
                unit_id=index.units[i].unit_id,
                sem_sim=sem,
                struct_sim=struct,
                score=alpha * sem + (1 - alpha) * struct,
                from_semantic=i in sem_ids,
                from_structural=i in struct_ids,
            )
        )
    merged.sort(key=lambda c: (-c.score, c.unit_id))
    pool = mmr_select(merged, index, alpha, k_c, mmr_lambda)
    pool.sort(key=lambda c: (-c.score, c.unit_id))
    return RetrievalResult(alpha, alpha_mode, merged, pool, pool[:k_f])
