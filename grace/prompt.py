"""Prompt construction under the paper's token budget.

Budget (paper): 2048 context tokens = 1024 local context + 1024 retrieved context. The retrieved
budget covers the retrieved code and the serialized graph together. The fixed instruction text of
the template is not counted.

Order of filling:
1. Local context: the most recent lines of the visible prefix (oldest lines are dropped first).
2. Retrieved context, rank by rank in rerank order: the unit's code snippet (cut at line granularity
   if it does not fit), then the unit's graph records (NODE before EDGE; align edges bring their
   query endpoint nodes). Remaining query-graph records come last. Atomic records that do not fit
   are skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import networkx as nx

from .serialize import Serializer, ordered_groups

TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "utils" / "prompt.md"
PLACEHOLDERS = ("repo_name", "current_file_path", "code_before_cursor", "code_context", "graph_context")
COMMENT = {"python": "#", "java": "//"}


# --------------------------------------------------------------------------------------------------
# Tokenizers
# --------------------------------------------------------------------------------------------------

class TokenCounter:
    name = "abstract"

    def count(self, text: str) -> int:
        raise NotImplementedError

    def tail(self, text: str, max_tokens: int) -> str:
        """Longest suffix of ``text`` with at most ``max_tokens`` tokens."""
        raise NotImplementedError


class RegexTokenCounter(TokenCounter):
    """Approximate tokenizer (identifiers, numbers, single symbols). For development only."""

    name = "regex-approximation"
    _TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S")

    def count(self, text: str) -> int:
        return len(self._TOKEN.findall(text or ""))

    def tail(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        matches = list(self._TOKEN.finditer(text or ""))
        if len(matches) <= max_tokens:
            return text
        return text[matches[-max_tokens].start():]


class HFTokenCounter(TokenCounter):
    def __init__(self, name: str):
        from transformers import AutoTokenizer

        self.name = f"hf:{name}"
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    def count(self, text: str) -> int:
        return len(self.tok.encode(text or "", add_special_tokens=False))

    def tail(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        ids = self.tok.encode(text or "", add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        out = self.tok.decode(ids[-max_tokens:])
        while out and self.count(out) > max_tokens:
            out = out[1:]
        return out


class TiktokenCounter(TokenCounter):
    def __init__(self, encoding: str):
        import tiktoken

        self.name = f"tiktoken:{encoding}"
        self.enc = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        return len(self.enc.encode(text or "", disallowed_special=()))

    def tail(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        ids = self.enc.encode(text or "", disallowed_special=())
        if len(ids) <= max_tokens:
            return text
        out = self.enc.decode(ids[-max_tokens:])
        while out and self.count(out) > max_tokens:
            out = out[1:]
        return out


def build_token_counter(spec: str) -> TokenCounter:
    if spec == "regex":
        return RegexTokenCounter()
    if spec.startswith("hf:"):
        return HFTokenCounter(spec[3:])
    if spec.startswith("tiktoken:"):
        return TiktokenCounter(spec[len("tiktoken:"):])
    raise ValueError(f"prompt_tokenizer must be 'regex', 'hf:<name>' or 'tiktoken:<encoding>', got {spec!r}")


# --------------------------------------------------------------------------------------------------
# Local context
# --------------------------------------------------------------------------------------------------

@dataclass
class LocalContext:
    text: str
    tokens: int
    kept_lines: int
    dropped_lines: int
    truncated: bool


def truncate_local_context(prefix: str, counter: TokenCounter, budget: int) -> LocalContext:
    """Keep the most recent lines of ``prefix`` within ``budget`` tokens."""
    lines = prefix.split("\n")
    if counter.count(prefix) <= budget:
        return LocalContext(prefix, counter.count(prefix), len(lines), 0, False)
    kept: List[str] = []
    for i in range(len(lines) - 1, -1, -1):
        candidate = "\n".join([lines[i]] + kept)
        if counter.count(candidate) <= budget:
            kept.insert(0, lines[i])
            continue
        if not kept:
            text = counter.tail(lines[i], budget)
            return LocalContext(text, counter.count(text), 1, len(lines) - 1, True)
        break
    text = "\n".join(kept)
    return LocalContext(text, counter.count(text), len(kept), len(lines) - len(kept), True)


# --------------------------------------------------------------------------------------------------
# Retrieved context
# --------------------------------------------------------------------------------------------------

@dataclass
class RetrievedBlock:
    header: str
    code: str
    node_ids: Sequence[str]


@dataclass
class RetrievedContext:
    code_context: str
    graph_context: str
    tokens: int
    id_map: Dict[int, str]
    stats: Dict[str, int] = field(default_factory=dict)


def _fit_lines(text: str, counter: TokenCounter, budget: int) -> Tuple[str, bool]:
    if counter.count(text) <= budget:
        return text, False
    kept: List[str] = []
    for line in text.split("\n"):
        candidate = "\n".join(kept + [line])
        if counter.count(candidate) > budget:
            break
        kept.append(line)
    return "\n".join(kept), True


def build_retrieved_context(graph: nx.MultiDiGraph, query_nodes: Sequence[str], blocks: Sequence[RetrievedBlock],
                            counter: TokenCounter, budget: int) -> RetrievedContext:
    ser = Serializer(graph)
    code_parts: List[str] = []
    graph_lines: List[str] = []
    used = 0
    emitted = set()
    stats = {"snippets": 0, "snippets_truncated": 0, "snippets_skipped": 0, "node_records": 0,
             "edge_records": 0, "records_skipped": 0}

    def sep_cost(parts: List[str], joiner: str) -> int:
        return counter.count(joiner) if parts else 0

    def try_node(nid: str) -> bool:
        nonlocal used
        if nid in emitted:
            return True
        provisional = dict(ser.ids)
        record = ser.node_record(nid)
        cost = counter.count(record) + sep_cost(graph_lines, "\n")
        if used + cost > budget:
            ser.ids = provisional
            stats["records_skipped"] += 1
            return False
        graph_lines.append(record)
        emitted.add(nid)
        used += cost
        stats["node_records"] += 1
        return True

    def try_edge(u: str, v: str, rel: str) -> None:
        nonlocal used
        if u not in emitted or v not in emitted:
            stats["records_skipped"] += 1
            return
        record = ser.edge_record(u, v, rel)
        cost = counter.count(record) + sep_cost(graph_lines, "\n")
        if used + cost > budget:
            stats["records_skipped"] += 1
            return
        graph_lines.append(record)
        used += cost
        stats["edge_records"] += 1

    groups = ordered_groups(graph, query_nodes, [b.node_ids for b in blocks])
    for group, nodes, edges in groups:
        if group >= 0:
            block = blocks[group]
            snippet = f"{block.header}\n{block.code}"
            remaining = budget - used - sep_cost(code_parts, "\n\n")
            text, cut = _fit_lines(snippet, counter, remaining)
            # Keep the snippet if it fits whole, or if at least one code line survives truncation.
            useful = bool(text.strip()) and (not cut or text.count("\n") >= 1)
            if useful:
                cost = counter.count(text) + sep_cost(code_parts, "\n\n")
                code_parts.append(text)
                used += cost
                stats["snippets"] += 1
                stats["snippets_truncated"] += int(cut)
            else:
                stats["snippets_skipped"] += 1
        for nid in nodes:
            try_node(nid)
        for u, v, rel in edges:
            if rel == "align" and u not in emitted:
                try_node(u)
            try_edge(u, v, rel)
    code_context = "\n\n".join(code_parts)
    graph_context = "\n".join(graph_lines)
    total = counter.count(code_context) + counter.count(graph_context)
    stats["tokens"] = total
    return RetrievedContext(code_context, graph_context, total, ser.id_map(), stats)


def snippet_header(language: str, rank: int, unit) -> str:
    mark = COMMENT.get(language, "#")
    return f"{mark} [{rank + 1}] {unit.file_path}:{unit.start_line}-{unit.end_line} ({unit.kind} {unit.name})"


# --------------------------------------------------------------------------------------------------
# Template
# --------------------------------------------------------------------------------------------------

def render_prompt(values: Dict[str, str], template_path: Optional[str] = None) -> str:
    path = Path(template_path) if template_path else TEMPLATE_PATH
    template = path.read_text(encoding="utf-8")
    pattern = re.compile(r"\{(" + "|".join(PLACEHOLDERS) + r")\}")
    # One pass, so placeholder-like text inside inserted code is never substituted again.
    return pattern.sub(lambda m: str(values.get(m.group(1), "")), template)
