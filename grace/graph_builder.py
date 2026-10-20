"""Multi-level code graph construction for Python and Java with Tree-sitter.

* repository level: folder/file containment (``contains``) and import edges (``imports``);
* module level: calls (``calls``), class inheritance (``inherits``), type dependencies (``type_ref``);
* function level: AST (``ast_root``/``ast_child``), CFG (``control_flow``) and DFG
  (``defines`` statement->variable, ``data_flow`` variable->use statement);
* cross-level: file/class/function -> declared definitions (``declares``), file-level dependency
  mapped to inter-function calls (``dep_call``), and type usages aligned with data-flow variables
  (``type_flow``). Statements are shared between the AST and the CFG, and variables connect them to
  the DFG.

Source files are parsed with the Tree-sitter Python and Java grammars (``tree-sitter-python``,
``tree-sitter-java``). Every node keeps its file and 1-based line span. The current file is cut at
the cursor and is usually incomplete; :func:`parse_source` repairs the tail (closing a trailing
Python block or open Java braces, then dropping trailing lines) so that the visible code yields an
error-free tree.

Unresolved symbols are not promoted to cross-file edges: a call/type edge is added only when the
name is defined in the same file or in a file reachable by an import (Java: same package or import).
"""

from __future__ import annotations

import ctypes
import functools
import importlib
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx

LOGGER = logging.getLogger(__name__)

PARSER_BACKEND = "tree-sitter"

NODE_TYPES = ("folder", "file", "class", "function", "statement", "variable")

RELATIONS = (
    "contains",
    "imports",
    "declares",
    "calls",
    "inherits",
    "type_ref",
    "ast_root",
    "ast_child",
    "control_flow",
    "defines",
    "data_flow",
    "dep_call",
    "type_flow",
    "align",
    "self",
    "other",
)
RELATION_TO_ID = {name: i for i, name in enumerate(RELATIONS)}

# Relations that stay inside the body of a definition (used to cut retrieval units).
BODY_RELATIONS = {"ast_root", "ast_child", "control_flow", "defines", "data_flow"}
# Relations that link definitions to other definitions (used for hop expansion).
SEMANTIC_RELATIONS = {"calls", "inherits", "type_ref", "imports", "dep_call", "type_flow", "declares"}


def relation_id(name: str) -> int:
    return RELATION_TO_ID.get(name, RELATION_TO_ID["other"])


# --------------------------------------------------------------------------------------------------
# Tree-sitter parsers
# --------------------------------------------------------------------------------------------------

_GRAMMARS = {"python": "tree_sitter_python", "java": "tree_sitter_java"}
_PARSERS: Dict[str, object] = {}
# Tail repair drops at most this many trailing lines; errors further up are left to Tree-sitter's
# error recovery (ERROR subtrees are skipped by the builders).
MAX_TAIL_REPAIR_LINES = 50
_COMMENTS = {"comment", "line_comment", "block_comment"}


def _grammar_pointer(grammar) -> int:
    """Address of the TSLanguage behind a grammar object (an int or a PyCapsule)."""
    if isinstance(grammar, int):
        return grammar
    api = ctypes.pythonapi
    api.PyCapsule_GetName.restype = ctypes.c_char_p
    api.PyCapsule_GetName.argtypes = [ctypes.py_object]
    api.PyCapsule_GetPointer.restype = ctypes.c_void_p
    api.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return int(api.PyCapsule_GetPointer(grammar, api.PyCapsule_GetName(grammar)))


def _load_language(language: str):
    import tree_sitter

    grammar = importlib.import_module(_GRAMMARS[language]).language()
    try:
        return tree_sitter.Language(grammar)  # py-tree-sitter >= 0.22
    except TypeError:
        pass
    pointer = _grammar_pointer(grammar)
    try:
        return tree_sitter.Language(pointer, language)  # py-tree-sitter 0.21
    except Exception:
        # py-tree-sitter 0.20 only loads shared libraries by path; bind the grammar pointer directly.
        lang = tree_sitter.Language.__new__(tree_sitter.Language)
        lang.name = language
        lang.language_id = pointer
        return lang


def get_parser(language: str):
    """Return a cached Tree-sitter parser for ``python`` or ``java``."""
    if language not in _GRAMMARS:
        raise ValueError(f"Unsupported language: {language}")
    parser = _PARSERS.get(language)
    if parser is None:
        import tree_sitter

        lang = _load_language(language)
        if hasattr(tree_sitter.Parser, "set_language"):  # py-tree-sitter < 0.23
            parser = tree_sitter.Parser()
            parser.set_language(lang)
        else:
            parser = tree_sitter.Parser(lang)
        _PARSERS[language] = parser
    return parser


def _text(node) -> str:
    return node.text.decode("utf-8", errors="replace") if node is not None else ""


def _line(node) -> int:
    return node.start_point[0] + 1


def _end_line(node) -> int:
    row, col = node.end_point
    return max(row + (1 if col > 0 else 0), _line(node))


def _col(node) -> int:
    return node.start_point[1]


def _key(node) -> Tuple[int, int, str]:
    return node.start_byte, node.end_byte, node.type


def _same(a, b) -> bool:
    return a is not None and b is not None and _key(a) == _key(b)


def _first_error_line(root) -> Optional[int]:
    if not root.has_error:
        return None
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "ERROR" or node.is_missing:
            return _line(node)
        if node.has_error:
            stack.extend(reversed(node.children))
    return _line(root)


def _python_with_pass(chunk: str) -> Optional[str]:
    lines = chunk.split("\n")
    for line in reversed(lines):
        if line.strip():
            if line.rstrip().endswith(":"):
                indent = line[: len(line) - len(line.lstrip())]
                return chunk + "\n" + indent + "    pass"
            return None
    return None


def _java_open_braces(text: str) -> int:
    """Number of unclosed ``{`` outside comments, strings and character literals."""
    depth, i, n = 0, 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if text.startswith('"""', i):
            j = text.find('"""', i + 3)
            i = n if j < 0 else j + 3
            continue
        if c in "\"'":
            j = i + 1
            while j < n and text[j] not in (c, "\n"):
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    return max(depth, 0)


def _repairs(language: str, chunk: str) -> List[str]:
    if language == "python":
        fixed = _python_with_pass(chunk)
        return [chunk] + ([fixed] if fixed else [])
    depth = _java_open_braces(chunk)
    return [chunk + "\n" + "}" * depth] if depth else [chunk]


@functools.lru_cache(maxsize=8192)
def parse_source(language: str, text: str):
    """Parse ``text`` and return ``(tree, kept_lines)``.

    If the tree has errors near the end (code cut at the cursor), the tail is repaired and trailing
    lines are dropped until the tree is error-free; ``kept_lines`` is the number of original lines
    that the tree covers, and nodes starting after them (for example an appended ``pass``) must be
    ignored by callers. Errors further up are left to Tree-sitter's error recovery.
    """
    parser = get_parser(language)
    lines = text.split("\n")
    n = len(lines)
    while n > 0 and not lines[n - 1].strip():
        n -= 1  # trailing blank lines carry no code; spans are clamped to the last code line
    floor = max(len(lines) - MAX_TAIL_REPAIR_LINES, 0)
    while n >= floor:
        chunk = "\n".join(lines[:n])
        err_line = None
        for candidate in _repairs(language, chunk):
            tree = parser.parse(candidate.encode("utf-8"))
            line = _first_error_line(tree.root_node)
            if line is None:
                return tree, n
            err_line = line if err_line is None else min(err_line, line)
        n = min(n - 1, err_line) if err_line else n - 1
    return parser.parse(text.encode("utf-8")), len(lines)


# --------------------------------------------------------------------------------------------------
# Shared graph writer
# --------------------------------------------------------------------------------------------------

class _BaseBuilder:
    language = ""

    def __init__(self, files: Dict[str, str]):
        self.files = files
        self.g = nx.MultiDiGraph()
        self.symbols: Dict[str, Dict[str, List[str]]] = {"function": defaultdict(list), "class": defaultdict(list)}
        self.def_file: Dict[str, str] = {}
        self.import_targets: Dict[str, set] = defaultdict(set)
        self.records: List[dict] = []
        self.inherit_records: List[Tuple[str, str, str]] = []
        self.var_index: Dict[Tuple[str, str], str] = {}
        self.lines: Dict[str, List[str]] = {rel: text.split("\n") for rel, text in files.items()}

    # -- primitives ---------------------------------------------------------------------------
    def node(self, nid: str, **attrs) -> None:
        clean = {k: v for k, v in attrs.items() if v is not None}
        if nid in self.g:
            for k, v in clean.items():
                self.g.nodes[nid].setdefault(k, v)
        else:
            self.g.add_node(nid, **clean)

    def edge(self, src: str, dst: str, relation: str) -> None:
        if src == dst or src not in self.g or dst not in self.g:
            return
        if not self.g.has_edge(src, dst, key=relation):
            self.g.add_edge(src, dst, key=relation, relation=relation)

    def text_span(self, rel: str, start: int, end: int, joiner: str = " ") -> str:
        lines = self.lines.get(rel, [])
        return joiner.join(ln.strip() for ln in lines[max(start - 1, 0): max(end, start)] if ln.strip())

    # -- repository level ---------------------------------------------------------------------
    def add_files(self) -> None:
        self.node("folder:.", type="folder", code=".", name=".", file_path=".", level="repository")
        for rel in sorted(self.files):
            parts = rel.split("/")
            parent = "folder:."
            for i in range(len(parts) - 1):
                folder = "/".join(parts[: i + 1])
                fid = f"folder:{folder}"
                self.node(fid, type="folder", code=folder, name=parts[i], file_path=folder, level="repository")
                self.edge(parent, fid, "contains")
                parent = fid
            nlines = len(self.lines[rel])
            self.node(
                f"file:{rel}",
                type="file",
                code=rel,
                name=parts[-1],
                file_path=rel,
                start_line=1,
                end_line=nlines,
                level="repository",
            )
            self.edge(parent, f"file:{rel}", "contains")

    def add_import_edge(self, src_rel: str, dst_rel: Optional[str]) -> None:
        if dst_rel and dst_rel != src_rel and dst_rel in self.files:
            self.import_targets[src_rel].add(dst_rel)
            self.edge(f"file:{src_rel}", f"file:{dst_rel}", "imports")

    # -- definitions --------------------------------------------------------------------------
    def add_definition(self, nid: str, kind: str, name: str, rel: str, parent: str, **attrs) -> str:
        if nid in self.g:
            nid = f"{nid}@{attrs.get('start_line')}"
        self.node(nid, type=kind, name=name, file_path=rel, level="module", **attrs)
        self.symbols[kind][name].append(nid)
        self.def_file[nid] = rel
        self.edge(parent, nid, "declares")
        return nid

    def visible_files(self, rel: str) -> set:
        return set(self.import_targets.get(rel, set()))

    def resolve(self, name: str, rel: str, kind: str) -> List[str]:
        cands = self.symbols[kind].get(name, [])
        if not cands:
            return []
        same = [c for c in cands if self.def_file[c] == rel]
        if same:
            return same
        visible = self.visible_files(rel)
        return [c for c in cands if self.def_file[c] in visible]

    # -- function level records -> DFG / calls / types ----------------------------------------
    def record(self, rel: str, sid: str, line: int, scope: Sequence[str], owner: str, defs, uses, calls, types) -> None:
        self.records.append(
            {
                "rel": rel,
                "sid": sid,
                "line": line,
                "scope": list(scope),
                "owner": owner,
                "defs": list(defs),
                "uses": list(dict.fromkeys(uses)),
                "calls": list(dict.fromkeys(calls)),
                "types": list(dict.fromkeys(types)),
            }
        )

    def finalize(self) -> None:
        for rec in self.records:
            for item in rec["defs"]:
                name, hint = item[0], item[1]
                rebind = len(item) > 2 and item[2]
                vid = None
                scopes = rec["scope"]
                if rebind:
                    # Assignment to an existing name (e.g. a Java field): reuse the enclosing definition.
                    # rebind == "field" (``this.x = ...``) skips function scopes so a parameter with the
                    # same name is not mistaken for the field.
                    if rebind == "field":
                        scopes = [s for s in scopes if not s.startswith("func:")]
                    for scope in scopes:
                        vid = self.var_index.get((scope, name))
                        if vid:
                            break
                if vid is None:
                    home = scopes[0] if rebind == "field" and scopes else rec["scope"][0]
                    vid = f"var:{home}:{name}"
                    self.node(
                        vid,
                        type="variable",
                        code=name,
                        name=name,
                        file_path=rec["rel"],
                        start_line=rec["line"],
                        end_line=rec["line"],
                        level="function",
                    )
                    self.var_index[(home, name)] = vid
                self.edge(rec["sid"], vid, "defines")
                if hint:
                    for cls in self.resolve(hint, rec["rel"], "class"):
                        self.g.nodes[vid].setdefault("type_name", self.g.nodes[cls].get("name"))
                        self.edge(vid, cls, "type_flow")
        for rec in self.records:
            for name in rec["uses"]:
                for scope in rec["scope"]:
                    vid = self.var_index.get((scope, name))
                    if vid:
                        self.edge(vid, rec["sid"], "data_flow")
                        break
            owner = rec["owner"]
            for name in rec["calls"]:
                for callee in self.resolve(name, rec["rel"], "function"):
                    self.edge(owner, callee, "calls")
                    callee_file = self.def_file[callee]
                    if callee_file != rec["rel"] and callee_file in self.import_targets.get(rec["rel"], ()):
                        self.edge(f"file:{rec['rel']}", callee, "dep_call")
            for name in rec["types"]:
                for cls in self.resolve(name, rec["rel"], "class"):
                    self.edge(owner, cls, "type_ref")
        for cls_id, base, rel in self.inherit_records:
            for parent in self.resolve(base, rel, "class"):
                self.edge(cls_id, parent, "inherits")


# --------------------------------------------------------------------------------------------------
# Python
# --------------------------------------------------------------------------------------------------

_PY_DEFS = {"function_definition", "class_definition"}
_PY_COMPOUND = {"if_statement", "for_statement", "while_statement", "try_statement", "with_statement",
                "match_statement"}
_PY_CLAUSES = {"elif_clause", "else_clause", "except_clause", "except_group_clause", "finally_clause"}
_PY_LOOPS = {"for_statement", "while_statement"}
_PY_STORE_CONTAINERS = {"pattern_list", "tuple_pattern", "list_pattern", "tuple", "list",
                        "parenthesized_expression", "list_splat_pattern", "as_pattern_target"}
_PY_NO_NAMES = {"global_statement", "nonlocal_statement", "import_statement", "import_from_statement",
                "future_import_statement", "lambda_parameters", "block", "ERROR"} | _COMMENTS


def _py_string_value(node) -> Optional[str]:
    parts = [_text(c) for c in node.children if c.type == "string_content"]
    value = "".join(parts).split(".")[-1]
    return value or None


def _py_name(expr) -> Optional[str]:
    if expr is None:
        return None
    if expr.type == "identifier":
        return _text(expr)
    if expr.type == "attribute":
        return _text(expr.child_by_field_name("attribute")) or None
    if expr.type == "subscript":
        return _py_name(expr.child_by_field_name("value"))
    if expr.type in ("generic_type", "type") and expr.named_children:
        return _py_name(expr.named_children[0])
    if expr.type == "string":
        return _py_string_value(expr)
    return None


def _py_annotation_names(node) -> List[str]:
    if node is None:
        return []
    names, stack = [], [node]
    while stack:
        cur = stack.pop()
        if cur.type == "identifier":
            names.append(_text(cur))
        elif cur.type == "attribute":
            attr = cur.child_by_field_name("attribute")
            if attr is not None:
                names.append(_text(attr))
            obj = cur.child_by_field_name("object")
            if obj is not None:
                stack.append(obj)
        elif cur.type == "string":
            value = _py_string_value(cur)
            if value:
                names.append(value)
        else:
            stack.extend(reversed(cur.named_children))
    return names


def _py_statements(container) -> List:
    """Statements directly inside a module or block (comments and unrecoverable errors skipped)."""
    return [c for c in container.named_children if c.type not in _COMMENTS and c.type != "ERROR"]


def _py_definition(stmt):
    """The function/class definition of ``stmt`` (unwrapping decorators), or ``None``."""
    if stmt.type == "decorated_definition":
        stmt = stmt.child_by_field_name("definition")
    return stmt if stmt is not None and stmt.type in _PY_DEFS else None


def _py_blocks(stmt) -> List:
    """Nested statement blocks of a compound statement, in source order."""
    if stmt.type == "match_statement":
        body = stmt.child_by_field_name("body")
        cases = [c for c in body.named_children if c.type == "case_clause"] if body is not None else []
        return [b for case in cases for b in case.children if b.type == "block"]
    blocks = []
    for child in stmt.children:
        if child.type == "block":
            blocks.append(child)
        elif child.type in _PY_CLAUSES:
            blocks.extend(c for c in child.children if c.type == "block")
    return blocks


def _py_first_block(stmt):
    for child in stmt.children:
        if child.type == "block":
            return child
    return None


class PythonGraphBuilder(_BaseBuilder):
    language = "python"

    def build(self) -> nx.MultiDiGraph:
        self.add_files()
        self.trees: Dict[str, Tuple[object, int]] = {}
        for rel, text in self.files.items():
            tree, kept = parse_source("python", text)
            self.trees[rel] = (tree.root_node, kept)
        self._suffix_index = defaultdict(list)
        for rel in self.files:
            parts = rel.split("/")
            for i in range(len(parts)):
                self._suffix_index["/".join(parts[i:])].append(rel)
        self.def_by_pos: Dict[Tuple[str, int, int], str] = {}
        for rel, (root, kept) in self.trees.items():
            self._collect_defs(rel, _py_statements(root), f"file:{rel}", [], kept)
        for rel, (root, kept) in self.trees.items():
            self._collect_imports(rel, root, kept)
        for rel, (root, kept) in self.trees.items():
            self._visit_block(rel, _py_statements(root), f"file:{rel}", [f"file:{rel}"], f"file:{rel}", None, kept)
        self.finalize()
        return self.g

    def _header(self, rel: str, stmt, kept: int) -> str:
        start = _line(stmt)
        body = _py_first_block(stmt)
        end = start
        if body is not None and _line(body) > start:
            end = _line(body) - 1
        return self.text_span(rel, start, min(end, kept))

    def _collect_defs(self, rel: str, body: List, parent: str, qual: List[str], kept: int) -> None:
        for stmt in body:
            if _line(stmt) > kept:
                continue
            definition = _py_definition(stmt)
            if definition is not None:
                name = _text(definition.child_by_field_name("name"))
                if not name:
                    continue
                kind = "class" if definition.type == "class_definition" else "function"
                q = qual + [name]
                prefix = "class" if kind == "class" else "func"
                nid = self.add_definition(
                    f"{prefix}:{rel}:{'.'.join(q)}",
                    kind,
                    name,
                    rel,
                    parent,
                    code=self._header(rel, definition, kept),
                    qualname=".".join(q),
                    start_line=_line(definition),
                    end_line=min(_end_line(definition), kept),
                )
                self.def_by_pos[(rel, _line(definition), _col(definition))] = nid
                inner = _py_first_block(definition)
                if inner is not None:
                    self._collect_defs(rel, _py_statements(inner), nid, q, kept)
            else:
                for block in _py_blocks(stmt):
                    self._collect_defs(rel, _py_statements(block), parent, qual, kept)

    def _module_file(self, rel: str, module: Optional[str], level: int) -> Optional[str]:
        if level:
            base = rel.split("/")[:-1]
            if level - 1:
                base = base[: max(len(base) - (level - 1), 0)]
            parts = base + (module.split(".") if module else [])
        else:
            if not module:
                return None
            parts = module.split(".")
        if not parts:
            return None
        path = "/".join(parts)
        for cand in (path + ".py", path + "/__init__.py"):
            if cand in self.files:
                return cand
        if not level:
            for cand in (path + ".py", path + "/__init__.py"):
                matches = self._suffix_index.get(cand, [])
                if len(matches) == 1:
                    return matches[0]
        return None

    @staticmethod
    def _imported_name(node) -> Optional[str]:
        if node.type == "aliased_import":
            node = node.child_by_field_name("name")
        return _text(node) if node is not None and node.type == "dotted_name" else None

    @staticmethod
    def _import_alias(node) -> Optional[str]:
        if node.type == "aliased_import":
            alias = node.child_by_field_name("alias")
            if alias is not None:
                return _text(alias)
            node = node.child_by_field_name("name")
        return _text(node).split(".")[0] if node is not None and node.type == "dotted_name" else None

    def _collect_imports(self, rel: str, root, kept: int) -> None:
        stack = [root]
        while stack:
            node = stack.pop()
            if node.type == "ERROR" or _line(node) > kept:
                continue
            if node.type == "import_statement":
                for name in node.children_by_field_name("name"):
                    module = self._imported_name(name)
                    if module:
                        self.add_import_edge(rel, self._module_file(rel, module, 0))
            elif node.type == "import_from_statement":
                module_node = node.child_by_field_name("module_name")
                module, level = None, 0
                if module_node is not None and module_node.type == "relative_import":
                    for child in module_node.named_children:
                        if child.type == "import_prefix":
                            level = _text(child).count(".")
                        elif child.type == "dotted_name":
                            module = _text(child)
                elif module_node is not None:
                    module = _text(module_node)
                self.add_import_edge(rel, self._module_file(rel, module, level))
                for name in node.children_by_field_name("name"):
                    imported = self._imported_name(name)
                    if imported:
                        sub = f"{module}.{imported}" if module else imported
                        self.add_import_edge(rel, self._module_file(rel, sub, level))
            else:
                stack.extend(node.named_children)

    def _stmt_code(self, rel: str, stmt, kept: int) -> str:
        if stmt.type in _PY_COMPOUND:
            return self._header(rel, stmt, kept)
        return self.text_span(rel, _line(stmt), min(_end_line(stmt), kept))

    def _record_statement(self, rel: str, sid: str, stmt, scope: List[str], owner: str) -> List[str]:
        defs, uses, calls, types, refs = [], [], [], [], []
        hint = None
        inner = stmt.named_children[0] if stmt.type == "expression_statement" and stmt.named_children else None
        if inner is not None and inner.type == "assignment":
            right = inner.child_by_field_name("right")
            if right is not None and right.type == "call":
                hint = _py_name(right.child_by_field_name("function"))
            ann_names = _py_annotation_names(inner.child_by_field_name("type"))
            if ann_names:
                hint = ann_names[0]
                types.extend(ann_names)
        if stmt.type in ("import_statement", "import_from_statement"):
            for name in stmt.children_by_field_name("name"):
                alias = self._import_alias(name)
                if alias:
                    defs.append((alias, None))
        stack = [(stmt, False)]
        while stack:
            node, store = stack.pop()
            t = node.type
            if t in _PY_NO_NAMES:
                continue
            children = node.children
            if t == "identifier":
                if store:
                    defs.append((_text(node), hint))
                else:
                    uses.append(_text(node))
                    refs.append(_text(node))
                continue
            if t == "attribute":
                attr, obj = node.child_by_field_name("attribute"), node.child_by_field_name("object")
                if attr is not None:
                    refs.append(_text(attr))
                if obj is not None:
                    stack.append((obj, False))
                continue
            if t == "call":
                name = _py_name(node.child_by_field_name("function"))
                if name:
                    calls.append(name)
                    types.append(name)  # constructor calls resolve to classes
            elif t == "keyword_argument":
                value = node.child_by_field_name("value")
                if value is not None:
                    stack.append((value, False))
                continue
            stores = []
            if t in ("assignment", "augmented_assignment", "for_statement", "for_in_clause"):
                stores = [node.child_by_field_name("left")]
            elif t == "named_expression":
                stores = [node.child_by_field_name("name")]
            elif t == "as_pattern":
                stores = [node.child_by_field_name("alias")]
            store_keys = {_key(s) for s in stores if s is not None}
            child_store = store and t in _PY_STORE_CONTAINERS
            for child in reversed(children):
                stack.append((child, child_store or _key(child) in store_keys))
        self.record(rel, sid, _line(stmt), scope, owner, defs, uses, calls, types)
        return refs

    def _function_params(self, definition) -> Tuple[List[tuple], List[str]]:
        defs, types = [], []
        params = definition.child_by_field_name("parameters")
        for param in params.named_children if params is not None else []:
            if param.type == "identifier":
                defs.append((_text(param), None))
                continue
            name = param.child_by_field_name("name")
            if name is None:
                name = next((c for c in param.named_children if c.type == "identifier"), None)
            if name is None:
                continue
            names = _py_annotation_names(param.child_by_field_name("type"))
            defs.append((_text(name), names[0] if names else None))
            types.extend(names)
        types.extend(_py_annotation_names(definition.child_by_field_name("return_type")))
        return defs, types

    def _visit_block(self, rel, body, parent, scope, owner, func_id, kept) -> Tuple[Optional[str], Optional[str]]:
        first = last = None
        pending_exits: List[str] = []
        for stmt in body:
            if _line(stmt) > kept:
                continue
            definition = _py_definition(stmt)
            if definition is not None:
                nid = self.def_by_pos.get((rel, _line(definition), _col(definition)))
                if nid is None:
                    continue
                inner = _py_first_block(definition)
                inner_body = _py_statements(inner) if inner is not None else []
                if definition.type == "class_definition":
                    bases = definition.child_by_field_name("superclasses")
                    for base in bases.named_children if bases is not None else []:
                        if base.type == "keyword_argument":
                            continue
                        name = _py_name(base)
                        if name:
                            self.inherit_records.append((nid, name, rel))
                    self._visit_block(rel, inner_body, nid, [nid] + list(scope), nid, None, kept)
                else:
                    fn_scope = [nid] + [s for s in scope if not s.startswith("class:")]
                    defs, types = self._function_params(definition)
                    self.record(rel, nid, _line(definition), fn_scope, nid, defs, [], [], types)
                    self._visit_block(rel, inner_body, nid, fn_scope, nid, nid, kept)
                continue
            sid = f"stmt:{rel}:{_line(stmt)}:{_col(stmt)}"
            self.node(
                sid,
                type="statement",
                code=self._stmt_code(rel, stmt, kept),
                file_path=rel,
                start_line=_line(stmt),
                end_line=min(_end_line(stmt), kept),
                level="function",
                syntax=stmt.type,
            )
            self.edge(parent, sid, "ast_root" if func_id is not None and parent == func_id else "ast_child")
            if last is not None:
                self.edge(last, sid, "control_flow")
            for exit_id in pending_exits:
                self.edge(exit_id, sid, "control_flow")
            pending_exits = []
            first = first or sid
            last = sid
            refs = self._record_statement(rel, sid, stmt, scope, owner)
            if refs:
                self.g.nodes[sid]["refs"] = " ".join(sorted(set(refs)))
            for block in _py_blocks(stmt):
                b_first, b_last = self._visit_block(rel, _py_statements(block), sid, scope, owner, func_id, kept)
                if b_first:
                    self.edge(sid, b_first, "control_flow")
                if b_last:
                    if stmt.type in _PY_LOOPS:
                        self.edge(b_last, sid, "control_flow")
                    pending_exits.append(b_last)
        return first, last


# --------------------------------------------------------------------------------------------------
# Java
# --------------------------------------------------------------------------------------------------

_JAVA_TYPES = {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration",
               "annotation_type_declaration"}
_JAVA_METHODS = {"method_declaration", "constructor_declaration", "compact_constructor_declaration"}
_JAVA_FIELDS = {"field_declaration", "constant_declaration"}
_JAVA_COMPOUND = {"if_statement", "while_statement", "do_statement", "for_statement",
                  "enhanced_for_statement", "try_statement", "try_with_resources_statement",
                  "switch_expression", "switch_statement", "synchronized_statement"}
_JAVA_LOOPS = {"while_statement", "do_statement", "for_statement", "enhanced_for_statement"}
_JAVA_SKIP_IN_BLOCK = _COMMENTS | {"ERROR", "class_declaration", "interface_declaration", "enum_declaration",
                                   "record_declaration", "local_class_declaration", ";"}
# Statement-level recording stops at nested statement bodies (they are separate statement nodes).
_JAVA_STOP = {"block", "constructor_body", "class_body", "switch_block", "expression_statement",
              "if_statement", "while_statement", "do_statement", "for_statement", "enhanced_for_statement",
              "try_statement", "try_with_resources_statement", "synchronized_statement", "return_statement",
              "throw_statement", "break_statement", "continue_statement", "yield_statement",
              "assert_statement", "labeled_statement", "switch_statement", "ERROR"} | _COMMENTS
_JAVA_PRIMITIVES = {"integral_type", "floating_point_type", "boolean_type", "void_type"}


def _java_type_name(node) -> Optional[str]:
    if node is None or node.type in _JAVA_PRIMITIVES:
        return None
    if node.type in ("type_identifier", "identifier"):
        return _text(node)
    if node.type == "scoped_type_identifier":
        names = [c for c in node.named_children if c.type == "type_identifier"]
        return _text(names[-1]) if names else None
    if node.type in ("generic_type", "array_type", "annotated_type"):
        for child in node.named_children:
            if child.type not in ("type_arguments", "dimensions", "marker_annotation", "annotation"):
                return _java_type_name(child)
    return None


def _java_types_in(node) -> List:
    """Type nodes listed under ``superclass`` / ``super_interfaces`` / ``extends_interfaces``."""
    if node is None:
        return []
    out = []
    for child in node.named_children:
        if child.type == "type_list":
            out.extend(child.named_children)
        else:
            out.append(child)
    return out


def _java_root_identifier(node) -> Optional[str]:
    while node is not None and node.type in ("field_access", "method_invocation"):
        node = node.child_by_field_name("object")
    return _text(node) if node is not None and node.type == "identifier" else None


class JavaGraphBuilder(_BaseBuilder):
    language = "java"

    def build(self) -> nx.MultiDiGraph:
        self.add_files()
        self.parsed: Dict[str, Tuple[object, int]] = {}
        for rel, text in self.files.items():
            tree, kept = parse_source("java", text)
            self.parsed[rel] = (tree.root_node, kept)
        self.package_of: Dict[str, str] = {}
        self.package_files: Dict[str, set] = defaultdict(set)
        self.fqn_file: Dict[str, str] = {}
        self.method_nodes: List[Tuple[str, object, str, List[str], str, int]] = []
        for rel, (root, kept) in self.parsed.items():
            pkg = ""
            for child in root.named_children:
                if child.type == "package_declaration":
                    name = next((c for c in child.named_children if c.type in ("scoped_identifier", "identifier")), None)
                    pkg = _text(name)
            self.package_of[rel] = pkg
            self.package_files[pkg].add(rel)
            for child in root.named_children:
                if child.type in _JAVA_TYPES:
                    self._collect_type(rel, child, f"file:{rel}", [], pkg, kept)
        for rel, (root, _kept) in self.parsed.items():
            self._collect_imports(rel, root)
        for method_id, decl, rel, scope, owner, kept in self.method_nodes:
            self._visit_method(method_id, decl, rel, scope, owner, kept)
        self.finalize()
        return self.g

    # -- declarations -------------------------------------------------------------------------
    @staticmethod
    def _members(decl) -> List:
        body = decl.child_by_field_name("body")
        if body is None:
            return []
        members = []
        for child in body.named_children:
            if child.type == "enum_body_declarations":
                members.extend(child.named_children)
            else:
                members.append(child)
        return members

    @staticmethod
    def _header_end(decl) -> int:
        body = decl.child_by_field_name("body")
        return _line(body) if body is not None else _end_line(decl)

    def _collect_type(self, rel, decl, parent, qual, pkg, kept) -> None:
        name = _text(decl.child_by_field_name("name"))
        if not name or _line(decl) > kept:
            return
        line = _line(decl)
        q = qual + [name]
        cls_id = self.add_definition(
            f"class:{rel}:{'.'.join(q)}",
            "class",
            name,
            rel,
            parent,
            code=self.text_span(rel, line, min(self._header_end(decl), kept)),
            qualname=".".join(q),
            start_line=line,
            end_line=min(_end_line(decl), kept),
        )
        self.fqn_file[".".join(([pkg] if pkg else []) + q)] = rel
        bases = _java_types_in(decl.child_by_field_name("superclass"))
        bases += _java_types_in(decl.child_by_field_name("interfaces"))
        for child in decl.named_children:
            if child.type == "extends_interfaces":
                bases += _java_types_in(child)
        for base in bases:
            base_name = _java_type_name(base)
            if base_name:
                self.inherit_records.append((cls_id, base_name, rel))
        class_scope = [cls_id] + ([parent] if parent.startswith("class:") else []) + [f"file:{rel}"]
        prev = None
        for member in self._members(decl):
            if _line(member) > kept:
                continue
            if member.type in _JAVA_METHODS:
                m_name = _text(member.child_by_field_name("name")) or name
                m_line = _line(member)
                m_id = self.add_definition(
                    f"func:{rel}:{'.'.join(q)}.{m_name}:{m_line}",
                    "function",
                    m_name,
                    rel,
                    cls_id,
                    code=self.text_span(rel, m_line, min(self._header_end(member), kept)),
                    qualname=".".join(q + [m_name]),
                    start_line=m_line,
                    end_line=min(_end_line(member), kept),
                )
                self.method_nodes.append((m_id, member, rel, [m_id] + class_scope, m_id, kept))
            elif member.type in _JAVA_FIELDS:
                sid = f"stmt:{rel}:{_line(member)}:{_col(member)}"
                end = min(_end_line(member), kept)
                self.node(
                    sid,
                    type="statement",
                    code=self.text_span(rel, _line(member), end),
                    file_path=rel,
                    start_line=_line(member),
                    end_line=end,
                    level="function",
                    syntax=member.type,
                )
                self.edge(cls_id, sid, "ast_child")
                if prev:
                    self.edge(prev, sid, "control_flow")
                prev = sid
                self._record_java(rel, sid, member, class_scope, cls_id)
            elif member.type in _JAVA_TYPES:
                self._collect_type(rel, member, cls_id, q, pkg, kept)

    def _collect_imports(self, rel: str, root) -> None:
        for imp in root.named_children:
            if imp.type != "import_declaration":
                continue
            name = next((c for c in imp.named_children if c.type in ("scoped_identifier", "identifier")), None)
            if name is None:
                continue
            path = _text(name)
            if any(c.type == "asterisk" for c in imp.children):
                for target in sorted(self.package_files.get(path, ())):
                    self.add_import_edge(rel, target)
                if path in self.fqn_file:
                    self.add_import_edge(rel, self.fqn_file[path])
                continue
            parts = path.split(".")
            for k in range(len(parts), 0, -1):
                fqn = ".".join(parts[:k])
                if fqn in self.fqn_file:
                    self.add_import_edge(rel, self.fqn_file[fqn])
                    break

    def visible_files(self, rel: str) -> set:
        visible = set(self.import_targets.get(rel, set()))
        pkg = self.package_of.get(rel)
        if pkg is not None:
            visible |= self.package_files.get(pkg, set())
        return visible

    # -- statements ---------------------------------------------------------------------------
    def _record_java(self, rel, sid, stmt, scope, owner) -> None:
        defs, uses, calls, types, refs = [], [], [], [], []

        def declare(name_node, type_node) -> None:
            hint = _java_type_name(type_node)
            if name_node is not None:
                defs.append((_text(name_node), hint))
            if hint:
                types.append(hint)

        stack = [stmt]
        while stack:
            node = stack.pop()
            t = node.type
            push = None  # children to visit; None = all children
            if t in ("local_variable_declaration", "field_declaration", "constant_declaration"):
                type_node = node.child_by_field_name("type")
                for decl in node.children_by_field_name("declarator"):
                    declare(decl.child_by_field_name("name"), type_node)
                push = [type_node] + [d.child_by_field_name("value") for d in node.children_by_field_name("declarator")]
            elif t in ("enhanced_for_statement", "resource", "catch_formal_parameter"):
                type_node = node.child_by_field_name("type")
                if type_node is None:  # catch (A | B e)
                    type_node = next((c for c in node.named_children if c.type == "catch_type"), None)
                    type_node = type_node.named_children[0] if type_node is not None and type_node.named_children else None
                declare(node.child_by_field_name("name"), type_node)
                push = [c for c in node.children if not _same(c, node.child_by_field_name("name"))]
            elif t == "assignment_expression":
                left = node.child_by_field_name("left")
                if left is not None and left.type == "identifier":
                    defs.append((_text(left), None, True))
                    push = [node.child_by_field_name("right")]
                elif left is not None and left.type == "field_access" and _text(left.child_by_field_name("object")) == "this":
                    defs.append((_text(left.child_by_field_name("field")), None, "field"))
                    push = [node.child_by_field_name("right")]
            elif t == "identifier":
                uses.append(_text(node))
                refs.append(_text(node))
                continue
            elif t == "field_access":
                refs.append(_text(node.child_by_field_name("field")))
                push = [node.child_by_field_name("object")]
            elif t == "method_invocation":
                name = _text(node.child_by_field_name("name"))
                if name:
                    calls.append(name)
                    refs.append(name)
                root = _java_root_identifier(node.child_by_field_name("object"))
                if root:
                    types.append(root)
                push = [node.child_by_field_name("object"), node.child_by_field_name("arguments")]
            elif t == "object_creation_expression":
                name = _java_type_name(node.child_by_field_name("type"))
                if name:
                    types.append(name)
                    refs.append(name)
                push = [node.child_by_field_name("arguments")]
            elif t in ("type_identifier", "scoped_type_identifier"):
                name = _java_type_name(node)
                if name:
                    types.append(name)
                continue
            elif t == "lambda_expression":
                push = [node.child_by_field_name("body")]
            children = node.children if push is None else [c for c in push if c is not None]
            for child in reversed(children):
                if child.type not in _JAVA_STOP or _same(child, stmt):
                    stack.append(child)
        if refs:
            self.g.nodes[sid]["refs"] = " ".join(sorted(set(r for r in refs if r)))
        self.record(rel, sid, _line(stmt), scope, owner, defs, uses, calls, types)

    @staticmethod
    def _statements(node) -> List:
        """Statements of a block-like node, or ``[node]`` for a single statement."""
        if node is None:
            return []
        if node.type in ("block", "constructor_body"):
            return [c for c in node.named_children if c.type not in _JAVA_SKIP_IN_BLOCK]
        return [node]

    def _sub_blocks(self, stmt) -> List[List]:
        t = stmt.type
        field = stmt.child_by_field_name
        if t == "if_statement":
            return [self._statements(field("consequence")), self._statements(field("alternative"))]
        if t in _JAVA_LOOPS:
            return [self._statements(field("body"))]
        if t in ("try_statement", "try_with_resources_statement"):
            blocks = [self._statements(field("body"))]
            for child in stmt.named_children:
                if child.type == "catch_clause":
                    blocks.append(self._statements(child.child_by_field_name("body")))
                elif child.type == "finally_clause":
                    blocks.append(self._statements(next((c for c in child.named_children if c.type == "block"), None)))
            return blocks
        if t in ("switch_expression", "switch_statement"):
            body = field("body")
            groups = body.named_children if body is not None else []
            out = []
            for group in groups:
                stmts = []
                for child in group.named_children:
                    if child.type == "switch_label" or child.type in _JAVA_SKIP_IN_BLOCK:
                        continue
                    stmts.extend(self._statements(child))
                out.append(stmts)
            return out
        if t == "synchronized_statement":
            return [self._statements(field("body") or next((c for c in stmt.named_children if c.type == "block"), None))]
        return []

    def _visit_method(self, method_id, decl, rel, scope, owner, kept) -> None:
        defs, types = [], []
        params = decl.child_by_field_name("parameters")
        for param in params.named_children if params is not None else []:
            if param.type == "formal_parameter":
                name = param.child_by_field_name("name")
                type_node = param.child_by_field_name("type")
            elif param.type == "spread_parameter":
                declarator = next((c for c in param.named_children if c.type == "variable_declarator"), None)
                name = declarator.child_by_field_name("name") if declarator is not None else None
                type_node = next((c for c in param.named_children if c.type not in ("variable_declarator", "modifiers")), None)
            else:
                continue
            hint = _java_type_name(type_node)
            if name is not None:
                defs.append((_text(name), hint))
            if hint:
                types.append(hint)
        ret = _java_type_name(decl.child_by_field_name("type"))
        if ret:
            types.append(ret)
        self.record(rel, method_id, _line(decl), scope, owner, defs, [], [], types)
        body = decl.child_by_field_name("body")
        if body is not None:
            self._visit_block(rel, self._statements(body), method_id, scope, owner, method_id, kept)

    def _visit_block(self, rel, stmts, parent, scope, owner, func_id, kept) -> Tuple[Optional[str], Optional[str]]:
        first = last = None
        pending: List[str] = []
        for stmt in stmts:
            while stmt.type == "labeled_statement" and stmt.named_children:
                stmt = stmt.named_children[-1]
            if stmt.type in _JAVA_SKIP_IN_BLOCK or _line(stmt) > kept:
                continue
            if stmt.type == "block":
                # A nested { ... } block is inlined into the enclosing statement sequence.
                b_first, b_last = self._visit_block(rel, self._statements(stmt), parent, scope, owner, func_id, kept)
                if b_first and last:
                    self.edge(last, b_first, "control_flow")
                for exit_id in pending if b_first else []:
                    self.edge(exit_id, b_first, "control_flow")
                if b_first:
                    pending = []
                first = first or b_first
                last = b_last or last
                continue
            compound = stmt.type in _JAVA_COMPOUND
            end = _line(stmt) if compound else min(_end_line(stmt), kept)
            sid = f"stmt:{rel}:{_line(stmt)}:{_col(stmt)}"
            self.node(
                sid,
                type="statement",
                code=self.text_span(rel, _line(stmt), end),
                file_path=rel,
                start_line=_line(stmt),
                end_line=end,
                level="function",
                syntax=stmt.type,
            )
            self.edge(parent, sid, "ast_root" if parent == func_id else "ast_child")
            if last is not None:
                self.edge(last, sid, "control_flow")
            for exit_id in pending:
                self.edge(exit_id, sid, "control_flow")
            pending = []
            first = first or sid
            last = sid
            self._record_java(rel, sid, stmt, scope, owner)
            max_end = end
            for block in self._sub_blocks(stmt):
                b_first, b_last = self._visit_block(rel, block, sid, scope, owner, func_id, kept)
                if b_first:
                    self.edge(sid, b_first, "control_flow")
                if b_last:
                    if stmt.type in _JAVA_LOOPS:
                        self.edge(b_last, sid, "control_flow")
                    pending.append(b_last)
                    max_end = max(max_end, int(self.g.nodes[b_last].get("end_line", end)))
            self.g.nodes[sid]["end_line"] = max_end
        return first, last


# --------------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------------

BUILDERS = {"python": PythonGraphBuilder, "java": JavaGraphBuilder}


def build_repo_graph(files: Dict[str, str], language: str) -> nx.MultiDiGraph:
    """Build the multi-level graph for ``{relative_path: source}`` (already leakage-masked)."""
    if language not in BUILDERS:
        raise ValueError(f"Unsupported language: {language}")
    graph = BUILDERS[language](files).build()
    graph.graph["language"] = language
    return graph


def build_query_graph(graph: nx.MultiDiGraph, current_file: str, first_line: int, last_line: int) -> nx.MultiDiGraph:
    """Query graph G_q: current-file nodes whose span starts inside the visible local window.

    ``first_line``/``last_line`` are 1-based inclusive bounds of the local context window. The current
    file has already been cut at the cursor, so no node can cover the hidden target.
    """
    keep = []
    for nid, data in graph.nodes(data=True):
        if data.get("file_path") != current_file or data.get("type") in ("folder", "file"):
            continue
        start = int(data.get("start_line", 0) or 0)
        end = int(data.get("end_line", start) or start)
        if end >= first_line and start <= last_line:
            keep.append(nid)
    return graph.subgraph(keep).copy()


def write_graphml(graph: nx.MultiDiGraph, path: str) -> None:
    nx.write_graphml(graph, path)


def read_graphml(path: str) -> nx.MultiDiGraph:
    graph = nx.read_graphml(path, force_multigraph=True)
    for _, data in graph.nodes(data=True):
        for key in ("start_line", "end_line"):
            if key in data:
                data[key] = int(data[key])
    return graph
