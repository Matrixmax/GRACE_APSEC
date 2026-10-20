"""Tree-sitter graph construction: parser backend, tail repair at the cursor, and language constructs."""

import pytest

from grace.graph_builder import PARSER_BACKEND, build_repo_graph, get_parser, parse_source


def edges(graph, relation):
    return {(u, v) for u, v, d in graph.edges(data=True) if d.get("relation") == relation}


def nodes_of(graph, kind):
    return {n for n, d in graph.nodes(data=True) if d.get("type") == kind}


def test_parsers_are_tree_sitter():
    assert PARSER_BACKEND == "tree-sitter"
    for language, root in (("python", "module"), ("java", "program")):
        parser = get_parser(language)
        assert type(parser).__module__.startswith("tree_sitter")
        assert parser.parse(b"").root_node.type == root
        assert get_parser(language) is parser
    with pytest.raises(ValueError):
        get_parser("cobol")


def test_python_tail_repair_keeps_visible_lines():
    text = "def f(x):\n    y = g(x)\n    if y:\n"
    tree, kept = parse_source("python", text)
    assert not tree.root_node.has_error and kept == 3
    text = "def f(x):\n    y = g(x)\n    z = h(y,\n"
    tree, kept = parse_source("python", text)
    assert not tree.root_node.has_error and kept == 2


def test_java_tail_repair_closes_braces():
    text = "class A {\n  int f(int x) {\n    int y = x + 1;\n    if (y > 0) {\n"
    tree, kept = parse_source("java", text)
    assert not tree.root_node.has_error and kept == 4
    # a brace inside a string or comment is not counted
    text = 'class A {\n  String s = "{";\n  // {\n  int f() {\n    return 1;\n'
    tree, kept = parse_source("java", text)
    assert not tree.root_node.has_error and kept == 5


PY_REPO = {
    "pkg/__init__.py": "",
    "pkg/base.py": "class Base:\n    def run(self):\n        return 1\n",
    "pkg/sub/__init__.py": "",
    "pkg/sub/impl.py": (
        "from ..base import Base\n"
        "import functools\n"
        "\n"
        "def helper(v: int) -> 'Base':\n"
        "    return Base()\n"
        "\n"
        "class Impl(Base, metaclass=type):\n"
        "    @functools.cache\n"
        "    def run(self, *args, k: Base = None, **kw):\n"
        "        total = 0\n"
        "        for i, item in enumerate(args):\n"
        "            total += helper(item)\n"
        "        match total:\n"
        "            case 0:\n"
        "                result = None\n"
        "            case _:\n"
        "                result: Base = helper(total)\n"
        "        with open('f') as fh:\n"
        "            data = fh.read()\n"
        "        return result\n"
    ),
}


def test_python_constructs():
    g = build_repo_graph(PY_REPO, "python")
    assert ("file:pkg/sub/impl.py", "file:pkg/base.py") in edges(g, "imports")
    assert ("class:pkg/sub/impl.py:Impl", "class:pkg/base.py:Base") in edges(g, "inherits")
    # the decorated method is found and calls the module-level helper
    assert "func:pkg/sub/impl.py:Impl.run" in nodes_of(g, "function")
    assert ("func:pkg/sub/impl.py:Impl.run", "func:pkg/sub/impl.py:helper") in edges(g, "calls")
    assert ("func:pkg/sub/impl.py:helper", "class:pkg/base.py:Base") in edges(g, "type_ref")
    variables = {g.nodes[n]["name"] for n in nodes_of(g, "variable")}
    assert {"total", "i", "item", "result", "fh", "data", "args", "k", "kw", "v"} <= variables
    # annotated assignment links the variable to its class
    assert any(v.endswith(":result") and c == "class:pkg/base.py:Base" for v, c in edges(g, "type_flow"))
    # match/case bodies become statements reachable through the CFG
    syntax = {d.get("syntax") for _, d in g.nodes(data=True)}
    assert {"match_statement", "for_statement", "with_statement", "return_statement"} <= syntax
    loop = next(n for n, d in g.nodes(data=True) if d.get("syntax") == "for_statement")
    assert any(u != loop and v == loop for u, v in edges(g, "control_flow"))  # loop back edge


JAVA_REPO = {
    "src/a/Base.java": "package a;\npublic abstract class Base implements Runnable {\n  public void run() {}\n}\n",
    "src/b/Impl.java": (
        "package b;\n"
        "import a.Base;\n"
        "import java.util.*;\n"
        "public class Impl extends Base {\n"
        "  private int count;\n"
        "  public Impl(int count) { this.count = count; }\n"
        "  @Override\n"
        "  public void run() {\n"
        "    List<Base> items = new ArrayList<>();\n"
        "    for (Base b : items) { helper(b); }\n"
        "    switch (count) {\n"
        "      case 0: count = 1; break;\n"
        "      default: count = 2;\n"
        "    }\n"
        "    try { helper(null); } catch (RuntimeException e) { count = 3; } finally { count = 4; }\n"
        "    Runnable r = () -> helper(null);\n"
        "  }\n"
        "  static Base helper(Base value) { return value; }\n"
        "}\n"
    ),
}


def test_java_constructs():
    g = build_repo_graph(JAVA_REPO, "java")
    assert ("file:src/b/Impl.java", "file:src/a/Base.java") in edges(g, "imports")
    assert ("class:src/b/Impl.java:Impl", "class:src/a/Base.java:Base") in edges(g, "inherits")
    run = next(n for n in nodes_of(g, "function") if ".run:" in n and "Impl" in n)
    helper = next(n for n in nodes_of(g, "function") if n.endswith(".helper:18"))
    assert (run, helper) in edges(g, "calls")
    assert (run, "class:src/a/Base.java:Base") in edges(g, "type_ref")
    variables = {g.nodes[n]["name"] for n in nodes_of(g, "variable")}
    assert {"items", "b", "e", "r", "value", "count"} <= variables
    # this.count = count defines the field, not the constructor parameter
    field = "var:class:src/b/Impl.java:Impl:count"
    assert field in g
    assert any(v == field for _, v in edges(g, "defines"))
    syntax = {d.get("syntax") for _, d in g.nodes(data=True)}
    assert {"enhanced_for_statement", "switch_expression", "try_statement", "field_declaration"} <= syntax
    # statements inside switch cases and try/catch/finally blocks are visited (lines 12, 13, 15)
    nested = {(d["start_line"], d["syntax"]) for n, d in g.nodes(data=True)
              if n.startswith("stmt:src/b/Impl.java:") and d.get("syntax") in ("expression_statement", "break_statement")}
    assert {(12, "expression_statement"), (12, "break_statement"), (13, "expression_statement")} <= nested
    in_try = [n for n, d in g.nodes(data=True)
              if n.startswith("stmt:src/b/Impl.java:15:") and d.get("syntax") == "expression_statement"]
    assert len(in_try) == 3  # try body, catch body, finally body


def test_incomplete_current_file_is_parsed_up_to_the_cursor():
    files = dict(JAVA_REPO)
    files["src/b/App.java"] = (
        "package b;\nimport a.Base;\npublic class App {\n  void go() {\n    Base x = Impl.helper(null);\n    x."
    )
    g = build_repo_graph(files, "java")
    assert "class:src/b/App.java:App" in g
    go = next(n for n in nodes_of(g, "function") if "App.go" in n)
    assert any(u == go for u, _ in edges(g, "ast_root"))
    assert all(d.get("end_line", 0) <= 6 for _, d in g.nodes(data=True) if d.get("file_path") == "src/b/App.java")
