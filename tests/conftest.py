"""Shared fixtures: tiny synthetic Python/Java repositories, the dev config, and test-only encoders.

Nothing here downloads models or calls an API. ``HashingTextEncoder`` stands in for CodeT5p and the
structural model is randomly initialised.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace.config import DEV_CONFIG, load_config  # noqa: E402
from grace.data import CompletionTask  # noqa: E402
from grace.encoder import EncoderSpec, StructuralRetrievalModel  # noqa: E402
from grace.graph_features import HashingTextEncoder  # noqa: E402

PY_FILES = {
    "pkg/__init__.py": "",
    "pkg/shapes.py": (
        "class Shape:\n"
        "    def __init__(self, name):\n"
        "        self.name = name\n"
        "\n"
        "    def area(self):\n"
        "        return 0\n"
        "\n"
        "\n"
        "class Circle(Shape):\n"
        "    def __init__(self, radius):\n"
        "        super().__init__('circle')\n"
        "        self.radius = radius\n"
        "\n"
        "    def area(self):\n"
        "        return 3.14 * self.radius * self.radius\n"
    ),
    "pkg/utils.py": (
        "def scale(value, factor):\n"
        "    return value * factor\n"
        "\n"
        "\n"
        "def describe(shape):\n"
        "    return shape.name\n"
    ),
}
PY_PREFIX = (
    "from pkg.shapes import Circle\n"
    "from pkg.utils import scale, describe\n"
    "\n"
    "\n"
    "def run(radius):\n"
    "    circle = Circle(radius)\n"
    "    label = describe(circle)\n"
    "    big = scale(circle.area(), 2)\n"
)
PY_TARGET = "    secret_target_value = compute_hidden_total(big, label)"
PY_SUFFIX = (
    "\n"
    "    return secret_target_value\n"
    "\n"
    "\n"
    "def compute_hidden_total(a, b):\n"
    "    return HIDDEN_SUFFIX_MARKER\n"
)
PY_FORBIDDEN = ["secret_target_value", "compute_hidden_total", "HIDDEN_SUFFIX_MARKER"]

JAVA_FILES = {
    "src/com/example/Shape.java": (
        "package com.example;\n"
        "\n"
        "public class Shape {\n"
        "    protected String name;\n"
        "\n"
        "    public Shape(String name) {\n"
        "        this.name = name;\n"
        "    }\n"
        "\n"
        "    public double area() {\n"
        "        return 0.0;\n"
        "    }\n"
        "}\n"
    ),
    "src/com/example/Circle.java": (
        "package com.example;\n"
        "\n"
        "public class Circle extends Shape {\n"
        "    private double radius;\n"
        "\n"
        "    public Circle(double radius) {\n"
        "        super(\"circle\");\n"
        "        this.radius = radius;\n"
        "    }\n"
        "\n"
        "    public double area() {\n"
        "        return 3.14 * radius * radius;\n"
        "    }\n"
        "}\n"
    ),
    "src/com/example/Util.java": (
        "package com.example;\n"
        "\n"
        "public class Util {\n"
        "    public static double scale(double value, double factor) {\n"
        "        return value * factor;\n"
        "    }\n"
        "}\n"
    ),
}
JAVA_PREFIX = (
    "package com.example;\n"
    "\n"
    "public class App {\n"
    "    public static double run(double radius) {\n"
    "        Circle circle = new Circle(radius);\n"
    "        double big = Util.scale(circle.area(), 2.0);\n"
)
JAVA_TARGET = "        double secretTargetValue = hiddenHelperMethod(big);"
JAVA_SUFFIX = (
    "\n"
    "        return secretTargetValue;\n"
    "    }\n"
    "\n"
    "    private static double hiddenHelperMethod(double v) {\n"
    "        return v + HIDDEN_SUFFIX_MARKER;\n"
    "    }\n"
    "}\n"
)
JAVA_FORBIDDEN = ["secretTargetValue", "hiddenHelperMethod", "HIDDEN_SUFFIX_MARKER"]


def write_repo(root: Path, files: dict) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def make_python_repo(root: Path) -> Path:
    files = dict(PY_FILES)
    files["main.py"] = PY_PREFIX + PY_TARGET + PY_SUFFIX
    return write_repo(root, files)


def make_java_repo(root: Path) -> Path:
    files = dict(JAVA_FILES)
    files["src/com/example/App.java"] = JAVA_PREFIX + JAVA_TARGET + JAVA_SUFFIX
    return write_repo(root, files)


def python_task(repo_dir: Path, repo_name: str = "example/pyrepo", task_id: str = "py/0") -> CompletionTask:
    return CompletionTask(task_id=task_id, dataset="grace_tasks", language="python", repo_name=repo_name,
                          file_path="main.py", prefix=PY_PREFIX, groundtruth=PY_TARGET, repo_dir=str(repo_dir),
                          right_context=PY_SUFFIX)


def java_task(repo_dir: Path, repo_name: str = "example/javarepo", task_id: str = "java/0") -> CompletionTask:
    return CompletionTask(task_id=task_id, dataset="grace_tasks", language="java", repo_name=repo_name,
                          file_path="src/com/example/App.java", prefix=JAVA_PREFIX, groundtruth=JAVA_TARGET,
                          repo_dir=str(repo_dir), right_context=JAVA_SUFFIX)


def crosscodeeval_row(task: CompletionTask) -> dict:
    return {"prompt": task.prefix, "groundtruth": task.groundtruth, "right_context": task.right_context,
            "metadata": {"task_id": task.task_id, "repository": task.repo_name, "file": task.file_path}}


def write_jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


@pytest.fixture
def dev_cfg():
    return load_config(str(DEV_CONFIG))


@pytest.fixture
def text_encoder():
    return HashingTextEncoder()


@pytest.fixture
def model(dev_cfg):
    import torch

    torch.manual_seed(0)
    return StructuralRetrievalModel(EncoderSpec.from_config(dev_cfg))


@pytest.fixture
def py_repo(tmp_path):
    return make_python_repo(tmp_path / "repos" / "pyrepo")


@pytest.fixture
def java_repo(tmp_path):
    return make_java_repo(tmp_path / "repos" / "javarepo")
