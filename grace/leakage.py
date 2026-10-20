"""Target-leakage control.

Before any graph is built or indexed, the current file is cut at the cursor: the target line and the
post-cursor suffix of the same file are removed. Other files come from the repository snapshot.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .data import CompletionTask

LOGGER = logging.getLogger(__name__)

SOURCE_EXTENSIONS = {"python": (".py",), "java": (".java",)}
SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv", ".tox", "build", "dist"}


def read_repo_sources(repo_dir: Optional[str], language: str) -> Dict[str, str]:
    """Read all source files of ``language`` under ``repo_dir`` as ``{relative_path: text}``."""
    files: Dict[str, str] = {}
    if not repo_dir:
        return files
    root = Path(repo_dir)
    exts = SOURCE_EXTENSIONS[language]
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for fname in sorted(filenames):
            if fname.endswith(exts):
                path = Path(dirpath) / fname
                rel = path.relative_to(root).as_posix()
                try:
                    files[rel] = path.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:  # pragma: no cover - filesystem errors
                    LOGGER.warning("Could not read %s: %s", path, exc)
    return files


def locate_cursor(file_text: str, prefix: str) -> Optional[int]:
    """Return the character offset of the cursor inside ``file_text`` or ``None`` if ambiguous.

    The benchmark prompt is the visible code before the cursor. The cursor is the end of the unique
    occurrence of the prompt in the file.
    """
    if not prefix:
        return 0
    if file_text.startswith(prefix):
        return len(prefix)
    first = file_text.find(prefix)
    if first < 0:
        return None
    if file_text.find(prefix, first + 1) >= 0:
        return None
    return first + len(prefix)


@dataclass
class MaskedSnapshot:
    files: Dict[str, str]
    current_file: str
    visible_text: str
    cursor_located: bool

    @property
    def visible_line_count(self) -> int:
        """Number of (possibly partial) lines visible in the current file."""
        if not self.visible_text:
            return 0
        return self.visible_text.count("\n") + (0 if self.visible_text.endswith("\n") else 1)


def make_masked_snapshot(task: CompletionTask, files: Optional[Dict[str, str]] = None) -> MaskedSnapshot:
    """Copy the repository sources and cut the current file at the cursor."""
    if files is None:
        files = read_repo_sources(task.repo_dir, task.language)
    masked = dict(files)
    original = files.get(task.file_path)
    located = False
    if original is not None:
        cursor = locate_cursor(original, task.prefix)
        if cursor is not None:
            visible = original[:cursor]
            located = True
        else:
            LOGGER.warning(
                "Cursor of task %s not uniquely located in %s; using the benchmark prompt as the only "
                "visible content of the current file.",
                task.task_id,
                task.file_path,
            )
            visible = task.prefix
    else:
        visible = task.prefix
    masked[task.file_path] = visible
    return MaskedSnapshot(files=masked, current_file=task.file_path, visible_text=visible, cursor_located=located)


def materialize_masked_repo(repo_dir: str, snapshot: MaskedSnapshot, dest: str) -> str:
    """Create a directory tree for builders that need a path: symlinks for unchanged files and a
    truncated copy of the current file. Returns ``dest``."""
    src_root = Path(repo_dir).resolve()
    dst_root = Path(dest)
    dst_root.mkdir(parents=True, exist_ok=True)
    for rel in snapshot.files:
        target = dst_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        if rel == snapshot.current_file:
            target.write_text(snapshot.visible_text, encoding="utf-8")
        else:
            os.symlink(src_root / rel, target)
    return str(dst_root)


def hidden_lines(task: CompletionTask) -> List[str]:
    """Non-empty stripped lines that must stay hidden (target line and optional right context)."""
    lines = [task.groundtruth]
    if task.right_context:
        lines.extend(task.right_context.splitlines())
    return [ln.strip() for ln in lines if ln.strip()]


def audit_leakage(texts: Dict[str, Iterable[str]], forbidden: Iterable[str]) -> List[str]:
    """Return ``"field: forbidden"`` for every forbidden string found in any of the texts.

    Used by tests and optional audits only; it never changes what is retrieved or prompted.
    """
    hits = []
    forbidden = [f for f in forbidden if f]
    for field, values in texts.items():
        for value in values:
            for item in forbidden:
                if item in (value or ""):
                    hits.append(f"{field}: {item}")
    return hits
