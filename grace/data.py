"""Benchmark loading, field mapping, and project-disjoint repository splits."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

LOGGER = logging.getLogger(__name__)

DATASET_ALIASES = {
    "crosscodeeval": "crosscodeeval",
    "cceval": "crosscodeeval",
    "repoeval_updated": "repoeval_updated",
    "repoeval-updated": "repoeval_updated",
    "grace_tasks": "grace_tasks",
}
LANGUAGES = ("python", "java")


def canonical_dataset(name: str) -> str:
    key = name.strip().lower()
    if key not in DATASET_ALIASES:
        raise ValueError(f"Unknown dataset '{name}'. Known: {sorted(set(DATASET_ALIASES.values()))}")
    return DATASET_ALIASES[key]


@dataclass
class CompletionTask:
    """One next-line completion instance.

    ``right_context`` is kept only for leakage audits; retrieval, graph construction and prompting never
    read it.
    """

    task_id: str
    dataset: str
    language: str
    repo_name: str
    file_path: str
    prefix: str
    groundtruth: str
    repo_dir: Optional[str] = None
    right_context: Optional[str] = None

    @property
    def safe_key(self) -> str:
        text = f"{self.task_id}"
        return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)

    def public_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data.pop("right_context", None)
        return data


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_repo_dir(repos_root: Optional[str], repo_name: str) -> Optional[str]:
    """Find the local snapshot for ``repo_name`` (tries full name, basename, and owner_repo)."""
    if not repos_root:
        return None
    root = Path(repos_root)
    candidates = [repo_name, repo_name.split("/")[-1], repo_name.replace("/", "_")]
    for cand in candidates:
        path = root / cand
        if path.is_dir():
            return str(path)
    return None


def _crosscodeeval_task(row: dict, language: str, repos_root: Optional[str], index: int) -> CompletionTask:
    meta = row["metadata"]
    repo = meta["repository"]
    return CompletionTask(
        task_id=str(meta.get("task_id", f"{repo}/{index}")),
        dataset="crosscodeeval",
        language=language,
        repo_name=repo,
        file_path=meta["file"],
        prefix=row["prompt"],
        groundtruth=row["groundtruth"],
        repo_dir=resolve_repo_dir(repos_root, repo),
        right_context=row.get("right_context"),
    )


def _repoeval_task(row: dict, language: str, repos_root: Optional[str], index: int) -> CompletionTask:
    """Map one RepoEval-Updated record.

    Two layouts are supported:

    * RepoCoder/GraphCoder layout: ``prompt`` + ``metadata.{task_id, ground_truth, fpath_tuple}``.
    * RepoBench-style layout (legacy folder ``repoeval_to_repobench``): ``repo_name``, ``file_path``,
      ``all_code``/``cropped_code`` and ``next_line``.
    """
    if "metadata" in row and "fpath_tuple" in row["metadata"]:
        meta = row["metadata"]
        fpath = list(meta["fpath_tuple"])
        repo = fpath[0]
        return CompletionTask(
            task_id=str(meta.get("task_id", f"{repo}/{index}")),
            dataset="repoeval_updated",
            language=language,
            repo_name=repo,
            file_path="/".join(fpath[1:]),
            prefix=row["prompt"],
            groundtruth=meta.get("ground_truth", row.get("groundtruth", "")),
            repo_dir=resolve_repo_dir(repos_root, repo),
        )
    if "repo_name" in row and "file_path" in row and "next_line" in row:
        repo = row["repo_name"]
        prefix = row.get("all_code") or row.get("cropped_code") or ""
        return CompletionTask(
            task_id=str(row.get("task_id", f"{repo}/{row['file_path']}:{index}")),
            dataset="repoeval_updated",
            language=language,
            repo_name=repo,
            file_path=row["file_path"],
            prefix=prefix,
            groundtruth=row["next_line"],
            repo_dir=resolve_repo_dir(repos_root, repo),
        )
    raise ValueError(
        "Unrecognised RepoEval-Updated record layout. Expected either metadata.fpath_tuple or "
        f"repo_name/file_path/next_line. Keys: {sorted(row.keys())}"
    )


def _grace_task(row: dict, language: str, repos_root: Optional[str], index: int) -> CompletionTask:
    repo = row["repo_name"]
    return CompletionTask(
        task_id=str(row.get("task_id", f"{repo}/{index}")),
        dataset=row.get("dataset", "grace_tasks"),
        language=row.get("language", language),
        repo_name=repo,
        file_path=row["file_path"],
        prefix=row["prefix"],
        groundtruth=row["groundtruth"],
        repo_dir=row.get("repo_dir") or resolve_repo_dir(repos_root, repo),
        right_context=row.get("right_context"),
    )


_LOADERS = {
    "crosscodeeval": _crosscodeeval_task,
    "repoeval_updated": _repoeval_task,
    "grace_tasks": _grace_task,
}


def default_data_file(dataset: str, dataset_root: str, language: str) -> str:
    """Default benchmark file locations (inherited from the legacy scripts; override with --data-file)."""
    dataset = canonical_dataset(dataset)
    root = Path(dataset_root)
    if dataset == "crosscodeeval":
        return str(root / "rawdata" / language / "line_completion.jsonl")
    if dataset == "repoeval_updated":
        # Legacy script read rawdata/<language>/line_level.python.test.jsonl for every language.
        return str(root / "rawdata" / language / f"line_level.{language}.test.jsonl")
    raise ValueError("grace_tasks has no default file; pass --data-file")


def load_tasks(dataset: str, data_file: str, language: str, repos_root: Optional[str] = None) -> List[CompletionTask]:
    if language not in LANGUAGES:
        raise ValueError(f"language must be one of {LANGUAGES}")
    dataset = canonical_dataset(dataset)
    loader = _LOADERS[dataset]
    tasks = [loader(row, language, repos_root, i) for i, row in enumerate(load_jsonl(data_file))]
    missing = sum(1 for t in tasks if t.repo_dir is None)
    if missing:
        LOGGER.warning("%d/%d tasks have no local repository snapshot under %s", missing, len(tasks), repos_root)
    return tasks


def tasks_to_jsonl_rows(tasks: Sequence[CompletionTask]) -> List[dict]:
    return [asdict(t) for t in tasks]


# --------------------------------------------------------------------------------------------------
# Project-disjoint splits
# --------------------------------------------------------------------------------------------------

def repo_identity_keys(name: str) -> set:
    """Conservative identity keys: full name and basename, lower-cased, without .git."""
    norm = name.strip().lower().rstrip("/")
    if norm.endswith(".git"):
        norm = norm[:-4]
    keys = {norm, norm.split("/")[-1]}
    return {k for k in keys if k}


class ProjectOverlapError(ValueError):
    pass


def assert_project_disjoint(groups: Dict[str, Iterable[str]]) -> None:
    """Raise if any repository appears in more than one group (e.g. fitting/tuning/test)."""
    owners: Dict[str, tuple] = {}
    for group, repos in groups.items():
        for repo in repos:
            for key in repo_identity_keys(repo):
                if key in owners and owners[key][0] != group:
                    other_group, other_repo = owners[key]
                    raise ProjectOverlapError(
                        f"Repository overlap between '{other_group}' ({other_repo}) and '{group}' ({repo}). "
                        "Development and test repositories must be project-disjoint."
                    )
                owners.setdefault(key, (group, repo))


def read_repo_list(path: str) -> List[str]:
    """Read a repository list: one name per line, ``#`` comments allowed."""
    repos = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if line:
                repos.append(line)
    return repos
