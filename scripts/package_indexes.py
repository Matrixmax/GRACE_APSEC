"""Package and restore preprocessed GRACE task indexes (artifact support).

Indexes are produced by ``python -m grace.cli index`` (one directory per task under ``--index-dir``).
They contain derived data (CodeT5p / GAT embeddings, HNSW graphs, unit metadata) and, by default, copies
of repository source code (unit text and node ``code``). Benchmark repositories may not be
redistributable, so ``pack --strip-code`` removes all source text and keeps only embeddings, ids,
line spans and graph structure. ``hydrate`` restores the text from local repository snapshots by
re-running the deterministic, leakage-masked graph construction and checking that unit ids and node
lists match exactly.

    python scripts/package_indexes.py pack --index-dir runs/index --out grace_indexes.tar.gz --strip-code
    python scripts/package_indexes.py unpack --archive grace_indexes.tar.gz --out runs/index
    python scripts/package_indexes.py hydrate --index-dir runs/index --dataset crosscodeeval \
        --language python --data-file ... --repos-root ... [--config ...] [--set subgraph_hops=...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace.config import load_config  # noqa: E402
from grace.data import load_tasks  # noqa: E402
from grace.graph_builder import read_graphml, write_graphml  # noqa: E402
from grace.index import INDEX_FORMAT  # noqa: E402

MANIFEST = "MANIFEST.json"
CODE_NODE_ATTRS = ("code",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_dirs(index_dir: Path) -> List[Path]:
    out = []
    for meta in sorted(index_dir.glob("*/meta.json")):
        with open(meta, "r", encoding="utf-8") as handle:
            if json.load(handle).get("format") == INDEX_FORMAT:
                out.append(meta.parent)
    return out


def _read_units(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_units(path: Path, units: Sequence[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for unit in units:
            handle.write(json.dumps(unit, ensure_ascii=False) + "\n")


def _update_meta(task_dir: Path, **updates) -> None:
    path = task_dir / "meta.json"
    with open(path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    meta.update(updates)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)


def strip_code(task_dir: Path) -> None:
    """Remove source text from one task index in place."""
    units = _read_units(task_dir / "units.jsonl")
    for unit in units:
        unit["text"] = ""
    _write_units(task_dir / "units.jsonl", units)
    graph = read_graphml(str(task_dir / "unit_nodes.graphml"))
    for _, data in graph.nodes(data=True):
        for attr in CODE_NODE_ATTRS:
            if attr in data:
                data[attr] = ""
    write_graphml(graph, str(task_dir / "unit_nodes.graphml"))
    _update_meta(task_dir, code_stripped=True)


def pack(index_dir: str, out: str, strip: bool) -> Dict[str, object]:
    src = Path(index_dir)
    dirs = task_dirs(src)
    if not dirs:
        raise ValueError(f"No GRACE task indexes found under {index_dir}")
    with tempfile.TemporaryDirectory(prefix="grace_pack_") as tmp:
        stage = Path(tmp) / "grace_indexes"
        files: Dict[str, str] = {}
        for task_dir in dirs:
            dest = stage / task_dir.name
            shutil.copytree(task_dir, dest)
            if strip:
                strip_code(dest)
            for path in sorted(dest.rglob("*")):
                if path.is_file():
                    files[path.relative_to(stage).as_posix()] = _sha256(path)
        manifest = {"format": INDEX_FORMAT, "num_tasks": len(dirs), "code_stripped": bool(strip), "files": files}
        with open(stage / MANIFEST, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(out, "w:gz") as tar:
            tar.add(stage, arcname="grace_indexes")
    return manifest


def unpack(archive: str, out: str) -> Dict[str, object]:
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="grace_unpack_") as tmp:
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                target = (Path(tmp) / member.name).resolve()
                if not str(target).startswith(str(Path(tmp).resolve())) or member.issym() or member.islnk():
                    raise ValueError(f"Unsafe archive member: {member.name}")
            tar.extractall(tmp)
        stage = Path(tmp) / "grace_indexes"
        with open(stage / MANIFEST, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        for rel, digest in manifest["files"].items():
            if _sha256(stage / rel) != digest:
                raise ValueError(f"Checksum mismatch for {rel}")
        for task_dir in sorted(p for p in stage.iterdir() if p.is_dir()):
            target = dest / task_dir.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(task_dir, target)
    return manifest


def hydrate(index_dir: str, tasks, cfg) -> int:
    """Fill source text back into stripped task indexes. Returns the number of hydrated tasks."""
    from grace.pipeline import GracePipeline

    pipe = GracePipeline(cfg, text_encoder=None, model=None, index_backend="exact")
    count = 0
    for task in tasks:
        task_dir = Path(index_dir) / task.safe_key
        if not (task_dir / "meta.json").exists():
            continue
        prepared = pipe.prepare(task)
        rebuilt = {u.unit_id: u for u in prepared.units}
        units = _read_units(task_dir / "units.jsonl")
        for unit in units:
            fresh = rebuilt.get(unit["unit_id"])
            if fresh is None or list(fresh.nodes) != list(unit["nodes"]):
                raise ValueError(f"{task.task_id}: unit {unit['unit_id']} does not match the local repository "
                                 "snapshot or configuration (subgraph_hops); cannot hydrate")
            unit["text"] = fresh.text
        graph = read_graphml(str(task_dir / "unit_nodes.graphml"))
        for nid, data in graph.nodes(data=True):
            if nid not in prepared.graph:
                raise ValueError(f"{task.task_id}: node {nid} missing from the rebuilt graph")
            for attr in CODE_NODE_ATTRS:
                if attr in data:
                    data[attr] = prepared.graph.nodes[nid].get(attr, "")
        _write_units(task_dir / "units.jsonl", units)
        write_graphml(graph, str(task_dir / "unit_nodes.graphml"))
        _update_meta(task_dir, code_stripped=False)
        count += 1
    return count


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--index-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--strip-code", action="store_true", help="remove repository source text before packaging")
    p = sub.add_parser("unpack")
    p.add_argument("--archive", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("hydrate")
    p.add_argument("--index-dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--language", required=True, choices=["python", "java"])
    p.add_argument("--data-file", required=True)
    p.add_argument("--repos-root", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--set", action="append", default=[])
    args = parser.parse_args(argv)

    if args.command == "pack":
        manifest = pack(args.index_dir, args.out, args.strip_code)
        print(f"packed {manifest['num_tasks']} task indexes into {args.out} (code_stripped={manifest['code_stripped']})")
    elif args.command == "unpack":
        manifest = unpack(args.archive, args.out)
        print(f"unpacked {manifest['num_tasks']} task indexes into {args.out}")
    else:
        cfg = load_config(args.config, args.set)
        tasks = load_tasks(args.dataset, args.data_file, args.language, args.repos_root)
        print(f"hydrated {hydrate(args.index_dir, tasks, cfg)} task indexes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
