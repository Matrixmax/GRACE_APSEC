"""Clone the repositories referenced by a benchmark (formerly ``donwload_repos.py``).

The released script cloned every ``repo_name`` of a local RepoBench-style Hugging Face dataset
(``dataset/hf_datasets/repobench_<language>_v1.1``, split ``cross_file_first``) into
``.../<split>/repos``. The same behaviour is the default here; paths, split and parallelism are now
arguments, and a plain repository list can be used instead of a dataset.

Credentials are optional and only read from the environment (``GIT_USERNAME`` / ``GIT_PASSWORD``);
nothing is stored in the code. Repositories are cloned shallowly (``depth=1``) from the default
branch.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Optional, Sequence


def repo_url(repo: str, username: Optional[str] = None, password: Optional[str] = None) -> str:
    if username and password:
        return f"https://{username}:{password}@github.com/{repo}.git"
    return f"https://github.com/{repo}.git"


def download_repo(repo: str, repo_folder: str) -> str:
    """Clone ``owner/name`` into ``repo_folder/name`` unless it already exists. Returns a status string."""
    import git

    file_name = repo.split("/")[-1]
    os.makedirs(repo_folder, exist_ok=True)
    if file_name in os.listdir(repo_folder):
        return f"exists {repo}"
    url = repo_url(repo, os.environ.get("GIT_USERNAME"), os.environ.get("GIT_PASSWORD"))
    try:
        git.Repo.clone_from(url, os.path.join(repo_folder, file_name), depth=1, single_branch=True)
        return f"cloned {repo}"
    except Exception as exc:  # network / auth errors are reported, not fatal
        return f"failed {repo}: {type(exc).__name__}"


def repos_from_dataset(dataset_path: str, split: str) -> List[str]:
    from datasets import load_dataset

    data = load_dataset(dataset_path)
    return sorted({sample["repo_name"] for sample in data[split]})


def repos_from_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [ln.split("#", 1)[0].strip() for ln in handle if ln.split("#", 1)[0].strip()]


def download_all(repos: Iterable[str], repo_folder: str, n_jobs: int = 4) -> List[str]:
    from joblib import Parallel, delayed
    from tqdm import tqdm

    return Parallel(n_jobs=n_jobs, prefer="threads")(delayed(download_repo)(r, repo_folder) for r in tqdm(list(repos)))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Clone benchmark repositories")
    parser.add_argument("--language", default="python", choices=["python", "java"])
    parser.add_argument("--split", default="cross_file_first")
    parser.add_argument("--dataset-path", default=None,
                        help="default: dataset/hf_datasets/repobench_<language>_v1.1")
    parser.add_argument("--repo-list", default=None, help="text file with owner/name per line (overrides dataset)")
    parser.add_argument("--out", default=None, help="default: <dataset-path>/<split>/repos")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)

    dataset_path = args.dataset_path or f"dataset/hf_datasets/repobench_{args.language}_v1.1"
    out = args.out or os.path.join(dataset_path, args.split, "repos")
    repos = repos_from_list(args.repo_list) if args.repo_list else repos_from_dataset(dataset_path, args.split)
    print(f"language: {args.language}, repositories: {len(repos)}, target: {out}")
    for status in download_all(repos, out, args.jobs):
        print(status)


if __name__ == "__main__":
    main()
