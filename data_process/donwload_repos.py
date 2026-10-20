"""Deprecated misspelled name kept for compatibility. Use ``data_process/download_repos.py``."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download_repos import download_all, download_repo, main, repo_url, repos_from_dataset, repos_from_list  # noqa: E402,F401

if __name__ == "__main__":
    main()
