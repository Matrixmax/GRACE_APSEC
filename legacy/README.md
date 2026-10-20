# Archived scripts

* `grace_coderag_original_release.py`: the originally released `model/grace_coderag.py`. Kept for
  reference; it requires `dgl`.
  The only change is that the hard-coded device `cuda:3` was replaced by `GRACE_DEVICE` / automatic
  selection. It expects `model/` on `sys.path` (it imports `graph_text_encoders`).
* `donwload_repos_original_release.py`: the originally released repository download script.

Use `grace/` (see `../README.md`) or the compatibility wrapper `model/grace_coderag.py` instead.
