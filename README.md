# GRACE

GRACE: Graph-Guided Repository-Aware Code Completion through Hierarchical Code Fusion.

The `grace/` package implements the full method: multi-level code graph (Tree-sitter), GAT
structural encoder trained with InfoNCE, gated hybrid retrieval with MMR, graph fusion with `align`
edges, `NODE`/`EDGE` serialization and token-budgeted prompting. All settings are in
`configs/paper_reproduction.yaml`.

## 0. Environment

```bash
uv sync                 # dependencies from pyproject.toml / uv.lock (refreshed with `uv lock` on 2026-10-20)
uv sync --group dev     # adds pytest
uv run pytest           # unit and end-to-end tests (mocked encoders and LLM, no downloads, no API calls)
```

Source files are parsed with `tree-sitter` 0.20.1 and the `tree-sitter-python` / `tree-sitter-java`
grammar packages (installed by `uv sync`; no grammar build step). `dgl` is not needed: the Laplacian
positional encoding is computed with numpy (scipy for graphs larger than 3000 nodes).

Environment variables (never put keys in files):

| Variable | Used by |
|---|---|
| `OPENAI_API_KEY`, `DASHSCOPE_API_KEY`, `DEEPSEEK_API_KEY`, `SILICONFLOW_API_KEY` | `grace.cli generate` (names configured under `llm.providers`) |
| `GRACE_DEVICE` | baseline and archived scripts (`utils/utils.py`, `model/generate.py`, `legacy/`); `grace/` reads `implementation.device` (`auto` = CUDA if available, else CPU) |
| `GIT_USERNAME`, `GIT_PASSWORD` (optional) | `data_process/download_repos.py` |
| `OPENAI_BASE_URL` (optional) | baseline `model/generate.py` (gpt-3.5 branch) |

## 1. Data layout

Defaults follow the released scripts; every path can be overridden on the command line.

```text
dataset/
  crosscodeeval/
    rawdata/{python,java}/line_completion.jsonl
    repos/<repository>/...                 # one snapshot per repository
  repoeval_updated/
    rawdata/{python,java}/line_level.<language>.test.jsonl
    repos/<repository>/...
```

Repository directories are found under `--repos-root` by full name, basename, or `owner_repo`.

Supported benchmark records (`grace/data.py`):

* `crosscodeeval`: `prompt`, `groundtruth`, `right_context`, `metadata.{task_id, repository, file}`.
* `repoeval_updated` (alias `repoeval-updated`), two layouts:
  `prompt` + `metadata.{task_id, ground_truth, fpath_tuple}`, or
  `repo_name`, `file_path`, `all_code`/`cropped_code`, `next_line`.
* `grace_tasks` (training queries): `task_id`, `repo_name`, `file_path`, `prefix`, `groundtruth`,
  optional `repo_dir`, optional `positive_units`.

Repositories can be cloned with `python data_process/download_repos.py --repo-list repos.txt --out dataset/<name>/repos`
(the misspelled `donwload_repos.py` still works). The cloned revision is the default branch (shallow).

## 2. Configuration

* `configs/paper_reproduction.yaml` (default):
  * `paper_reported`: settings stated in the paper (source lines in the comments);
  * `method_hyperparameters`: indexing, fusion, structural encoder, training, retrieval and decoding;
  * `implementation`: I/O, retries, device;
  * `llm`: providers and models.
  A key may occur in only one section. A stage that needs a value set to `null` stops with
  `MissingParameter` until the value is supplied.
* `configs/dev_smoke_test.yaml`: inherits the paper file and replaces the method values with small
  values for the unit tests (section `test_overrides`). Every output records `config_name`.
* Override any value with `--set key=value` (repeatable). Overriding a `paper_reported` value logs a
  warning. Every retrieval output is accompanied by `<output>.config.json` (full configuration snapshot).

| Group | Values |
|---|---|
| Paper settings | context 2048 = 1024 local + 1024 retrieved tokens, `max_new_tokens` 100, K_c 10, K_f 3, theta 0.4, HNSW M 32 / ef_search 256, encoder `Salesforce/codet5p-110m-embedding` (256-d) + 5-d LapPE |
| Indexing and fusion | `hnsw_ef_construction` 400, `code_encoder_max_length` 512, `gamma` 0.5 (with a softmax over K_f = 3 subgraphs at most one subgraph exceeds 0.5) |
| Structural encoder | `gat_layers` 2, `gat_heads` 4, `gat_hidden_dim` 256, `gat_dropout` 0.1, `gat_edge_type_mode` relation_attention_bias, `gat_add_reverse_edges` true, `graph_pooling` mean, `subgraph_hops` 1, `type_compatibility` same_node_type_or_reference |
| Training | `infonce_temperature` 0.07, `optimizer` adamw, `learning_rate` 0.001, `weight_decay` 0.01, `batch_size` 16, `max_epochs` 20, `early_stopping_patience` 3, `num_random_negatives` 8, `num_hard_negatives` 8, `train_queries_per_repo` 50, `random_seed` 42 |
| Retrieval | `mmr_lambda` 0.7 |
| Prompt and decoding | `prompt_tokenizer` tiktoken:o200k_base, `llm_temperature` 0.0, `llm_top_p` omit, `llm_seed` omit, `system_prompt` "" |
| Models | `gpt-4o-mini`, `gpt3.5` -> `gpt-3.5-turbo`, `qwen2.5-coder-14b-instruct` (DashScope), `deepseek-v3` -> `deepseek-chat` (DeepSeek) |

## 3. Pipeline

All commands run from `grace_code/`. Without `--set`, the configured values are used;
`--set key=value` overrides a value and is recorded in the configuration snapshot.

```bash
CFG="--config configs/paper_reproduction.yaml"
TASK="--dataset crosscodeeval --language python --dataset-root dataset/crosscodeeval \
      --repos-root dataset/crosscodeeval/repos"
```

1. **Leakage-masked graphs** (optional cache; otherwise built on the fly):
   `python -m grace.cli build-graphs $CFG $TASK --out runs/graphs`
2. **Train the structural encoder + gate** on development repositories that are project-disjoint
   from the test benchmarks:
   ```bash
   python -m grace.train_encoder $CFG \
     --language python --train-queries dev/train_queries.jsonl --repos-root dev/repos \
     --fit-repos dev/fit_repos.txt --tune-repos dev/tune_repos.txt \
     --test-repos dataset/crosscodeeval/rawdata/python/line_completion.jsonl \
                  dataset/repoeval_updated/rawdata/python/line_level.python.test.jsonl \
     --output-dir runs/encoder_python
   ```
   The script stops with `ProjectOverlapError` if a repository (full name or basename) occurs in more
   than one of fit / tune / test, and refuses training queries from repositories outside fit/tune.
   It writes `best.pt` (best tuning Recall@K_c), `last.pt` and `training_log.json`.
3. **Index** (per task; CodeT5p embeddings + GAT embeddings + HNSW):
   `python -m grace.cli index $CFG $TASK --checkpoint runs/encoder_python/best.pt --graph-dir runs/graphs --index-dir runs/index`
4. **Retrieve, fuse, serialize, build prompts**:
   `python -m grace.cli retrieve $CFG $TASK --checkpoint runs/encoder_python/best.pt --graph-dir runs/graphs --index-dir runs/index --output runs/retrieval.jsonl`
5. **Generate** (uses `llm_temperature`, `llm_top_p`, `llm_seed`, `system_prompt`; each may be a
   value or `omit` = not sent; the model entry must have a `model_id`):
   `python -m grace.cli generate $CFG --retrieval runs/retrieval.jsonl --model qwen2.5-coder-14b-instruct --output-dir results/crosscodeeval/python/grace`
   Output: `prediction.jsonl` (fields `task_id`, `pred`, `code_context`, `groundtruth`, plus provider,
   `model_id`, request parameters, UTC timestamps, parse status) and
   `failures.jsonl`.
6. **Metrics** (unchanged evaluation script):
   `python model/grace_eval.py --language python --output_dir results/crosscodeeval/python/grace --ts_lib build/python-lang-parser.so`

Compatibility entry point: `python model/grace_coderag.py --model ... --dataset ... --language ... --checkpoint ...`
runs steps 4 and 5 with the old default directories (see the module docstring for the mapping of
`--k`, `--alpha`, `--fusion-type`, `--device`).

### Preprocessed indexes (artifact)

```bash
python scripts/package_indexes.py pack --index-dir runs/index --out grace_indexes.tar.gz --strip-code
python scripts/package_indexes.py unpack --archive grace_indexes.tar.gz --out runs/index
python scripts/package_indexes.py hydrate --index-dir runs/index $CFG --dataset crosscodeeval --language python \
  --data-file dataset/crosscodeeval/rawdata/python/line_completion.jsonl --repos-root dataset/crosscodeeval/repos
```

`--strip-code` removes all repository source text (unit text and node `code`) and keeps embeddings,
ids, spans and graph structure, so non-redistributable repository content is not shipped. `hydrate`
rebuilds the masked graphs from local snapshots and refills the text, failing if unit ids or node
lists differ. The pipeline refuses stripped indexes. The archive has a SHA-256 manifest.
No index archive is included in this repository.

## 4. Method

**Leakage control** (`grace/leakage.py`). Before building any graph or index, the current file is cut
at the cursor (end of the unique occurrence of the benchmark prompt in the file; if not unique or not
found, the prompt alone is used as the visible current file). The target line and the post-cursor
suffix therefore never reach the graph, index, snippets or prompt. Indexes are built per task.

**Multi-level graph** (`grace/graph_builder.py`, Tree-sitter Python and Java grammars). Nodes: folder,
file, class, function, statement, variable.
Relations: `contains`, `imports` (repository); `calls`, `inherits`, `type_ref` (module); `ast_root`,
`ast_child`, `control_flow`, `defines`, `data_flow` (function); `declares`, `dep_call`, `type_flow`
(cross-level). Calls/types are linked only when resolvable in the same file or an imported file
(Java: same package or import). The current file ends at the cursor and is usually incomplete;
before parsing, a trailing open Python block is closed with `pass` and open Java braces are closed,
and trailing lines are dropped (at most 50) until the tree has no syntax error; otherwise the full
text is parsed with Tree-sitter's error recovery.

**Node features**: `x_v = [CodeT5p(code) (256, L2-normalised); LapPE (5)]` = 261 dims, asserted at
runtime. LapPE: eigenvectors of the symmetric normalised Laplacian of the undirected simple graph,
zero eigenvalues skipped (disconnected graphs), zero padding for small graphs, deterministic signs.

**Query graph** `G_q`: current-file nodes overlapping the local-context window kept by the token budget.

**Retrieval units** (`grace/subgraphs.py`): one unit per function/class: body nodes + `subgraph_hops` hops over semantic relations; induced edges
keep their original relations. Units in the current file overlapping the query window are skipped.

**Structural encoder** (`grace/encoder.py`): `gat_layers` GAT layers, `gat_heads` heads, output
`gat_hidden_dim`, pooled with `graph_pooling` (mean|max). Edge types (`gat_edge_type_mode`):
`homogeneous` ignores relations; `relation_attention_bias` adds a learned per-(relation, head) bias to
attention logits. Self-loops always; reverse edges if `gat_add_reverse_edges`. Pure PyTorch.

**Gate**: `alpha = sigmoid(W [v_q; h_Gq] + b)`, trained jointly with the GAT; `v_q` is the CodeT5p
embedding of the local context. `alpha_mode: fixed` + `fixed_alpha` is an ablation.

**Training** (`grace/train_encoder.py`): score `alpha*cos(v_q,v_c) + (1-alpha)*cos(h_Gq,h_Gc)`, InfoNCE
with temperature (several positives: `-log(sum_pos/sum_all)`). Positives: `positive_units` if given,
otherwise units whose names match identifiers in the reference line. Negatives: random units of the same repository outside the
semantic top-K_c pool, and hard negatives = non-positive units of that pool. Early stopping on
Recall@K_c over tuning queries.

**Hybrid retrieval** (`grace/retrieve.py`): top-K_c from the semantic path and from the structural path
(HNSW with `M`, `ef_search`, `ef_construction`), union scored with the formula above, MMR
(`mmr_lambda`, redundancy = the same alpha mix of cosines) keeps K_c, the pool is reranked by score and the top K_f are fused.

**Fusion** (`grace/fusion.py`, Eqs. 7-12): `alpha_i^F = softmax(scores)`; `beta = softmax_u(h_v.h_u/sqrt(d))`;
only subgraphs with `alpha_i^F > gamma` update `h_v` and can create edges; an `align` edge `v -> u` is
added iff `alpha_i^F > gamma`, `beta > theta` and `TypeCompatible(v,u)`. `G_f` = `G_q` + all top-K_f
subgraphs (original nodes/edges, merged by node id) + `align` edges. `TypeCompatible`
(`type_compatibility`): `same_node_type` (equal node types) or `same_node_type_or_reference` (equal types, or `v` references
`u` by name, or `v` is a variable whose inferred type name is `u`'s name).

**Serialization** (`grace/serialize.py`): `NODE(<id>, <type>, "<code>")` and
`EDGE(<source>, <relation>, <target>)`; code is a JSON string; native relations keep their names;
virtual edges are `align`. `model/graph_text_encoders.py` keeps the old encoder and adds
`encode_code_graph_paper_format`.

**Prompt budget** (`grace/prompt.py`): 2048 = 1024 local + 1024 retrieved tokens, counted with
`prompt_tokenizer` (`hf:<name>`, `tiktoken:<encoding>`, or `regex` for tests).
Local context keeps the most recent lines. Retrieved context is filled rank by rank: the unit snippet
(cut at line granularity), then its NODE/EDGE records (align edges bring their query endpoint); the
remaining query-graph records last. The template instructions (`utils/prompt.md`) are not counted.
Truncation statistics are written to `prompt_stats`.

**Response parsing** (`grace/llm.py`, written to `parse_status`):
1. The first JSON object (in a code fence or anywhere in the text) with a `completed_code` key is
   used; if it has several non-empty lines only the first is kept (`json_multiline_truncated`).
2. Otherwise the parse error is recorded and `parse_failure_policy` applies: `first_code_line`
   (default) = first non-empty line that is not a code fence, a language tag or a boilerplate phrase;
   `empty` = empty prediction.

**API failures**: bounded retries with exponential backoff; a failed request yields no text and is written to `failures.jsonl`. `on_api_error: skip` (default)
leaves it out of `prediction.jsonl`; `record_empty` writes `pred: ""` with `parse_status: api_error`.

## 5. Released scripts

| File | Role |
|---|---|
| `model/grace_coderag.py` | compatibility wrapper around `grace.cli`; the original script is archived as `legacy/grace_coderag_original_release.py` (hard-coded `cuda:3` replaced by `GRACE_DEVICE`/automatic selection). |
| `data_process/multilevel_graph_builder.py` | earlier graph format; supports Python and Java, CrossCodeEval and RepoEval-Updated, and builds on a leakage-masked copy of the repository. `grace/` uses `grace/graph_builder.py`. |
| `model/graph_text_encoders.py` | graph serializers, including the paper format `encode_code_graph_paper_format`. |
| `data_process/download_repos.py` | repository cloning (arguments instead of hard-coded settings, credentials only from environment); `donwload_repos.py` is kept as a wrapper. |
| `utils/utils.py`, `model/generate.py` | device from `GRACE_DEVICE`/automatic selection; `generate.py` endpoint overridable via `OPENAI_BASE_URL`. |
| `model/grace_eval.py`, `eval_*`, baselines | evaluation and baseline scripts. |

## Tests

`uv run pytest` (or `python -m pytest tests`) covers: Tree-sitter graph construction for Python and
Java (all node types and relations, tail repair of incomplete current files); 261-d features and LapPE
edge cases; GAT shapes, edge-type modes and checkpoint round-trip; a one-step InfoNCE run that saves
checkpoints; project-disjoint fit/tune/test splits; the alpha gate range and formula; candidate merge,
MMR, K_c and K_f; `align` edge conditions; serialization format; prompt budgets; leakage of the target
line and suffix into graph/index/snippets/prompt; Python and Java end-to-end runs on small
repositories; mocked LLM provider, JSON-parse fallbacks and API failures; configuration sections and
unset parameters; the CLI, the compatibility wrapper, index packaging and the earlier graph builder
(skipped if its optional dependencies are missing). Tests use a hashing text encoder and an untrained GAT.
