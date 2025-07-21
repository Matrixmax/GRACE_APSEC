# GRACE
GRACE: Graph-Guided Repository-Aware Code Completion through Hierarchical Code Fusion


### 0. environment preparation
1. install uv
2. uv sync
### 1. Raw Data Acquisition
1. repoeval_updated dataset: Run the code data_process/donwload_repos.py to download the corresponding repos. The cceval dataset does not need to be acquired.
### 2. Graph Construction
1. Run the code data_process/multilevel_graph_builder.py to generate the corresponding graph.
### 3. Run RAG Process
1. Run the code model/grace_coderag.py to get the generated results.
### 4. Metric Calculation
1. Run the code model/grace_eval.py for testing.