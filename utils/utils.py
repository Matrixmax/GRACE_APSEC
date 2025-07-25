import glob
import networkx as nx
import pickle
import json
import os
from networkx.readwrite import json_graph
import tiktoken

device = "cuda"


class CodeGenTokenizer:
    def __init__(self, tokenizer_raw):
        self.tokenizer = tokenizer_raw

    def tokenize(self, text):
        return self.tokenizer(text, return_tensors="pt")['input_ids'][0].to(device)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


class StarCoderTokenizer:
    def __init__(self, tokenizer_raw):
        self.tokenizer = tokenizer_raw

    def tokenize(self, text):
        return self.tokenizer.encode(text, return_tensors="pt")[0].to(device)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


class CONSTANTS:
    max_hop = 5
    max_search_top_k = 10
    max_statement = 20
 
    repo_base_dir = f"dataset/hf_datasets/repobench_python_v1.1/cross_file_first/repos"
    graph_database_save_dir = f"dataset/hf_datasets/repobench_python_v1.1/cross_file_first/repos_graphs_labeled_link_with_called_imported_edges"
    query_graph_save_dir = "dataset/hf_datasets/repobench_python_v1.1/cross_file_first/repos_call_graphs"
    search_results_save_dir= "dataset/hf_datasets/repobench_python_v1.1/cross_file_first/repos_search_results"
    generate_save_dir = "dataset/hf_datasets/repobench_python_v1.1/cross_file_first/repos_generation_results"
    dataset_dir = "dataset/hf_datasets/repobench_python_v1.1/cross_file_first/repos"
    repos = os.listdir(repo_base_dir)
    repos_language =  {item: "python" for item in repos}



class CodexTokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo-instruct")
        # self.tokenizer = tiktoken.get_encoding("p50k_base")

    def tokenize(self, text):
        # return self.tokenizer.encode(text)
        return self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)


class CodeGenTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)


def iterate_repository_file(base_dir, repo):
    if CONSTANTS.repos_language[repo] == "python":
        pattern = os.path.join(f'{base_dir}/{repo}', "**", f"*.py")
    elif CONSTANTS.repos_language[repo] == "java":
        pattern = os.path.join(f'{base_dir}/{repo}', "**", f"*.java")
    else:
        raise NotImplementedError
    files = glob.glob(pattern, recursive=True)
    return files


def iterate_repository_json_file(base_dir, repo):
    pattern = os.path.join(f'{base_dir}/{repo}', "**", "*.json")
    files = glob.glob(pattern, recursive=True)
    return files


def make_needed_dir(file_path):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return


def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def dump_jsonl(obj, fname):
    with open(fname, 'w', encoding='utf8') as f:
        for item in obj:
            f.write(json.dumps(item, default=set_default) + '\n')


def load_jsonl(fname):
    with open(fname, 'r', encoding='utf8') as f:
        lines = []
        for line in f:
            lines.append(json.loads(line))
        return lines




def graph_to_json(obj: nx.MultiDiGraph):
    return json.dumps(json_graph.node_link_data(obj), default=set_default)


def json_to_graph(json_format):
    graph_js = json.loads(json_format)
    graph = json_graph.node_link_graph(graph_js)
    return graph


def tokenize(code):
    tokenizer = CodexTokenizer()
    return tokenizer.tokenize(code)
