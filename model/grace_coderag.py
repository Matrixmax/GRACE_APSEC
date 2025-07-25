import os
import torch
import networkx as nx
import numpy as np
import hnswlib
import json
import requests
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import argparse
from transformers import AutoModel,AutoTokenizer, AutoModelForCausalLM
from transformers import PreTrainedTokenizer, PreTrainedModel
from datasets import load_dataset
import tqdm
from string import Template
import openai
from openai import OpenAI

import dgl
from dgl import LapPE
import logging
import pickle
import ast
import parso
import graph_text_encoders
# from graph_text_encoders import encode_code_graph
import dashscope
from http import HTTPStatus

def _calculate_laplacian_pe(graph: nx.Graph, node_id: str) -> np.ndarray:
    return dgl.lap_pe(graph, k=5, padding=True)

def _parse_code_to_ast_graph(code_snippet: str) -> nx.Graph:
    node_id_counter = [0]
    # parso_to_nx_map = {}

    def parso_to_networkx_recursive(p_node, nx_graph, parent_nx_id=None):
        current_nx_id = node_id_counter[0]
        node_id_counter[0] += 1

        node_type = p_node.type
        try:
            node_code = p_node.get_code().strip()
        except Exception:
            node_code = f"<{node_type}>"

        nx_graph.add_node(current_nx_id, type=node_type, code=node_code, parso_type=str(type(p_node)))

        if parent_nx_id is not None:
            nx_graph.add_edge(parent_nx_id, current_nx_id)

        if hasattr(p_node, 'children') and p_node.children:
            for child_node in p_node.children:
                parso_to_networkx_recursive(child_node, nx_graph, current_nx_id)
    G = nx.DiGraph()
    module_node = parso.parse(code_snippet)
    parso_to_networkx_recursive(module_node, G)
    return G

def filter_llm_response(code:str, language:str="python"): 

    # check if the language is valid
    assert language in ["python", "java"], "language must be one of [python, java]"


    # first remove the \n at the beginning of the code
    code = code.lstrip('\n')

    lines = code.split('\n')
    
    # --- Start: Added noise filtering ---
    filtered_lines = []
    start_processing = False
    # Define patterns for noise lines to ignore at the beginning
    # Matches lines like: ```, ```python, python, java, ''', """, the next code is:
    noise_patterns = [
        re.compile(r"^\s*```(\s*\w*)?\s*$"), # Code fence start (optional language)
        re.compile(r"^\s*(python|java)\s*$", re.IGNORECASE), # Language name only
        re.compile(r"^\s*('''|\"\"\")\s*$"), # Triple quotes only
        re.compile(r"^\s*the next code is:", re.IGNORECASE), # Introductory phrase
    ]
    
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        stripped_line = line.strip()

        # If we already started processing, keep the line
        if start_processing:
            filtered_lines.append(line)
            line_index += 1
            continue
            
        # Skip empty lines before code starts
        if not stripped_line:
            line_index += 1
            continue

        # Check if the line matches any noise pattern
        is_noise = False
        for pattern in noise_patterns:
            if pattern.match(stripped_line):
                is_noise = True
                break
        
        if is_noise:
            # It's a noise line, skip it
            line_index += 1
            continue
        else:
            # Not noise, not empty -> start processing from here
            start_processing = True
            filtered_lines.append(line)
            line_index += 1
            
    # If all lines were filtered out as noise or empty
    if not filtered_lines:
        # Return empty string or maybe the original first line if that's preferred?
        # Let's return empty string for now.
        return ""
        
    # --- End: Added noise filtering ---

    # Now process the filtered lines
    lines = filtered_lines # Use the filtered lines for the rest of the logic
    in_multiline_comment = False

    if language == "python":
        for line in lines:
            # if the line is empty, then skip (already handled by initial filter, but keep for safety)
            if not line.strip():
                continue
            # if the line is a start of a multiline comment, then set the in_multiline_comment to True and skip
            # Check if it STARTS with quotes but isn't JUST quotes (handled by noise filter)
            if not in_multiline_comment and (line.strip().startswith('"""') or line.strip().startswith("'''")):
                 # Check if it also ends on the same line and isn't just the quotes
                if not (line.strip().endswith('"""') or line.strip().endswith("'''")) or len(line.strip()) <= 3:
                    in_multiline_comment = True
                # If it's a single-line docstring/comment like """ comment """, don't enter multi-line mode
                continue 
            # if the line is the end of a multiline comment, then set the in_multiline_comment to False and skip
            if in_multiline_comment and (line.strip().endswith('"""') or line.strip().endswith("'''")):
                in_multiline_comment = False
                continue
            # if the line is in a multiline comment, then skip
            if in_multiline_comment:
                continue
            # if the line is a single line comment, then skip
            if line.strip().startswith('#'):
                continue
            # if the line is not a comment, then return the line
            return line

    elif language == "java":
        for line in lines:
            # if the line is empty, then skip
            if not line.strip():
                continue
            # if the line is a start of a multiline comment, then set the in_multiline_comment to True and skip
            if not in_multiline_comment and line.strip().startswith('/*'):
                 # Check if it also ends on the same line
                if not line.strip().endswith('*/') or len(line.strip()) <= 2:
                     in_multiline_comment = True
                # If it's a single-line block comment /* comment */, don't enter multi-line mode
                continue
            # if the line is the end of a multiline comment, then set the in_multiline_comment to False and skip
            if in_multiline_comment and line.strip().endswith('*/'):
                in_multiline_comment = False
                continue
            # if the line is in a multiline comment, then skip
            if in_multiline_comment:
                continue
            # if the line is a single line comment, then skip
            if line.strip().startswith('//'):
                continue
            # if the line is not a comment, then return the line
            return line


    # if we cannot find a line that is not a comment or noise, return the first line of the *filtered* list
    # (or empty string if filtering removed everything)
    return lines[0] if lines else ""


def retrieved_graph_serialized(self, retrieved_graph: nx.DiGraph) -> List[str]:
    """
    Serialize the retrieved graph into a list of code snippets for the LLM.
    
    Args:
        retrieved_graph: The fused graph with query and snippets
        
    Returns:
        List[str]: Serialized code snippets from the graph
    """
    serialized = []
    
    # Add node code attributes to the serialized output
    for node, data in retrieved_graph.nodes(data=True):
        if 'code' in data:
            code = data['code']
            # Add node type prefix if available
            if 'type' in data:
                prefix = f"[{data['type']}] "
            else:
                prefix = ""
            
            serialized.append(f"{prefix}{code}")
    
    
    return serialized

class LLMModel:
    def __init__(self, model_name: str="qwen2.5-coder-3b-instruct"):
        self.model_name = model_name
    
    def code_complete(self,out_path,data):        
        skipped = []
        with open(out_path, 'w') as f:
            for d in data:
                try:
                    prompt = self.prepare_prompt(d)
                    if "gpt" in self.model_name:
                        response = self._call_openai_LLM(prompt, self.model_name)
                    elif "qwen" in self.model_name:
                        response = self._call_qwen_LLM(prompt, self.model_name)
                    elif "deepseek" in self.model_name:
                        response = self._call_deepseek_LLM(prompt, self.model_name)
                except Exception as e:
                    print('Unknown error', e)
                    raise

                if response is not None:
                    d['pred_raw'] = response
                    json_str = response.strip()
                    if json_str.startswith('```json'):
                        # Handles ```json ... ```
                        json_str = '\n'.join(json_str.split('\n')[1:]).rstrip('`').strip()
                    elif json_str.startswith('```'):
                        # Handles ``` ... ```
                        json_str = '\n'.join(json_str.split('\n')[1:]).rstrip('`').strip()

                    try:
                        # Parse the JSON and extract relevant fields
                        response_json = json.loads(json_str)
                        d['pred'] = response_json.get("completed_code", "")
                        d['explanation'] = response_json.get("explanation", "")
                        d['confidence_score'] = response_json.get("confidence_score", -1)
                        d['referenced_nodes'] = response_json.get("referenced_nodes", [])
                    except (json.JSONDecodeError, AttributeError):
                        # Fallback for non-JSON responses or if response_json is not a dict
                        d['pred'] = json_str # Use the cleaned string as prediction

                    # d['api_response'] = str(response)
                    d['prompt_used'] = prompt  # records the augmented prompt
                    d['task_id'] = d['task_id']  # adding for compatibility with eval script
                    # print("真实值：",d['groundtruth'])
                    # print("预测值：",d['pred'])

                    print(json.dumps(d, default=lambda o: o.item() if isinstance(o, np.generic) else (o.tolist() if isinstance(o, np.ndarray) else str(o))), file=f, flush=True)
                else:
                    skipped.append(d['task_id'])
                    print(f'Skipped {d["task_id"]}')

        return skipped    


    def _call_openai_LLM(self, inputs, contexts):
        url = "https://api.chatanywhere.tech/v1/chat/completions"
        OPENAI_KEY = os.getenv("OPENAI_API_KEY")
        payload = json.dumps({
        "model": self.model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": self.format(inputs, contexts)
            }
        ]
        })
        headers = {
        'Authorization': f'Bearer {OPENAI_KEY}',
        'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        response_dict = json.loads(response.text)

        generated_content = response_dict['choices'][0]['message']['content']
        next_line = generated_content.split('\n')[0].strip()
        return generated_content

    def _call_siliconflow_deepseek_LLM(self, prompt: str, specific_model=None) -> dict:
        url = "https://api.siliconflow.cn/v1/chat/completions"
        payload = {
            "model": specific_model if specific_model else self.llm_model,
            "messages": [
                {"role": "system", "content": "You are a text parsing assistant. Return only pure JSON format results."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.6
        }
        headers = {
            "Authorization": "Bearer " + os.getenv("SILICIONFLOW_API_KEY", ""),
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                response_json = response.json()
                content = response_json['choices'][0]['message']['content']
                if '```json' in content:
                    json_str = content.split('```json\n')[1].split('```')[0].strip()
                else:
                    json_str = content.strip()
                try:
                    result = json.loads(json_str)
                    return result
                except json.JSONDecodeError:
                    print(f"Failed to parse content as JSON: {json_str}")
                    return {}
            else:
                print(f"Error from DeepSeek API: {response.text}")
                return {}
        except requests.exceptions.RequestException as e:
            print(f"Request error calling DeepSeek API: {str(e)}")
            return {}
        except Exception as e:
            print(f"Unexpected error calling DeepSeek API: {str(e)}")
            return {}
    
    def _call_qwen_LLM(self, prompt: str, specific_model = None) -> dict:
        try:
            client = OpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            response = client.chat.completions.create(
                model=specific_model if specific_model else self.llm_model,
                messages=[
                    {"role": "system", "content": "You are Codex, a code completion language model. Continue the code presented to you."}, 
                    {"role": "user", "content": prompt}
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Qwen API: {str(e)}")
            return {}

    def _call_groq_LLM(self, prompt: str, specific_model=None) -> dict:
        try:
            from groq import Groq
            client = Groq(
                api_key=os.environ.get("GROQ_API_KEY"),
            )
            response = client.chat.completions.create(
                model=specific_model if specific_model else self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a text parsing assistant. You must respond in JSON format. Do not include any explanatory text."}, 
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            json_string = response.choices[0].message.content
            json_object = json.loads(json_string)
            return json_object
        except Exception as e:
            print(f"Error calling Groq API: {str(e)}")
            return {}
    def _encode_text_openai(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
        )
        return np.array(response.data[0].embedding)
    def _encode_text_netease(self, text: str) -> np.ndarray:
        url = "https://api.siliconflow.cn/v1/embeddings"
    
        payload = {
            "input": text,
            "model": "netease-youdao/bce-embedding-base_v1",
            "encoding_format": "float"
        }
        headers = {
            "Authorization": "Bearer " + os.getenv("SILICIONFLOW_API_KEY", ""),
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return np.array(response.json()['data'][0]['embedding'])
        else:
            raise RuntimeError(f"Failed to get embedding from Netease API: {response.text}")
    
    def _encode_text_qwen(self,text:str)->np.ndarray:
        input = [{'text': text}]
        resp = dashscope.MultiModalEmbedding.call(
            model="multimodal-embedding-v1",
            input=input
        )

        if resp.status_code == HTTPStatus.OK:
            print(json.dumps(resp.output, ensure_ascii=False, indent=4))
            return np.array(resp.output['embeddings'][0]['embedding'])
        else:
            raise RuntimeError(f"Failed to get embedding from Qwen API: {resp.text}")
        

    def format(self, inputs, contexts):
        return f"Given following context: {contexts} and your need to complete following {inputs} in one line:"
    
    def get_embeddings(self, text):
        import numpy as np
        import torch
        from hashlib import md5
        
        hash_object = md5(text.encode())
        hash_hex = hash_object.hexdigest()
        
        hash_ints = [int(hash_hex[i:i+2], 16) for i in range(0, len(hash_hex), 2)]
        
        np.random.seed(sum(hash_ints))
        embedding = torch.tensor(np.random.randn(1, 768)).float()
        
        embedding = embedding / torch.norm(embedding)
        
        return embedding
    
    def _encode_text_netease(self, text: str) -> np.ndarray:
        url = "https://api.siliconflow.cn/v1/embeddings"
    
        payload = {
            "input": text,
            "model": "netease-youdao/bce-embedding-base_v1",
            "encoding_format": "float"
        }
        headers = {
            "Authorization": "Bearer " + os.getenv("SILICIONFLOW_API_KEY", ""),
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return np.array(response.json()['data'][0]['embedding'])
        else:
            raise RuntimeError(f"Failed to get embedding from Netease API: {response.text}")
    
    def load_codet5(self, checkpoint: str = "Salesforce/codet5p-110m-embedding"):
        """
        Load the CodeT5 model from the specified checkpoint.
        
        Args:
            checkpoint (str): Path to the CodeT5 model checkpoint.
            
        Returns:
            Tuple[AutoTokenizer, AutoModel]: The tokenizer and model.
        """
        device = "cuda:3" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained("./pretrain_models", trust_remote_code=True)
        model = AutoModel.from_pretrained("./pretrain_models", trust_remote_code=True).to(device)

        return tokenizer, model

    def _encode_text_codet5(self, text: str, tokenizer, model) -> np.ndarray:
        """
        Encodes text using the loaded CodeT5 model to get embeddings.

        Args:
            text: The input text (code snippet).

        Returns:
            np.ndarray: The embedding vector for the text.
        """

        with torch.no_grad(): # Ensure no gradients are calculated
            device = "cuda:3" if torch.cuda.is_available() else "cpu"
            inputs = tokenizer.encode(text, return_tensors="pt").to(device)
            embedding = model(inputs)[0]

            return embedding.cpu().numpy().astype(np.float32)



    @staticmethod
    def prepare_prompt(data: dict) -> str:
        """
        Constructs a prompt for code completion by filling a template
        with context from the data dictionary using string.Template.

        Args:
            data (dict): A dictionary containing the necessary data.

        Returns:
            str: The fully constructed prompt.
        """
        # Determine the correct path to the prompt template file.
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # template_path = os.path.join(current_dir, '..', 'utils', 'prompt.md')
        template_path = "utils/prompt.md"

        with open(template_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        # Known placeholders in the template
        placeholders = [
            "repo_name",
            "current_file_path",
            "code_before_cursor",
            "code_context",
            "graph_context",
        ]

        escaped_template = prompt_template.replace("{", "{{").replace("}", "}}")
        for ph in placeholders:
            escaped_template = escaped_template.replace("{{" + ph + "}}", "{" + ph + "}")

        # Populate the template with the provided context from the data dictionary.
        filled_prompt = escaped_template.format(
            repo_name=data.get("repo_name", ""),
            current_file_path=data.get("file_path", ""),
            code_before_cursor=data.get("code_context", ""),
            code_context=data.get("retrieved_snippets", ""),
            graph_context=data.get("graph_context", "")
        )

        return filled_prompt


class GraphRAGPipeline:
    def __init__(
        self,
        args,
        model_name: str,
        index_path: str,
        dim: int = 768,
        ef_construction: int = 400,
        M: int = 64
    ):
        self.model = LLMModel(model_name)
        self.dim = dim
        self.index_path = Path(index_path)
        self.args = args
        


    def _chunk_code(self, file_path: Path, chunk_size: int = 50, overlap: int = 5):
        """
        Chunks code from a file. Simple line-based chunking.
        Yields tuples of (chunk_content, chunk_id).
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            logging.warning(f"Could not read file {file_path}: {e}")
            return

        for i in range(0, len(lines), chunk_size - overlap):
            chunk_lines = lines[i : i + chunk_size]
            if not chunk_lines:
                continue
            chunk_content = "".join(chunk_lines)
            start_line = i + 1
            end_line = i + len(chunk_lines)
            chunk_id = f"{file_path}:{start_line}-{end_line}"
            yield chunk_content, chunk_id

    def encode_code(self, code: str) -> torch.Tensor:
        try:
            if not isinstance(code, str):
                code = str(code) if code is not None else "" 
            
            device = self.model.device if hasattr(self.model, 'device') else 'cpu'

            if not code.strip():
                return torch.zeros(self.code_embedding_dim, device=device)

            embedding = self.model.get_embeddings(code)
            
            if embedding.ndim == 2 and embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)
            
            if embedding.shape[0] != self.code_embedding_dim:
                final_embedding = torch.zeros(self.code_embedding_dim, device=embedding.device)
                copy_len = min(embedding.shape[0], self.code_embedding_dim)
                final_embedding[:copy_len] = embedding[:copy_len]
                return final_embedding
            return embedding
        except Exception as e:
            logging.error(f"Error encoding code: '{code[:50]}...': {e}", exc_info=True)
            device = self.model.device if hasattr(self.model, 'device') else 'cpu'
            return torch.zeros(self.code_embedding_dim, device=device)

    def _encode_text_codet5_batch(self, text_list, tokenizer, model, device=None, max_len=512):
        """
        text_list: List[str]
        return     (B, hid_dim)  numpy.float32
        """
        # Determine device if not explicitly provided
        if device is None:
            device = model.device if hasattr(model, "device") else "cpu"

        enc = tokenizer(text_list, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
         
        with torch.no_grad():
            out = model.encoder(**enc).last_hidden_state         # (B, L, d)
            emb = out[:, 0, :]                                   # (B, d)

        return emb.cpu().numpy().astype(np.float32)

    def _encode_graph_nodes(self,
                        graph: nx.Graph,
                        llm_tools,
                        tokenizer,
                        model,
                        batch_size: int = 512,
                        lappe_k: int = 5):
        dgl_graph = dgl.from_networkx(graph)
        lap_pe = dgl.lap_pe(dgl_graph, k=lappe_k, padding=True).cpu().numpy()

        node_ids, codes = zip(*[(nid, data["code"]) for nid, data in graph.nodes(data=True)])

        text_emb_chunks = []
        for i in range(0, len(codes), batch_size):
            chunk = codes[i:i + batch_size]
            text_emb_chunks.append(
                self._encode_text_codet5_batch(chunk, tokenizer, model)
            )
        text_emb = np.vstack(text_emb_chunks)                    # (N, d1)

        final_emb = np.concatenate([text_emb, lap_pe], axis=1)   # (N, d1 + d2)

        return final_emb, list(node_ids), list(codes)

    def old_encode_graph_nodes(self,graph: nx.Graph, llm_tools: LLMModel, tokenizer, model, calculate_lappe: bool = True, lappe_k: int = 5) -> (np.ndarray, list):
        dgl_graph = dgl.from_networkx(graph)
        lappe_embeddings = dgl.lap_pe(dgl_graph, k=5, padding=True)
        i = 0
        graph_node_embeddings_for_hnsw = np.array([])
        graph_node_ids = []
        graph_node_snippets = []
        for node_id, node_data in graph.nodes.items():
            node_emb = llm_tools._encode_text_codet5(node_data['code'], tokenizer, model) # Use CodeT5
            struct_emb = lappe_embeddings[i]
            i += 1
            final_emb = np.concatenate((node_emb, struct_emb)).astype(np.float32)
            if graph_node_embeddings_for_hnsw.size != 0:
                graph_node_embeddings_for_hnsw = np.concatenate([graph_node_embeddings_for_hnsw,[final_emb]], axis=0)
            else:
                graph_node_embeddings_for_hnsw = np.array([final_emb])
            graph_node_ids.append(node_id) # Store the original node ID
            graph_node_snippets.append(node_data['code']) # Added this line
        return graph_node_embeddings_for_hnsw, graph_node_ids, graph_node_snippets # Added graph_node_snippets to return


    def build_index(self, graph_path: str, repo_name: str, repo_path: Path, index_save_path: str, code_snippet: str = None):
        Path(index_save_path).mkdir(parents=True, exist_ok=True)
        
        graph_file_path = Path(graph_path) / "repo_multi_graph.graphml"
        if not graph_file_path.exists():
            logging.error(f"GraphML file not found at {graph_file_path}")
            return

        combined_graph = nx.read_graphml(graph_file_path)
        print(f'Successfully loaded graph from {graph_file_path}')


        dgl_graph = dgl.from_networkx(combined_graph)
        lappe_embeddings = dgl.lap_pe(dgl_graph, k=5, padding=True)

        code_metadata_list = []
        graph_node_metadata_list = []
        
        code_embeddings_for_hnsw = []
        
        current_code_id = 0
        current_graph_node_id = 0

        i = 0
        llm_tools = LLMModel()
        tokenizer, model = llm_tools.load_codet5()

        graph_node_embeddings_for_hnsw, graph_node_ids, graph_node_snippets = self._encode_graph_nodes(combined_graph, llm_tools, tokenizer, model)

        # 4. Build HNSW index for graph nodes
        if graph_node_embeddings_for_hnsw.size != 0:
            graph_index = hnswlib.Index(space='cosine', dim=graph_node_embeddings_for_hnsw.shape[-1])
            graph_index.init_index(max_elements=1000000, ef_construction=400, M=64)
            graph_index.add_items(graph_node_embeddings_for_hnsw, np.arange(len(graph_node_embeddings_for_hnsw))) # Use sequential IDs for HNSW

            # Save the graph index and the mapping from HNSW ID to original node ID
            graph_index_file = f"{index_save_path}/{repo_name}_graph_index.bin"
            graph_map_file = f"{index_save_path}/{repo_name}_graph_map.pkl"
            graph_snippets_file = f"{index_save_path}/{repo_name}_graph_node_snippets.pkl" # Added this line
            graph_index.save_index(str(graph_index_file))
            with open(graph_map_file, 'wb') as f_map:
                pickle.dump(graph_node_ids, f_map) # Save the list mapping index 0..N-1 to original node IDs
            with open(graph_snippets_file, 'wb') as f_snippets: # Added this block
                pickle.dump(graph_node_snippets, f_snippets) # Added this line
            logging.info(f"Graph index saved to {graph_index_file}")
            logging.info(f"Graph node ID map saved to {graph_map_file}")
            logging.info(f"Graph node snippets saved to {graph_snippets_file}") # Added this line
        else:
            logging.warning("No valid graph node embeddings generated. Skipping graph index creation.")

        
        # 5. Find source files, chunk, embed, and add to index
        # Assuming Python files for now
        logging.info("Starting code chunk index building...")
        code_chunk_embeddings = np.array([])
        code_chunk_ids = []
        all_code_snippets = [] # Added this line

        source_files = list(repo_path.rglob('*.py')) 
        logging.info(f"Found {len(source_files)} Python files for code chunking.")

        for py_file in source_files:
            for chunk_content, chunk_id in self._chunk_code(py_file):
                # Compute semantic embedding for the code chunk
                chunk_emb = llm_tools._encode_text_codet5(chunk_content, tokenizer, model) # Use CodeT5
                if code_chunk_embeddings.size != 0:
                    code_chunk_embeddings = np.concatenate([code_chunk_embeddings,[chunk_emb]], axis=0)
                else:
                    code_chunk_embeddings = np.array([chunk_emb])
                code_chunk_ids.append(chunk_id) # Store file:start-end ID
                all_code_snippets.append(chunk_content) # Added this line

        # 6. Build HNSW index for code chunks
        if code_chunk_embeddings.size != 0:
            code_index = hnswlib.Index(space='cosine', dim=chunk_emb.shape[-1])
            code_index.init_index(max_elements=1000000, ef_construction=400, M=64)
            code_index.add_items(code_chunk_embeddings, np.arange(len(code_chunk_embeddings))) # Use sequential IDs

            # Save the code index and the mapping from HNSW ID to chunk ID
            code_index_file = f"{index_save_path}/{repo_name}_code_index.bin"
            code_map_file = f"{index_save_path}/{repo_name}_code_map.pkl"
            code_snippets_file = f"{index_save_path}/{repo_name}_code_snippets.pkl" # Added this line
            code_index.save_index(str(code_index_file))
            with open(code_map_file, 'wb') as f_map:
                 pickle.dump(code_chunk_ids, f_map) # Save list mapping index 0..N-1 to chunk IDs
            with open(code_snippets_file, 'wb') as f_snippets: # Added this block
                pickle.dump(all_code_snippets, f_snippets) # Added this line
            logging.info(f"Code index saved to {code_index_file}")
            logging.info(f"Code chunk ID map saved to {code_map_file}")
            logging.info(f"Code snippets saved to {code_snippets_file}") # Added this line
        else:
            logging.warning("No valid code chunk embeddings generated. Skipping code index creation.")
            
        logging.info(f"Index building finished for repo: {repo_name}")


    def retrieve(
        self,
        query_code: str,
        query_graph: nx.DiGraph,
        repo_name: str,
        k: int = 5,
        alpha: float = 0.5,
        llm_tools: LLMModel = None,
        tokenizer = None,
        model = None,
        sample_idx: int = None
    ) -> Tuple[List[str], List[float]]:
        """Retrieve relevant code snippets using both code and graph similarity from a specific repository"""
        repo_sample_name = f'{sample_idx}_{repo_name}' if sample_idx is not None else repo_name
        repo_index_path = Path(self.args.index_path) / repo_sample_name
        
        if not (repo_index_path / f'{repo_name}_code_index.bin').exists() or not (repo_index_path / f'{repo_name}_graph_index.bin').exists():
            print(f'Index files missing for repository {repo_name} at {repo_index_path}')
            return [], []
        
        code_index = hnswlib.Index(space='cosine', dim=256)
        graph_index = hnswlib.Index(space='cosine', dim=261)

        code_index.load_index(str(repo_index_path / f'{repo_name}_code_index.bin'))
        graph_index.load_index(str(repo_index_path / f'{repo_name}_graph_index.bin'))
                
        code_emb = llm_tools._encode_text_codet5(query_code, tokenizer, model) # Use CodeT5
        graph_emb, _, _ = self._encode_graph_nodes(query_graph, llm_tools, tokenizer, model)
        
        k = min(k, graph_index.get_current_count())
        k = min(k, code_index.get_current_count())
        code_ids, code_distances = code_index.knn_query(code_emb, k=k)
        graph_ids, graph_distances = graph_index.knn_query(np.mean(graph_emb, axis=0), k=k)
        
        code_snippets_file = f"{repo_index_path}/{repo_name}_code_snippets.pkl"
        graph_snippets_file = f"{repo_index_path}/{repo_name}_graph_node_snippets.pkl"
        
        all_code_snippets = []
        if os.path.exists(code_snippets_file):
            try:
                with open(code_snippets_file, 'rb') as f_snippets:
                    all_code_snippets = pickle.load(f_snippets)
                print(f"Loaded {len(all_code_snippets)} code snippets from {code_snippets_file}")
            except Exception as e:
                print(f"Error loading code snippets: {e}")
        else:
            print(f"Warning: Code snippets file not found: {code_snippets_file}")
        
        all_graph_snippets = []
        if os.path.exists(graph_snippets_file):
            try:
                with open(graph_snippets_file, 'rb') as f_snippets:
                    all_graph_snippets = pickle.load(f_snippets)
                print(f"Loaded {len(all_graph_snippets)} graph snippets from {graph_snippets_file}")
            except Exception as e:
                print(f"Error loading graph snippets: {e}")
        else:
            print(f"Warning: Graph snippets file not found: {graph_snippets_file}")
        

        
        code_results = []
        for i in range(len(code_ids[0])):
            code_idx = code_ids[0][i]
            if code_idx < len(all_code_snippets):
                code_content = all_code_snippets[code_idx]
                code_results.append({
                    'id': str(code_idx),
                    'content': code_content,
                    'score': alpha * (1 - code_distances[0][i]),
                    'type': 'code'
                })
        
        graph_results = []
        for i in range(len(graph_ids[0])):
            graph_idx = graph_ids[0][i]
            if graph_idx < len(all_graph_snippets):
                graph_content = all_graph_snippets[graph_idx]
                graph_results.append({
                    'id': str(graph_idx),
                    'content': graph_content,
                    'score': (1 - alpha) * (1 - graph_distances[0][i]),
                    'type': 'graph'
                })
        
        print(f'Retrieved {len(code_results)} code snippets and {len(graph_results)} graph snippets for {repo_name}')
        
        code_results.sort(key=lambda x: x['score'], reverse=True)
        graph_results.sort(key=lambda x: x['score'], reverse=True)
        merged_results = []
        for i in range(k):
            if i < len(code_results):
                merged_results.append(code_results[i])
            if i < len(graph_results):
                merged_results.append(graph_results[i])
        
        merged_results.sort(key=lambda x: x['score'], reverse=True)
        top_results = merged_results[:k] if len(merged_results) >= k else merged_results
        
        retrieved_snippets = []
        scores = []

        for result in top_results:
            retrieved_snippets.append(result['content'])
            scores.append(result['score'])
        
        retrieved_snippets, scores = self.rerank(retrieved_snippets, scores)

        return retrieved_snippets, scores
    

    def rerank(self, snippets: List[str], scores: List[float]) -> List[str]:
        scored_snippets = []
        for snippet, score in zip(snippets, scores):
            features = self.feature_extractor.get_features(snippet)
            score = sum(self.weights[fname] * fvalue for fname, fvalue in features.items())
            scored_snippets.append((snippet, score, features))

        scored_snippets.sort(key=lambda x: x[1], reverse=True)
        return [snippet for snippet, score, features in scored_snippets], [score for snippet, score, features in scored_snippets]


    def graph_fusion(
        self,
        snippets: List[str],
        query_graph: nx.DiGraph,
        fusion_type: str = 'attention',
        llm_tools: LLMModel = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model: Optional[PreTrainedModel] = None
    ) -> torch.Tensor:
        """Fuse code and graph embeddings using different fusion strategies"""
        if fusion_type == 'concat':
            # Simple concatenation followed by linear projection
            fused = torch.cat([code_emb, graph_emb], dim=-1)
            # Project back to original dimension
            projection = torch.nn.Linear(self.dim * 2, self.dim).to(code_emb.device)
            return projection(fused)
            
        elif fusion_type == 'attention':
            # Calculate cross-attention between query and retrieved graph embeddings
            # Convert graph structures to node feature matrices
            # 1. Build snippets graph
            snippets_graph = nx.DiGraph()
            for i, snippet in enumerate(snippets):
                snippets_graph.add_node(i, code=snippet)

            snippet_embeddings = []
            
            for i, snippet in enumerate(snippets):
                emb = llm_tools._encode_text_codet5(snippet, tokenizer, model)  # Get code embedding
                if isinstance(emb, np.ndarray):
                    emb = torch.from_numpy(emb).to(torch.float32)
                snippet_embeddings.append(emb)
            
            # Create edges between similar snippets (similarity > 0.5)
            for i in range(len(snippets)-1):
                for j in range(i+1, len(snippets)):
                    sim = torch.cosine_similarity(snippet_embeddings[i], snippet_embeddings[j], dim=0)
                    if sim > 0.5:
                        snippets_graph.add_edge(f"snippet_{i}", f"snippet_{j}", weight=sim.item())
                        snippets_graph.add_edge(f"snippet_{j}", f"snippet_{i}", weight=sim.item())
            
            # 2. Encode query graph nodes
            query_node_embeddings = {}
            for node in query_graph.nodes():
                if 'code' in query_graph.nodes[node]:
                    code = query_graph.nodes[node]['code']
                    emb = llm_tools._encode_text_codet5(code, tokenizer, model)
                    if isinstance(emb, np.ndarray):
                        emb = torch.from_numpy(emb).to(torch.float32)
                    query_node_embeddings[node] = emb

            # 3. Create fusion graph by copying query_graph
            fused_graph = query_graph.copy()
            
            # 4. Compute cross-attention between query nodes and snippet nodes
            device = next(iter(query_node_embeddings.values())).device
            
            # Prepare query features
            query_nodes = list(query_node_embeddings.keys())
            query_features = torch.stack([query_node_embeddings[node] for node in query_nodes])
            
            # Prepare snippet features
            snippets_nodes = list(snippets_graph.nodes())
            snippets_features = torch.stack(snippet_embeddings)
            
            # Calculate cross-attention
            cross_attention = torch.matmul(query_features, snippets_features.transpose(-2, -1)) / np.sqrt(self.dim)
            attention_weights = torch.softmax(cross_attention, dim=-1)
            
            # 5. Add snippet nodes to fused graph based on attention weights
            for i, q_node in enumerate(query_nodes):
                for j, s_node in enumerate(snippets_nodes[:len(snippets_nodes) // 2]):
                    if attention_weights[i, j] > 0.5:  # Only add if similarity > 0.5
                        # Add snippet node to fused graph if not already there
                        s_node_id = f"snippet_{j}"
                        if s_node_id not in fused_graph:
                            fused_graph.add_node(s_node_id, code=snippets_graph.nodes[s_node]['code'],type='retrieved_snippet')
                        
                        # Add edge from query node to snippet node
                        fused_graph.add_edge(q_node, s_node_id,weight=attention_weights[i, j].item(),type='query_to_snippet')
            
            # 6. Add edges between snippets in the fused graph
            for u, v, data in snippets_graph.edges(data=True):
                if u in fused_graph and v in fused_graph:
                    fused_graph.add_edge(u, v, weight=data['weight'], type='snippet_to_snippet')
            
            return fused_graph
            
        elif fusion_type == 'sum':
            # Simple element-wise sum
            return code_emb + graph_emb
            
        else:
            raise ValueError(f'Unknown fusion type: {fusion_type}')
    
    

def load_jsonl(fname):
    with open(fname, 'r', encoding='utf8') as f:
        lines = []
        for line in f:
            lines.append(json.loads(line))
        return lines

def convert(obj):
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(i) for i in obj]
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    else:
        return obj

def main():
    parser = argparse.ArgumentParser(description='Run the Graph-guided RAG pipeline')
    parser.add_argument('--model', type=str, default='qwen2.5-coder-14b-instruct',choices=['gpt3.5','gpt40-mini', 'qwen2.5-coder-3b-instruct','qwen2.5-coder-14b-instruct','qwen-coder-turbo','deepseek-v3'], help='Name of the LLM model to use')
    parser.add_argument('--dataset', type=str, default='crosscodeeval', choices=['crosscodeeval', 'repoeval_updated'], help='Dataset to use for evaluation')
    parser.add_argument('--language', type=str, default='python', choices=['python', 'java'], help='Language to use for evaluation')
    parser.add_argument('--fusion-type', type=str, default='attention', choices=['concat', 'attention', 'sum'], help='Type of fusion to use for code and graph embeddings')
    parser.add_argument('--k', type=int, default=5, help='Number of neighbors to retrieve')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for code vs graph similarity')
    parser.add_argument('--device', type=str, default='cuda:3' if torch.cuda.is_available() else 'cpu', help='Device to use for model')

    
    args = parser.parse_args()

    args.dataset_root = Path(f"dataset/{args.dataset}")
    args.repo_path = args.dataset_root / "repos"
    args.raw_dataset_root = args.dataset_root / "rawdata"/args.language
    # args.processed_dataset_root = args.dataset_root / "processed_data"/args.language
    # args.repo_graph_path = args.processed_dataset_root / "graphs"
    # args.index_path = args.processed_dataset_root / "index"
    # args.output_path = args.processed_dataset_root / "grace_search/line_search_results.jsonl"
    args.repo_graph_path = args.dataset_root / "graphs"
    args.index_path = args.dataset_root / "index"
    # args.index_path = Path("dataset/crosscodeeval/old_processed_data/python/index")
    args.output_path = args.dataset_root / f"grace_search/line_search_results_{args.language}_{args.model}.jsonl"

    print(args)
    
    pipeline = GraphRAGPipeline(
        args = args,
        model_name=args.model,
        index_path=args.index_path
    )
    
    Path(args.index_path).mkdir(parents=True, exist_ok=True)


    if args.dataset == 'crosscodeeval':
        dataset = load_jsonl(f"{args.raw_dataset_root}/line_completion.jsonl")
    elif args.dataset == 'repoeval_updated':
        dataset = load_jsonl(f"{args.raw_dataset_root}/line_level.python.test.jsonl")
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    # dataset=dataset[:200]

    print(f'Loaded dataset with {len(dataset)} samples')

    processed_repos = set()
    
    for repo_dir in Path(args.index_path).iterdir():
        if repo_dir.is_dir():
            repo_name = '_'.join(repo_dir.name.split('_')[1:])
            if (repo_dir / f"{repo_name}_graph_index.bin").exists():
                processed_repos.add(repo_dir.name)
                print(f'Found existing index for repository: {repo_dir.name}')
    print(f'Found {len(processed_repos)} repositories with existing indexes')
    
    total_samples = 0
    correct_samples = 0
    total_metric = 0.0
    results = []

    search_result = []
    for idx, sample in enumerate(dataset):
        if args.dataset == 'crosscodeeval':
            repo_name = sample['metadata']['repository']
            file_path = sample['metadata']['file']
            code_context = sample['prompt']
            next_line = sample['groundtruth']
            right_context = sample['right_context']
        elif args.dataset == "repobench":
            repo_path = args.repo_path / repo_name
            temp_repo_path = args.repo_path / 'temp' / repo_name
            file_path = sample['file_path']
            all_code_context = sample['all_code']
            next_line = sample['next_line']
            code_context = sample['cropped_code']
            all_code = sample['all_code']
        elif args.dataset == "repoeval_updated":
            pass
            raise ValueError(f"unimplemented")
        else:
            raise ValueError(f"unsupported dataset: {args.dataset}")
        

        print(f'\nProcessing sample {idx+1}/{len(dataset)}: {repo_name} - {file_path}')
        if  str(idx)+str("_")+repo_name not in processed_repos:
            processed_repos.add(str(idx)+str("_")+repo_name)
            repo_index_path = args.index_path / Path(str(idx)+str("_")+repo_name)
            print(f'Building index for idx {idx} repository: {repo_name}')
            repo_graph_path = args.repo_graph_path / Path(str(idx)+str("_")+repo_name)
            if repo_graph_path.exists() and (repo_graph_path / "repo_multi_graph.graphml").exists():
                print(f'Found processed graph for {repo_name} at {repo_graph_path}')
                pipeline.build_index(
                    str(repo_graph_path),
                    repo_name,
                    args.repo_path / repo_name,
                    str(repo_index_path),
                    code_snippet=code_context
                )
            else:
                print(f'Warning: No processed graph found for {repo_name} at {repo_graph_path}')

        repo_dir = args.repo_graph_path / Path(str(idx)+str("_")+repo_name)
        if not repo_dir.exists():
            print(f'Repository graph {repo_name} not found at {repo_dir}')
            continue

        query_graph = _parse_code_to_ast_graph(code_context)

        llm_tools = LLMModel()
        tokenizer, model = llm_tools.load_codet5()
        snippets, scores = pipeline.retrieve(
            query_code=code_context,
            query_graph=query_graph,
            repo_name=repo_name,
            k=args.k,
            alpha=args.alpha,
            llm_tools=llm_tools,
            tokenizer=tokenizer,
            model=model,
            sample_idx=str(idx)
        )
        
        fused_graph = pipeline.graph_fusion(
            snippets=snippets,
            query_graph=query_graph,
            fusion_type='attention',
            llm_tools=llm_tools,
            tokenizer=tokenizer,
            model=model
        )

        new_context = graph_text_encoders.encode_code_graph(fused_graph, node_encoder="integer", edge_encoder="incident")
        search_result.append({
            "repo_name": repo_name,
            "file_path": file_path,
            "graph_context": new_context,
            "code_context": code_context,
            "next_line": next_line,
            "retrieved_snippets": snippets,
            "retrieval_scores": scores,
            "right_context": right_context,
            "task_id": sample['metadata']['task_id'],
            "groundtruth": sample['groundtruth'],
        })

    with open(args.output_path, "w", encoding="utf-8") as f:
        for item in search_result:
            f.write(json.dumps(convert(item), ensure_ascii=False) + "\n")
    
    llm = LLMModel()
    
    results_root = Path(f"results/{args.dataset}/{args.language}/grace")
    results_root.mkdir(parents=True, exist_ok=True)
    generate_result_path = results_root / "prediction.jsonl"
    llm.code_complete(generate_result_path, search_result)
    

if __name__ == '__main__':
    main()
