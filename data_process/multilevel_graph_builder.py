import os
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
import ast
from dataclasses import dataclass
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from staticfg import CFGBuilder
import astroid
from mypy import build
from mypy.nodes import MypyFile
import json
import libcst as cst
from multiprocessing import Pool  # Added missing import

import subprocess
import xml.etree.ElementTree as ET
import re
import javalang
import argparse

@dataclass
class MultiLevelGraph:
    # Repo Level
    folder_structure: nx.DiGraph
    cross_file_deps: nx.DiGraph
    
    # Module Level
    call_graph: nx.DiGraph
    type_deps: nx.DiGraph
    class_inheritance: nx.DiGraph
    
    # Function Level
    ast: nx.DiGraph
    cfg: nx.DiGraph
    dfg: nx.DiGraph

    combined_graph: Optional[nx.DiGraph] = None

class MultiLevelGraphBuilder:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.graphs = MultiLevelGraph(
            folder_structure=nx.DiGraph(),
            cross_file_deps=nx.DiGraph(),
            call_graph=nx.DiGraph(),
            type_deps=nx.DiGraph(),
            class_inheritance=nx.DiGraph(),
            ast=nx.DiGraph(),
            cfg=nx.DiGraph(),
            dfg=nx.DiGraph()
        )
        
    def build_repo_level(self):
        self._build_folder_structure()
        self._build_cross_file_deps()
        
    def build_module_level(self):

        self._build_call_graph()
        self._build_type_deps()
        self._build_class_inheritance()
        
    def build_function_level(self):
        self._build_ast_graph()
        self._build_cfg()
        self._build_dfg()
        
    def _add_node_with_code(self, graph, node_id, code_content, file_path=None, extra_attrs=None):
        attrs = {'code': code_content}
        if file_path:
            attrs['file_path'] = file_path
        if extra_attrs:
            attrs.update(extra_attrs)
            
        if not graph.has_node(node_id):
            graph.add_node(node_id, **attrs)
        else:
            graph.nodes[node_id].update(attrs)

    def _build_folder_structure(self):
        for root, dirs, files in os.walk(self.repo_path):
            root_path_str = str(Path(root).relative_to(self.repo_path))
            
            if not self.graphs.folder_structure.has_node(root_path_str):
                self.graphs.folder_structure.add_node(root_path_str, code=root_path_str)
            else:
                if 'code' not in self.graphs.folder_structure.nodes[root_path_str]:
                    self.graphs.folder_structure.nodes[root_path_str]['code'] = root_path_str

            for d in dirs:
                child_path_str = str(Path(root, d).relative_to(self.repo_path))
                if not self.graphs.folder_structure.has_node(child_path_str):
                    self.graphs.folder_structure.add_node(child_path_str, code=child_path_str)
                self.graphs.folder_structure.add_edge(root_path_str, child_path_str)
                self.graphs.folder_structure.nodes[root_path_str]['code'] = root_path_str
                self.graphs.folder_structure.nodes[child_path_str]['code'] = child_path_str

            for f in files:
                # if f.endswith('.py'): # Keep or remove based on whether all files or only .py are needed
                child_path_str = str(Path(root, f).relative_to(self.repo_path))
                if not self.graphs.folder_structure.has_node(child_path_str):
                    self.graphs.folder_structure.add_node(child_path_str, code=child_path_str)
                self.graphs.folder_structure.add_edge(root_path_str, child_path_str)
                self.graphs.folder_structure.nodes[root_path_str]['code'] = root_path_str
                self.graphs.folder_structure.nodes[child_path_str]['code'] = child_path_str

    def _build_cross_file_deps(self):
        python_files = {}
        
        for py_file in self.repo_path.rglob('*.py'):
            rel_path = str(py_file.relative_to(self.repo_path))
            python_files[rel_path] = py_file
            if not self.graphs.cross_file_deps.has_node(rel_path):
                self.graphs.cross_file_deps.add_node(rel_path, code=rel_path)
        
        for rel_path, py_file in python_files.items():
            try:
                with open(py_file) as f:
                    try:
                        tree = ast.parse(f.read())
                    except SyntaxError as se:
                        logging.warning(f"Syntax error in {py_file}: {se}. Skipping this file for cross-file dependencies.")
                        continue
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            imported_module = name.name
                            self._add_dependency(rel_path, imported_module, python_files)
                    
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        self._add_dependency(rel_path, node.module, python_files)
            
            except Exception as e:
                logging.warning(f"Failed to analyze imports in {py_file}: {str(e)}")
    
    def _add_dependency(self, source_file, imported_module, python_files):
        parts = imported_module.split('.')
        potential_paths = []
        potential_paths.append(f"{'/'.join(parts)}.py")
        potential_paths.append(f"{'/'.join(parts)}/__init__.py")
        if len(parts) > 1:
            potential_paths.append(f"{'/'.join(parts[:-1])}/{parts[-1]}.py")
        
        for target_file_candidate in potential_paths:
            if target_file_candidate in python_files:
                if not self.graphs.cross_file_deps.has_node(source_file):
                    self.graphs.cross_file_deps.add_node(source_file, code=source_file)
                else:
                     self.graphs.cross_file_deps.nodes[source_file]['code'] = source_file
                
                if not self.graphs.cross_file_deps.has_node(target_file_candidate):
                    self.graphs.cross_file_deps.add_node(target_file_candidate, code=target_file_candidate)
                else:
                    self.graphs.cross_file_deps.nodes[target_file_candidate]['code'] = target_file_candidate
                
                self.graphs.cross_file_deps.add_edge(source_file, target_file_candidate)
                return
        
        prefix = '/'.join(parts)
        for file_path_candidate in python_files.keys():
            if file_path_candidate.startswith(prefix + '/') or file_path_candidate == prefix + '.py':
                if not self.graphs.cross_file_deps.has_node(source_file):
                    self.graphs.cross_file_deps.add_node(source_file, code=source_file)
                else:
                    self.graphs.cross_file_deps.nodes[source_file]['code'] = source_file

                if not self.graphs.cross_file_deps.has_node(file_path_candidate):
                    self.graphs.cross_file_deps.add_node(file_path_candidate, code=file_path_candidate)
                else:
                    self.graphs.cross_file_deps.nodes[file_path_candidate]['code'] = file_path_candidate

                self.graphs.cross_file_deps.add_edge(source_file, file_path_candidate)
                return

    def _build_call_graph(self):
        for py_file in self.repo_path.rglob('*.py'):
            try:
                with open(py_file, encoding='utf-8') as f:
                    try:
                        tree = ast.parse(f.read())
                    except SyntaxError as se:
                        logging.warning(f"Syntax error in {py_file}: {se}. Skipping this file for call graph.")
                        continue
                    
                class CallVisitor(ast.NodeVisitor):
                    def __init__(self, graph, file_path_str):
                        self.calls = []
                        self.current_func = None
                        self.graph = graph
                        self.file_path_str = file_path_str 

                    def visit_FunctionDef(self, node):
                        old_func = self.current_func
                        self.current_func = node.name 
                        
                        if not self.graph.has_node(self.current_func):
                            self.graph.add_node(self.current_func, code=self.current_func)
                        else: 
                            if 'code' not in self.graph.nodes[self.current_func]:
                                 self.graph.nodes[self.current_func]['code'] = self.current_func

                        self.generic_visit(node)
                        self.current_func = old_func
                        
                    def visit_Call(self, node):
                        callee_name = None
                        if isinstance(node.func, ast.Name):
                            callee_name = node.func.id
                        elif isinstance(node.func, ast.Attribute):
                            callee_name = node.func.attr 
                        
                        if callee_name and self.current_func:
                            self.calls.append((self.current_func, callee_name))
                        self.generic_visit(node)
                
                visitor = CallVisitor(self.graphs.call_graph, str(py_file.name))
                visitor.visit(tree)
                
                for caller, callee in visitor.calls:
                    if not self.graphs.call_graph.has_node(callee):
                        self.graphs.call_graph.add_node(callee, code=callee)
                    else:
                        if 'code' not in self.graphs.call_graph.nodes[callee]:
                            self.graphs.call_graph.nodes[callee]['code'] = callee
                    
                    self.graphs.call_graph.add_edge(caller, callee)
            except Exception as e: 
                logging.warning(f"Failed to process call graph for {py_file}: {e}", exc_info=True)

    def _build_type_deps(self):
        # Ensure mypy is installed and relevant types are imported
        # from mypy.nodes import MypyFile, ClassDef, FuncDef (adjust as needed)
        # from mypy.types import Type, Instance, UnionType, TupleType, CallableType, AnyType, NoneType
        # from mypy.build import build, BuildSource (or appropriate API for your mypy version)

        for py_file in self.repo_path.rglob('*.py'):
            try:
                # This is a placeholder for actual mypy invocation
                # You'll need to replace this with your mypy API usage if it's different
                # For example, using mypy.api.run or a similar newer API if available
                # The `build.build` API might be older or specific to certain mypy versions.
                # result = build.build(sources=[BuildSource(str(py_file), None, None)], options=self.mypy_options) # Pass options if needed
                # tree = result.files.get(str(py_file.resolve())) # Use resolved absolute path for key
                
                # Placeholder: to avoid breaking if mypy isn't set up, we'll log and skip.
                # Replace with your actual mypy integration.
                logging.info(f"Mypy type analysis for {py_file} - Placeholder, actual mypy call needed.")
                # if tree:
                #    self._process_mypy_tree(tree)
                # else:
                #    logging.warning(f"Mypy did not produce a tree for {py_file}")
                pass # Remove this pass when mypy integration is active

            except Exception as e:
                logging.warning(f"Failed to process types in {py_file} (mypy integration): {e}", exc_info=True)

    def _build_class_inheritance(self):
        for py_file in self.repo_path.rglob('*.py'):
            try:
                with open(py_file, encoding='utf-8') as f:
                    try:
                        tree = ast.parse(f.read())
                    except SyntaxError as se:
                        logging.warning(f"Syntax error in {py_file}: {se}. Skipping this file for class inheritance.")
                        continue
                    
                class InheritanceVisitor(ast.NodeVisitor):
                    def __init__(self, graph):
                        self.inheritance = []
                        self.graph = graph
                        
                    def visit_ClassDef(self, node):
                        child_class_name = node.name
                        if not self.graph.has_node(child_class_name):
                            self.graph.add_node(child_class_name, code=child_class_name)
                        else:
                            if 'code' not in self.graph.nodes[child_class_name]:
                                self.graph.nodes[child_class_name]['code'] = child_class_name

                        for base in node.bases:
                            parent_class_name = None
                            if isinstance(base, ast.Name):
                                parent_class_name = base.id
                            elif isinstance(base, ast.Attribute):
                                parent_class_name = base.attr 
                            
                            if parent_class_name:
                                self.inheritance.append((child_class_name, parent_class_name))
                        self.generic_visit(node)
                
                visitor = InheritanceVisitor(self.graphs.class_inheritance)
                visitor.visit(tree)
                
                for child, parent in visitor.inheritance:
                    if not self.graphs.class_inheritance.has_node(parent):
                        self.graphs.class_inheritance.add_node(parent, code=parent)
                    else:
                        if 'code' not in self.graphs.class_inheritance.nodes[parent]:
                             self.graphs.class_inheritance.nodes[parent]['code'] = parent
                    
                    self.graphs.class_inheritance.add_edge(child, parent)
            except Exception as e:
                logging.warning(f"Failed to process inheritance in {py_file}: {e}", exc_info=True)

    def old_build_ast_graph(self):
        def get_node_uid(node, filename_str):
                if isinstance(node, ast.Module):
                    return f"ast:{filename_str}:module"
                lineno = getattr(node, 'lineno', 0)
                col_offset = getattr(node, 'col_offset', 0)
                return f"ast:{filename_str}:{lineno}:{col_offset}:{type(node).__name__}"

        def get_code_snippet_for_node(node, full_source_code):
            if hasattr(ast, 'get_source_segment'):
                segment = ast.get_source_segment(full_source_code, node)
                return segment if segment is not None else ""
            return ""
        
        def add_edges_recursively(parent_ast_node):
            parent_uid = get_node_uid(parent_ast_node, py_file.name)
            for child_ast_node in ast.iter_child_nodes(parent_ast_node):
                child_uid = get_node_uid(child_ast_node, py_file.name)
                self.graphs.ast.add_edge(parent_uid, child_uid)
                add_edges_recursively(child_ast_node)


        for py_file in self.repo_path.rglob('*.py'):
            with open(py_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
                try:
                    tree = ast.parse(source_code)
                except SyntaxError as se:
                    logging.warning(f"Syntax error in {py_file}: {se}. Skipping this file for AST graph.")
                    continue
            
            for node_obj in ast.walk(tree):
                node_identifier = get_node_uid(node_obj, py_file.name)
                
                attributes = {
                    'node_type': type(node_obj).__name__,
                    'file_path': str(py_file.relative_to(self.repo_path.parent)),
                    'code': get_code_snippet_for_node(node_obj, source_code)
                }
                if hasattr(node_obj, 'name'):
                    attributes['name'] = node_obj.name
                elif hasattr(node_obj, 'id'):
                    attributes['name'] = node_obj.id
                elif hasattr(node_obj, 'attr'):
                    attributes['name'] = node_obj.attr
                
                if isinstance(node_obj, ast.Constant) and not isinstance(node_obj.value, ast.AST):
                    attributes['value'] = str(node_obj.value)
                elif isinstance(node_obj, ast.Str):
                    attributes['value'] = node_obj.s
                elif isinstance(node_obj, ast.Num):
                    attributes['value'] = str(node_obj.n)

                self._add_node_with_code(self.graphs.ast, node_identifier, attributes['code'], str(py_file.relative_to(self.repo_path.parent)), extra_attrs=attributes)

            add_edges_recursively(tree)

    def _build_ast_graph(self):
        all_nodes = []
        all_edges = []
        
        for py_file in self.repo_path.rglob('*.py'):
            nodes, edges = self._process_file_for_ast(py_file)
            all_nodes.extend(nodes)
            all_edges.extend(edges)
        
        if all_nodes:
            self.graphs.ast.add_nodes_from(all_nodes)
        
        if all_edges:
            self.graphs.ast.add_edges_from(all_edges)

    def _process_file_for_ast(self, py_file: Path):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = cst.parse_module(source_code)
            
            nodes, edges = self._extract_ast_nodes_and_edges(tree, py_file, source_code)
            
            return nodes, edges
            
        except Exception as e:
            logging.warning(f"Error processing {py_file} for AST: {e}")
            return [], []

    def _extract_ast_nodes_and_edges(self, tree, py_file: Path, source_code: str):
        nodes = []
        edges = []
        relative_path = str(py_file.relative_to(self.repo_path.parent))
        
        def add_nodes_recursive(node, parent_id=None):
            node_id = self._get_ast_node_uid(node, py_file)
            
            attrs = {
                'node_type': type(node).__name__,
                'file_path': relative_path,
                'code': self._get_code_snippet(node, source_code)
            }
            
            name = self._extract_node_name(node)
            if name:
                attrs['name'] = name
            
            value = self._extract_node_value(node)
            if value:
                attrs['value'] = value
            
            nodes.append((node_id, attrs))
            
            if parent_id:
                edges.append((parent_id, node_id))
            
            for child in node.children:
                if isinstance(child, cst.CSTNode):
                    add_nodes_recursive(child, node_id)
        
        add_nodes_recursive(tree)
        return nodes, edges

    def _get_ast_node_uid(self, node, py_file: Path):
        if isinstance(node, cst.Module):
            return f"ast:{py_file.name}:module"
        
        node_type = type(node).__name__
        unique_id = id(node)
        return f"ast:{py_file.name}:{unique_id}:{node_type}"

    def _get_code_snippet(self, node, source_code: str):
        try:
            return cst.Module([]).code_for_node(node).strip()
        except Exception:
            return f"<{type(node).__name__}>"

    def _extract_node_name(self, node):
        try:
            if isinstance(node, (cst.FunctionDef, cst.ClassDef)):
                return node.name.value
            elif isinstance(node, cst.Name):
                return node.value
            elif isinstance(node, cst.Attribute):
                return node.attr.value
            elif isinstance(node, cst.Arg):
                if hasattr(node, 'name') and hasattr(node.name, 'value'):
                    return node.name.value
                elif hasattr(node, 'name'):
                    return str(node.name)
            return None
        except AttributeError:
            return None

    def _extract_node_value(self, node):
        try:
            if isinstance(node, cst.SimpleString):
                return node.value
            elif isinstance(node, cst.Integer):
                return node.value
            elif isinstance(node, cst.Float):
                return node.value
            elif isinstance(node, cst.ConcatenatedString):
                return str(node)
            return None
        except AttributeError:
            return None

    def _build_cfg(self):
        try:
            from staticfg import CFGBuilder
        except ImportError:
            logging.error("staticfg library not found. CFG will not be built. pip install staticfg")
            return

        for py_file in self.repo_path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                source_lines = source_code.splitlines()

                try:
                    # Using str(py_file.resolve()) for path consistency
                    cfg = CFGBuilder().build_from_file(py_file.name, str(py_file.resolve())) 
                except SyntaxError as se:
                    logging.warning(f"Syntax error when building CFG for {py_file}: {se}")
                    continue 
                except Exception as e_cfg:
                    logging.warning(f"Failed to build CFG (initialization) for {py_file}: {e_cfg}", exc_info=True)
                    continue

                file_rel_path_str = str(py_file.relative_to(self.repo_path.parent))
                
                # In staticfg, the CFG object has blocks as a dictionary, not a list
                # The keys are the block IDs and the values are the Block objects
                for block_id, block in cfg.blocks.items():
                    # Extract code content for this block
                    block_code_content = ""
                    if hasattr(block, 'source') and block.source:
                        # In staticfg, blocks have a 'source' attribute with the code
                        block_code_content = block.source
                    else:
                        # Fallback: try to extract code from line numbers if available
                        if hasattr(block, 'at') and block.at:
                            try:
                                # 'at' attribute contains line numbers (1-based)
                                block_code_lines = [source_lines[line_num - 1] for line_num in sorted(block.at) 
                                                   if 0 < line_num <= len(source_lines)]
                                block_code_content = "\n".join(block_code_lines)
                            except (IndexError, AttributeError) as e:
                                logging.warning(f"Error extracting code for block {block_id} in {py_file}: {e}")
                    
                    # Create a unique ID for this block
                    block_uid = f"cfg:{file_rel_path_str}:{block_id}"

                    # Add the block as a node in our graph
                    node_attrs = {
                        'code': block_code_content, 
                        'file_path': file_rel_path_str, 
                        'block_id_orig': block_id
                    }
                    
                    if not self.graphs.cfg.has_node(block_uid):
                        self.graphs.cfg.add_node(block_uid, **node_attrs)
                    else: 
                        self.graphs.cfg.nodes[block_uid].update(node_attrs)

                    # Add edges to successor blocks
                    # In staticfg, blocks have 'exits' which are the edges to successor blocks
                    if hasattr(block, 'exits'):
                        for exit_edge in block.exits:
                            if hasattr(exit_edge, 'target') and exit_edge.target:
                                # Get the target block ID
                                succ_block_id = exit_edge.target.id
                                succ_block = exit_edge.target
                                
                                # Create a unique ID for the successor block
                                succ_block_uid = f"cfg:{file_rel_path_str}:{succ_block_id}"
                                
                                # Extract code content for the successor block
                                succ_block_code_content = ""
                                if hasattr(succ_block, 'source') and succ_block.source:
                                    succ_block_code_content = succ_block.source
                                else:
                                    # Fallback: try to extract code from line numbers
                                    if hasattr(succ_block, 'at') and succ_block.at:
                                        try:
                                            succ_block_code_lines = [source_lines[line_num - 1] 
                                                                    for line_num in sorted(succ_block.at) 
                                                                    if 0 < line_num <= len(source_lines)]
                                            succ_block_code_content = "\n".join(succ_block_code_lines)
                                        except (IndexError, AttributeError) as e:
                                            logging.warning(f"Error extracting code for successor block {succ_block_id} in {py_file}: {e}")
                                
                                # Add the successor block as a node in our graph
                                succ_node_attrs = {
                                    'code': succ_block_code_content, 
                                    'file_path': file_rel_path_str, 
                                    'block_id_orig': succ_block_id
                                }
                                
                                if not self.graphs.cfg.has_node(succ_block_uid):
                                    self.graphs.cfg.add_node(succ_block_uid, **succ_node_attrs)
                                else:
                                    self.graphs.cfg.nodes[succ_block_uid].update(succ_node_attrs)
                                
                                # Add an edge from the current block to the successor block
                                self.graphs.cfg.add_edge(block_uid, succ_block_uid)
            
            except Exception as e:
                logging.warning(f"Generic error in _build_cfg for {py_file}: {e}", exc_info=True)

    def old_build_dfg(self):

        def get_node_uid(node, filename_str):
            if isinstance(node, ast.Module): return f"dfg:{filename_str}:module"
            lineno = getattr(node, 'lineno', 0)
            col_offset = getattr(node, 'col_offset', 0)
            return f"dfg:{filename_str}:{lineno}:{col_offset}:{type(node).__name__}"

        def get_code_snippet_for_node(node, full_source_code):
            if hasattr(ast, 'get_source_segment'):
                segment = ast.get_source_segment(full_source_code, node)
                return segment if segment is not None else ""
            return ""

        class DataFlowVisitor(ast.NodeVisitor):
            def __init__(self, graph_obj, filename_str, full_source_code, file_path_for_nodes, builder):
                self.graph = graph_obj
                self.filename = filename_str
                self.source_code = full_source_code
                self.file_path = file_path_for_nodes
                self.builder = builder  # Reference to the MultiLevelGraphBuilder instance
                self.definitions = {}

            def _ensure_node_in_graph(self, ast_node_obj):
                node_uid = get_node_uid(ast_node_obj, self.filename)
                if not self.graph.has_node(node_uid):
                    attributes = {
                        'node_type': type(ast_node_obj).__name__,
                        'file_path': self.file_path,
                        'code': get_code_snippet_for_node(ast_node_obj, self.source_code)
                    }
                    if hasattr(ast_node_obj, 'name'): attributes['name'] = ast_node_obj.name
                    elif hasattr(ast_node_obj, 'id'): attributes['name'] = ast_node_obj.id
                    elif hasattr(ast_node_obj, 'attr'): attributes['name'] = ast_node_obj.attr
                    if isinstance(ast_node_obj, ast.Constant) and not isinstance(ast_node_obj.value, ast.AST):
                        attributes['value'] = str(ast_node_obj.value)
                    self.builder._add_node_with_code(self.graph, node_uid, attributes['code'], self.file_path, extra_attrs=attributes)
                return node_uid

            def visit_Assign(self, node):
                rhs_node_obj = node.value 
                rhs_uid = self._ensure_node_in_graph(rhs_node_obj)

                for target in node.targets:
                    if isinstance(target, ast.Name):
                        target_uid = self._ensure_node_in_graph(target)
                        self.definitions[target.id] = target_uid
                        self.graph.add_edge(rhs_uid, target_uid, type='assignment_flow')
                self.generic_visit(node)

            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load) and node.id in self.definitions:
                    use_uid = self._ensure_node_in_graph(node)
                    def_uid = self.definitions[node.id]
                    self.graph.add_edge(def_uid, use_uid, type='data_flow')
                self.generic_visit(node)
        

        for py_file in self.repo_path.rglob('*.py'):
            with open(py_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
            try:
                tree = ast.parse(source_code)
            except SyntaxError as se:
                logging.warning(f"Syntax error in {py_file}: {se}. Skipping this file for DFG.")
                continue
            file_path_str = str(py_file.relative_to(self.repo_path.parent))
            dfg_visitor = DataFlowVisitor(self.graphs.dfg, py_file.name, source_code, file_path_str, self)
            dfg_visitor.visit(tree)


    def _build_dfg(self):

        class DFGBuildingVisitor(cst.CSTVisitor):
            # METADATA_DEPENDENCIES = (cst.metadata.ScopeProvider, cst.metadata.CodeRangeProvider)

            def __init__(self, graph: nx.DiGraph, file_path: str, wrapper: cst.MetadataWrapper):
                self.graph = graph
                self.file_path = file_path
                self.wrapper = wrapper

            def _get_node_uid(self, node: cst.CSTNode) -> str:
                prefix = f"dfg:{self.file_path}"
                if isinstance(node, cst.Module):
                    return f"{prefix}:module"
                pos = self.wrapper.get_metadata(cst.metadata.CodeRangeProvider, node)
                return f"{prefix}:{pos.start.line}:{pos.start.column}:{type(node).__name__}:{id(node)}"

            def _ensure_node_in_graph(self, node: cst.CSTNode) -> str:
                node_uid = self._get_node_uid(node)
                if not self.graph.has_node(node_uid):
                    code_range = self.wrapper.get_metadata(cst.metadata.CodeRangeProvider, node)
                    attributes = {
                        'node_type': type(node).__name__,
                        'file_path': self.file_path,
                        'code': self.wrapper.module.code_for_node(node),
                        'start_line': code_range.start.line,
                    }
                    if hasattr(node, 'value'): attributes['name'] = node.value
                    if hasattr(node, 'name') and hasattr(node.name, 'value'): attributes['name'] = node.name.value
                    self.graph.add_node(node_uid, **attributes)
                return node_uid

            def visit_Name(self, node: cst.Name) -> None:
                try:
                    scope = self.get_metadata(cst.metadata.ScopeProvider, node)
                    assignments = scope.get_assignments(node)
                    if not assignments: return

                    use_uid = self._ensure_node_in_graph(node)
                    
                    for assignment in assignments:
                        def_uid = self._ensure_node_in_graph(assignment)
                        self.graph.add_edge(def_uid, use_uid, type='data_flow')
                except Exception as e:
                    logging.debug(f"Could not resolve DFG scope for '{node.value}' in {self.file_path}: {e}")

        for py_file in self.repo_path.rglob('*.py'):
            relative_path = str(py_file.relative_to(self.repo_path))
            logging.debug(f"Building DFG for: {relative_path}")
            try:
                source_code = py_file.read_text(encoding='utf-8')
                module_wrapper = cst.MetadataWrapper(cst.parse_module(source_code))
                visitor = DFGBuildingVisitor(self.graphs.dfg, relative_path, module_wrapper)
                module_wrapper.visit(visitor)
            except Exception as e:
                logging.warning(f"Could not build DFG for {relative_path}: {e}")



    def _get_referenced_types(self, type_obj):
        referenced_types = [] 
        if type_obj is None: return []

        # Simple name attribute check (might be too generic, but a fallback)
        if hasattr(type_obj, 'name') and isinstance(type_obj.name, str):
            referenced_types.append(type_obj.name)
        
        if hasattr(type_obj, 'fullname') and isinstance(type_obj.fullname, str):
            referenced_types.append(type_obj.fullname)

        # Specific Mypy type handling (requires mypy.types to be importable)
        try:
            from mypy.types import Instance, UnionType, TupleType, CallableType, AnyType, NoneType as MypyNoneType, TypeVarType
            if isinstance(type_obj, Instance):
                if type_obj.type and hasattr(type_obj.type, 'fullname'): # type_obj.type is TypeInfo
                    referenced_types.append(type_obj.type.fullname)
                for arg_type in type_obj.args:
                    referenced_types.extend(self._get_referenced_types(arg_type))
            elif isinstance(type_obj, UnionType):
                for item_type in type_obj.items:
                    referenced_types.extend(self._get_referenced_types(item_type))
            elif isinstance(type_obj, TupleType):
                for item_type in type_obj.items:
                    referenced_types.extend(self._get_referenced_types(item_type))
            elif isinstance(type_obj, CallableType):
                for arg_t in type_obj.arg_types:
                    referenced_types.extend(self._get_referenced_types(arg_t))
                referenced_types.extend(self._get_referenced_types(type_obj.ret_type))
            elif isinstance(type_obj, TypeVarType): # e.g. _T = TypeVar('_T')
                if type_obj.fullname:
                    referenced_types.append(type_obj.fullname)
            elif isinstance(type_obj, (AnyType, MypyNoneType)):
                pass
        except ImportError:
            logging.debug("Mypy types not available for detailed type parsing in _get_referenced_types.")
            # Fallback to basic name if mypy types can't be imported (e.g. mypy not fully installed/available)
            if hasattr(type_obj, 'name') and isinstance(type_obj.name, str) and type_obj.name not in referenced_types:
                 referenced_types.append(type_obj.name)
            if hasattr(type_obj, 'fullname') and isinstance(type_obj.fullname, str) and type_obj.fullname not in referenced_types:
                 referenced_types.append(type_obj.fullname)

        # Filter out None, empty strings and non-string types, then unique
        return list(set(name for name in referenced_types if name and isinstance(name, str)))
    
    def build_combined_graph(self):
        self.combined_graph = nx.DiGraph()
        self.graphs.combined_graph = self.combined_graph
        
        for graph_name, graph in [
            ('folder_structure', self.graphs.folder_structure),
            ('cross_file_deps', self.graphs.cross_file_deps),
            ('call_graph', self.graphs.call_graph),
            ('type_deps', self.graphs.type_deps),
            ('class_inheritance', self.graphs.class_inheritance),
            ('ast', self.graphs.ast),
            ('cfg', self.graphs.cfg),
            ('dfg', self.graphs.dfg)
        ]:
            for node in graph.nodes():
                self.combined_graph.add_node(f"{graph_name}:{node}", 
                                            code=graph.nodes[node]['code'],
                                            graph_type=graph_name,
                                            original_id=node)
            
            for src, dst in graph.edges():
                self.combined_graph.add_edge(f"{graph_name}:{src}", 
                                           f"{graph_name}:{dst}",
                                           edge_type=f"internal_{graph_name}")
        
        # 建立跨层次的连接
        self._connect_repo_to_module_level()
        self._connect_module_to_function_level()
        self._connect_function_level_internally()
        
        return self.combined_graph
    
    def _connect_repo_to_module_level(self):
        for file_node in self.graphs.folder_structure.nodes():
            if file_node.endswith('.py'):
                file_path = self.repo_path / file_node
                try:
                    with open(file_path) as f:
                        tree = ast.parse(f.read())
                        
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            self.combined_graph.add_edge(
                                f"folder_structure:{file_node}",
                                f"call_graph:{node.name}",
                                edge_type="file_location_constraint"
                            )
                except Exception as e:
                    logging.warning(f"Failed to connect file to methods for {file_path}: {str(e)}")
        
        for src, dst in self.graphs.cross_file_deps.edges():
            for caller in self.graphs.call_graph.nodes():
                for callee in self.graphs.call_graph.successors(caller):
                    self.combined_graph.add_edge(
                        f"cross_file_deps:{src}",
                        f"call_graph:{caller}",
                        edge_type="cross_module_call"
                    )
        
        for src, dst in self.graphs.cross_file_deps.edges():
            for type_src, type_dst in self.graphs.type_deps.edges():
                self.combined_graph.add_edge(
                    f"cross_file_deps:{src}",
                    f"type_deps:{type_src}",
                    edge_type="type_cross_file_reference"
                )
        
        for src, dst in self.graphs.cross_file_deps.edges():
            for child, parent in self.graphs.class_inheritance.edges():
                self.combined_graph.add_edge(
                    f"cross_file_deps:{src}",
                    f"class_inheritance:{child}",
                    edge_type="interface_inheritance"
                )
    
    def _connect_module_to_function_level(self):
        for func_node in self.graphs.call_graph.nodes():
            for ast_node in self.graphs.ast.nodes():
                self.combined_graph.add_edge(
                    f"call_graph:{func_node}",
                    f"ast:{ast_node}",
                    edge_type="method_body_structure"
                )
                break
        
        for type_src, type_dst in self.graphs.type_deps.edges():
            for dfg_src, dfg_dst in self.graphs.dfg.edges():
                self.combined_graph.add_edge(
                    f"type_deps:{type_src}",
                    f"dfg:{dfg_src}",
                    edge_type="variable_type_constraint"
                )
                break
        
        for child, parent in self.graphs.class_inheritance.edges():
            for caller, callee in self.graphs.call_graph.edges():
                self.combined_graph.add_edge(
                    f"class_inheritance:{child}",
                    f"call_graph:{caller}",
                    edge_type="method_inheritance"
                )
                break
    
    def _connect_function_level_internally(self):
        for ast_node in self.graphs.ast.nodes():
            for cfg_node in self.graphs.cfg.nodes():
                self.combined_graph.add_edge(
                    f"ast:{ast_node}",
                    f"cfg:{cfg_node}",
                    edge_type="syntax_structure"
                )
                break
        
        for cfg_node in self.graphs.cfg.nodes():
            for dfg_node in self.graphs.dfg.nodes():
                self.combined_graph.add_edge(
                    f"cfg:{cfg_node}",
                    f"dfg:{dfg_node}",
                    edge_type="control_dependency"
                )
                break


class MultiLevelJavaGraphBuilder:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.graphs = MultiLevelGraph(
            folder_structure=nx.DiGraph(),
            cross_file_deps=nx.DiGraph(),
            call_graph=nx.DiGraph(),
            type_deps=nx.DiGraph(),
            class_inheritance=nx.DiGraph(),
            ast=nx.DiGraph(),
            cfg=nx.DiGraph(),
            dfg=nx.DiGraph()
        )
        
        # Java项目特有的配置
        self.source_dirs = self._find_source_directories()
        self.classpath = self._build_classpath()
        
    def build_repo_level(self):
        self._build_folder_structure()
        self._build_cross_file_deps()
        print("build_repo_level success")
        
    def build_module_level(self):
        self._build_call_graph()
        self._build_type_deps()
        print("build_module_level success")
        
    def build_function_level(self):
        self._build_ast_graph()
        self._build_cfg()
        self._build_dfg()
        print("build_function_level success")
        
    def _find_source_directories(self) -> List[Path]:
        source_dirs = []
        
        maven_src = self.repo_path / "src" / "main" / "java"
        if maven_src.exists():
            source_dirs.append(maven_src)
            
        gradle_src = self.repo_path / "src" / "main" / "java"
        if gradle_src.exists() and gradle_src not in source_dirs:
            source_dirs.append(gradle_src)
            
        for root, dirs, files in os.walk(self.repo_path):
            if any(f.endswith('.java') for f in files):
                path = Path(root)
                if path not in source_dirs:
                    source_dirs.append(path)
                    
        return source_dirs
    
    def _build_classpath(self) -> str:
        classpath_parts = []
        
        target_classes = self.repo_path / "target" / "classes"
        if target_classes.exists():
            classpath_parts.append(str(target_classes))
            
        lib_dir = self.repo_path / "lib"
        if lib_dir.exists():
            for jar_file in lib_dir.glob("*.jar"):
                classpath_parts.append(str(jar_file))
                
        return os.pathsep.join(classpath_parts)
    
    def _add_node_with_code(self, graph, node_id, code_content, file_path=None, extra_attrs=None):
        attrs = {'code': code_content}
        if file_path:
            attrs['file_path'] = file_path
        if extra_attrs:
            attrs.update(extra_attrs)
            
        if not graph.has_node(node_id):
            graph.add_node(node_id, **attrs)
        else:
            graph.nodes[node_id].update(attrs)

    def _build_folder_structure(self):
        for root, dirs, files in os.walk(self.repo_path):
            root_path_str = str(Path(root).relative_to(self.repo_path))
            
            if not self.graphs.folder_structure.has_node(root_path_str):
                self.graphs.folder_structure.add_node(root_path_str, code=root_path_str)

            for d in dirs:
                child_path_str = str(Path(root, d).relative_to(self.repo_path))
                if not self.graphs.folder_structure.has_node(child_path_str):
                    self.graphs.folder_structure.add_node(child_path_str, code=child_path_str)
                self.graphs.folder_structure.add_edge(root_path_str, child_path_str)

            for f in files:
                if f.endswith('.java'):
                    child_path_str = str(Path(root, f).relative_to(self.repo_path))
                    if not self.graphs.folder_structure.has_node(child_path_str):
                        self.graphs.folder_structure.add_node(child_path_str, code=child_path_str)
                    self.graphs.folder_structure.add_edge(root_path_str, child_path_str)

    def _build_package_deps(self):
        package_files = {}
        
        for java_file in self.repo_path.rglob('*.java'):
            try:
                with open(java_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                package_match = re.search(r'package\s+([\w.]+)\s*;', content)
                if package_match:
                    package_name = package_match.group(1)
                    rel_path = str(java_file.relative_to(self.repo_path))
                    
                    if package_name not in package_files:
                        package_files[package_name] = []
                    package_files[package_name].append(rel_path)
                    
                    if not self.graphs.package_deps.has_node(package_name):
                        self.graphs.package_deps.add_node(package_name, code=package_name)
                
                import_matches = re.findall(r'import\s+([\w.*]+)\s*;', content)
                current_package = package_match.group(1) if package_match else "default"
                
                for import_stmt in import_matches:
                    if '.' in import_stmt:
                        imported_package = '.'.join(import_stmt.split('.')[:-1])
                        if imported_package != current_package:
                            if not self.graphs.package_deps.has_node(imported_package):
                                self.graphs.package_deps.add_node(imported_package, code=imported_package)
                            self.graphs.package_deps.add_edge(current_package, imported_package)
                            
            except Exception as e:
                logging.warning(f"Failed to analyze package deps in {java_file}: {e}")

    def _build_cross_file_deps(self):
        java_files = {}
        
        for java_file in self.repo_path.rglob('*.java'):
            rel_path = str(java_file.relative_to(self.repo_path))
            java_files[rel_path] = java_file
            
            if not self.graphs.cross_file_deps.has_node(rel_path):
                self.graphs.cross_file_deps.add_node(rel_path, code=rel_path)
        
        for rel_path, java_file in java_files.items():
            try:
                with open(java_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                import_matches = re.findall(r'import\s+([\w.*]+)\s*;', content)
                
                for import_stmt in import_matches:
                    self._add_file_dependency(rel_path, import_stmt, java_files)
                    
            except Exception as e:
                logging.warning(f"Failed to analyze imports in {java_file}: {e}")
    
    def _add_file_dependency(self, source_file, import_stmt, java_files):
        if import_stmt.endswith('.*'):
            return
            
        class_path = import_stmt.replace('.', '/') + '.java'
        
        for src_dir in self.source_dirs:
            potential_file = src_dir / class_path
            if potential_file.exists():
                target_rel_path = str(potential_file.relative_to(self.repo_path))
                if target_rel_path in java_files:
                    self.graphs.cross_file_deps.add_edge(source_file, target_rel_path)
                    return

    def _build_call_graph(self):
        try:
            import javalang
        except ImportError:
            logging.error("javalang not installed. Call graph will not be built.")
            return
            
        for java_file in self.repo_path.rglob('*.java'):
            try:
                with open(java_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    tree = javalang.parse.parse(content)
                except Exception as e:
                    logging.warning(f"Failed to parse {java_file}: {e}")
                    continue
                
                self._extract_method_calls(tree, java_file)
                
            except Exception as e:
                logging.warning(f"Failed to process call graph for {java_file}: {e}")
    
    def _extract_method_calls(self, tree, java_file):
        current_class = None
        current_method = None
        
        for path, node in tree:
            if isinstance(node, javalang.tree.ClassDeclaration):
                current_class = node.name
                
            elif isinstance(node, javalang.tree.MethodDeclaration):
                current_method = f"{current_class}.{node.name}" if current_class else node.name
                if not self.graphs.call_graph.has_node(current_method):
                    self.graphs.call_graph.add_node(current_method, code=current_method)
                    
            elif isinstance(node, javalang.tree.MethodInvocation):
                if current_method:
                    called_method = node.member
                    if node.qualifier:
                        if hasattr(node.qualifier, 'member'):
                            called_method = f"{node.qualifier.member}.{called_method}"
                    
                    if not self.graphs.call_graph.has_node(called_method):
                        self.graphs.call_graph.add_node(called_method, code=called_method)
                    
                    self.graphs.call_graph.add_edge(current_method, called_method)

    def _build_type_deps(self):
        try:
            import javalang
        except ImportError:
            logging.error("javalang not installed. Type dependencies will not be built.")
            return
            
        for java_file in self.repo_path.rglob('*.java'):
            try:
                with open(java_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    tree = javalang.parse.parse(content)
                except Exception as e:
                    logging.warning(f"Failed to parse {java_file}: {e}")
                    continue
                
                self._extract_type_dependencies(tree)
                
            except Exception as e:
                logging.warning(f"Failed to process type deps for {java_file}: {e}")
    
    def _extract_type_dependencies(self, tree):
        current_type = None
        
        for path, node in tree:
            if isinstance(node, (javalang.tree.ClassDeclaration, 
                               javalang.tree.InterfaceDeclaration)):
                current_type = node.name
                if not self.graphs.type_deps.has_node(current_type):
                    self.graphs.type_deps.add_node(current_type, code=current_type)
                    
            elif isinstance(node, javalang.tree.VariableDeclaration):
                if current_type and hasattr(node, 'type'):
                    type_name = self._extract_type_name(node.type)
                    if type_name and type_name != current_type:
                        if not self.graphs.type_deps.has_node(type_name):
                            self.graphs.type_deps.add_node(type_name, code=type_name)
                        self.graphs.type_deps.add_edge(current_type, type_name)
    
    def _extract_type_name(self, type_node):
        if hasattr(type_node, 'name'):
            return type_node.name
        elif hasattr(type_node, 'element_type'):
            return self._extract_type_name(type_node.element_type)
        return None

    def _build_class_inheritance(self):
        try:
            import javalang
        except ImportError:
            logging.error("javalang not installed. Class inheritance will not be built.")
            return
            
        for java_file in self.repo_path.rglob('*.java'):
            try:
                with open(java_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    tree = javalang.parse.parse(content)
                except Exception as e:
                    logging.warning(f"Failed to parse {java_file}: {e}")
                    continue
                
                self._extract_inheritance(tree)
                
            except Exception as e:
                logging.warning(f"Failed to process inheritance for {java_file}: {e}")
    
    def _extract_inheritance(self, tree):
        for path, node in tree:
            if isinstance(node, javalang.tree.ClassDeclaration):
                child_class = node.name
                if not self.graphs.class_inheritance.has_node(child_class):
                    self.graphs.class_inheritance.add_node(child_class, code=child_class)
                
                if node.extends:
                    parent_class = node.extends.name
                    if not self.graphs.class_inheritance.has_node(parent_class):
                        self.graphs.class_inheritance.add_node(parent_class, code=parent_class)
                    self.graphs.class_inheritance.add_edge(child_class, parent_class)

    def _build_annotation_graph(self):
        try:
            import javalang
        except ImportError:
            logging.error("javalang not installed. Annotation graph will not be built.")
            return
            
        for java_file in self.repo_path.rglob('*.java'):
            try:
                with open(java_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    tree = javalang.parse.parse(content)
                except Exception as e:
                    logging.warning(f"Failed to parse {java_file}: {e}")
                    continue
                
                self._extract_annotations(tree)
                
            except Exception as e:
                logging.warning(f"Failed to process annotations for {java_file}: {e}")
    
    def _extract_annotations(self, tree):
        current_element = None
        
        for path, node in tree:
            if isinstance(node, (javalang.tree.ClassDeclaration,
                               javalang.tree.MethodDeclaration,
                               javalang.tree.FieldDeclaration)):
                if isinstance(node, javalang.tree.ClassDeclaration):
                    current_element = f"class:{node.name}"
                elif isinstance(node, javalang.tree.MethodDeclaration):
                    current_element = f"method:{node.name}"
                elif isinstance(node, javalang.tree.FieldDeclaration):
                    for declarator in node.declarators:
                        current_element = f"field:{declarator.name}"
                
                if hasattr(node, 'annotations') and node.annotations:
                    for annotation in node.annotations:
                        annotation_name = annotation.name
                        
                        if not self.graphs.annotation_graph.has_node(current_element):
                            self.graphs.annotation_graph.add_node(current_element, code=current_element)
                        if not self.graphs.annotation_graph.has_node(annotation_name):
                            self.graphs.annotation_graph.add_node(annotation_name, code=annotation_name)
                        
                        self.graphs.annotation_graph.add_edge(current_element, annotation_name)

    def _build_ast_graph(self):
        try:
            import javalang
        except ImportError:
            logging.error("javalang not installed. AST graph will not be built.")
            return
            
        for java_file in self.repo_path.rglob('*.java'):
            try:
                with open(java_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    tree = javalang.parse.parse(content)
                except Exception as e:
                    logging.warning(f"Failed to parse {java_file}: {e}")
                    continue
                
                self._build_ast_nodes_and_edges(tree, java_file, content)
                
            except Exception as e:
                logging.warning(f"Failed to process AST for {java_file}: {e}")
    
    def _build_ast_nodes_and_edges(self, tree, java_file, content):
        file_rel_path = str(java_file.relative_to(self.repo_path))
        source_lines = content.splitlines()
        
        def get_node_uid(node, file_path):
            node_type = type(node).__name__
            position = getattr(node, 'position', None)
            if position:
                return f"ast:{file_path}:{position.line}:{position.column}:{node_type}"
            else:
                return f"ast:{file_path}:0:0:{node_type}:{id(node)}"
        
        def get_code_snippet(node):
            if hasattr(node, 'position') and node.position:
                try:
                    line_idx = node.position.line - 1
                    if 0 <= line_idx < len(source_lines):
                        return source_lines[line_idx].strip()
                except:
                    pass
            return ""
        
        for path, node in tree:
            node_uid = get_node_uid(node, file_rel_path)
            code_snippet = get_code_snippet(node)
            
            attrs = {
                'node_type': type(node).__name__,
                'file_path': file_rel_path,
                'code': code_snippet
            }
            
            if hasattr(node, 'name'):
                attrs['name'] = node.name
            
            self._add_node_with_code(self.graphs.ast, node_uid, code_snippet, file_rel_path, extra_attrs=attrs)
        
        def add_edges_recursively(parent_path, parent_node):
            parent_uid = get_node_uid(parent_node, file_rel_path)
            
            for child_name, child_node in parent_node:
                if hasattr(child_node, '__iter__') and not isinstance(child_node, str):
                    try:
                        for item in child_node:
                            if hasattr(item, 'position'):
                                child_uid = get_node_uid(item, file_rel_path)
                                self.graphs.ast.add_edge(parent_uid, child_uid)
                                add_edges_recursively(parent_path + [child_name], item)
                    except:
                        pass
                elif hasattr(child_node, 'position'):
                    child_uid = get_node_uid(child_node, file_rel_path)
                    self.graphs.ast.add_edge(parent_uid, child_uid)
                    add_edges_recursively(parent_path + [child_name], child_node)
        
        for path, node in tree.filter(javalang.tree.CompilationUnit):
            add_edges_recursively([], node)

    def _build_cfg(self):
        try:
            import javalang
        except ImportError:
            logging.error("javalang not installed. CFG will not be built.")
            return

        for java_file in self.repo_path.rglob('*.java'):
            file_rel = str(java_file.relative_to(self.repo_path))
            try:
                source = java_file.read_text(encoding='utf-8', errors='ignore')
                tree = javalang.parse.parse(source)
            except Exception as e:
                logging.warning(f"Skip CFG for {java_file}: {e}")
                continue

            source_lines = source.splitlines()

            def code_of(node):
                if hasattr(node, 'position') and node.position:
                    line = node.position.line - 1
                    if 0 <= line < len(source_lines):
                        return source_lines[line].strip()
                return type(node).__name__

            def stmt_uid(method_uid, idx):
                return f"cfg:{file_rel}:{method_uid}:{idx}"

            for _, method in tree.filter(javalang.tree.MethodDeclaration):
                if method.body is None:
                    continue
                method_uid = f"{method.name}"
                prev_stmt_node = None
                idx = 0
                for stmt in method.body.statements:
                    node_id = stmt_uid(method_uid, idx)
                    self._add_node_with_code(
                        self.graphs.cfg,
                        node_id,
                        code_of(stmt),
                        file_rel,
                        extra_attrs={
                            'node_type': type(stmt).__name__,
                            'method': method.name
                        }
                    )
                    if prev_stmt_node is not None:
                        self.graphs.cfg.add_edge(prev_stmt_node, node_id)
                    prev_stmt_node = node_id

                    if isinstance(stmt, (javalang.tree.IfStatement, javalang.tree.WhileStatement, javalang.tree.ForStatement)):
                        children_blocks = []
                        if hasattr(stmt, 'then_statement') and stmt.then_statement:
                            children_blocks.append(stmt.then_statement)
                        if hasattr(stmt, 'else_statement') and stmt.else_statement:
                            children_blocks.append(stmt.else_statement)
                        if hasattr(stmt, 'body') and stmt.body:  # loop
                            children_blocks.append(stmt.body)
                        for blk in children_blocks:
                            first_child = None
                            if isinstance(blk, javalang.tree.BlockStatement) and blk.statements:
                                first_child = blk.statements[0]
                            elif not isinstance(blk, javalang.tree.BlockStatement):
                                first_child = blk
                            if first_child is not None:
                                child_id = stmt_uid(method_uid, f"{idx}_b")
                                self._add_node_with_code(
                                    self.graphs.cfg,
                                    child_id,
                                    code_of(first_child),
                                    file_rel,
                                    extra_attrs={'node_type': type(first_child).__name__, 'method': method.name}
                                )
                                self.graphs.cfg.add_edge(node_id, child_id)
                    idx += 1


        def _build_dfg(self):
            try:
                import javalang
            except ImportError:
                logging.error("javalang not installed. DFG will not be built.")
                return
                
            for java_file in self.repo_path.rglob('*.java'):
                try:
                    with open(java_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    try:
                        tree = javalang.parse.parse(content)
                    except Exception as e:
                        logging.warning(f"Failed to parse {java_file}: {e}")
                        continue
                    
                    self._extract_data_flow(tree, java_file)
                    
                except Exception as e:
                    logging.warning(f"Failed to process DFG for {java_file}: {e}")
        
        def _extract_data_flow(self, tree, java_file):
            file_rel_path = str(java_file.relative_to(self.repo_path))
            variable_defs = {}
            
            for path, node in tree:
                if isinstance(node, javalang.tree.VariableDeclarator):
                    var_name = node.name
                    var_uid = f"dfg:{file_rel_path}:{var_name}_def"
                    
                    if not self.graphs.dfg.has_node(var_uid):
                        self.graphs.dfg.add_node(var_uid, code=var_name)
                    
                    variable_defs[var_name] = var_uid
                
                elif isinstance(node, javalang.tree.MemberReference):
                    if node.member in variable_defs:
                        use_uid = f"dfg:{file_rel_path}:{node.member}_use_{id(node)}"
                        if not self.graphs.dfg.has_node(use_uid):
                            self.graphs.dfg.add_node(use_uid, code=node.member)
                        
                        self.graphs.dfg.add_edge(variable_defs[node.member], use_uid)

        def build_combined_graph(self):
            self.combined_graph = nx.DiGraph()
            self.graphs.combined_graph = self.combined_graph
            
            for graph_name, graph in [
                ('folder_structure', self.graphs.folder_structure),
                ('cross_file_deps', self.graphs.cross_file_deps),
                ('call_graph', self.graphs.call_graph),
                ('type_deps', self.graphs.type_deps),
                ('ast', self.graphs.ast),
                ('cfg', self.graphs.cfg),
                ('dfg', self.graphs.dfg)
            ]:
                for node in graph.nodes():
                    node_attrs = graph.nodes[node].copy()
                    node_attrs['graph_type'] = graph_name
                    node_attrs['original_id'] = node
                    
                    self.combined_graph.add_node(f"{graph_name}:{node}", **node_attrs)
                
                for src, dst in graph.edges():
                    self.combined_graph.add_edge(f"{graph_name}:{src}", 
                                            f"{graph_name}:{dst}",
                                            edge_type=f"internal_{graph_name}")
            self._connect_java_repo_to_module_level()
            self._connect_java_module_to_function_level()
            
            return self.combined_graph

        def _connect_java_repo_to_module_level(self):
            import javalang

            # 1. 文件 → 方法 (file_location_constraint)
            for file_node in self.graphs.folder_structure.nodes():
                if not str(file_node).endswith('.java'):
                    continue
                file_path = self.repo_path / file_node
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    tree = javalang.parse.parse(content)
                    for _, md in tree.filter(javalang.tree.MethodDeclaration):
                        method_name = md.name
                        if self.graphs.call_graph.has_node(method_name):
                            self.combined_graph.add_edge(
                                f"folder_structure:{file_node}",
                                f"call_graph:{method_name}",
                                edge_type="file_location_constraint"
                            )
                except Exception as e:
                    logging.debug(f"Skip parsing {file_path}: {e}")

            for src_file, _ in self.graphs.cross_file_deps.edges():
                for caller in self.graphs.call_graph.nodes():
                    self.combined_graph.add_edge(
                        f"cross_file_deps:{src_file}",
                        f"call_graph:{caller}",
                        edge_type="cross_module_call"
                    )

            for src_file, _ in self.graphs.cross_file_deps.edges():
                for type_src, _ in self.graphs.type_deps.edges():
                    self.combined_graph.add_edge(
                        f"cross_file_deps:{src_file}",
                        f"type_deps:{type_src}",
                        edge_type="type_cross_file_reference"
                    )

        def _connect_java_module_to_function_level(self):
            for func_node in self.graphs.call_graph.nodes():
                for ast_node in self.graphs.ast.nodes():
                    self.combined_graph.add_edge(
                        f"call_graph:{func_node}",
                        f"ast:{ast_node}",
                        edge_type="method_body_structure"
                    )
                    break

            for type_src, _ in self.graphs.type_deps.edges():
                for dfg_src, _ in self.graphs.dfg.edges():
                    self.combined_graph.add_edge(
                        f"type_deps:{type_src}",
                        f"dfg:{dfg_src}",
                        edge_type="variable_type_constraint"
                    )
                    break

            for func_node in self.graphs.call_graph.nodes():
                for cfg_node in self.graphs.cfg.nodes():
                    self.combined_graph.add_edge(
                        f"call_graph:{func_node}",
                        f"cfg:{cfg_node}",
                        edge_type="method_control_flow"
                    )
                    break


def preprocess_repobench_data(data_sample, repo_path, temp_repo_path):

    next_line = data_sample.get('next_line', '')
    all_code = data_sample.get('all_code', '')
    file_path = data_sample.get('file_path', '')
    
    processed_sample = dict(data_sample) if data_sample else {}

    try:
        repo_path = Path(repo_path)
        if not repo_path.exists() or not repo_path.is_dir():
            logging.error(f"Repository path does not exist or is not a directory: {repo_path}")
            return processed_sample
            
        temp_repo_path.mkdir(exist_ok=True, parents=True)
        
        found_files = []
        
        logging.info(f"Scanning repository {repo_path} for next_line: {next_line[:50]}...")
        
        for py_file in repo_path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                if next_line in file_content:
                    logging.info(f"Found next_line in file: {py_file}")

                    
                    next_line_pos = file_content.find(next_line)
                    
                    filtered_content = file_content[:next_line_pos].rstrip() + '\n'
                    
                    rel_path = py_file.relative_to(repo_path)
                    temp_file_path = temp_repo_path /rel_path
                    temp_file_path.parent.mkdir(exist_ok=True, parents=True)
                    
                    with open(temp_file_path, 'w', encoding='utf-8') as f:
                        f.write(filtered_content)
                    
                    # found_files.append({
                    #     'original_file': str(py_file),
                    #     'filtered_file': str(temp_file_path),
                    #     'relative_path': str(rel_path)
                    # })
                    
                    logging.info(f"Created filtered file: {temp_file_path}")
            except Exception as e:
                logging.warning(f"Error processing file {py_file}: {e}")
        
    except Exception as e:
        logging.error(f"Error processing repository: {e}")


def process_repo_to_graph(repo_path: str, lang: str = 'python') -> MultiLevelGraph:
    if lang == 'python':
        builder = MultiLevelGraphBuilder(repo_path)
    elif lang == 'java':
        builder = MultiLevelJavaGraphBuilder(repo_path)
    else:
        raise ValueError(f"Unsupported language: {lang}")
    
    builder.build_repo_level()
    builder.build_module_level()
    builder.build_function_level()
    
    builder.build_combined_graph()
    
    return builder.graphs

def load_jsonl(fname):
    with open(fname, 'r', encoding='utf8') as f:
        lines = []
        for line in f:
            lines.append(json.loads(line))
        return lines

def visualize_and_save_graphs(graphs, output_dir):
    def save_and_visualize_graph(graph, name, file_ext=".gpickle", figsize=(20, 20)):
        nx.write_graphml(graph, output_dir / f"{name}{file_ext}")
        
        try:
            plt.figure(figsize=figsize)
            pos = nx.spring_layout(graph)
            nx.draw(graph, pos, with_labels=True, node_size=30,
                    node_color="blue", font_size=8, alpha=0.6)
            plt.title(name)
            plt.savefig(output_dir / f"{name}_viz.png", dpi=100)
            plt.close()
        except Exception as e:
            logging.warning(f"Failed to visualize {name} graph: {str(e)}")
    
    graph_mapping = {
        "folder_structure": graphs.folder_structure,
        "cross_file_deps": graphs.cross_file_deps,
        "call_graph": graphs.call_graph,
        "type_deps": graphs.type_deps,
        "class_inheritance": graphs.class_inheritance,
        "ast": graphs.ast,
        "cfg": graphs.cfg,
        "dfg": graphs.dfg
    }

    for name, graph in graph_mapping.items():
        save_and_visualize_graph(graph, name)
    
    save_and_visualize_graph(graphs.combined_graph, "combined_graph", file_ext=".graphml", figsize=(100, 100))


def main(dataset_name,language):
    logging.basicConfig(level=logging.INFO)
    print("start")
    print(f"dataset_name: {dataset_name}, language: {language}")

    if dataset_name == "repoeval-updated":
        repo_path = Path(f"dataset/dataset_repoeval_updated/repos")
        repo_root = Path(f"dataset/dataset_repoeval_updated/repoeval_to_repobench")
        dataset=load_jsonl(f"{repo_root}/line_level.python.test.jsonl")
        
    elif dataset_name == "crosscodeeval":
        repo_path = Path(f"dataset/crosscodeeval/repos")
        repo_root = Path(f"dataset/crosscodeeval/")
        dataset=load_jsonl(f"{repo_root}/rawdata/{language}/line_completion.jsonl")

    else:
        print(f"Unknown dataset name: {dataset_name}")
        return
    

    processed_dir = repo_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    graphs_dir = repo_root / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    if dataset is None:
        logging.error("No dataset provided for processing")
        return
    
    logging.info(f"Processing {len(dataset)} samples from dataset")
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
            
    for idx, sample in enumerate(dataset):
        if dataset_name == "crosscodeeval":
            repo_name = sample['metadata']['repository']
            
        elif dataset_name == "repoeval_updated":
            repo_author, repo_name = repo_full_name.split('/')
        temp_repo_path = repo_root / 'temp' / f"{repo_name}"
        
        sample_id = f"{idx}_{repo_name}"
        sample_processed_dir = processed_dir / sample_id
        sample_processed_dir.mkdir(parents=True, exist_ok=True)
        sample_graph_dir = graphs_dir / sample_id
        sample_graph_dir.mkdir(parents=True, exist_ok=True)

        if (sample_graph_dir / "repo_multi_graph.graphml").exists():
            print(f"Sample {sample_id} has already been processed, skipping...")
            continue


        try:    
            graphs = process_repo_to_graph(str(repo_path/repo_name), lang=language)

            nx.write_graphml_lxml(graphs.combined_graph, sample_graph_dir / "repo_multi_graph.graphml")
            
            processed_count += 1
            logging.info(f"Successfully processed sample {idx}")
            
            if idx % 10 == 0 and idx > 0:
                logging.info(f"Progress: {idx}/{len(dataset)} samples processed")
                
        except Exception as e:
            error_count += 1
            logging.error(f"Failed to process sample {idx}: {str(e)}")
    
    logging.info(f"Finished processing dataset")
    logging.info(f"Total samples: {len(dataset)}")
    logging.info(f"Successfully processed: {processed_count}")
    logging.info(f"Skipped: {skipped_count}")
    logging.info(f"Errors: {error_count}")



if __name__ == "__main__":
    # main("crosscodeeval","python")
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", choices=["crosscodeeval", "repoeval-updated"])
    parser.add_argument("language",choices=["python", "java"])
    args = parser.parse_args()

    main(args.dataset_name, args.language)