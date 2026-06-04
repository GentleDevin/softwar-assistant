# -*- coding: utf-8 -*-
import html
import json
import os
from dotenv import load_dotenv
load_dotenv()
import pickle
import re
import traceback
from collections import deque
from typing import List, Dict, Any, Tuple, Optional

import PyPDF2
import numpy as np
from langchain.embeddings.base import Embeddings
from langchain_classic import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangchainDocument
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
# Langchain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Neo4j direct import instead of using LangChain's wrapper
from neo4j import GraphDatabase

# Import the agent coordinator from our refactored agents.py
from agents import AgentCoordinator

# 常量定义
ENTITY_TYPES = [
    "概念", "方法", "工具",
    "模型", "原则", "阶段",
    "角色", "工件", "技术",
    "框架", "模式", "流程",
    "标准", "实践", "语言"
]

RELATIONSHIP_TYPES = [
    "属于", "包含", "使用",
    "定义", "实现", "创建",
    "前置", "后置", "依赖",
    "派生", "应用于", "结合",
    "基于", "替代", "优化",
    "参与", "生成", "验证",
    "遵循", "扩展", "关联"
]

# --- 嵌入模型配置 ---
RAG_EMBEDDING_MODEL = "text-embedding-v3"
RAG_EMBEDDING_DIMENSION = 1536
ENTITY_EMBEDDING_MODEL = "text-embedding-v3"
ENTITY_EMBEDDING_DIMENSION = 1536

class RAGManager:
    """
    管理文档处理和检索的类 (使用Langchain和FAISS)
    """
    _instance = None
    _initialized = False

    @classmethod
    def get_instance(cls, embeddings: Optional[Embeddings] = None):
        """
        获取RAGManager的单例实例。首次调用必须提供embeddings。
        """
        if cls._instance is None:
            if embeddings is None:
                raise ValueError("首次创建 RAGManager 实例时必须提供 embeddings。")
            print("使用 embeddings 创建新的 RAGManager 实例。")
            cls._instance = cls(embeddings)
        elif embeddings is not None and not cls._initialized:
             print("为现有的 RAGManager 实例设置 embeddings。")
             cls._instance.embeddings = embeddings
             cls._initialized = True
        return cls._instance

    def __init__(self, embeddings: Embeddings):
        """
        初始化RAG管理器 (需要OpenAI嵌入模型)
        """
        if RAGManager._initialized:
            return
        
        print("使用 Langchain 和 FAISS 初始化 RAGManager...")
        self.embeddings = embeddings
        self.vector_store = None
        self.document_sources = [] # 存储每个文本块的来源文件名
        self.is_knowledge_base_loaded = False # 知识库是否加载成功的标志
        RAGManager._initialized = True
        print(f"RAGManager 初始化完成，使用 FAISS 和 Langchain embeddings")

    def extract_text_from_pdf(self, file_path):
        """从PDF文件中提取文本"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                if reader.is_encrypted:
                     try: reader.decrypt('')
                     except Exception as decrypt_err: print(f"  警告：无法解密 PDF {os.path.basename(file_path)}: {decrypt_err}")
                for page_num in range(len(reader.pages)):
                    try:
                        page = reader.pages[page_num]
                        extracted = page.extract_text()
                        if extracted: text += extracted + "\n"
                    except Exception as page_err: print(f"  警告：从 PDF {os.path.basename(file_path)} 的第 {page_num+1} 页提取文本时出错: {page_err}")
            print(f"  从 PDF 提取了 {len(text)} 个字符: {os.path.basename(file_path)}")
            return text
        except Exception as e:
            print(f"  处理 PDF 文件 {os.path.basename(file_path)} 时出错: {e}")
            traceback.print_exc()
            return ""

    def extract_text_from_txt(self, file_path):
        """从TXT文件中提取文本"""
        text = ""
        encodings_to_try = ['utf-8', 'gbk', 'utf-16']
        for enc in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=enc) as file: text = file.read()
                print(f"  使用编码 {enc} 成功读取 TXT 文件 {os.path.basename(file_path)}")
                return text
            except UnicodeDecodeError: continue
            except Exception as e:
                print(f"  使用编码 {enc} 处理 TXT 文件 {os.path.basename(file_path)} 时出错: {e}")
                traceback.print_exc()
                return ""
        print(f"  无法使用尝试的编码解码 TXT 文件 {os.path.basename(file_path)}。")
        return ""

    def process_uploaded_files(self, files):
        """处理上传的文件：提取、分块、并使用Langchain进行向量化"""
        print("为新上传重置 RAGManager 状态...")
        self.vector_store = None
        self.document_sources = []
        self.is_knowledge_base_loaded = False

        if not files: 
            return "未上传任何文件"

        status_texts = []
        print(f"开始处理 {len(files)} 个上传的文件...")
        raw_docs_content = []
        raw_docs_sources = []

        try:
            # --- 提取文本 ---
            for file_obj in files:
                temp_file_path = file_obj.name
                original_file_name = getattr(file_obj, 'orig_name', os.path.basename(temp_file_path))
                file_ext = os.path.splitext(original_file_name)[1].lower()
                print(f"处理文件: {original_file_name}...")
                text = ""
                if file_ext == '.pdf': 
                    text = self.extract_text_from_pdf(temp_file_path)
                elif file_ext == '.txt': 
                    text = self.extract_text_from_txt(temp_file_path)
                else:
                    status_texts.append(f"❌ 不支持的文件类型: {original_file_name}")
                    continue
                
                if text and len(text.strip()) > 50:
                    raw_docs_content.append(text)
                    raw_docs_sources.append(original_file_name)
                    status_texts.append(f"✅ 成功提取文本: {original_file_name}")
                else: 
                    status_texts.append(f"⚠️ 文件内容为空或过短，已跳过: {original_file_name}")

            if not raw_docs_content:
                status_texts.append("❌ 未能成功提取任何有效文本内容。")
                return "\n".join(status_texts)

            # --- 文档分块 - 使用Langchain的RecursiveCharacterTextSplitter ---
            print("使用 Langchain 分块文档...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150,
                length_function=len,
            )
            
            langchain_docs = []
            for doc_idx, doc_text in enumerate(raw_docs_content):
                source = raw_docs_sources[doc_idx]
                chunks = text_splitter.split_text(doc_text)
                for chunk in chunks:
                    if len(chunk.strip()) > 20:  # 确保块不是太小
                        langchain_doc = LangchainDocument(
                            page_content=chunk,
                            metadata={"source": source}
                        )
                        langchain_docs.append(langchain_doc)
                        self.document_sources.append(source)
            
            if not langchain_docs:
                status_texts.append("❌ 提取的文本内容无法分割成有效的文本块。")
                return "\n".join(status_texts)
            
            print(f"将文档分割成 {len(langchain_docs)} 个块。")
            status_texts.append(f"📄 已将文档分割为 {len(langchain_docs)} 个文本块")

            # --- 创建向量存储 ---
            print("使用 FAISS 创建向量存储...")
            try:
                self.vector_store = FAISS.from_documents(
                    langchain_docs, 
                    self.embeddings
                )
                print(f"FAISS 向量存储创建完成，包含 {len(langchain_docs)} 个嵌入")
                status_texts.append(f"📊 成功创建向量存储，包含 {len(langchain_docs)} 个文本块的向量")
                self.is_knowledge_base_loaded = True
            except Exception as e:
                error_msg = f"创建向量存储时出错: {str(e)}"
                status_texts.append(f"❌ {error_msg}")
                print(error_msg)
                traceback.print_exc()
                self.is_knowledge_base_loaded = False

        except Exception as e:
            error_msg = f"处理文件时发生严重错误: {str(e)}"
            status_texts.append(f"❌ {error_msg}")
            print(error_msg)
            traceback.print_exc()
            self.is_knowledge_base_loaded = False

        status_result = "\n".join(status_texts)
        print(f"文件处理完成。知识库加载状态: {self.is_knowledge_base_loaded}")
        if not self.is_knowledge_base_loaded:
             status_result += "\n\n⚠️ 知识库未能成功加载，文档检索功能将不可用。"
        return status_result

    def search_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        在上传的文档中搜索相关内容 (使用Langchain和FAISS)
        """
        print(f"--- 文档搜索 (Langchain FAISS) ---")
        print(f"查询: '{query}'")
        print(f"知识库加载状态: {self.is_knowledge_base_loaded}")
        
        # --- 前置检查 ---
        if not self.is_knowledge_base_loaded or self.vector_store is None:
            print("搜索中止：知识库向量存储未加载。")
            return []
        if not query or not query.strip():
             print("搜索中止：查询为空。")
             return []

        try:
            # 使用Langchain的相似度搜索
            print(f"执行向量搜索，top_k={top_k}...")
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k
            )
            
            # 转换为原始格式
            results = []
            min_similarity_threshold = 0.3  # 相似度阈值
            
            for doc, score in docs_with_scores:
                # FAISS返回的是距离，需要转换为相似度
                # 距离越小越相似，需要转换一下
                similarity = 1.0 / (1.0 + score)
                
                if similarity > min_similarity_threshold:
                    results.append({
                        "source": doc.metadata.get("source", "未知来源"),
                        "text": doc.page_content,
                        "similarity": float(similarity)
                    })
            
            print(f"找到 {len(results)} 个高于阈值的相关文档块。")
            return results

        except Exception as e:
            print(f"文档搜索期间出错: {str(e)}")
            traceback.print_exc()
            return []

    def format_search_results_as_html(self, results: List[Dict[str, Any]]) -> str:
        """将搜索结果格式化为HTML"""
        if not results:
            if not self.is_knowledge_base_loaded:
                 return "<p>⚠️ 知识库尚未加载或加载失败。请先上传有效的知识库文档。</p>"
            else:
                 return "<p>✅ 知识库已加载，但在文档中未找到与查询语义相关的段落 (阈值 > 0.3)。</p>"

        html_result = "<div class='search-results'>"
        html_result += f"<h4>在知识库文档中找到 {len(results)} 个相关段落：</h4>"
        for i, res in enumerate(results):
            source_escaped = html.escape(res.get('source', '未知来源'))
            text_escaped = html.escape(res.get('text', ''))
            similarity = res.get('similarity', 0.0)
            html_result += f"""
            <div class='result-item'>
                <h5>相关段落 {i+1} (来自: {source_escaped}, 相似度: {similarity:.3f})</h5>
                <div class='result-text'>{text_escaped}</div>
            </div>"""
        html_result += "</div>"
        css = """
        <style>
        .search-results {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 10px 0;
        }
        .search-results h4 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.1em;
            font-weight: 600;
        }
        .result-item {
            background-color: #ffffff;
            border: 1px solid #eef2f6;
            border-left: 4px solid #2a81e3;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .result-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(42, 129, 227, 0.12);
        }
        .result-item h5 {
            margin: 0 0 10px 0;
            color: #2c3e50;
            font-size: 14px;
            font-weight: bold;
        }
        .result-text {
            color: #555555;
            font-size: 13.5px;
            line-height: 1.6;
            background-color: #fafbfc;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid #f1f3f5;
            white-space: pre-wrap;
        }
        </style>
        """
        html_result = f"{css}{html_result}"
        return html_result


class EntityExtractor:
    """从用户问题中提取实体"""
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
    def extract_entities(self, question: str) -> List[Dict[str, str]]:
        """使用LLM从问题中提取实体"""
        prompt_template = PromptTemplate(
            template="""
            从以下问题中提取关于软件工程的实体。
            实体类型必须是以下之一: {entity_types}
            输出严格按照 JSON 格式: [{{"name": "实体名", "type": "实体类型"}}, ...]
            问题: {question}
            JSON:
            """,
            input_variables=["question", "entity_types"]
        )
        
        try:
            chain = LLMChain(
                llm=self.llm, 
                prompt=prompt_template
            )
            
            response = chain.invoke({
                "question": question,
                "entity_types": ", ".join(ENTITY_TYPES)
            })
            
            content = response["text"].strip()
            print(f"实体提取原始响应: '{content}'")
            
            entities = []
            try: 
                entities = json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r'\[\s*(\{.*?\}\s*(,\s*\{.*?\})*)\s*\]', content)
                if match:
                     try: 
                        entities = json.loads(match.group(0))
                     except json.JSONDecodeError: 
                        entities = []
                else: 
                    entities = []
            
            valid_entities = []
            for item in entities:
                 if isinstance(item, dict) and "name" in item and "type" in item and item["type"] in ENTITY_TYPES:
                      valid_entities.append({"name": str(item["name"]), "type": str(item["type"])})
                 else: 
                    print(f"跳过无效实体: {item}")
            
            print(f"提取的实体 (已验证): {valid_entities}")
            return valid_entities
            
        except Exception as e:
            print(f"实体提取过程中出错: {e}")
            traceback.print_exc()
            return []


class Neo4jHandler:
    """处理知识图谱的Neo4j数据库操作"""
    def __init__(self, uri, username, password):
        self.uri = uri; self.username = username; self.password = password
        self.driver = None; self._connect()

    def _connect(self):
        if self.driver: self.close()
        try:
            print(f"尝试连接到 Neo4j 于 {self.uri}...")
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            with self.driver.session(database="neo4j") as session:
                result = session.run("RETURN 1")
                if result.single()[0] == 1: print("成功连接到 Neo4j 数据库。")
                else: raise Exception("连接测试查询失败。")
        except Exception as e:
            print(f"连接到 Neo4j 失败: {e}"); self.driver = None; raise

    def close(self):
        if self.driver:
            try: self.driver.close(); print("Neo4j 连接已关闭。")
            except Exception as e: print(f"关闭 Neo4j 连接时出错: {e}")
            finally: self.driver = None

    def _ensure_connection(self):
         if not self.driver: print("Neo4j 驱动程序不可用。正在重新连接..."); self._connect()

    def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        self._ensure_connection()
        if not self.driver: return []
        results = []
        print(f"执行 Cypher: {query[:150]}... 参数: {params}")
        try:
            with self.driver.session(database="neo4j") as session:
                cypher_result = session.run(query, parameters=params or {})
                results = [record.data() for record in cypher_result]
                print(f"查询完成。返回 {len(results)} 行。")
        except Exception as e:
            print(f"执行 Neo4j 查询时出错: {e}"); traceback.print_exc(); self.close()
        return results

    def get_all_entities(self) -> List[Dict[str, Any]]:
        print("Neo4jHandler: 获取所有实体...")
        query = "MATCH (n) WHERE n.name IS NOT NULL AND n.type IS NOT NULL RETURN n.name AS name, n.type AS type, labels(n) AS labels ORDER BY n.name"
        entities = self.execute_query(query)
        print(f"Neo4jHandler: get_all_entities 返回了 {len(entities)} 个实体。")
        return entities

    def get_entities_by_name(self, name: str) -> List[Dict[str, Any]]:
        query = "MATCH (n {name: $name}) WHERE n.type IS NOT NULL RETURN n.name AS name, n.type AS type, labels(n) AS labels"
        return self.execute_query(query, {"name": name})

    def get_entities_by_name_and_type(self, name: str, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        if entity_type:
            query = "MATCH (n {name: $name, type: $type}) RETURN n.name AS name, n.type AS type, labels(n) AS labels"
            params = {"name": name, "type": entity_type}
            return self.execute_query(query, params)
        else: return self.get_entities_by_name(name)

    def get_entity_relationships(self, entity_name: str) -> Dict[str, Any]:
        entity_info_list = self.get_entities_by_name(entity_name)
        if not entity_info_list: return {}
        entity_info = entity_info_list[0]
        out_query = "MATCH (n {name: $name})-[r]->(m) WHERE m.name IS NOT NULL AND m.type IS NOT NULL RETURN type(r) AS relationship, properties(r) AS rel_props, m.name AS target, m.type AS target_type"
        out_rels_raw = self.execute_query(out_query, {"name": entity_name})
        in_query = "MATCH (m)-[r]->(n {name: $name}) WHERE m.name IS NOT NULL AND m.type IS NOT NULL RETURN m.name AS source, m.type AS source_type, type(r) AS relationship, properties(r) AS rel_props"
        in_rels_raw = self.execute_query(in_query, {"name": entity_name})
        relationships = []
        for rel in out_rels_raw: relationships.append({"direction": "outgoing", "relationship": rel["relationship"], "rel_name": rel.get("rel_props", {}).get("name"), "target": rel["target"], "target_type": rel["target_type"]})
        for rel in in_rels_raw: relationships.append({"direction": "incoming", "source": rel["source"], "source_type": rel["source_type"], "relationship": rel["relationship"], "rel_name": rel.get("rel_props", {}).get("name")})
        return {"entity": entity_info, "relationships": relationships}

    def get_path_between_entities(self, source_name: str, target_name: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        path_query = f"MATCH path = allShortestPaths((src {{name: $source_name}})-[*1..{max_depth}]-(tgt {{name: $target_name}})) WHERE src.name IS NOT NULL AND tgt.name IS NOT NULL RETURN path LIMIT 5"
        path_results = self.execute_query(path_query, {"source_name": source_name, "target_name": target_name})
        paths = []
        processed_paths_count = 0

        for record in path_results:
            path_obj = record.get("path")
            if not path_obj:
                print("警告: 查询结果中未找到路径值。")
                continue

            if hasattr(path_obj, 'nodes') and hasattr(path_obj, 'relationships'):
                # --- 处理 neo4j.graph.Path 对象 (之前的逻辑) ---
                try:
                    path_info = {"nodes": [], "relationships": []}
                    for node in path_obj.nodes:
                        node_properties = dict(node.items())
                        path_info["nodes"].append({
                            "name": node_properties.get("name", "?"),
                            "type": node_properties.get("type", "?"),
                            "labels": list(node.labels)
                        })
                    for rel in path_obj.relationships:
                        rel_properties = dict(rel.items())
                        start_node_properties = dict(rel.start_node.items())
                        end_node_properties = dict(rel.end_node.items())
                        path_info["relationships"].append({
                            "source": start_node_properties.get("name", "?"),
                            "target": end_node_properties.get("name", "?"),
                            "type": rel.type,
                            "name": rel_properties.get("name", "")
                        })
                    paths.append(path_info)
                    processed_paths_count += 1
                except Exception as e:
                    print(f"警告: 处理 Path 对象时发生意外错误: {e}。路径值: {path_obj}")
                    traceback.print_exc()
                    continue
            elif isinstance(path_obj, list) and len(path_obj) >= 3 and len(path_obj) % 2 != 0:
                # --- 新增逻辑：处理列表表示的路径 ---
                # 假设结构是 [节点字典, 关系字符串, 节点字典, ...]
                try:
                    path_info = {"nodes": [], "relationships": []}
                    temp_nodes = []
                    temp_rels = []
                    valid_path_structure = True

                    for i, item in enumerate(path_obj):
                        if i % 2 == 0: # 偶数索引 - 节点
                            if isinstance(item, dict) and 'name' in item:
                                temp_nodes.append({
                                    "name": item.get("name", "?"),
                                    "type": item.get("type", "?"),
                                    "labels": item.get("labels", []) # 尝试获取 labels
                                })
                            else:
                                print(f"警告: 路径列表在索引 {i} 处期望节点字典，但找到: {type(item)}。跳过此路径。")
                                valid_path_structure = False
                                break
                        else: # 奇数索引 - 关系 (类型字符串)
                            if isinstance(item, str):
                                temp_rels.append(item)
                            else:
                                print(f"警告: 路径列表在索引 {i} 处期望关系字符串，但找到: {type(item)}。跳过此路径。")
                                valid_path_structure = False
                                break

                    if valid_path_structure:
                        path_info["nodes"] = temp_nodes
                        for i, rel_type in enumerate(temp_rels):
                            source_node = temp_nodes[i]
                            target_node = temp_nodes[i+1]
                            path_info["relationships"].append({
                                "source": source_node.get("name", "?"),
                                "target": target_node.get("name", "?"),
                                "type": rel_type,
                                "name": rel_type # 使用类型作为备用名称
                            })
                        paths.append(path_info)
                        processed_paths_count += 1

                except Exception as e:
                    print(f"警告: 解析列表表示的路径时发生错误: {e}。列表值: {path_obj}")
                    traceback.print_exc()
                    continue
            else:
                # --- 处理其他未知类型 ---
                print(f"警告: 返回的路径值既不是 Path 对象，也不是预期的列表结构。类型: {type(path_obj)}，值: {path_obj}。跳过此路径。")
                continue

        print(f"在 '{source_name}' 和 '{target_name}' 之间成功处理了 {processed_paths_count} 条路径。")
        return paths


class EntityMatcher:
    """将提取的实体与知识图谱实体匹配"""

    def __init__(self, embeddings: Embeddings, neo4j_handler, embedding_cache_path="entity_embeddings.pkl"):
        self.embeddings = embeddings
        self.neo4j_handler = neo4j_handler
        self.embedding_cache_path = embedding_cache_path
        self.embeddings_cache = self._load_embeddings_cache()
        self.kg_entities = []
        # 使用一致的嵌入模型和维度
        self.embedding_dim = ENTITY_EMBEDDING_DIMENSION

    def load_and_cache_kg_entities(self):
         """从 Neo4j 加载实体并缓存嵌入"""
         print("EntityMatcher: 加载 KG 实体...")
         all_entities_from_db = self.neo4j_handler.get_all_entities()
         self.kg_entities = [e for e in all_entities_from_db if e.get("name")]
         print(f"EntityMatcher: 存储了 {len(self.kg_entities)} 个带名称的实体。")
         if not self.kg_entities: return
         
         entity_names = [entity["name"] for entity in self.kg_entities]
         print(f"EntityMatcher: 确保 {len(entity_names)} 个名称的嵌入...")
         self._ensure_embeddings(entity_names)
         print("EntityMatcher: KG 实体已加载并确保嵌入。")

    def _load_embeddings_cache(self) -> Dict[str, list]:
        """从文件加载实体嵌入缓存"""
        cache = {}
        if os.path.exists(self.embedding_cache_path):
            try:
                with open(self.embedding_cache_path, 'rb') as f: 
                    cache = pickle.load(f)
                print(f"从缓存加载了 {len(cache)} 个嵌入: {self.embedding_cache_path}")
            except Exception as e: 
                print(f"加载嵌入缓存时出错: {e}")
        return cache

    def _save_embeddings_cache(self):
        """保存实体嵌入缓存到文件"""
        try:
            print(f"保存 {len(self.embeddings_cache)} 个嵌入到缓存: {self.embedding_cache_path}...")
            with open(self.embedding_cache_path, 'wb') as f: 
                pickle.dump(self.embeddings_cache, f)
            print("嵌入已保存。")
        except Exception as e: 
            print(f"保存嵌入缓存时出错: {e}")

    def get_embedding(self, text: str, force_refresh=False) -> Optional[list]:
        """获取文本的嵌入，使用缓存"""
        if not text: 
            return None
        
        cache_key = text
        if not force_refresh and cache_key in self.embeddings_cache:
            return self.embeddings_cache[cache_key]

        print(f"为 '{text}' 生成嵌入")
        try:
            embedding = self.embeddings.embed_query(text)
            self.embeddings_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            print(f"获取 '{text}' 的嵌入时出错: {e}")
            traceback.print_exc()
            return None

    def _ensure_embeddings(self, texts: List[str]):
        """确保所有文本在缓存中都有嵌入"""
        unique_texts = list(set(filter(None, texts)))
        texts_to_embed = [text for text in unique_texts if text not in self.embeddings_cache]
        
        if not texts_to_embed:
            print("所有必需的 KG 实体嵌入都已缓存。")
            return

        print(f"为 {len(texts_to_embed)} 个新的 KG 实体生成嵌入...")
        
        # 使用批处理来提高效率
        batch_size = 10
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:min(i+batch_size, len(texts_to_embed))]
            try:
                embeddings = self.embeddings.embed_documents(batch)
                for j, text in enumerate(batch):
                    self.embeddings_cache[text] = embeddings[j]
            except Exception as e:
                print(f"生成批次嵌入时出错: {e}")
        
        print(f"生成了 {len(texts_to_embed)} 个新嵌入。")
        self._save_embeddings_cache()

    def match_entity(self, entity: Dict[str, str], similarity_threshold=0.85) -> List[Dict[str, Any]]:
        """将提取的实体与已加载的 KG 实体列表进行匹配"""
        entity_name = entity.get("name")
        entity_type = entity.get("type")
        
        if not entity_name: 
            return []
            
        print(f"--- 实体匹配 ---")
        print(f"尝试匹配: '{entity_name}' (类型: {entity_type})")
        print(f"与 {len(self.kg_entities)} 个已加载的 KG 实体进行比较。")
        
        if not self.kg_entities: 
            print("警告：KG 实体列表为空。")
            return []

        # 步骤 1: 精确名称匹配
        print("步骤 1: 精确名称匹配 (在加载的列表中)...")
        exact_matches = [kg for kg in self.kg_entities if kg.get("name") == entity_name and (not entity_type or kg.get("type") == entity_type)]
        if exact_matches:
            print(f"找到 {len(exact_matches)} 个精确匹配。")
            return [{"entity": match, "match_type": "exact", "score": 1.0} for match in exact_matches]

        # 步骤 2: 向量相似度匹配
        print(f"步骤 2: 向量相似度匹配 (阈值: {similarity_threshold})...")
        entity_embedding = self.get_embedding(entity_name)
        if not entity_embedding: 
            print("无法获取查询嵌入。")
            return []

        matches = []
        print(f"与 {len(self.kg_entities)} 个 KG 实体比较向量...")
        
        for kg_entity in self.kg_entities:
            kg_entity_name = kg_entity.get("name")
            if not kg_entity_name: 
                continue
                
            kg_entity_embedding = self.embeddings_cache.get(kg_entity_name)
            if kg_entity_embedding:
                try:
                    # 使用np.dot计算余弦相似度，并增加模长非零安全保护
                    norm_a = np.linalg.norm(entity_embedding)
                    norm_b = np.linalg.norm(kg_entity_embedding)
                    if norm_a > 0 and norm_b > 0:
                        similarity = np.dot(entity_embedding, kg_entity_embedding) / (norm_a * norm_b)
                    else:
                        similarity = 0.0
                    
                    if similarity >= similarity_threshold:
                        matches.append({"entity": kg_entity, "match_type": "vector", "score": float(similarity)})
                except Exception as e: 
                    print(f"计算 '{kg_entity_name}' 的相似度时出错: {e}")

        matches.sort(key=lambda x: x["score"], reverse=True)
        top_matches = matches[:5]
        print(f"找到 {len(top_matches)} 个高于阈值的向量匹配。")
        return top_matches


class ResponseGenerator:
    """使用LLM、知识图谱上下文和文档检索结果生成响应"""
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def generate_response(self, question: str, kg_context: Dict[str, Any], 
                         entities: List[Dict[str, str]], doc_results: List[Dict[str, Any]]) -> str:
        """使用LLM生成响应"""
        print("--- 响应生成 ---")
        kg_context_str = self._format_kg_context(kg_context)
        doc_context_str = self._format_doc_context(doc_results)
        entities_str = json.dumps(entities, ensure_ascii=False, indent=2)
        
        prompt_template = PromptTemplate(
            template="""
            任务：作为软件工程课程助手，根据【知识图谱信息】和【相关文档片段】回答【用户问题】。
            【用户问题】: {question}
            【提取实体】: {entities}
            【知识图谱信息】: {kg_context}
            【相关文档片段】: {doc_context}
            回答要求:
            1. 回答要准确、全面、简洁明了
            2. 使用专业且易于理解的语言
            3. 如有必要，提供实例或实践建议
            4. 如果问题涉及多个概念，要清晰区分各个概念
            5. 如果问题是编程相关的，可以提供简单的代码示例
            请生成回答:
            """,
            input_variables=["question", "entities", "kg_context", "doc_context"]
        )
        
        try:
            print("调用 LLM 生成响应...")
            
            messages = [
                SystemMessage(content="你是专业的软件工程课程助手，精通各种软件工程概念、方法、工具和最佳实践。你的回答简洁明了、专业权威，并能根据学生的问题提供恰当的解释和指导。"),
                HumanMessage(content=prompt_template.format(
                    question=question,
                    entities=entities_str,
                    kg_context=kg_context_str or "无相关知识图谱信息。",
                    doc_context=doc_context_str or "无相关文档片段。"
                ))
            ]
            
            response = self.llm.invoke(messages)
            final_answer = response.content
            print(f"LLM 响应已接收。")
            return final_answer
            
        except Exception as e:
            print(f"调用 LLM 时出错: {e}")
            traceback.print_exc()
            return f"抱歉，在生成回答时遇到了问题：{e}。"

    def _format_kg_context(self, kg_context: Dict[str, Any]) -> str:
        if not kg_context or (not kg_context.get("entities") and not kg_context.get("paths")): 
            return ""
            
        formatted_parts = []
        if kg_context.get("entities"):
            formatted_parts.append("--- 知识图谱实体与关系 ---")
            for entity_data in kg_context["entities"]:
                entity = entity_data.get("entity", {})
                relationships = entity_data.get("relationships", [])
                
                if not entity or not entity.get("name"): 
                    continue
                    
                entity_name = entity["name"]
                entity_type = entity.get("type", "?")
                formatted_parts.append(f"\n实体: {entity_name} (类型: {entity_type})")
                
                if relationships:
                    formatted_parts.append("  关系:")
                    for rel in relationships:
                        rel_name = rel.get("rel_name") or rel.get("relationship", "?")
                        if rel.get("direction") == "outgoing": 
                            target = rel.get("target", "?")
                            target_type = rel.get("target_type", "?")
                            formatted_parts.append(f"    - {entity_name} --[{rel_name}]--> {target} (类型: {target_type})")
                        elif rel.get("direction") == "incoming": 
                            source = rel.get("source", "?")
                            source_type = rel.get("source_type", "?")
                            formatted_parts.append(f"    - {source} (类型: {source_type}) --[{rel_name}]--> {entity_name}")
        
        if kg_context.get("paths"):
            formatted_parts.append("\n--- 知识图谱实体间路径 ---")
            for i, path_data in enumerate(kg_context["paths"]):
                nodes = path_data.get("nodes", [])
                rels = path_data.get("relationships", [])
                
                if not nodes or not rels: 
                    continue
                    
                path_str = f"\n路径 {i+1}: "
                elements = []
                
                for j, node in enumerate(nodes):
                    elements.append(f"({node.get('name', '?')})")
                    if j < len(rels): 
                        rel = rels[j]
                        rel_name = rel.get('name') or rel.get('type', '?')
                        elements.append(f"--[{rel_name}]-->" if rel.get('source') == node.get('name') else f"<--[{rel_name}]--")
                
                formatted_parts.append(path_str + "".join(elements))
        
        return "\n".join(formatted_parts)

    def _format_doc_context(self, doc_results: List[Dict[str, Any]]) -> str:
        if not doc_results: 
            return ""
            
        formatted_parts = ["--- 相关文档片段 ---"]
        for i, doc in enumerate(doc_results):
            source = doc.get("source", "?")
            text = doc.get("text", "").strip()
            similarity = doc.get("similarity", 0.0)
            max_len = 300
            truncated = text[:max_len] + ("..." if len(text) > max_len else "")
            formatted_parts.append(f"\n片段 {i+1} (来源: {source}, Sim: {similarity:.3f}):\n---\n{truncated}\n---")
        
        return "\n".join(formatted_parts)


class SoftwareEngineeringQASystem:
    """整合知识图谱、本地文档与LLM的软件工程问答系统"""

    def __init__(self, neo4j_uri, neo4j_username, neo4j_password):
        """初始化QA系统"""
        print("初始化 SoftwareEngineeringQASystem...")
        
        # 优先从环境变量加载 API 密钥与 base_url，若无则使用默认值
        self.openai_api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if not self.openai_api_key: 
            raise ValueError("未找到 API 密钥。")
        
        # 设置 Langchain 组件
        self.llm = self._setup_llm()
        self.embeddings = self._setup_embeddings()
        
        # 设置 Neo4j
        try: 
            self.neo4j_handler = Neo4jHandler(neo4j_uri, neo4j_username, neo4j_password)
        except Exception as e: 
            print(f"致命错误：初始化 Neo4j Handler 失败: {e}")
            raise

        # 初始化 RAG 管理器
        self.rag_manager = RAGManager.get_instance(embeddings=self.embeddings)

        # 实体匹配器
        self.entity_matcher = EntityMatcher(self.embeddings, self.neo4j_handler)
        try: 
            self.entity_matcher.load_and_cache_kg_entities()
        except Exception as e: 
            print(f"初始 KG 实体加载期间出错: {e}")

        # 其他组件
        self.entity_extractor = EntityExtractor(self.llm)
        self.response_generator = ResponseGenerator(self.llm)  # 保留原始生成器用于回退
        self.agent_coordinator = AgentCoordinator(self.llm)

        # UI 状态
        self.current_kg_context = {"entities": [], "paths": []}
        self.current_doc_results_raw = []
        self.current_agent_name = None  # 跟踪使用的智能体
        
        # 添加对话历史管理
        self.conversation_history = deque(maxlen=10)  # 最多保留10轮对话
        
        print("SoftwareEngineeringQASystem 初始化成功。")

    def _setup_llm(self) -> ChatOpenAI:
        """设置Langchain的LLM组件"""
        try:
            llm = ChatOpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url,
                temperature=0.5,
                model="qwen-plus"
            )
            print("Langchain LLM 设置成功。")
            return llm
        except Exception as e: 
            print(f"设置 LLM 失败: {e}")
            raise

    def _setup_embeddings(self) -> OpenAIEmbeddings:
        """设置Langchain的Embeddings组件"""
        try:
            embeddings = OpenAIEmbeddings(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url,
                model=RAG_EMBEDDING_MODEL,
                dimensions=RAG_EMBEDDING_DIMENSION
            )
            print("Langchain Embeddings 设置成功。")
            return embeddings
        except Exception as e: 
            print(f"设置 Embeddings 失败: {e}")
            raise

    def answer_question(self, question: str) -> dict:
        """处理问题，整合 KG 和 RAG 生成答案，支持多轮对话"""
        print(f"\n===== 回答问题 =====")
        print(f"问题: {question}")
        print(f"对话历史长度: {len(self.conversation_history)}")
        
        self.current_kg_context = {"entities": [], "paths": []}
        self.current_doc_results_raw = []
        self.current_agent_name = None

        # 1. 提取实体
        print("\n步骤 1: 提取实体...")
        entities = self.entity_extractor.extract_entities(question)
        if not entities: 
            print("未提取到实体。")

        # 2. 匹配实体并构建 KG 上下文
        print("\n步骤 2: 匹配实体并检索 KG 上下文...")
        all_matched_kg_entity_names = set()
        
        if entities:
            for entity in entities:
                entity_name = entity.get("name")
                if not entity_name: 
                    continue
                    
                print(f"  匹配实体: '{entity_name}' (类型: {entity.get('type')})")
                matches = self.entity_matcher.match_entity(entity)
                
                if matches:
                    top_match = matches[0]
                    kg_entity_name = top_match["entity"]["name"]
                    print(f"    -> 最佳匹配: '{kg_entity_name}' (分数: {top_match['score']:.3f}, 方法: {top_match['match_type']})")
                    entity_data = self.neo4j_handler.get_entity_relationships(kg_entity_name)
                    
                    if entity_data: 
                        self.current_kg_context["entities"].append(entity_data)
                        all_matched_kg_entity_names.add(kg_entity_name)
                else: 
                    print(f"    -> 未找到 KG 匹配。")
        
        print(f"为 {len(self.current_kg_context['entities'])} 个匹配的实体找到 KG 上下文。")

        # 3. 查找实体之间的路径
        if len(all_matched_kg_entity_names) > 1:
            print("\n步骤 3: 查找匹配实体之间的路径...")
            entity_list = list(all_matched_kg_entity_names)
            
            for i in range(len(entity_list)):
                for j in range(i + 1, len(entity_list)):
                    source, target = entity_list[i], entity_list[j]
                    print(f"  查找 '{source}' 和 '{target}' 之间的路径...")
                    paths = self.neo4j_handler.get_path_between_entities(source, target)
                    if paths: 
                        self.current_kg_context["paths"].extend(paths)

        # 4. 文档检索 (RAG) - 使用Langchain
        print("\n步骤 4: 检索相关文档 (RAG)...")
        self.current_doc_results_raw = self.rag_manager.search_documents(question)
        print(f"检索到 {len(self.current_doc_results_raw)} 个相关文档块。")

        # 5. 使用智能体协调器生成最终响应
        print("\n步骤 5: 使用智能体生成最终响应...")
        has_kg = bool(self.current_kg_context.get("entities") or self.current_kg_context.get("paths"))
        has_docs = bool(self.current_doc_results_raw)

        if not has_kg and not has_docs:
            print("未找到 KG 上下文或相关文档。生成回退响应。")
            reason = "未能从问题中提取到有效实体，也未找到相关文档。" if not entities else "未能在知识图谱或文档中找到相关信息。"
            final_answer = self._generate_fallback_response(question, reason)
            self.current_agent_name = "回退响应生成器"
        else:
            # 使用智能体协调器处理问题
            final_answer, agent_name = self.agent_coordinator.process_question(
                question=question, 
                kg_context=self.current_kg_context,
                entities=entities, 
                doc_results=self.current_doc_results_raw,
                conversation_history=list(self.conversation_history)
            )
            self.current_agent_name = agent_name

        print(f"\n===== 答案生成完成 (由 {self.current_agent_name} 提供) =====")
        
        # 更新对话历史
        self.conversation_history.append((question, final_answer))
        
        # 返回增强的结果
        return {
            "answer": final_answer,
            "agent_name": self.current_agent_name,
            "conversation_history": list(self.conversation_history)
        }

    def _generate_fallback_response(self, question: str, reason: str) -> str:
        """生成基于通用知识的回退响应，考虑对话历史"""
        print(f"生成回退响应。原因: {reason}")
        
        # 格式化对话历史
        history_text = ""
        if self.conversation_history:
            history_text = "\n".join([f"用户: {q}\n助手: {a}" for q, a in list(self.conversation_history)[-5:]])
        else:
            history_text = "这是新的对话"
        
        prompt_template = PromptTemplate(
            template="""
            您是软件工程课程的智能助手。学生问了以下问题，但我们没有在知识库中找到相关信息。
            请基于您的通用软件工程知识提供最佳回答。
            
            对话历史:
            {history}
            
            当前问题: {question}
            
            请提供专业、准确且有帮助的回答，使用易于理解的语言解释软件工程概念。如果问题不清楚或超出软件工程范围，请礼貌地说明。
            根据对话历史，保持回答的连贯性。
            """,
            input_variables=["question", "history"]
        )
        
        try:
            messages = [
                SystemMessage(content="你是专业的软件工程课程助手，精通各种软件工程概念、方法、工具和最佳实践。你会根据对话历史提供连贯的回答。"),
                HumanMessage(content=prompt_template.format(
                    question=question,
                    history=history_text
                ))
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            print(f"生成回退响应时出错: {e}")
            return f"抱歉，我无法回答这个问题（{reason}）。尝试生成通用回答时也出错：{e}"

    def clear_conversation_history(self):
        """清除对话历史"""
        self.conversation_history.clear()
        print("对话历史已清除。")
    
    def get_conversation_history(self) -> List[Tuple[str, str]]:
        """获取对话历史"""
        return list(self.conversation_history)
    
    def close(self):
        """关闭资源"""
        print("关闭 SoftwareEngineeringQASystem 资源...")
        # Neo4j连接关闭
        if hasattr(self, 'neo4j_handler'): 
            self.neo4j_handler.close()
        print("系统已关闭。")


# --- 导出供UI使用的函数 ---

qa_system_instance: Optional[SoftwareEngineeringQASystem] = None

def initialize_qa_system(uri, user, pwd):
    """初始化全局QA系统实例 (供UI调用)"""
    global qa_system_instance
    if qa_system_instance is None:
        print("UI 正在初始化 QA 系统...")
        try:
             qa_system_instance = SoftwareEngineeringQASystem(uri, user, pwd)
        except Exception as e:
             print(f"致命错误：在 UI 中初始化 QA 系统失败: {e}")
             raise
    return qa_system_instance

def get_qa_system_instance() -> SoftwareEngineeringQASystem:
     """获取全局QA系统实例"""
     if qa_system_instance is None: 
        raise RuntimeError("QA 系统尚未初始化。")
     return qa_system_instance

def process_uploaded_files(files):
    """UI调用的文档处理函数"""
    print("UI 触发文件处理...")
    # 确保 QA 系统已初始化
    qa_system = get_qa_system_instance()
    rag_manager = qa_system.rag_manager
    return rag_manager.process_uploaded_files(files)

def search_documents(query, top_k=3):
    """UI调用的文档搜索函数 (仅用于独立显示结果)"""
    print(f"UI 触发文档搜索，查询: '{query}'")
    # 确保 QA 系统已初始化
    qa_system = get_qa_system_instance()
    rag_manager = qa_system.rag_manager
    results_raw = rag_manager.search_documents(query, top_k)
    results_html = rag_manager.format_search_results_as_html(results_raw)
    return results_html


if __name__ == "__main__":
    # 测试代码可以在这里添加
    pass