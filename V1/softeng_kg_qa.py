# -*- coding: utf-8 -*-
import html
import json
import logging
import os
import pickle
import re
import threading
import time
import traceback
import hashlib
from collections import deque
from typing import List, Dict, Any, Tuple, Optional

from dotenv import load_dotenv

load_dotenv()

import PyPDF2
import numpy as np
from langchain.embeddings.base import Embeddings
from langchain_classic import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangchainDocument
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
# Langchain imports
ChatOpenAI = None
OpenAIEmbeddings = None
ChatTongyi = None
DashScopeEmbeddings = None

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    pass

try:
    from langchain_dashscope import ChatTongyi, DashScopeEmbeddings
except ImportError:
    pass
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Neo4j direct import instead of using LangChain's wrapper
from neo4j import GraphDatabase

# Import the agent coordinator from our refactored agents.py
from agents import AgentCoordinator

# 导入统一配置和工具模块
from config import get_log_config
from utils import setup_logger
from utils.formatters import format_search_results_as_html as format_doc_results_html

# 配置日志（包含文件输出）
log_config = get_log_config()
logger = setup_logger(__name__, log_file=log_config.log_file, log_level="DEBUG")


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
RAG_EMBEDDING_MODEL = "text-embedding-v4"
RAG_EMBEDDING_DIMENSION = 1024
ENTITY_EMBEDDING_MODEL = "text-embedding-v4"
ENTITY_EMBEDDING_DIMENSION = 1024


class RAGManager:
    """
    管理文档处理和检索的类 (使用Langchain和FAISS)
    """
    _instance = None
    _initialized = False
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, embeddings: Optional[Embeddings] = None):
        """
        获取RAGManager的单例实例。首次调用必须提供embeddings。
        """
        with cls._lock:
            if cls._instance is None:
                if embeddings is None:
                    logger.warning("未提供 embeddings，RAGManager 实例将为 None。")
                    return None
                logger.info("使用 embeddings 创建新的 RAGManager 实例。")
                cls._instance = cls(embeddings)
            return cls._instance


    def __init__(self, embeddings: Embeddings):
        """
        初始化RAG管理器 (需要OpenAI嵌入模型)
        注意：此方法应由 get_instance 调用，已确保线程安全
        """
        if RAGManager._initialized:
            return

        logger.info("使用 Langchain 和 FAISS 初始化 RAGManager...")
        self.embeddings = embeddings
        self.vector_store = None
        self.document_sources = []  # 存储每个文本块的来源文件名
        self.is_knowledge_base_loaded = False  # 知识库是否加载成功的标志
        
        # FAISS 持久化相关
        self.faiss_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_cache")
        self.file_hashes_file = os.path.join(self.faiss_cache_dir, "file_hashes.pkl")
        self.faiss_index_file = os.path.join(self.faiss_cache_dir, "faiss_index")
        self.file_hashes = {}  # {filename: (file_hash, file_size, last_modified)}
        
        # 确保缓存目录存在
        os.makedirs(self.faiss_cache_dir, exist_ok=True)
        
        # 尝试加载已有的文件哈希记录
        self._load_file_hashes()
        
        # 尝试加载已有的FAISS索引
        self._load_faiss_index()
        
        RAGManager._initialized = True
        logger.info("RAGManager 初始化完成，使用 FAISS 和 Langchain embeddings")

    def _get_content_hash(self, content: str) -> str:
        """计算文本内容的 SHA256 哈希值"""
        hash_sha256 = hashlib.sha256()
        try:
            hash_sha256.update(content.encode('utf-8'))
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"计算内容哈希失败: {e}")
            return ""

    def _save_file_hashes(self):
        """保存文件哈希记录到磁盘"""
        try:
            with open(self.file_hashes_file, "wb") as f:
                pickle.dump(self.file_hashes, f)
            logger.info(f"文件哈希记录已保存到 {self.file_hashes_file}，共 {len(self.file_hashes)} 个文件")
        except Exception as e:
            logger.error(f"保存文件哈希记录失败: {e}")

    def _load_file_hashes(self):
        """从磁盘加载文件哈希记录"""
        try:
            if os.path.exists(self.file_hashes_file):
                with open(self.file_hashes_file, "rb") as f:
                    self.file_hashes = pickle.load(f)
                logger.info(f"已加载 {len(self.file_hashes)} 个文件哈希记录")
            else:
                logger.info("文件哈希记录不存在，将创建新记录")
        except Exception as e:
            logger.error(f"加载文件哈希记录失败: {e}")
            self.file_hashes = {}

    def _save_faiss_index(self):
        """保存 FAISS 索引到磁盘"""
        try:
            if self.vector_store is not None:
                self.vector_store.save_local(self.faiss_index_file)
                logger.info(f"FAISS 索引已保存到 {self.faiss_index_file}")
        except Exception as e:
            logger.error(f"保存 FAISS 索引失败: {e}")

    def _load_faiss_index(self):
        """从磁盘加载 FAISS 索引"""
        try:
            if os.path.exists(self.faiss_index_file):
                self.vector_store = FAISS.load_local(
                    self.faiss_index_file, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.is_knowledge_base_loaded = True
                logger.info(f"已从 {self.faiss_index_file} 加载 FAISS 索引")
            else:
                logger.info("FAISS 索引不存在，将创建新索引")
        except Exception as e:
            logger.warning(f"加载 FAISS 索引失败（可能不存在）: {e}")
            self.vector_store = None
            self.is_knowledge_base_loaded = False

    def _is_content_cached(self, original_filename: str, content: str) -> bool:
        """检查文件内容是否已缓存（基于文件名和内容哈希）"""
        try:
            content_hash = self._get_content_hash(content)
            
            if original_filename in self.file_hashes:
                stored_hash = self.file_hashes[original_filename]
                if stored_hash == content_hash:
                    logger.info(f"文件 {original_filename} 内容未变化，使用缓存")
                    return True
                else:
                    logger.info(f"文件 {original_filename} 内容已变化，需要重新处理")
            else:
                logger.info(f"文件 {original_filename} 是新文件，需要处理")
        except Exception as e:
            logger.warning(f"检查缓存失败 {original_filename}: {e}")
        return False

    def _update_content_hash(self, original_filename: str, content: str):
        """更新文件内容哈希记录"""
        try:
            content_hash = self._get_content_hash(content)
            self.file_hashes[original_filename] = content_hash
            self._save_file_hashes()
            logger.info(f"已更新文件哈希记录: {original_filename}")
        except Exception as e:
            logger.error(f"更新文件哈希记录失败 {original_filename}: {e}")

    def extract_text_from_pdf(self, file_path):
        """从PDF文件中提取文本"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                if reader.is_encrypted:
                    try:
                        reader.decrypt('')
                    except Exception as decrypt_err:
                        logger.warning("无法解密 PDF %s: %s", os.path.basename(file_path), decrypt_err)
                for page_num in range(len(reader.pages)):
                    try:
                        page = reader.pages[page_num]
                        extracted = page.extract_text()
                        if extracted: text += extracted + "\n"
                    except Exception as page_err:
                        logger.warning(
                            "从 PDF %s 的第 %s 页提取文本时出错: %s", os.path.basename(file_path), page_num + 1, page_err)
            logger.info(f"从 PDF 提取了 {len(text)} 个字符: {os.path.basename(file_path)}")
            return text
        except Exception as e:
            logger.error("处理 PDF 文件 %s 时出错: %s", os.path.basename(file_path), e)
            logger.debug(traceback.format_exc())
            return ""

    def extract_text_from_txt(self, file_path):
        """从TXT文件中提取文本"""
        text = ""
        encodings_to_try = ['utf-8', 'gbk', 'utf-16']
        for enc in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=enc) as file:
                    text = file.read()
                logger.info("使用编码 %s 成功读取 TXT 文件 %s", enc, os.path.basename(file_path))
                return text
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error("使用编码 %s 处理 TXT 文件 %s 时出错: %s", enc, os.path.basename(file_path), e)
                logger.debug(traceback.format_exc())
                return ""
        logger.warning("无法使用尝试的编码解码 TXT 文件 %s。", os.path.basename(file_path))
        return ""

    def process_uploaded_files(self, files):
        """处理上传的文件：提取、分块、并使用Langchain进行向量化"""
        if not files:
            return "未上传任何文件"

        status_texts = []
        logger.info("开始处理 %s 个上传的文件...", len(files))
        
        # 先提取所有文件的文本内容
        extracted_files = []  # (original_filename, content)
        for file_obj in files:
            temp_file_path = getattr(file_obj, 'name', None) or file_obj
            original_file_name = getattr(file_obj, 'orig_name', os.path.basename(str(temp_file_path)))
            file_ext = os.path.splitext(original_file_name)[1].lower()
            
            text = ""
            if file_ext == '.pdf':
                text = self.extract_text_from_pdf(temp_file_path)
            elif file_ext == '.txt':
                text = self.extract_text_from_txt(temp_file_path)
            else:
                status_texts.append(f"❌ 不支持的文件类型: {original_file_name}")
                continue
            
            if text and len(text.strip()) > 50:
                extracted_files.append((original_file_name, text))
            else:
                status_texts.append(f"⚠️ 文件内容为空或过短，已跳过: {original_file_name}")
        
        # 检查哪些文件内容已缓存
        files_to_process = []
        cached_files = []
        
        for original_filename, content in extracted_files:
            if self._is_content_cached(original_filename, content):
                cached_files.append(original_filename)
                status_texts.append(f"✅ 缓存命中，跳过重复向量化: {original_filename}")
            else:
                files_to_process.append((original_filename, content))
                status_texts.append(f"✅ 成功提取文本: {original_filename}")
        
        # 如果有缓存的文件且没有新文件需要处理，直接使用已加载的索引
        if not files_to_process and cached_files and self.is_knowledge_base_loaded:
            status_texts.append(f"📊 使用已缓存的 FAISS 索引，无需重新向量化")
            return "\n".join(status_texts)
        
        # 处理新文件
        raw_docs_content = []
        raw_docs_sources = []
        for original_filename, content in files_to_process:
            raw_docs_content.append(content)
            raw_docs_sources.append(original_filename)
        
        try:
            # 如果有新文件需要处理
            if raw_docs_content:
                # --- 文档分块 - 使用Langchain的RecursiveCharacterTextSplitter ---
                logger.info("使用 Langchain 分块文档...")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=150,
                    length_function=len,
                )

                langchain_docs = []
                for doc_idx, doc_text in enumerate(raw_docs_content):
                    source = raw_docs_sources[doc_idx]
                    doc_text_str = str(doc_text) if doc_text is not None else ""
                    chunks = text_splitter.split_text(doc_text_str)
                    for chunk in chunks:
                        if len(chunk.strip()) > 20:
                            langchain_doc = LangchainDocument(
                                page_content=str(chunk),
                                metadata={"source": str(source)}
                            )
                            langchain_docs.append(langchain_doc)
                            self.document_sources.append(source)

                if langchain_docs:
                    logger.info("将文档分割成 %s 个块。", len(langchain_docs))
                    status_texts.append(f"📄 已将新文档分割为 {len(langchain_docs)} 个文本块")

                    # --- 创建或更新向量存储 ---
                    logger.info("使用 FAISS 更新向量存储...")
                    try:
                        if self.vector_store is None:
                            # 创建新的向量存储
                            self.vector_store = FAISS.from_documents(
                                langchain_docs,
                                self.embeddings
                            )
                        else:
                            # 追加到现有向量存储
                            self.vector_store.add_documents(langchain_docs)
                        
                        logger.info("FAISS 向量存储更新完成")
                        status_texts.append(f"📊 成功更新向量存储")
                        self.is_knowledge_base_loaded = True
                        
                        # 保存到磁盘
                        self._save_faiss_index()
                        
                        # 更新文件哈希记录
                        for original_filename, content in files_to_process:
                            self._update_content_hash(original_filename, content)
                        
                    except Exception as e:
                        error_msg = f"创建向量存储时出错: {str(e)}"
                        status_texts.append(f"❌ {error_msg}")
                        logger.error(error_msg)
                        logger.debug(traceback.format_exc())
                        self.is_knowledge_base_loaded = False
            elif cached_files:
                # 只有缓存的文件，确保向量存储已加载
                if self.is_knowledge_base_loaded:
                    status_texts.append(f"📊 所有文件均使用缓存，无需重新向量化")
                else:
                    status_texts.append(f"⚠️ 缓存文件存在但向量存储未加载，请重新上传")

        except Exception as e:
            error_msg = f"处理文件时发生严重错误: {str(e)}"
            status_texts.append(f"❌ {error_msg}")
            logger.error(error_msg)
            logger.debug(traceback.format_exc())

        status_result = "\n".join(status_texts)
        logger.info("文件处理完成。知识库加载状态: %s", self.is_knowledge_base_loaded)
        if not self.is_knowledge_base_loaded:
            status_result += "\n\n⚠️ 知识库未能成功加载，文档检索功能将不可用。"
        return status_result

    def search_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        在上传的文档中搜索相关内容 (使用Langchain和FAISS)
        """
        logger.info("--- 文档搜索 (Langchain FAISS) ---")
        logger.info("查询: '%s'", query)
        logger.info("知识库加载状态: %s", self.is_knowledge_base_loaded)

        # --- 前置检查 ---
        if not self.is_knowledge_base_loaded or self.vector_store is None:
            logger.warning("搜索中止：知识库向量存储未加载。")
            return []
        if not query or not query.strip():
            logger.warning("搜索中止：查询为空。")
            return []

        try:
            # 使用Langchain的相似度搜索
            logger.info("执行向量搜索，top_k=%s...", top_k)
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

            logger.info("找到 %s 个高于阈值的相关文档块。", len(results))
            # 按相似度降序返回
            results.sort(key=lambda item: item.get('similarity', 0.0), reverse=True)
            return results

        except Exception as e:
            logger.error("文档搜索期间出错: %s", e)
            logger.debug(traceback.format_exc())
            return []

    def format_search_results_as_html(self, results: List[Dict[str, Any]]) -> str:
        """将搜索结果格式化为HTML（使用统一的格式化工具）"""
        return format_doc_results_html(results, self.is_knowledge_base_loaded)


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
            logger.debug(f"实体提取原始响应: '{content}'")

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
                    logger.debug(f"跳过无效实体: {item}")

            logger.debug(f"提取的实体 (已验证): {valid_entities}")
            return valid_entities

        except Exception as e:
            logger.error(f"实体提取过程中出错: {e}")
            traceback.print_exc()
            return []


class Neo4jHandler:
    """处理知识图谱的Neo4j数据库操作"""

    def __init__(self, uri, username, password):
        self.uri = uri;
        self.username = username;
        self.password = password
        self.driver = None;
        self._connect()

    def _connect(self):
        if self.driver:
            self.close()
        try:
            logger.info("尝试连接到 Neo4j 于 %s...", self.uri)
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            with self.driver.session(database="neo4j") as session:
                result = session.run("RETURN 1")
                if result.single()[0] == 1:
                    logger.info("成功连接到 Neo4j 数据库。")
                else:
                    raise Exception("连接测试查询失败。")
        except Exception as e:
            logger.error("连接到 Neo4j 失败: %s", e)
            logger.debug(traceback.format_exc())
            self.driver = None
            raise

    def close(self):
        if self.driver:
            try:
                self.driver.close();
                logger.info("Neo4j 连接已关闭。")
            except Exception as e:
                logger.error(f"关闭 Neo4j 连接时出错: {e}")
            finally:
                self.driver = None

    def _ensure_connection(self):
        if not self.driver:
            logger.warning("Neo4j 驱动程序不可用。正在重新连接...")
            try:
                self._connect()
            except Exception as e:
                logger.error("Neo4j 重新连接失败: %s", e)

    def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        self._ensure_connection()
        if not self.driver: return []
        results = []
        logger.debug(f"执行 Cypher: {query[:150]}... 参数: {params}")
        try:
            with self.driver.session(database="neo4j") as session:
                cypher_result = session.run(query, parameters=params or {})
                results = [record.data() for record in cypher_result]
                logger.info("查询完成。返回 %s 行。", len(results))
        except Exception as e:
            logger.error("执行 Neo4j 查询时出错: %s", e)
            logger.debug(traceback.format_exc())
            self.close()
        return results

    def get_all_entities(self) -> List[Dict[str, Any]]:
        logger.debug("Neo4jHandler: 获取所有实体...")
        query = "MATCH (n) WHERE n.name IS NOT NULL AND n.type IS NOT NULL RETURN n.name AS name, n.type AS type, labels(n) AS labels ORDER BY n.name"
        entities = self.execute_query(query)
        logger.debug(f"Neo4jHandler: get_all_entities 返回了 {len(entities)} 个实体。")
        return entities

    def get_entities_by_name(self, name: str) -> List[Dict[str, Any]]:
        query = "MATCH (n {name: $name}) WHERE n.type IS NOT NULL RETURN n.name AS name, n.type AS type, labels(n) AS labels"
        return self.execute_query(query, {"name": name})

    def get_entities_by_name_and_type(self, name: str, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        if entity_type:
            query = "MATCH (n {name: $name, type: $type}) RETURN n.name AS name, n.type AS type, labels(n) AS labels"
            params = {"name": name, "type": entity_type}
            return self.execute_query(query, params)
        else:
            return self.get_entities_by_name(name)

    def get_entity_relationships(self, entity_name: str) -> Dict[str, Any]:
        entity_info_list = self.get_entities_by_name(entity_name)
        if not entity_info_list: return {}
        entity_info = entity_info_list[0]
        out_query = "MATCH (n {name: $name})-[r]->(m) WHERE m.name IS NOT NULL AND m.type IS NOT NULL RETURN type(r) AS relationship, properties(r) AS rel_props, m.name AS target, m.type AS target_type"
        out_rels_raw = self.execute_query(out_query, {"name": entity_name})
        in_query = "MATCH (m)-[r]->(n {name: $name}) WHERE m.name IS NOT NULL AND m.type IS NOT NULL RETURN m.name AS source, m.type AS source_type, type(r) AS relationship, properties(r) AS rel_props"
        in_rels_raw = self.execute_query(in_query, {"name": entity_name})
        relationships = []
        for rel in out_rels_raw: relationships.append({"direction": "outgoing", "relationship": rel["relationship"],
                                                       "rel_name": rel.get("rel_props", {}).get("name"),
                                                       "target": rel["target"], "target_type": rel["target_type"]})
        for rel in in_rels_raw: relationships.append(
            {"direction": "incoming", "source": rel["source"], "source_type": rel["source_type"],
             "relationship": rel["relationship"], "rel_name": rel.get("rel_props", {}).get("name")})
        return {"entity": entity_info, "relationships": relationships}

    def get_path_between_entities(self, source_name: str, target_name: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        path_query = f"MATCH path = allShortestPaths((src {{name: $source_name}})-[*1..{max_depth}]-(tgt {{name: $target_name}})) WHERE src.name IS NOT NULL AND tgt.name IS NOT NULL RETURN path LIMIT 5"
        path_results = self.execute_query(path_query, {"source_name": source_name, "target_name": target_name})
        paths = []
        processed_paths_count = 0

        for record in path_results:
            path_obj = record.get("path")
            if not path_obj:
                logger.warning("警告: 查询结果中未找到路径值。")
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
                    logger.warning(f"警告: 处理 Path 对象时发生意外错误: {e}。路径值: {path_obj}")
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
                        if i % 2 == 0:  # 偶数索引 - 节点
                            if isinstance(item, dict) and 'name' in item:
                                temp_nodes.append({
                                    "name": item.get("name", "?"),
                                    "type": item.get("type", "?"),
                                    "labels": item.get("labels", [])  # 尝试获取 labels
                                })
                            else:
                                logger.warning(f"警告: 路径列表在索引 {i} 处期望节点字典，但找到: {type(item)}。跳过此路径。")
                                valid_path_structure = False
                                break
                        else:  # 奇数索引 - 关系 (类型字符串)
                            if isinstance(item, str):
                                temp_rels.append(item)
                            else:
                                logger.warning(f"警告: 路径列表在索引 {i} 处期望关系字符串，但找到: {type(item)}。跳过此路径。")
                                valid_path_structure = False
                                break

                    if valid_path_structure:
                        path_info["nodes"] = temp_nodes
                        for i, rel_type in enumerate(temp_rels):
                            source_node = temp_nodes[i]
                            target_node = temp_nodes[i + 1]
                            path_info["relationships"].append({
                                "source": source_node.get("name", "?"),
                                "target": target_node.get("name", "?"),
                                "type": rel_type,
                                "name": rel_type  # 使用类型作为备用名称
                            })
                        paths.append(path_info)
                        processed_paths_count += 1

                except Exception as e:
                    logger.warning(f"警告: 解析列表表示的路径时发生错误: {e}。列表值: {path_obj}")
                    traceback.print_exc()
                    continue
            else:
                # --- 处理其他未知类型 ---
                logger.warning(
                    f"返回的路径值既不是 Path 对象，也不是预期的列表结构。类型: {type(path_obj)}，值: {path_obj}。跳过此路径。")
                continue

        logger.debug(f"在 '{source_name}' 和 '{target_name}' 之间成功处理了 {processed_paths_count} 条路径。")
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
        logger.info("EntityMatcher: 加载 KG 实体...")
        all_entities_from_db = self.neo4j_handler.get_all_entities()
        self.kg_entities = [e for e in all_entities_from_db if e.get("name")]
        logger.info("EntityMatcher: 存储了 %s 个带名称的实体。", len(self.kg_entities))
        if not self.kg_entities: return

        entity_names = [entity["name"] for entity in self.kg_entities]
        logger.info("EntityMatcher: 确保 %s 个名称的嵌入...", len(entity_names))
        self._ensure_embeddings(entity_names)
        logger.debug("EntityMatcher: KG 实体已加载并确保嵌入。")

    def _load_embeddings_cache(self) -> Dict[str, list]:
        """从文件加载实体嵌入缓存"""
        cache = {}
        if os.path.exists(self.embedding_cache_path):
            try:
                with open(self.embedding_cache_path, 'rb') as f:
                    cache = pickle.load(f)
                logger.debug(f"从缓存加载了 {len(cache)} 个嵌入: {self.embedding_cache_path}")
            except Exception as e:
                logger.error(f"加载嵌入缓存时出错: {e}")
        return cache

    def _save_embeddings_cache(self):
        """保存实体嵌入缓存到文件"""
        try:
            logger.info(f"保存 {len(self.embeddings_cache)} 个嵌入到缓存: {self.embedding_cache_path}...")
            with open(self.embedding_cache_path, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            logger.info("嵌入已保存。")
        except Exception as e:
            logger.error(f"保存嵌入缓存时出错: {e}")

    def get_embedding(self, text: str, force_refresh=False) -> Optional[list]:
        """获取文本的嵌入，使用缓存"""
        if not text or not text.strip():
            return None

        cache_key = text.strip()
        if not force_refresh and cache_key in self.embeddings_cache:
            return self.embeddings_cache[cache_key]

        logger.info("为 '%s' 生成嵌入", cache_key)
        try:
            embedding = self.embeddings.embed_query(cache_key)
            self.embeddings_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            logger.error("获取 '%s' 的嵌入时出错: %s", cache_key, e)
            logger.debug(traceback.format_exc())
            return None

    def _ensure_embeddings(self, texts: List[str]):
        """确保所有文本在缓存中都有嵌入"""
        unique_texts = list(set(filter(None, texts)))
        texts_to_embed = [text for text in unique_texts if text not in self.embeddings_cache]

        if not texts_to_embed:
            logger.debug("所有必需的 KG 实体嵌入都已缓存。")
            return

        logger.info(f"为 {len(texts_to_embed)} 个新的 KG 实体生成嵌入...")
        logger.debug(f"示例实体名称: {texts_to_embed[:3] if len(texts_to_embed) > 0 else '无'}")
        logger.debug(f"示例实体名称类型: {[type(text) for text in texts_to_embed[:3]]}")

        # 确保所有文本都是字符串类型
        texts_to_embed = [str(text) if text is not None else "" for text in texts_to_embed]

        # 使用批处理来提高效率
        batch_size = 10
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:min(i + batch_size, len(texts_to_embed))]
            try:
                logger.debug(f"处理批次: {batch}")
                embeddings = self.embeddings.embed_documents(batch)
                for j, text in enumerate(batch):
                    self.embeddings_cache[text] = embeddings[j]
            except Exception as e:
                logger.error(f"生成批次嵌入时出错: {e}")
                logger.debug(f"错误批次内容: {batch}")

        logger.debug(f"生成了 {len(texts_to_embed)} 个新嵌入。")
        self._save_embeddings_cache()

    def match_entity(self, entity: Dict[str, str], similarity_threshold=0.85) -> List[Dict[str, Any]]:
        """将提取的实体与已加载的 KG 实体列表进行匹配"""
        entity_name = entity.get("name")
        entity_type = entity.get("type")

        if not entity_name:
            return []

        logger.debug(f"--- 实体匹配 ---")
        logger.debug(f"尝试匹配: '{entity_name}' (类型: {entity_type})")
        logger.debug(f"与 {len(self.kg_entities)} 个已加载的 KG 实体进行比较。")

        if not self.kg_entities:
            logger.warning("警告：KG 实体列表为空。")
            return []

        # 步骤 1: 精确名称匹配
        logger.debug("步骤 1: 精确名称匹配 (在加载的列表中)...")
        exact_matches = [kg for kg in self.kg_entities if
                         kg.get("name") == entity_name and (not entity_type or kg.get("type") == entity_type)]
        if exact_matches:
            logger.debug(f"找到 {len(exact_matches)} 个精确匹配。")
            return [{"entity": match, "match_type": "exact", "score": 1.0} for match in exact_matches]

        # 步骤 2: 向量相似度匹配
        logger.debug(f"步骤 2: 向量相似度匹配 (阈值: {similarity_threshold})...")
        entity_embedding = self.get_embedding(entity_name)
        if not entity_embedding:
            logger.warning("无法获取查询嵌入。")
            return []

        matches = []
        logger.debug(f"与 {len(self.kg_entities)} 个 KG 实体比较向量...")

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
                    logger.warning(f"计算 '{kg_entity_name}' 的相似度时出错: {e}")

        matches.sort(key=lambda x: x["score"], reverse=True)
        top_matches = matches[:5]
        logger.debug(f"找到 {len(top_matches)} 个高于阈值的向量匹配。")
        return top_matches


class ResponseGenerator:
    """使用LLM、知识图谱上下文和文档检索结果生成响应"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.logger = logging.getLogger(__name__)

    def generate_response(self, question: str, kg_context: Dict[str, Any],
                          entities: List[Dict[str, str]], doc_results: List[Dict[str, Any]]) -> str:
        """使用LLM生成响应"""
        self.logger.info("--- 响应生成 ---")
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
            logger.debug("调用 LLM 生成响应...")

            messages = [
                SystemMessage(
                    content="你是专业的软件工程课程助手，精通各种软件工程概念、方法、工具和最佳实践。你的回答简洁明了、专业权威，并能根据学生的问题提供恰当的解释和指导。"),
                HumanMessage(content=prompt_template.format(
                    question=question,
                    entities=entities_str,
                    kg_context=kg_context_str or "无相关知识图谱信息。",
                    doc_context=doc_context_str or "无相关文档片段。"
                ))
            ]

            response = self.llm.invoke(messages)
            final_answer = response.content
            logger.debug(f"LLM 响应已接收。")
            return final_answer

        except Exception as e:
            logger.error(f"调用 LLM 时出错: {e}")
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
                            formatted_parts.append(
                                f"    - {entity_name} --[{rel_name}]--> {target} (类型: {target_type})")
                        elif rel.get("direction") == "incoming":
                            source = rel.get("source", "?")
                            source_type = rel.get("source_type", "?")
                            formatted_parts.append(
                                f"    - {source} (类型: {source_type}) --[{rel_name}]--> {entity_name}")

        if kg_context.get("paths"):
            formatted_parts.append("\n--- 知识图谱实体间路径 ---")
            for i, path_data in enumerate(kg_context["paths"]):
                nodes = path_data.get("nodes", [])
                rels = path_data.get("relationships", [])

                if not nodes or not rels:
                    continue

                path_str = f"\n路径 {i + 1}: "
                elements = []

                for j, node in enumerate(nodes):
                    elements.append(f"({node.get('name', '?')})")
                    if j < len(rels):
                        rel = rels[j]
                        rel_name = rel.get('name') or rel.get('type', '?')
                        elements.append(
                            f"--[{rel_name}]-->" if rel.get('source') == node.get('name') else f"<--[{rel_name}]--")

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
            formatted_parts.append(f"\n片段 {i + 1} (来源: {source}, Sim: {similarity:.3f}):\n---\n{truncated}\n---")

        return "\n".join(formatted_parts)


class SoftwareEngineeringQASystem:
    """整合知识图谱、本地文档与LLM的软件工程问答系统"""

    def __init__(self, neo4j_uri, neo4j_username, neo4j_password):
        """初始化QA系统"""
        logger.info("初始化 SoftwareEngineeringQASystem...")

        # 优先从环境变量加载 API 密钥与 base_url，若无则使用默认值
        self.openai_api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if not self.openai_api_key:
            logger.warning("未找到 API 密钥，部分功能可能无法正常工作。")

        # 设置 Langchain 组件
        self.llm = self._setup_llm()
        self.embeddings = self._setup_embeddings()

        # 设置 Neo4j
        try:
            self.neo4j_handler = Neo4jHandler(neo4j_uri, neo4j_username, neo4j_password)
        except Exception as e:
            logger.error("致命错误：初始化 Neo4j Handler 失败: %s", e)
            logger.debug(traceback.format_exc())
            raise

        # 初始化 RAG 管理器（如果有 embeddings）
        if self.embeddings:
            logger.info("初始化 RAGManager...")
            self.rag_manager = RAGManager.get_instance(embeddings=self.embeddings)
            logger.info("RAGManager 初始化完成。")
        else:
            self.rag_manager = None
            logger.warning("未设置 embeddings，RAG 功能将不可用。")

        # 实体匹配器（如果有 embeddings）
        if self.embeddings:
            logger.info("初始化 EntityMatcher...")
            self.entity_matcher = EntityMatcher(self.embeddings, self.neo4j_handler)
            logger.info("EntityMatcher 初始化完成，开始加载 KG 实体...")
            try:
                self.entity_matcher.load_and_cache_kg_entities()
                logger.info("KG 实体加载完成。")
            except Exception as e:
                logger.warning("初始 KG 实体加载期间出错: %s", e)
                logger.debug(traceback.format_exc())
        else:
            self.entity_matcher = None
            logger.warning("未设置 embeddings，实体匹配功能将不可用。")

        # 其他组件（如果有 LLM）
        if self.llm:
            self.entity_extractor = EntityExtractor(self.llm)
            self.response_generator = ResponseGenerator(self.llm)  # 保留原始生成器用于回退
            self.agent_coordinator = AgentCoordinator(self.llm)
        else:
            self.entity_extractor = None
            self.response_generator = None
            self.agent_coordinator = None
            logger.warning("未设置 LLM，问答功能将不可用。")

        # UI 状态
        self.current_kg_context = {"entities": [], "paths": []}
        self.current_doc_results_raw = []
        self.current_agent_name = None  # 跟踪使用的智能体

        # 添加对话历史管理
        self.conversation_history = deque(maxlen=10)  # 最多保留10轮对话

        logger.info("SoftwareEngineeringQASystem 初始化成功。")

    def _setup_llm(self) -> Optional[Any]:
        """设置Langchain的LLM组件"""
        if not self.openai_api_key:
            logger.warning("未设置 API 密钥，跳过 LLM 初始化。")
            return None
            
        try:
            try:
                # 尝试使用通义千问的 langchain 集成
                llm = ChatTongyi(
                    api_key=self.openai_api_key,
                    temperature=0.5,
                    model="qwen-plus"
                )
                logger.info("Langchain ChatTongyi 设置成功。")
            except Exception:
                # 回退到使用 OpenAI 兼容接口
                llm = ChatOpenAI(
                    api_key=self.openai_api_key,
                    base_url=self.openai_base_url,
                    temperature=0.5,
                    model="qwen-plus"
                )
                logger.info("Langchain ChatOpenAI (兼容模式) 设置成功。")
            return llm
        except Exception as e:
            logger.error("设置 LLM 失败: %s", e)
            logger.debug(traceback.format_exc())
            return None

    def _setup_embeddings(self) -> Optional[Embeddings]:
        """设置Langchain的Embeddings组件"""
        if not self.openai_api_key:
            logger.warning("未设置 API 密钥，跳过 Embeddings 初始化。")
            return None
            
        try:
            try:
                # 尝试使用通义千问的 langchain 集成
                embeddings = DashScopeEmbeddings(
                    api_key=self.openai_api_key,
                    model=RAG_EMBEDDING_MODEL
                )
                logger.info("Langchain DashScopeEmbeddings 设置成功。")
            except Exception:
                # 回退到使用 OpenAI 兼容接口，但需要注意：通义千问的 text-embedding-v3 需要特殊处理
                logger.warning("DashScopeEmbeddings 导入失败，尝试直接使用 dashscope SDK...")
                # 创建一个自定义的 Embeddings 类，直接使用 dashscope SDK
                from langchain.embeddings.base import Embeddings as BaseEmbeddings
                
                class CustomDashScopeEmbeddings(BaseEmbeddings):
                    def __init__(self, api_key: str, model: str):
                        self.api_key = api_key
                        self.model = model
                        import dashscope
                        dashscope.api_key = api_key
                    
                    def embed_documents(self, texts: List[str]) -> List[List[float]]:
                        import dashscope
                        results = []
                        for text in texts:
                            try:
                                resp = dashscope.TextEmbedding.call(
                                    model=self.model,
                                    input=text
                                )
                                if resp.status_code == 200:
                                    results.append(resp.output['embeddings'][0]['embedding'])
                                elif resp.status_code == 403 and hasattr(resp, 'code') and resp.code == 'AllocationQuota.FreeTierOnly':
                                    raise Exception("通义千问 API 错误：免费额度已用完。"
                                                    "请在管理控制台关闭'仅使用免费额度'模式，或使用其他 API 密钥。")
                                else:
                                    raise Exception(f"获取嵌入失败 (状态码 {resp.status_code}): {resp}")
                            except Exception as e:
                                if "FreeTierOnly" in str(e):
                                    raise e
                                raise Exception(f"调用通义千问嵌入 API 时出错: {str(e)}")
                        return results
                    
                    def embed_query(self, text: str) -> List[float]:
                        return self.embed_documents([text])[0]
                
                embeddings = CustomDashScopeEmbeddings(
                    api_key=self.openai_api_key,
                    model=RAG_EMBEDDING_MODEL
                )
                logger.info("Custom DashScopeEmbeddings 设置成功。")
            return embeddings
        except Exception as e:
            logger.error("设置 Embeddings 失败: %s", e)
            logger.debug(traceback.format_exc())
            return None

    def answer_question(self, question: str) -> dict:
        """处理问题，整合 KG 和 RAG 生成答案，支持多轮对话"""
        logger.info("===== 回答问题 =====")
        logger.info("问题: %s", question)
        logger.info("对话历史长度: %s", len(self.conversation_history))

        self.current_kg_context = {"entities": [], "paths": []}
        self.current_doc_results_raw = []
        self.current_agent_name = None

        # 1. 提取实体
        logger.info("\n步骤 1: 提取实体...")
        entities = self.entity_extractor.extract_entities(question)
        if not entities:
            logger.info("未提取到实体。")

        # 2. 匹配实体并构建 KG 上下文
        logger.info("\n步骤 2: 匹配实体并检索 KG 上下文...")
        all_matched_kg_entity_names = set()

        if entities:
            for entity in entities:
                entity_name = entity.get("name")
                if not entity_name:
                    continue

                logger.info("匹配实体: '%s' (类型: %s)", entity_name, entity.get('type'))
                matches = self.entity_matcher.match_entity(entity)

                if matches:
                    top_match = matches[0]
                    kg_entity_name = top_match["entity"]["name"]
                    logger.info("最佳匹配: '%s' (分数: %.3f, 方法: %s)", kg_entity_name, top_match['score'], top_match['match_type'])
                    entity_data = self.neo4j_handler.get_entity_relationships(kg_entity_name)

                    if entity_data:
                        self.current_kg_context["entities"].append(entity_data)
                        all_matched_kg_entity_names.add(kg_entity_name)
                else:
                    logger.debug("未找到 KG 匹配。")

        logger.info("为 %s 个匹配的实体找到 KG 上下文。", len(self.current_kg_context['entities']))

        # 3. 查找实体之间的路径
        if len(all_matched_kg_entity_names) > 1:
            logger.info("\n步骤 3: 查找匹配实体之间的路径...")
            entity_list = list(all_matched_kg_entity_names)

            for i in range(len(entity_list)):
                for j in range(i + 1, len(entity_list)):
                    source, target = entity_list[i], entity_list[j]
                    logger.debug("查找 '%s' 和 '%s' 之间的路径...", source, target)
                    paths = self.neo4j_handler.get_path_between_entities(source, target)
                    if paths:
                        self.current_kg_context["paths"].extend(paths)

        # 4. 文档检索 (RAG) - 使用Langchain
        logger.info("\n步骤 4: 检索相关文档 (RAG)...")
        self.current_doc_results_raw = self.rag_manager.search_documents(question)
        logger.info("检索到 %s 个相关文档块。", len(self.current_doc_results_raw))

        # 5. 使用智能体协调器生成最终响应
        logger.info("\n步骤 5: 使用智能体生成最终响应...")
        has_kg = bool(self.current_kg_context.get("entities") or self.current_kg_context.get("paths"))
        has_docs = bool(self.current_doc_results_raw)

        if not has_kg and not has_docs:
            logger.warning("未找到 KG 上下文或相关文档。生成回退响应。")
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

        logger.info("===== 答案生成完成 (由 %s 提供) =====" , self.current_agent_name)

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
        logger.warning("生成回退响应。原因: %s", reason)

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
                SystemMessage(
                    content="你是专业的软件工程课程助手，精通各种软件工程概念、方法、工具和最佳实践。你会根据对话历史提供连贯的回答。"),
                HumanMessage(content=prompt_template.format(
                    question=question,
                    history=history_text
                ))
            ]

            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            logger.error("生成回退响应时出错: %s", e)
            logger.debug(traceback.format_exc())
            return f"抱歉，我无法回答这个问题（{reason}）。尝试生成通用回答时也出错：{e}"

    def clear_conversation_history(self):
        """清除对话历史"""
        self.conversation_history.clear()
        logger.info("对话历史已清除。")

    def get_conversation_history(self) -> List[Tuple[str, str]]:
        """获取对话历史"""
        return list(self.conversation_history)

    def close(self):
        """关闭资源"""
        logger.info("关闭 SoftwareEngineeringQASystem 资源...")
        # Neo4j连接关闭
        if hasattr(self, 'neo4j_handler'):
            self.neo4j_handler.close()
        logger.info("系统已关闭。")


# --- 导出供UI使用的函数 ---

qa_system_instance: Optional[SoftwareEngineeringQASystem] = None


def initialize_qa_system(uri, user, pwd):
    """初始化全局QA系统实例 (供UI调用)"""
    global qa_system_instance
    if qa_system_instance is None:
        logger.info("UI 正在初始化 QA 系统...")
        try:
            qa_system_instance = SoftwareEngineeringQASystem(uri, user, pwd)
        except Exception as e:
            logger.error("致命错误：在 UI 中初始化 QA 系统失败: %s", e)
            logger.debug(traceback.format_exc())
            raise
    return qa_system_instance


def get_qa_system_instance() -> SoftwareEngineeringQASystem:
    """获取全局QA系统实例"""
    if qa_system_instance is None:
        raise RuntimeError("QA 系统尚未初始化。")
    return qa_system_instance


def process_uploaded_files(files):
    """UI调用的文档处理函数"""
    logger.debug("UI 触发文件处理...")
    try:
        # 确保 QA 系统已初始化
        qa_system = get_qa_system_instance()
        rag_manager = qa_system.rag_manager
        if rag_manager is None:
            return "❌ 错误：RAG 管理器未初始化。请检查 API 密钥配置是否正确。"
        return rag_manager.process_uploaded_files(files)
    except RuntimeError as e:
        return f"❌ 系统错误：{str(e)}"
    except Exception as e:
        logger.error(f"处理文件上传时出错: {e}")
        logger.debug(traceback.format_exc())
        return f"❌ 文件上传处理错误: {str(e)}"


def search_documents(query, top_k=3):
    """UI调用的文档搜索函数 (仅用于独立显示结果)"""
    logger.debug(f"UI 触发文档搜索，查询: '{query}'")
    # 确保 QA 系统已初始化
    qa_system = get_qa_system_instance()
    rag_manager = qa_system.rag_manager
    results_raw = rag_manager.search_documents(query, top_k)
    results_html = rag_manager.format_search_results_as_html(results_raw)
    return results_html


if __name__ == "__main__":
    # 测试代码可以在这里添加
    pass
