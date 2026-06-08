# -*- coding: utf-8 -*-
"""
重构后的软件工程问答系统
集成了改进的错误处理、配置管理、性能监控和连接池
"""
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
from datetime import datetime

from dotenv import load_dotenv

import pypdf
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangchainDocument
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 导入新的核心模块
from core import (
    QASystemConfig, ErrorHandler, ErrorContext, ErrorType, ErrorSeverity,
    RAGException, LLMException, Neo4jException,
    PerformanceMonitor, PerformanceMetrics,
    ConnectionPool, ConnectionPoolException
)
from models import Entity, QuestionInput, AnswerOutput, SearchResult, FileUploadResult

# 导入原始模块中需要的部分
# 使用 importlib 来避免与 agents 目录冲突
import importlib.util
import os
current_file_dir = os.path.dirname(os.path.abspath(__file__))
agents_py_path = os.path.join(current_file_dir, 'agents.py')
spec = importlib.util.spec_from_file_location('agents_module', agents_py_path)
agents_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agents_module)

# 从 agents 模块导入需要的类
AgentCoordinator = getattr(agents_module, 'AgentCoordinator')

from config import get_log_config
from utils import setup_logger
from utils.formatters import format_search_results_as_html as format_doc_results_html

# 配置日志
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

# 嵌入模型配置
RAG_EMBEDDING_MODEL = "text-embedding-v4"
RAG_EMBEDDING_DIMENSION = 1024
ENTITY_EMBEDDING_MODEL = "text-embedding-v4"
ENTITY_EMBEDDING_DIMENSION = 1024


class RAGManager:
    """
    管理文档处理和检索的类 (使用Langchain和FAISS)
    优化：集成了改进的错误处理和性能监控
    """
    _instance = None
    _initialized = False
    _lock = threading.Lock()

    def __init__(self, embeddings: Embeddings, error_handler: ErrorHandler = None, perf_monitor: PerformanceMonitor = None):
        """
        初始化RAG管理器
        """
        if RAGManager._initialized:
            return

        self.embeddings = embeddings
        self.vector_store = None
        self.document_sources = []
        self.is_knowledge_base_loaded = False
        
        self.error_handler = error_handler or ErrorHandler(logger)
        self.perf_monitor = perf_monitor or PerformanceMonitor(logger)
        
        # FAISS 持久化相关
        self.faiss_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_cache")
        self.file_hashes_file = os.path.join(self.faiss_cache_dir, "file_hashes.pkl")
        self.faiss_index_file = os.path.join(self.faiss_cache_dir, "faiss_index")
        self.file_hashes = {}
        
        os.makedirs(self.faiss_cache_dir, exist_ok=True)
        self._load_file_hashes()
        self._load_faiss_index()
        
        RAGManager._initialized = True
        logger.info("RAGManager 初始化完成")

    @classmethod
    def get_instance(cls, embeddings: Embeddings = None, error_handler: ErrorHandler = None, perf_monitor: PerformanceMonitor = None):
        """获取RAGManager的单例实例"""
        with cls._lock:
            if cls._instance is None:
                if embeddings is None:
                    logger.warning("未提供 embeddings，RAGManager 实例将为 None。")
                    return None
                cls._instance = cls(embeddings, error_handler, perf_monitor)
            return cls._instance

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
            logger.info(f"文件哈希记录已保存到 {self.file_hashes_file}")
        except Exception as e:
            logger.error(f"保存文件哈希记录失败: {e}")

    def _load_file_hashes(self):
        """从磁盘加载文件哈希记录"""
        try:
            if os.path.exists(self.file_hashes_file):
                with open(self.file_hashes_file, "rb") as f:
                    self.file_hashes = pickle.load(f)
                logger.info(f"已加载 {len(self.file_hashes)} 个文件哈希记录")
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
        except Exception as e:
            logger.warning(f"加载 FAISS 索引失败: {e}")
            self.vector_store = None
            self.is_knowledge_base_loaded = False

    def _is_content_cached(self, original_filename: str, content: str) -> bool:
        """检查文件内容是否已缓存"""
        try:
            content_hash = self._get_content_hash(content)
            if original_filename in self.file_hashes:
                stored_hash = self.file_hashes[original_filename]
                if stored_hash == content_hash:
                    logger.info(f"文件 {original_filename} 内容未变化，使用缓存")
                    return True
        except Exception as e:
            logger.warning(f"检查缓存失败 {original_filename}: {e}")
        return False

    def _update_content_hash(self, original_filename: str, content: str):
        """更新文件内容哈希记录"""
        try:
            content_hash = self._get_content_hash(content)
            self.file_hashes[original_filename] = content_hash
            self._save_file_hashes()
        except Exception as e:
            logger.error(f"更新文件哈希记录失败: {e}")

    def extract_text_from_pdf(self, file_path):
        """从PDF文件中提取文本"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                if reader.is_encrypted:
                    try:
                        reader.decrypt('')
                    except Exception as decrypt_err:
                        logger.warning(f"无法解密 PDF: {decrypt_err}")
                for page_num in range(len(reader.pages)):
                    try:
                        page = reader.pages[page_num]
                        extracted = page.extract_text()
                        if extracted: 
                            text += extracted + "\n"
                    except Exception as page_err:
                        logger.warning(f"从 PDF 第 {page_num + 1} 页提取文本时出错: {page_err}")
            logger.info(f"从 PDF 提取了 {len(text)} 个字符")
            return text
        except Exception as e:
            error_ctx = ErrorContext(
                error_type=ErrorType.RAG_INDEXING,
                severity=ErrorSeverity.ERROR,
                message=f"处理 PDF 文件时出错: {e}",
                component="RAGManager.extract_text_from_pdf",
                timestamp=datetime.now(),
                exception=e
            )
            self.error_handler.handle(error_ctx)
            return ""

    def extract_text_from_txt(self, file_path):
        """从TXT文件中提取文本"""
        text = ""
        encodings_to_try = ['utf-8', 'gbk', 'utf-16']
        for enc in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=enc) as file:
                    text = file.read()
                logger.info(f"使用编码 {enc} 成功读取 TXT 文件")
                return text
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"使用编码 {enc} 处理 TXT 文件时出错: {e}")
                return ""
        logger.warning(f"无法使用尝试的编码解码 TXT 文件")
        return ""

    def process_uploaded_files(self, files) -> str:
        """处理上传的文件"""
        if not files:
            return "未上传任何文件"

        status_texts = []
        logger.info(f"开始处理 {len(files)} 个上传的文件...")
        
        with self.perf_monitor.measure("文件处理"):
            # 先提取所有文件的文本内容
            extracted_files = []
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
            
            # 处理新文件
            if files_to_process:
                raw_docs_content = []
                raw_docs_sources = []
                for original_filename, content in files_to_process:
                    raw_docs_content.append(content)
                    raw_docs_sources.append(original_filename)
                
                try:
                    # 文档分块
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800,
                        chunk_overlap=150,
                        length_function=len,
                    )
                    langchain_docs = []
                    for doc_idx, doc_text in enumerate(raw_docs_content):
                        source = raw_docs_sources[doc_idx]
                        chunks = text_splitter.split_text(str(doc_text))
                        for chunk in chunks:
                            if len(chunk.strip()) > 20:
                                langchain_doc = LangchainDocument(
                                    page_content=str(chunk),
                                    metadata={"source": str(source)}
                                )
                                langchain_docs.append(langchain_doc)
                                self.document_sources.append(source)
                    
                    if langchain_docs:
                        status_texts.append(f"📄 已将新文档分割为 {len(langchain_docs)} 个文本块")
                        
                        # 创建或更新向量存储
                        if self.vector_store is None:
                            self.vector_store = FAISS.from_documents(langchain_docs, self.embeddings)
                        else:
                            self.vector_store.add_documents(langchain_docs)
                        
                        status_texts.append(f"📊 成功更新向量存储")
                        self.is_knowledge_base_loaded = True
                        
                        self._save_faiss_index()
                        for original_filename, content in files_to_process:
                            self._update_content_hash(original_filename, content)
                except Exception as e:
                    error_ctx = ErrorContext(
                        error_type=ErrorType.RAG_INDEXING,
                        severity=ErrorSeverity.ERROR,
                        message=f"创建向量存储时出错: {e}",
                        component="RAGManager.process_uploaded_files",
                        timestamp=datetime.now(),
                        exception=e
                    )
                    self.error_handler.handle(error_ctx)
                    status_texts.append(f"❌ 创建向量存储时出错: {str(e)}")
            elif cached_files and self.is_knowledge_base_loaded:
                status_texts.append(f"📊 使用已缓存的 FAISS 索引，无需重新向量化")
        
        return "\n".join(status_texts)

    def search_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """在上传的文档中搜索相关内容"""
        with self.perf_monitor.measure("文档检索"):
            if not self.is_knowledge_base_loaded or self.vector_store is None:
                logger.warning("搜索中止：知识库向量存储未加载。")
                return []
            if not query or not query.strip():
                logger.warning("搜索中止：查询为空。")
                return []

            try:
                docs_with_scores = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=top_k
                )
                results = []
                min_similarity_threshold = 0.3

                for doc, score in docs_with_scores:
                    similarity = 1.0 / (1.0 + score)
                    if similarity > min_similarity_threshold:
                        results.append({
                            "source": doc.metadata.get("source", "未知来源"),
                            "text": doc.page_content,
                            "similarity": float(similarity)
                        })

                logger.info(f"找到 {len(results)} 个高于阈值的相关文档块。")
                results.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)
                return results
            except Exception as e:
                error_ctx = ErrorContext(
                    error_type=ErrorType.RAG_SEARCH,
                    severity=ErrorSeverity.ERROR,
                    message=f"文档搜索时出错: {e}",
                    component="RAGManager.search_documents",
                    timestamp=datetime.now(),
                    exception=e
                )
                self.error_handler.handle(error_ctx)
                return []

    def format_search_results_as_html(self, results: List[Dict[str, Any]]) -> str:
        """将搜索结果格式化为HTML"""
        return format_doc_results_html(results, self.is_knowledge_base_loaded)


class Neo4jHandler:
    """
    处理知识图谱的Neo4j数据库操作
    优化：支持连接池和改进的错误处理
    """

    def __init__(self, uri, username, password, connection_pool: ConnectionPool = None, error_handler: ErrorHandler = None):
        """初始化 Neo4jHandler"""
        self.uri = uri
        self.username = username
        self.password = password
        self.error_handler = error_handler or ErrorHandler(logger)
        
        if connection_pool:
            self.connection_pool = connection_pool
            self.driver = None
        else:
            self.connection_pool = None
            self.driver = None
            self._connect()

    def _connect(self):
        """建立到 Neo4j 的连接"""
        try:
            from neo4j import GraphDatabase
            logger.info(f"尝试连接到 Neo4j 于 {self.uri}...")
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            with self.driver.session(database="neo4j") as session:
                result = session.run("RETURN 1")
                if result.single()[0] == 1:
                    logger.info("成功连接到 Neo4j 数据库。")
        except Exception as e:
            error_ctx = ErrorContext(
                error_type=ErrorType.NEO4J_CONNECTION,
                severity=ErrorSeverity.CRITICAL,
                message=f"连接到 Neo4j 失败: {e}",
                component="Neo4jHandler._connect",
                timestamp=datetime.now(),
                exception=e
            )
            self.error_handler.handle(error_ctx)
            raise

    def close(self):
        """关闭资源"""
        if self.driver:
            try:
                self.driver.close()
                logger.info("Neo4j 连接已关闭。")
            except Exception as e:
                logger.error(f"关闭 Neo4j 连接时出错: {e}")
        if self.connection_pool:
            self.connection_pool.close_all()

    def _ensure_connection(self):
        """确保连接可用"""
        if not self.driver and not self.connection_pool:
            logger.warning("Neo4j 连接不可用。正在重新连接...")
            try:
                self._connect()
            except Exception as e:
                error_ctx = ErrorContext(
                    error_type=ErrorType.NEO4J_CONNECTION,
                    severity=ErrorSeverity.ERROR,
                    message=f"重新连接 Neo4j 失败: {e}",
                    component="Neo4jHandler._ensure_connection",
                    timestamp=datetime.now(),
                    exception=e
                )
                self.error_handler.handle(error_ctx)

    def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """执行 Cypher 查询"""
        self._ensure_connection()
        if not self.driver and not self.connection_pool:
            return []
        
        results = []
        try:
            session = None
            if self.connection_pool:
                conn = self.connection_pool.get_connection()
                session = conn.session(database="neo4j")
            else:
                session = self.driver.session(database="neo4j")
            
            with session:
                cypher_result = session.run(query, parameters=params or {})
                results = [record.data() for record in cypher_result]
                logger.debug(f"查询完成。返回 {len(results)} 行。")
        except Exception as e:
            error_ctx = ErrorContext(
                error_type=ErrorType.NEO4J_QUERY,
                severity=ErrorSeverity.ERROR,
                message=f"执行 Neo4j 查询时出错: {e}",
                component="Neo4jHandler.execute_query",
                timestamp=datetime.now(),
                exception=e
            )
            self.error_handler.handle(error_ctx)
            if self.driver:
                self.close()
        
        return results

    def get_all_entities(self) -> List[Dict[str, Any]]:
        """获取所有实体"""
        logger.debug("Neo4jHandler: 获取所有实体...")
        query = "MATCH (n) WHERE n.name IS NOT NULL AND n.type IS NOT NULL RETURN n.name AS name, n.type AS type, labels(n) AS labels ORDER BY n.name"
        entities = self.execute_query(query)
        logger.debug(f"Neo4jHandler: get_all_entities 返回了 {len(entities)} 个实体。")
        return entities

    def get_entity_relationships(self, entity_name: str) -> Dict[str, Any]:
        """获取实体的关系"""
        entity_info_list = self.execute_query(
            "MATCH (n {name: $name}) WHERE n.type IS NOT NULL RETURN n.name AS name, n.type AS type, labels(n) AS labels",
            {"name": entity_name}
        )
        if not entity_info_list:
            return {}
        
        entity_info = entity_info_list[0]
        out_query = "MATCH (n {name: $name})-[r]->(m) WHERE m.name IS NOT NULL AND m.type IS NOT NULL RETURN type(r) AS relationship, properties(r) AS rel_props, m.name AS target, m.type AS target_type"
        out_rels_raw = self.execute_query(out_query, {"name": entity_name})
        in_query = "MATCH (m)-[r]->(n {name: $name}) WHERE m.name IS NOT NULL AND m.type IS NOT NULL RETURN m.name AS source, m.type AS source_type, type(r) AS relationship, properties(r) AS rel_props"
        in_rels_raw = self.execute_query(in_query, {"name": entity_name})
        
        relationships = []
        for rel in out_rels_raw:
            relationships.append({
                "direction": "outgoing",
                "relationship": rel["relationship"],
                "rel_name": rel.get("rel_props", {}).get("name"),
                "target": rel["target"],
                "target_type": rel["target_type"]
            })
        for rel in in_rels_raw:
            relationships.append({
                "direction": "incoming",
                "source": rel["source"],
                "source_type": rel["source_type"],
                "relationship": rel["relationship"],
                "rel_name": rel.get("rel_props", {}).get("name")
            })
        
        return {"entity": entity_info, "relationships": relationships}

    def get_path_between_entities(self, source_name: str, target_name: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """获取实体之间的路径"""
        path_query = f"MATCH path = allShortestPaths((src {{name: $source_name}})-[*1..{max_depth}]-(tgt {{name: $target_name}})) WHERE src.name IS NOT NULL AND tgt.name IS NOT NULL RETURN path LIMIT 5"
        path_results = self.execute_query(path_query, {"source_name": source_name, "target_name": target_name})
        paths = []
        
        for record in path_results:
            path_obj = record.get("path")
            if not path_obj:
                continue
            
            if hasattr(path_obj, 'nodes') and hasattr(path_obj, 'relationships'):
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
                except Exception as e:
                    logger.warning(f"处理路径对象时出错: {e}")
        
        return paths


class EntityExtractor:
    """从用户问题中提取实体 - 优化版本，支持关键词匹配和 LLM 两种方式"""

    def __init__(self, llm, error_handler: ErrorHandler = None):
        self.llm = llm
        self.error_handler = error_handler or ErrorHandler(logger)
        # 预定义关键词到实体的映射（快速匹配）
        self.keyword_map = {
            "UML": {"name": "UML", "type": "模型"},
            "uml": {"name": "UML", "type": "模型"},
            "统一建模语言": {"name": "UML", "type": "模型"},
            "UML图": {"name": "UML", "type": "模型"},
            "类图": {"name": "类图", "type": "模型"},
            "用例图": {"name": "用例图", "type": "模型"},
            "时序图": {"name": "时序图", "type": "模型"},
            "活动图": {"name": "活动图", "type": "模型"},
            "状态图": {"name": "状态图", "type": "模型"},
            "组件图": {"name": "组件图", "type": "模型"},
            "部署图": {"name": "部署图", "type": "模型"},
            "包图": {"name": "包图", "type": "模型"},
            "对象图": {"name": "对象图", "type": "模型"},
            "协作图": {"name": "协作图", "type": "模型"},
            "敏捷开发": {"name": "敏捷开发", "type": "方法"},
            "Scrum": {"name": "Scrum", "type": "方法"},
            "瀑布模型": {"name": "瀑布模型", "type": "模型"},
            "设计模式": {"name": "设计模式", "type": "概念"},
            "单例模式": {"name": "单例模式", "type": "模式"},
            "工厂模式": {"name": "工厂模式", "type": "模式"},
            "观察者模式": {"name": "观察者模式", "type": "模式"},
            "策略模式": {"name": "策略模式", "type": "模式"},
            "适配器模式": {"name": "适配器模式", "type": "模式"},
            "装饰器模式": {"name": "装饰器模式", "type": "模式"},
            "单元测试": {"name": "单元测试", "type": "方法"},
            "集成测试": {"name": "集成测试", "type": "方法"},
            "系统测试": {"name": "系统测试", "type": "方法"},
            "验收测试": {"name": "验收测试", "type": "方法"},
            "回归测试": {"name": "回归测试", "type": "方法"},
            "函数式编程": {"name": "函数式编程", "type": "范式"},
            "面向对象": {"name": "面向对象编程", "type": "范式"},
            "面向对象编程": {"name": "面向对象编程", "type": "范式"},
            "微服务": {"name": "微服务架构", "type": "架构"},
            "微服务架构": {"name": "微服务架构", "type": "架构"},
            "MVC": {"name": "MVC", "type": "架构"},
            "分层架构": {"name": "分层架构", "type": "架构"},
            "事件驱动": {"name": "事件驱动架构", "type": "架构"},
            "SOLID": {"name": "SOLID原则", "type": "原则"},
            "单一职责": {"name": "单一职责原则", "type": "原则"},
            "开闭原则": {"name": "开闭原则", "type": "原则"},
            "里氏替换": {"name": "里氏替换原则", "type": "原则"},
            "接口隔离": {"name": "接口隔离原则", "type": "原则"},
            "依赖倒置": {"name": "依赖倒置原则", "type": "原则"},
        }

    def extract_entities(self, question: str) -> List[Dict[str, str]]:
        """提取实体 - 先尝试关键词快速匹配，失败则使用 LLM"""
        # 1. 先尝试关键词快速匹配
        quick_entities = self._quick_keyword_match(question)
        if quick_entities:
            logger.debug(f"快速匹配到实体: {quick_entities}")
            return quick_entities
        
        # 2. 关键词匹配失败，使用 LLM
        return self._extract_entities_with_llm(question)
    
    def _quick_keyword_match(self, question: str) -> List[Dict[str, str]]:
        """使用关键词快速匹配"""
        matched = []
        lower_question = question.lower()
        
        for keyword, entity in self.keyword_map.items():
            if keyword.lower() in lower_question:
                matched.append(entity)
        
        # 去重
        unique_entities = []
        seen = set()
        for e in matched:
            key = (e["name"], e["type"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(e)
        
        return unique_entities

    def _extract_entities_with_llm(self, question: str) -> List[Dict[str, str]]:
        """使用LLM从问题中提取实体"""
        prompt_template = PromptTemplate(
            template="""
            从以下问题中提取关于软件工程的实体。
            实体类型必须是以下之一: {entity_types}
            输出严格按照 JSON 格式: [{{"name": "实体名", "type": "实体类型"}}]
            问题: {question}
            JSON:
            """,
            input_variables=["question", "entity_types"]
        )

        try:
            # 使用新的 RunnableSequence 方式代替已弃用的 LLMChain
            chain = prompt_template | self.llm
            response = chain.invoke({
                "question": question,
                "entity_types": ", ".join(ENTITY_TYPES)
            })
            content = response.content if hasattr(response, 'content') else str(response)
            
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
            
            valid_entities = []
            for item in entities:
                if isinstance(item, dict) and "name" in item and "type" in item and item["type"] in ENTITY_TYPES:
                    valid_entities.append({"name": str(item["name"]), "type": str(item["type"])})
            
            logger.debug(f"LLM 提取的实体: {valid_entities}")
            return valid_entities
        except Exception as e:
            error_ctx = ErrorContext(
                error_type=ErrorType.ENTITY_EXTRACTION,
                severity=ErrorSeverity.WARNING,
                message=f"实体提取时出错: {e}",
                component="EntityExtractor.extract_entities",
                timestamp=datetime.now(),
                exception=e
            )
            self.error_handler.handle(error_ctx)
            return []


class EntityMatcher:
    """将提取的实体与知识图谱实体匹配"""

    def __init__(self, embeddings: Embeddings, neo4j_handler, embedding_cache_path="entity_embeddings.pkl", error_handler: ErrorHandler = None, perf_monitor: PerformanceMonitor = None):
        self.embeddings = embeddings
        self.neo4j_handler = neo4j_handler
        self.embedding_cache_path = embedding_cache_path
        self.embeddings_cache = self._load_embeddings_cache()
        self.kg_entities = []
        self.error_handler = error_handler or ErrorHandler(logger)
        self.perf_monitor = perf_monitor or PerformanceMonitor(logger)

    def load_and_cache_kg_entities(self):
        """从 Neo4j 加载实体并缓存嵌入"""
        logger.info("EntityMatcher: 加载 KG 实体...")
        all_entities_from_db = self.neo4j_handler.get_all_entities()
        self.kg_entities = [e for e in all_entities_from_db if e.get("name")]
        logger.info(f"EntityMatcher: 存储了 {len(self.kg_entities)} 个带名称的实体。")
        
        if self.kg_entities:
            entity_names = [entity["name"] for entity in self.kg_entities]
            self._ensure_embeddings(entity_names)

    def _load_embeddings_cache(self) -> Dict[str, list]:
        """从文件加载实体嵌入缓存"""
        cache = {}
        if os.path.exists(self.embedding_cache_path):
            try:
                with open(self.embedding_cache_path, 'rb') as f:
                    cache = pickle.load(f)
                logger.debug(f"从缓存加载了 {len(cache)} 个嵌入")
            except Exception as e:
                logger.error(f"加载嵌入缓存时出错: {e}")
        return cache

    def _save_embeddings_cache(self):
        """保存实体嵌入缓存到文件"""
        try:
            with open(self.embedding_cache_path, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            logger.debug(f"已保存 {len(self.embeddings_cache)} 个嵌入到缓存")
        except Exception as e:
            logger.error(f"保存嵌入缓存时出错: {e}")

    def get_embedding(self, text: str, force_refresh=False) -> Optional[list]:
        """获取文本的嵌入，使用缓存"""
        if not text or not text.strip():
            return None
        
        cache_key = text.strip()
        if not force_refresh and cache_key in self.embeddings_cache:
            return self.embeddings_cache[cache_key]
        
        try:
            embedding = self.embeddings.embed_query(cache_key)
            self.embeddings_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            error_ctx = ErrorContext(
                error_type=ErrorType.EMBEDDINGS,
                severity=ErrorSeverity.ERROR,
                message=f"获取 '{text}' 的嵌入时出错: {e}",
                component="EntityMatcher.get_embedding",
                timestamp=datetime.now(),
                exception=e
            )
            self.error_handler.handle(error_ctx)
            return None

    def _ensure_embeddings(self, texts: List[str]):
        """确保所有文本在缓存中都有嵌入"""
        unique_texts = list(set(filter(None, texts)))
        texts_to_embed = [text for text in unique_texts if text not in self.embeddings_cache]
        
        if not texts_to_embed:
            return
        
        logger.info(f"为 {len(texts_to_embed)} 个新的 KG 实体生成嵌入...")
        texts_to_embed = [str(text) if text is not None else "" for text in texts_to_embed]
        
        batch_size = 10
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:min(i + batch_size, len(texts_to_embed))]
            try:
                embeddings = self.embeddings.embed_documents(batch)
                for j, text in enumerate(batch):
                    self.embeddings_cache[text] = embeddings[j]
            except Exception as e:
                logger.error(f"生成批次嵌入时出错: {e}")
        
        self._save_embeddings_cache()

    def match_entity(self, entity: Dict[str, str], similarity_threshold=0.85) -> List[Dict[str, Any]]:
        """将提取的实体与已加载的 KG 实体列表进行匹配"""
        with self.perf_monitor.measure("实体匹配"):
            entity_name = entity.get("name")
            entity_type = entity.get("type")
            
            if not entity_name or not self.kg_entities:
                return []
            
            # 精确名称匹配
            exact_matches = [kg for kg in self.kg_entities if
                            kg.get("name") == entity_name and (not entity_type or kg.get("type") == entity_type)]
            if exact_matches:
                return [{"entity": match, "match_type": "exact", "score": 1.0} for match in exact_matches]
            
            # 向量相似度匹配
            entity_embedding = self.get_embedding(entity_name)
            if not entity_embedding:
                return []
            
            matches = []
            for kg_entity in self.kg_entities:
                kg_entity_name = kg_entity.get("name")
                if not kg_entity_name:
                    continue
                
                kg_entity_embedding = self.embeddings_cache.get(kg_entity_name)
                if kg_entity_embedding:
                    try:
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
            return matches[:5]


class ResponseGenerator:
    """使用LLM、知识图谱上下文与文档检索结果生成响应"""

    def __init__(self, llm, error_handler: ErrorHandler = None):
        self.llm = llm
        self.error_handler = error_handler or ErrorHandler(logger)

    def generate_response(self, question: str, kg_context: Dict[str, Any],
                        entities: List[Dict[str, str]], doc_results: List[Dict[str, Any]]) -> str:
        """使用LLM生成响应"""
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
            5. 如果问题是编程相关，可以提供简单的代码示例
            请生成回答:
            """,
            input_variables=["question", "entities", "kg_context", "doc_context"]
        )
        
        try:
            messages = [
                SystemMessage(content="你是专业的软件工程课程助手，精通各种软件工程概念、方法、工具和最佳实践。"),
                HumanMessage(content=prompt_template.format(
                    question=question,
                    entities=entities_str,
                    kg_context=kg_context_str or "无相关知识图谱信息。",
                    doc_context=doc_context_str or "无相关文档片段。"
                ))
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            error_ctx = ErrorContext(
                error_type=ErrorType.LLM_API,
                severity=ErrorSeverity.ERROR,
                message=f"调用 LLM 时出错: {e}",
                component="ResponseGenerator.generate_response",
                timestamp=datetime.now(),
                exception=e
            )
            self.error_handler.handle(error_ctx)
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
    """
    整合知识图谱、本地文档与LLM的软件工程问答系统
    优化版本：集成了错误处理、性能监控和改进的配置管理
    """

    def __init__(self, config: QASystemConfig = None):
        """初始化QA系统"""
        logger.info("初始化 SoftwareEngineeringQASystem...")
        
        # 加载配置
        self.config = config or QASystemConfig.from_env_file()
        config_errors = self.config.validate()
        if config_errors:
            for err in config_errors:
                logger.warning(err)
        
        # 初始化核心组件
        self.error_handler = ErrorHandler(logger)
        self.perf_monitor = PerformanceMonitor(logger)
        
        # 设置 Langchain 组件
        self.llm = None
        self.embeddings = None
        self.neo4j_handler = None
        self.rag_manager = None
        self.entity_matcher = None
        self.entity_extractor = None
        self.response_generator = None
        self.agent_coordinator = None
        
        # 初始化
        self._setup_llm()
        self._setup_embeddings()
        self._setup_neo4j()
        self._setup_rag_and_entities()
        
        # UI 状态
        self.current_kg_context = {"entities": [], "paths": []}
        self.current_doc_results_raw = []
        self.current_agent_name = None
        self.conversation_history = deque(maxlen=self.config.max_conversation_history)
        
        # 问题响应缓存 - 提高重复问题的响应速度
        from collections import OrderedDict
        self.response_cache = OrderedDict()
        self.max_cache_size = 100  # 最多缓存100个问题
        
        logger.info("SoftwareEngineeringQASystem 初始化成功。")

    def _setup_llm(self):
        """设置Langchain的LLM组件"""
        if not self.config.llm.api_key:
            logger.warning("未设置 API 密钥，跳过 LLM 初始化。")
            return
        
        try:
            ChatTongyi = None
            ChatOpenAI = None
            try:
                from langchain_openai import ChatOpenAI
                from langchain_dashscope import ChatTongyi
            except ImportError:
                pass
            
            try:
                llm = ChatTongyi(
                    api_key=self.config.llm.api_key,
                    temperature=self.config.llm.temperature,
                    model=self.config.llm.model,
                    request_timeout=self.config.llm.timeout,
                    max_retries=self.config.llm.max_retries
                )
                logger.info("Langchain ChatTongyi 设置成功。")
            except Exception:
                llm = ChatOpenAI(
                    api_key=self.config.llm.api_key,
                    base_url=self.config.llm.base_url,
                    temperature=self.config.llm.temperature,
                    model=self.config.llm.model,
                    request_timeout=self.config.llm.timeout,
                    max_retries=self.config.llm.max_retries
                )
                logger.info("Langchain ChatOpenAI (兼容模式) 设置成功。")
            
            self.llm = llm
        except Exception as e:
            error_ctx = ErrorContext(
                error_type=ErrorType.LLM_API,
                severity=ErrorSeverity.ERROR,
                message=f"设置 LLM 失败: {e}",
                component="SoftwareEngineeringQASystem._setup_llm",
                timestamp=datetime.now(),
                exception=e
            )
            self.error_handler.handle(error_ctx)

    def _setup_embeddings(self):
        """设置Langchain的Embeddings组件"""
        if not self.config.llm.api_key:
            logger.warning("未设置 API 密钥，跳过 Embeddings 初始化。")
            return
        
        try:
            DashScopeEmbeddings = None
            try:
                from langchain_dashscope import DashScopeEmbeddings
            except ImportError:
                pass
            
            try:
                embeddings = DashScopeEmbeddings(
                    api_key=self.config.llm.api_key,
                    model=self.config.rag.embedding_model
                )
                logger.info("Langchain DashScopeEmbeddings 设置成功。")
            except Exception:
                from langchain_core.embeddings import Embeddings as BaseEmbeddings
                
                class CustomDashScopeEmbeddings(BaseEmbeddings):
                    def __init__(self, api_key: str, model: str):
                        self.api_key = api_key
                        self.model = model
                        self.dimension = None
                        # 根据模型确定合适的维度
                        if model in ['text-embedding-v3', 'text-embedding-v4']:
                            self.dimension = 1024  # 使用支持的维度
                        import dashscope
                        dashscope.api_key = api_key
                    
                    def embed_documents(self, texts: List[str]) -> List[List[float]]:
                        import dashscope
                        results = []
                        # 逐个处理文本，避免 batch 处理的问题
                        for text in texts:
                            try:
                                # 确保文本是字符串
                                if not isinstance(text, str):
                                    text = str(text)
                                if not text or not text.strip():
                                    # 空文本返回零向量
                                    results.append([0.0] * 1024)
                                    continue
                                
                                resp = dashscope.TextEmbedding.call(
                                    model=self.model, 
                                    input=text,
                                    dimension=self.dimension
                                )
                                if resp.status_code == 200:
                                    results.append(resp.output['embeddings'][0]['embedding'])
                                elif resp.status_code == 403 and hasattr(resp, 'code') and resp.code == 'AllocationQuota.FreeTierOnly':
                                    raise Exception("免费额度已用完")
                                else:
                                    raise Exception(f"获取嵌入失败: {resp}")
                            except Exception as e:
                                if "FreeTierOnly" in str(e):
                                    raise e
                                logger.error(f"调用嵌入 API 时出错: {str(e)}, 文本: {text}")
                                # 出错时返回零向量
                                results.append([0.0] * 1024)
                        return results
                    
                    def embed_query(self, text: str) -> List[float]:
                        return self.embed_documents([text])[0]
                
                embeddings = CustomDashScopeEmbeddings(
                    api_key=self.config.llm.api_key,
                    model=self.config.rag.embedding_model
                )
                logger.info("Custom DashScopeEmbeddings 设置成功。")
            
            self.embeddings = embeddings
        except Exception as e:
            error_ctx = ErrorContext(
                error_type=ErrorType.EMBEDDINGS,
                severity=ErrorSeverity.ERROR,
                message=f"设置 Embeddings 失败: {e}",
                component="SoftwareEngineeringQASystem._setup_embeddings",
                timestamp=datetime.now(),
                exception=e
            )
            self.error_handler.handle(error_ctx)

    def _setup_neo4j(self):
        """设置 Neo4j 连接"""
        try:
            self.neo4j_handler = Neo4jHandler(
                self.config.neo4j.uri,
                self.config.neo4j.username,
                self.config.neo4j.password,
                error_handler=self.error_handler
            )
        except Exception as e:
            error_ctx = ErrorContext(
                error_type=ErrorType.NEO4J_CONNECTION,
                severity=ErrorSeverity.ERROR,
                message=f"初始化 Neo4j Handler 失败: {e}",
                component="SoftwareEngineeringQASystem._setup_neo4j",
                timestamp=datetime.now(),
                exception=e
            )
            self.error_handler.handle(error_ctx)

    def _setup_rag_and_entities(self):
        """设置 RAG 和实体相关组件"""
        if self.embeddings:
            self.rag_manager = RAGManager.get_instance(
                embeddings=self.embeddings,
                error_handler=self.error_handler,
                perf_monitor=self.perf_monitor
            )
            
            if self.neo4j_handler:
                self.entity_matcher = EntityMatcher(
                    self.embeddings,
                    self.neo4j_handler,
                    error_handler=self.error_handler,
                    perf_monitor=self.perf_monitor
                )
                try:
                    self.entity_matcher.load_and_cache_kg_entities()
                    logger.info("KG 实体加载完成。")
                except Exception as e:
                    logger.warning(f"初始 KG 实体加载期间出错: {e}")
        
        if self.llm:
            self.entity_extractor = EntityExtractor(self.llm, error_handler=self.error_handler)
            self.response_generator = ResponseGenerator(self.llm, error_handler=self.error_handler)
            self.agent_coordinator = AgentCoordinator(self.llm)

    def answer_question(self, question: str, web_search: bool = False, 
                       table_output: bool = False, multi_hop: bool = False) -> dict:
        """处理问题，整合 KG 和 RAG 生成答案"""
        start_time = time.time()
        logger.info("===== 回答问题 =====")
        logger.info(f"问题: {question}")
        logger.info(f"参数: web_search={web_search}, table_output={table_output}, multi_hop={multi_hop}")
        
        # 构建带参数的缓存键
        cache_key = f"{question.strip().lower()}|{web_search}|{table_output}|{multi_hop}"
        
        # 检查缓存
        if cache_key in self.response_cache:
            logger.info(f"从缓存中获取答案，命中缓存: {question}")
            cached_data = self.response_cache[cache_key]
            self.response_cache.move_to_end(cache_key)
            
            self.conversation_history.append((question, cached_data["answer"]))
            self.current_agent_name = cached_data["agent_name"]
            self.current_kg_context = cached_data["kg_context"]
            self.current_doc_results_raw = cached_data["doc_results"]
            
            result = {
                "answer": cached_data["answer"],
                "agent_name": cached_data["agent_name"],
                "conversation_history": list(self.conversation_history),
                "performance": {
                    "total": f"{time.time() - start_time:.3f}s (缓存)",
                    "from_cache": True
                }
            }
            
            logger.info(f"===== 缓存答案返回 (由 {self.current_agent_name} 提供，耗时 {time.time() - start_time:.3f}s) =====")
            return result
        
        # 缓存未命中，正常处理
        self.current_kg_context = {"entities": [], "paths": []}
        self.current_doc_results_raw = []
        self.current_agent_name = None
        
        metrics = PerformanceMetrics()
        perf_ctx = TimingContext(self.perf_monitor)
        perf_ctx.__enter__()
        
        try:
            # 1. 提取实体
            entities = []
            with perf_ctx.measure("实体提取"):
                if self.entity_extractor:
                    entities = self.entity_extractor.extract_entities(question)
            
            # 2. 并行执行 KG 查询和 RAG 搜索
            import concurrent.futures
            kg_future = None
            rag_future = None
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # 提交 KG 相关任务
                def process_kg():
                    kg_result = {"entities": [], "paths": [], "matched_names": set()}
                    if entities and self.entity_matcher and self.neo4j_handler:
                        for entity in entities:
                            entity_name = entity.get("name")
                            if not entity_name:
                                continue
                            
                            matches = self.entity_matcher.match_entity(entity)
                            if matches:
                                top_match = matches[0]
                                kg_entity_name = top_match["entity"]["name"]
                                try:
                                    entity_data = self.neo4j_handler.get_entity_relationships(kg_entity_name)
                                    if entity_data:
                                        kg_result["entities"].append(entity_data)
                                        kg_result["matched_names"].add(kg_entity_name)
                                except Exception as e:
                                    logger.debug(f"获取实体关系失败 {kg_entity_name}: {e}")
                    
                    # 多跳推理
                    if multi_hop and len(kg_result["matched_names"]) > 0 and self.neo4j_handler:
                        logger.info("执行多跳推理...")
                        extended_entities = set(kg_result["matched_names"])
                        
                        for _ in range(2):
                            new_entities = set()
                            for entity_name in extended_entities:
                                try:
                                    entity_data = self.neo4j_handler.get_entity_relationships(entity_name)
                                    if entity_data:
                                        relationships = entity_data.get("relationships", [])
                                        for rel in relationships:
                                            if rel.get("target"):
                                                new_entities.add(rel["target"])
                                except Exception as e:
                                    logger.debug(f"多跳查询失败 {entity_name}: {e}")
                        
                        for new_entity in new_entities:
                            if new_entity not in extended_entities:
                                extended_entities.add(new_entity)
                                try:
                                    entity_data = self.neo4j_handler.get_entity_relationships(new_entity)
                                    if entity_data and entity_data not in kg_result["entities"]:
                                        kg_result["entities"].append(entity_data)
                                except Exception as e:
                                    logger.debug(f"多跳扩展失败 {new_entity}: {e}")
                    
                    # 实体路径查询
                    if len(kg_result["matched_names"]) > 1 and self.neo4j_handler:
                        entity_list = list(kg_result["matched_names"])
                        for i in range(len(entity_list)):
                            for j in range(i + 1, len(entity_list)):
                                source, target = entity_list[i], entity_list[j]
                                try:
                                    paths = self.neo4j_handler.get_path_between_entities(source, target)
                                    if paths:
                                        kg_result["paths"].extend(paths)
                                except Exception as e:
                                    logger.debug(f"路径查询失败 {source} -> {target}: {e}")
                    
                    return kg_result
                
                # 提交 RAG 搜索任务
                def process_rag():
                    if self.rag_manager:
                        return self.rag_manager.search_documents(question)
                    return []
                
                kg_future = executor.submit(process_kg)
                rag_future = executor.submit(process_rag)
                
                # 等待结果
                with perf_ctx.measure("KG检索"):
                    kg_result = kg_future.result(timeout=30)  # KG查询超时30秒
                
                with perf_ctx.measure("RAG搜索"):
                    self.current_doc_results_raw = rag_future.result(timeout=30)  # RAG搜索超时30秒
                
                self.current_kg_context["entities"] = kg_result["entities"]
                self.current_kg_context["paths"] = kg_result["paths"]
                all_matched_kg_entity_names = kg_result["matched_names"]
            
            # 3. 联网搜索（可选）
            web_search_results = []
            if web_search:
                with perf_ctx.measure("联网搜索"):
                    try:
                        web_search_results = self._perform_web_search(question)
                        logger.info(f"联网搜索完成，获取 {len(web_search_results)} 条结果")
                    except Exception as e:
                        logger.warning(f"联网搜索失败: {e}")
            
            # 4. 使用智能体协调器生成最终响应
            with perf_ctx.measure("智能体选择"):
                has_kg = bool(self.current_kg_context.get("entities") or self.current_kg_context.get("paths"))
                has_docs = bool(self.current_doc_results_raw)
                
                if not has_kg and not has_docs:
                    final_answer = self._generate_fallback_response(question, "未能在知识图谱或文档中找到相关信息")
                    self.current_agent_name = "回退响应生成器"
                elif self.agent_coordinator:
                    with perf_ctx.measure("LLM响应"):
                        final_answer, agent_name = self.agent_coordinator.process_question(
                            question=question,
                            kg_context=self.current_kg_context,
                            entities=entities,
                            doc_results=self.current_doc_results_raw,
                            conversation_history=list(self.conversation_history),
                            web_search_results=web_search_results,
                            table_output=table_output
                        )
                        self.current_agent_name = agent_name
                else:
                    final_answer = self.response_generator.generate_response(
                        question, self.current_kg_context, entities, self.current_doc_results_raw
                    )
                    self.current_agent_name = "基础响应生成器"
        
        except Exception as e:
            error_ctx = ErrorContext(
                error_type=ErrorType.UNKNOWN,
                severity=ErrorSeverity.CRITICAL,
                message=f"回答问题时出错: {e}",
                component="SoftwareEngineeringQASystem.answer_question",
                timestamp=datetime.now(),
                exception=e
            )
            self.error_handler.handle(error_ctx)
            final_answer = f"抱歉，处理您的问题时遇到了错误：{e}"
            self.current_agent_name = "错误处理器"
        
        perf_ctx.__exit__(None, None, None)
        final_metrics = perf_ctx.get_metrics()
        final_metrics.total_time = time.time() - start_time
        
        # 更新对话历史
        self.conversation_history.append((question, final_answer))
        
        # 缓存结果
        self.response_cache[cache_key] = {
            "answer": final_answer,
            "agent_name": self.current_agent_name,
            "kg_context": self.current_kg_context,
            "doc_results": self.current_doc_results_raw.copy()
        }
        
        # 限制缓存大小
        if len(self.response_cache) > self.max_cache_size:
            self.response_cache.popitem(last=False)  # 删除最旧的
        
        logger.info(f"===== 答案生成完成 (由 {self.current_agent_name} 提供，耗时 {final_metrics.total_time:.3f}s) =====")
        
        return {
            "answer": final_answer,
            "agent_name": self.current_agent_name,
            "conversation_history": list(self.conversation_history),
            "performance": final_metrics.to_dict()
        }

    def _perform_web_search(self, query: str) -> List[Dict[str, str]]:
        """执行联网搜索获取最新信息"""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            # 使用 DuckDuckGo 搜索 API
            search_url = "https://html.duckduckgo.com/html/"
            params = {"q": query, "kl": "zh-CN"}
            
            # 添加浏览器请求头以避免被识别为自动化请求
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Referer": "https://duckduckgo.com/",
            }
            
            response = requests.get(search_url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # 解析搜索结果（尝试多种可能的class名称）
            for result in soup.find_all('div', class_='result'):
                title_tag = result.find('a', class_='result__a')
                url_tag = result.find('a', class_='result__url')
                snippet_tag = result.find('span', class_='result__snippet')
                
                if title_tag and url_tag:
                    results.append({
                        "title": title_tag.get_text(strip=True),
                        "url": url_tag['href'] if url_tag.has_attr('href') else url_tag.get_text(strip=True),
                        "snippet": snippet_tag.get_text(strip=True) if snippet_tag else ""
                    })
                
                if len(results) >= 5:
                    break
            
            return results
        except Exception as e:
            logger.error(f"联网搜索失败: {e}")
            return []

    def _generate_fallback_response(self, question: str, reason: str) -> str:
        """生成基于通用知识的回退响应"""
        logger.warning(f"生成回退响应。原因: {reason}")
        
        history_text = ""
        if self.conversation_history:
            history_text = "\n".join([f"用户: {q}\n助手: {a}" for q, a in list(self.conversation_history)[-5:]])
        else:
            history_text = "这是新的对话"
        
        if not self.llm:
            return f"抱歉，我无法回答这个问题（{reason}）。"
        
        try:
            prompt_template = PromptTemplate(
                template="""
                您是软件工程课程的智能助手。学生问了以下问题，但我们没有在知识库中找到相关信息。
                请基于您的通用软件工程知识提供最佳回答。
                
                对话历史: {history}
                当前问题: {question}
                
                请提供专业、准确且有帮助的回答，使用易于理解的语言解释软件工程概念。
                """,
                input_variables=["question", "history"]
            )
            
            messages = [
                SystemMessage(content="你是专业的软件工程课程助手，精通各种软件工程概念、方法、工具和最佳实践。"),
                HumanMessage(content=prompt_template.format(question=question, history=history_text))
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"生成回退响应时出错: {e}")
            return f"抱歉，我无法回答这个问题（{reason}）。"

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
        if hasattr(self, 'neo4j_handler') and self.neo4j_handler:
            self.neo4j_handler.close()
        logger.info("系统已关闭。")


class TimingContext:
    """计时上下文管理器"""
    def __init__(self, perf_monitor: PerformanceMonitor):
        self.perf_monitor = perf_monitor
        self.metrics = PerformanceMetrics()
        self._start_time = time.time()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.metrics.total_time = time.time() - self._start_time
    
    def measure(self, step_name: str):
        """测量步骤耗时"""
        return self.perf_monitor.measure(step_name)
    
    def get_metrics(self) -> PerformanceMetrics:
        """获取度量"""
        if self.metrics.total_time == 0.0:
            self.metrics.total_time = time.time() - self._start_time
        return self.metrics


# --- 导出供UI使用的函数 ---

qa_system_instance: Optional[SoftwareEngineeringQASystem] = None


def initialize_qa_system(uri, user, pwd):
    """初始化全局QA系统实例"""
    global qa_system_instance
    if qa_system_instance is None:
        logger.info("UI 正在初始化 QA 系统...")
        try:
            config = QASystemConfig.from_env_file()
            config.neo4j.uri = uri
            config.neo4j.username = user
            config.neo4j.password = pwd
            qa_system_instance = SoftwareEngineeringQASystem(config)
        except Exception as e:
            logger.error(f"致命错误：初始化 QA 系统失败: {e}")
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
        qa_system = get_qa_system_instance()
        if qa_system.rag_manager is None:
            return "❌ 错误：RAG 管理器未初始化。请检查 API 密钥配置。"
        return qa_system.rag_manager.process_uploaded_files(files)
    except RuntimeError as e:
        return f"❌ 系统错误：{str(e)}"
    except Exception as e:
        logger.error(f"处理文件上传时出错: {e}")
        return f"❌ 文件上传处理错误: {str(e)}"


def search_documents(query, top_k=3):
    """UI调用的文档搜索函数"""
    logger.debug(f"UI 触发文档搜索，查询: '{query}'")
    try:
        qa_system = get_qa_system_instance()
        if qa_system.rag_manager is None:
            return ""
        results_raw = qa_system.rag_manager.search_documents(query, top_k)
        return qa_system.rag_manager.format_search_results_as_html(results_raw)
    except Exception as e:
        logger.error(f"文档搜索时出错: {e}")
        return ""


if __name__ == "__main__":
    pass
