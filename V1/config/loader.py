# -*- coding: utf-8 -*-
"""
配置加载器
负责 .env 文件加载和配置初始化逻辑
"""

import os
from typing import Optional

from .settings import (
    AppConfig, Neo4jConfig, LLMConfig, RAGConfig, LogConfig
)

# ========== 全局状态 ==========
_config: Optional[AppConfig] = None
_env_loaded: bool = False


def load_dotenv():
    """
    加载 .env 文件
    只加载一次
    """
    global _env_loaded
    if _env_loaded:
        return
    
    try:
        from dotenv import load_dotenv as _load_dotenv
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 加载当前目录和上级目录的 .env 文件
        for dir_path in [current_dir, os.path.dirname(current_dir)]:
            env_path = os.path.join(dir_path, '.env')
            if os.path.exists(env_path):
                _load_dotenv(env_path, override=True)
                break
    except ImportError:
        pass  # 如果没有 dotenv 库，直接使用系统环境变量
    
    _env_loaded = True


def load_config() -> AppConfig:
    """
    从环境变量加载配置
    
    Returns:
        AppConfig: 完整的应用配置对象
    """
    # 确保 .env 文件已加载
    load_dotenv()
    
    # Neo4j配置
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "")
    
    # LLM配置
    dashscope_key = os.getenv("DASHSCOPE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    llm_api_key = dashscope_key or openai_key or ""
    llm_base_url = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    llm_model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    
    # RAG配置
    rag_embedding_model = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-v4")
    rag_embedding_dimension = int(os.getenv("RAG_EMBEDDING_DIMENSION", "1024"))
    rag_chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    rag_chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
    
    # 日志配置
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE", "logs/app.log")
    
    # 验证必要配置 - 仅在环境变量不存在时警告
    neo4j_password_env_exists = "NEO4J_PASSWORD" in os.environ
    if not neo4j_password_env_exists and not neo4j_password:
        import warnings
        warnings.warn("NEO4J_PASSWORD 环境变量未设置，使用默认密码")
        neo4j_password = "neo4j123"
    
    dashscope_key_exists = "DASHSCOPE_API_KEY" in os.environ
    openai_key_exists = "OPENAI_API_KEY" in os.environ
    if not dashscope_key_exists and not openai_key_exists and not llm_api_key:
        import warnings
        warnings.warn("DASHSCOPE_API_KEY 或 OPENAI_API_KEY 环境变量未设置，部分功能可能无法正常工作")
    
    return AppConfig(
        neo4j=Neo4jConfig(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password
        ),
        llm=LLMConfig(
            api_key=llm_api_key,
            base_url=llm_base_url,
            model_name=llm_model_name
        ),
        rag=RAGConfig(
            embedding_model=rag_embedding_model,
            embedding_dimension=rag_embedding_dimension,
            chunk_size=rag_chunk_size,
            chunk_overlap=rag_chunk_overlap
        ),
        log=LogConfig(
            log_level=log_level,
            log_file=log_file
        )
    )


def init_config():
    """初始化全局配置实例"""
    global _config
    _config = load_config()


def get_config() -> AppConfig:
    """获取配置，如果未初始化则先初始化"""
    global _config
    if _config is None:
        init_config()
    return _config


def get_neo4j_config() -> Neo4jConfig:
    """获取Neo4j配置"""
    return get_config().neo4j


def get_llm_config() -> LLMConfig:
    """获取LLM配置"""
    return get_config().llm


def get_rag_config() -> RAGConfig:
    """获取RAG配置"""
    return get_config().rag


def get_log_config() -> LogConfig:
    """获取日志配置"""
    return get_config().log
