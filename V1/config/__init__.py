# -*- coding: utf-8 -*-
"""
统一配置管理模块 - 公共 API 层
从环境变量加载配置，支持 Neo4j、LLM、RAG 等配置项
"""

# 配置类
from .settings import (
    AppConfig, Neo4jConfig, LLMConfig, RAGConfig, LogConfig
)

# 配置加载和访问函数
from .loader import (
    load_config, init_config, get_config,
    get_neo4j_config, get_llm_config, get_rag_config, get_log_config
)

__all__ = [
    # 配置类
    "AppConfig", "Neo4jConfig", "LLMConfig", "RAGConfig", "LogConfig",
    # 配置函数
    "load_config", "init_config", "get_config",
    "get_neo4j_config", "get_llm_config", "get_rag_config", "get_log_config"
]
