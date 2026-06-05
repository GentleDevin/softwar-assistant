# -*- coding: utf-8 -*-
"""
配置数据模型定义
只包含配置类的定义，不包含初始化逻辑
"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class Neo4jConfig:
    """Neo4j数据库配置"""
    uri: str
    username: str
    password: str


@dataclass
class LLMConfig:
    """大语言模型配置"""
    api_key: str
    base_url: str
    model_name: str = "gpt-4o-mini"


@dataclass
class RAGConfig:
    """RAG配置"""
    embedding_model: str = "text-embedding-v3"
    embedding_dimension: int = 1536
    chunk_size: int = 800
    chunk_overlap: int = 150


@dataclass
class LogConfig:
    """日志配置"""
    log_level: str = "INFO"
    log_file: str = "logs/app.log"


@dataclass
class AppConfig:
    """应用配置"""
    neo4j: Neo4jConfig
    llm: LLMConfig
    rag: RAGConfig
    log: LogConfig
