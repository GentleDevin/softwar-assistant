# -*- coding: utf-8 -*-
"""
Configuration loader (DEPRECATED - use core.config directly)
This is kept for backward compatibility only.
"""
from core.config import QASystemConfig


# Wrappers for backward compatibility
def load_dotenv():
    """Load .env file (wrapper)"""
    # The QASystemConfig.from_env_file already handles this
    pass


def load_config():
    """Load config from env (wrapper)"""
    return QASystemConfig.from_env_file('.env')


def init_config():
    """Init config (wrapper)"""
    pass


def get_config():
    """Get config (wrapper)"""
    return QASystemConfig.from_env_file('.env')


def get_neo4j_config():
    """Get Neo4j config (wrapper)"""
    return QASystemConfig.from_env_file('.env').neo4j


def get_llm_config():
    """Get LLM config (wrapper)"""
    return QASystemConfig.from_env_file('.env').llm


def get_rag_config():
    """Get RAG config (wrapper)"""
    return QASystemConfig.from_env_file('.env').rag


def get_log_config():
    """Get Log config (wrapper - use QASystemConfig directly)"""
    config = QASystemConfig.from_env_file('.env')
    return type('LogConfig', (), {
        'log_level': config.log_level,
        'log_file': config.log_file
    })()
