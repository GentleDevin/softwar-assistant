# -*- coding: utf-8 -*-
"""
Configuration module (BACKWARD COMPATIBILITY LAYER)
For new code, import directly from core.config instead!
"""
# Import from core.config for new unified configuration
from core.config import (
    QASystemConfig, LLMConfig, Neo4jConfig, RAGConfig,
    EntityMatchingConfig, Environment
)

# Backward compatibility wrappers
from .loader import (
    load_config, init_config, get_config,
    get_neo4j_config, get_llm_config, get_rag_config, get_log_config
)

# Alias QASystemConfig for backward compatibility
AppConfig = QASystemConfig


__all__ = [
    # New unified config
    "QASystemConfig", "LLMConfig", "Neo4jConfig", "RAGConfig",
    "EntityMatchingConfig", "Environment",
    # Backward compatibility
    "AppConfig",
    "load_config", "init_config", "get_config",
    "get_neo4j_config", "get_llm_config", "get_rag_config", "get_log_config"
]
