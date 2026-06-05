# -*- coding: utf-8 -*-
"""
Configuration data models (DEPRECATED - use core.config instead)
This is kept for backward compatibility only.
"""
from core.config import (
    AppConfig as _AppConfig,
    Neo4jConfig as _Neo4jConfig,
    LLMConfig as _LLMConfig,
    RAGConfig as _RAGConfig,
    LogConfig as _LogConfig,
    QASystemConfig,
    Environment,
    EntityMatchingConfig
)


# Alias for backward compatibility
AppConfig = _AppConfig
Neo4jConfig = _Neo4jConfig
LLMConfig = _LLMConfig
RAGConfig = _RAGConfig
LogConfig = _LogConfig


# Expose QASystemConfig for backward compatibility
__all__ = [
    'AppConfig', 'Neo4jConfig', 'LLMConfig', 'RAGConfig', 'LogConfig',
    'QASystemConfig', 'Environment', 'EntityMatchingConfig'
]
