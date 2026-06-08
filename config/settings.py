# -*- coding: utf-8 -*-
"""
Configuration data models (DEPRECATED - use core.config instead)
This is kept for backward compatibility only.
"""
from core.config import (
    Neo4jConfig as _Neo4jConfig,
    LLMConfig as _LLMConfig,
    RAGConfig as _RAGConfig,
    QASystemConfig,
    Environment,
    EntityMatchingConfig
)


# Alias for backward compatibility
Neo4jConfig = _Neo4jConfig
LLMConfig = _LLMConfig
RAGConfig = _RAGConfig


# Expose QASystemConfig for backward compatibility
__all__ = [
    'Neo4jConfig', 'LLMConfig', 'RAGConfig',
    'QASystemConfig', 'Environment', 'EntityMatchingConfig'
]
