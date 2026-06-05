"""
Core module for QA system.
This module contains error handling, configuration, monitoring, and connection pool.
"""
from .config import QASystemConfig, LLMConfig, Neo4jConfig, RAGConfig, EntityMatchingConfig, Environment
from .error_handling import (
    ErrorHandler, ErrorContext, ErrorType, ErrorSeverity,
    QASystemException, EntityExtractionException, EntityMatchingException,
    Neo4jException, RAGException, LLMException
)
from .monitoring import PerformanceMonitor, PerformanceMetrics
from .connection_pool import ConnectionPool, ConnectionPoolException

__all__ = [
    "QASystemConfig", "LLMConfig", "Neo4jConfig", "RAGConfig", "EntityMatchingConfig", "Environment",
    "ErrorHandler", "ErrorContext", "ErrorType", "ErrorSeverity",
    "QASystemException", "EntityExtractionException", "EntityMatchingException",
    "Neo4jException", "RAGException", "LLMException",
    "PerformanceMonitor", "PerformanceMetrics",
    "ConnectionPool", "ConnectionPoolException"
]
