"""
Improved error handling system.
"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
import traceback
import logging
from datetime import datetime


class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorType(Enum):
    """Error types."""
    ENTITY_EXTRACTION = "entity_extraction"
    ENTITY_MATCHING = "entity_matching"
    NEO4J_CONNECTION = "neo4j_connection"
    NEO4J_QUERY = "neo4j_query"
    RAG_INDEXING = "rag_indexing"
    RAG_SEARCH = "rag_search"
    LLM_API = "llm_api"
    EMBEDDINGS = "embeddings"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Error context information."""
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    component: str
    timestamp: datetime
    exception: Optional[Exception] = None
    context_data: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert error context to dictionary."""
        return {
            'error_type': self.error_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'component': self.component,
            'timestamp': self.timestamp.isoformat(),
            'stack_trace': traceback.format_exc() if self.exception else None,
            'context_data': self.context_data or {}
        }


class QASystemException(Exception):
    """Base QA system exception."""
    def __init__(self, error_ctx: ErrorContext):
        self.error_ctx = error_ctx
        super().__init__(error_ctx.message)


class EntityExtractionException(QASystemException):
    """Entity extraction exception."""
    pass


class EntityMatchingException(QASystemException):
    """Entity matching exception."""
    pass


class Neo4jException(QASystemException):
    """Neo4j exception."""
    pass


class RAGException(QASystemException):
    """RAG exception."""
    pass


class LLMException(QASystemException):
    """LLM exception."""
    pass


class ErrorHandler:
    """Unified error handler."""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger('qa_system_errors')
        self.error_history = []
        self.max_history = 100
    
    def handle(self, error_ctx: ErrorContext) -> str:
        """
        Handle error and return user-friendly message.
        
        Args:
            error_ctx: Error context
            
        Returns:
            User-friendly error message
        """
        # Log the error
        self._log_error(error_ctx)
        self._store_error(error_ctx)
        
        # Take action based on error severity
        if error_ctx.severity == ErrorSeverity.CRITICAL:
            self._alert_critical_error(error_ctx)
        
        # Return user-friendly message
        return self._get_user_message(error_ctx)
    
    def _log_error(self, error_ctx: ErrorContext) -> None:
        """Log the error."""
        log_method = {
            ErrorSeverity.INFO: self.logger.info,
            ErrorSeverity.WARNING: self.logger.warning,
            ErrorSeverity.ERROR: self.logger.error,
            ErrorSeverity.CRITICAL: self.logger.critical,
        }[error_ctx.severity]
        
        # 创建安全的 extra 字典，避免与 logging 保留字段冲突
        extra_dict = error_ctx.to_dict()
        # 移除可能冲突的字段
        if 'message' in extra_dict:
            del extra_dict['message']
        
        log_method(
            f"[{error_ctx.component}] {error_ctx.message}",
            extra=extra_dict
        )
    
    def _store_error(self, error_ctx: ErrorContext) -> None:
        """Store error for later analysis."""
        self.error_history.append(error_ctx)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
    
    def _alert_critical_error(self, error_ctx: ErrorContext) -> None:
        """
        Alert on critical errors.
        Could be extended to send alerts via email, Slack, etc.
        """
        self.logger.critical(f"CRITICAL ERROR: {error_ctx.message}", extra=error_ctx.to_dict())
    
    def _get_user_message(self, error_ctx: ErrorContext) -> str:
        """Return user-friendly error message."""
        messages = {
            ErrorType.ENTITY_EXTRACTION: "无法从问题中提取关键词，请尝试重新表述问题。",
            ErrorType.ENTITY_MATCHING: "无法在知识库中找到相关的概念，请尝试其他问法。",
            ErrorType.NEO4J_CONNECTION: "知识库连接异常，请稍后重试。",
            ErrorType.NEO4J_QUERY: "知识库查询失败，请稍后重试。",
            ErrorType.RAG_INDEXING: "文档索引创建失败，请重新上传文档。",
            ErrorType.RAG_SEARCH: "文档检索失败，可能是知识库为空。",
            ErrorType.LLM_API: "AI服务暂时不可用，请稍后重试。",
            ErrorType.EMBEDDINGS: "向量化服务异常，请稍后重试。",
            ErrorType.CONFIGURATION: "系统配置错误，请联系管理员。",
            ErrorType.UNKNOWN: "系统出现未知错误，请稍后重试。",
        }
        return messages.get(error_ctx.error_type, messages[ErrorType.UNKNOWN])
    
    def get_recent_errors(self, limit: int = 10) -> list:
        """Get recent error history."""
        return self.error_history[-limit:]
