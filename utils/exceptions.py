# -*- coding: utf-8 -*-
"""
统一异常处理模块
定义项目中使用的自定义异常类
"""


class QAException(Exception):
    """问答系统基础异常"""
    
    def __init__(self, message: str, error_code: int = 500):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
    
    def __str__(self):
        return f"[{self.error_code}] {self.message}"


class Neo4jConnectionError(QAException):
    """Neo4j连接异常"""
    
    def __init__(self, message: str):
        super().__init__(message, error_code=503)


class Neo4jQueryError(QAException):
    """Neo4j查询异常"""
    
    def __init__(self, message: str):
        super().__init__(message, error_code=400)


class LLMError(QAException):
    """LLM调用异常"""
    
    def __init__(self, message: str):
        super().__init__(message, error_code=503)


class EntityExtractionError(QAException):
    """实体提取异常"""
    
    def __init__(self, message: str):
        super().__init__(message, error_code=400)


class DocumentProcessingError(QAException):
    """文档处理异常"""
    
    def __init__(self, message: str):
        super().__init__(message, error_code=400)


class ConfigurationError(QAException):
    """配置错误异常"""
    
    def __init__(self, message: str):
        super().__init__(message, error_code=500)


class CacheError(QAException):
    """缓存操作异常"""
    
    def __init__(self, message: str):
        super().__init__(message, error_code=500)


def handle_exception(func):
    """
    异常处理装饰器
    捕获并记录异常，提供优雅的错误响应
    
    Args:
        func: 要包装的函数
        
    Returns:
        包装后的函数
    """
    import functools
    import logging
    
    logger = logging.getLogger(__name__)
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Neo4jConnectionError as e:
            logger.error(f"Neo4j连接错误: {e}")
            raise
        except Neo4jQueryError as e:
            logger.error(f"Neo4j查询错误: {e}")
            raise
        except LLMError as e:
            logger.error(f"LLM调用错误: {e}")
            # 实现降级策略
            return f"抱歉，服务暂时不可用，请稍后重试。"
        except (EntityExtractionError, DocumentProcessingError) as e:
            logger.warning(f"数据处理错误: {e}")
            raise
        except ConfigurationError as e:
            logger.critical(f"配置错误: {e}")
            raise
        except Exception as e:
            logger.error(f"未知错误: {e}", exc_info=True)
            raise QAException(f"系统内部错误: {str(e)}")
    
    return wrapper
