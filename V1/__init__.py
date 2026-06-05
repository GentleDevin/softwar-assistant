# -*- coding: utf-8 -*-
"""
Software Engineering QA System
企业级软件工程问答系统

统一导入入口，提供简洁的 API 访问方式
"""

__version__ = "1.0.0"
__author__ = "QA System Team"

# ========== 核心模块导入 ==========
from core import (
    # 配置
    QASystemConfig, LLMConfig, Neo4jConfig, RAGConfig, EntityMatchingConfig, Environment,
    # 错误处理
    ErrorHandler, ErrorContext, ErrorType, ErrorSeverity,
    QASystemException, EntityExtractionException, EntityMatchingException,
    Neo4jException, RAGException, LLMException,
    # 监控
    PerformanceMonitor, PerformanceMetrics,
    # 连接池
    ConnectionPool, ConnectionPoolException
)

# ========== 工具模块导入 ==========
from utils import (
    # 格式化工具
    format_kg_context, format_doc_context, format_conversation_history,
    # 日志工具
    setup_logger, get_logger,
    # 异常类
    QAException, Neo4jConnectionError, LLMError, EntityExtractionError, DocumentProcessingError
)

# ========== 数据模型导入 ==========
from models import (
    Entity, EntityType, QuestionInput, AnswerOutput, SearchResult, FileUploadResult
)

# ========== 配置管理导入 ==========
from config import (
    AppConfig, Neo4jConfig as AppNeo4jConfig, LLMConfig as AppLLMConfig,
    RAGConfig as AppRAGConfig, LogConfig,
    load_config, init_config, get_config,
    get_neo4j_config, get_llm_config, get_rag_config, get_log_config
)

# ========== 智能体系统导入 ==========
from agents import (
    Agent,
    ConceptExplanationAgent, RequirementsAnalysisAgent, SoftwareDesignAgent,
    SoftwareTestingAgent, ProjectManagementAgent, CodeImplementationAgent, SoftwareEthicsAgent,
    AgentCoordinator,
    select_agents_function, synthesize_answers_function
)

# ========== 主系统导入（延迟导入以避免自动初始化） ==========
def __getattr__(name):
    """
    延迟导入主系统组件
    避免在导入 V1 包时就初始化整个 QA 系统
    """
    if name in [
        "RAGManager", "Neo4jHandler", "EntityExtractor", "EntityMatcher", 
        "AnswerGenerator", "SoftwareEngineeringQASystem", "initialize_qa_system"
    ]:
        from softeng_kg_qa import (
            RAGManager, Neo4jHandler, EntityExtractor, EntityMatcher, 
            AnswerGenerator, SoftwareEngineeringQASystem, initialize_qa_system
        )
        return locals()[name]
    raise AttributeError(f"module {__name__} has no attribute {name}")

# ========== 公开 API 列表 ==========
__all__ = [
    # 版本信息
    "__version__", "__author__",
    
    # 核心模块
    "QASystemConfig", "LLMConfig", "Neo4jConfig", "RAGConfig", "EntityMatchingConfig", "Environment",
    "ErrorHandler", "ErrorContext", "ErrorType", "ErrorSeverity",
    "QASystemException", "EntityExtractionException", "EntityMatchingException",
    "Neo4jException", "RAGException", "LLMException",
    "PerformanceMonitor", "PerformanceMetrics",
    "ConnectionPool", "ConnectionPoolException",
    
    # 工具模块
    "format_kg_context", "format_doc_context", "format_conversation_history",
    "setup_logger", "get_logger",
    "QAException", "Neo4jConnectionError", "LLMError", "EntityExtractionError", "DocumentProcessingError",
    
    # 数据模型
    "Entity", "EntityType", "QuestionInput", "AnswerOutput", "SearchResult", "FileUploadResult",
    
    # 配置管理
    "AppConfig", "AppNeo4jConfig", "AppLLMConfig", "AppRAGConfig", "LogConfig",
    "load_config", "init_config", "get_config",
    "get_neo4j_config", "get_llm_config", "get_rag_config", "get_log_config",
    
    # 智能体系统
    "Agent",
    "ConceptExplanationAgent", "RequirementsAnalysisAgent", "SoftwareDesignAgent",
    "SoftwareTestingAgent", "ProjectManagementAgent", "CodeImplementationAgent", "SoftwareEthicsAgent",
    "AgentCoordinator",
    "select_agents_function", "synthesize_answers_function",
    
    # 主系统
    "RAGManager", "Neo4jHandler", "EntityExtractor", "EntityMatcher", "AnswerGenerator",
    "SoftwareEngineeringQASystem", "initialize_qa_system"
]
