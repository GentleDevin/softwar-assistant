# 项目优化代码示例

这个文件包含了针对原项目的具体优化代码实现

---

## 1. 改进的错误处理系统

```python
# improved_error_handling.py

from enum import Enum
from dataclasses import dataclass
from typing import Optional
import traceback
import logging
from datetime import datetime

class ErrorSeverity(Enum):
    """错误严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ErrorType(Enum):
    """错误类型"""
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
    """错误上下文信息"""
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    component: str
    timestamp: datetime
    exception: Optional[Exception] = None
    context_data: dict = None
    
    def to_dict(self):
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
    """基础异常"""
    def __init__(self, error_ctx: ErrorContext):
        self.error_ctx = error_ctx
        super().__init__(error_ctx.message)

class EntityExtractionException(QASystemException):
    """实体提取异常"""
    pass

class EntityMatchingException(QASystemException):
    """实体匹配异常"""
    pass

class Neo4jException(QASystemException):
    """Neo4j异常"""
    pass

class RAGException(QASystemException):
    """RAG异常"""
    pass

class LLMException(QASystemException):
    """LLM异常"""
    pass

class ErrorHandler:
    """统一的错误处理器"""
    
    def __init__(self):
        self.logger = logging.getLogger('qa_system_errors')
        self.error_history = []
        self.max_history = 100
    
    def handle(self, error_ctx: ErrorContext) -> str:
        """
        处理错误并返回用户友好的消息
        """
        # 记录错误
        self._log_error(error_ctx)
        self._store_error(error_ctx)
        
        # 根据错误类型采取不同的行动
        if error_ctx.severity == ErrorSeverity.CRITICAL:
            self._alert_critical_error(error_ctx)
        
        # 返回用户友好的消息
        return self._get_user_message(error_ctx)
    
    def _log_error(self, error_ctx: ErrorContext):
        """记录错误到日志"""
        log_method = {
            ErrorSeverity.INFO: self.logger.info,
            ErrorSeverity.WARNING: self.logger.warning,
            ErrorSeverity.ERROR: self.logger.error,
            ErrorSeverity.CRITICAL: self.logger.critical,
        }[error_ctx.severity]
        
        log_method(f"[{error_ctx.component}] {error_ctx.message}", 
                  extra=error_ctx.to_dict())
    
    def _store_error(self, error_ctx: ErrorContext):
        """存储错误供后续分析"""
        self.error_history.append(error_ctx)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
    
    def _alert_critical_error(self, error_ctx: ErrorContext):
        """发送关键错误告警 (可接入告警系统)"""
        # 这里可以接入企业级告警系统
        pass
    
    def _get_user_message(self, error_ctx: ErrorContext) -> str:
        """返回用户友好的错误消息"""
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

# 使用示例
def safe_extract_entities(question: str, entity_extractor, error_handler) -> list:
    """安全的实体提取"""
    try:
        return entity_extractor.extract_entities(question)
    except Exception as e:
        error_ctx = ErrorContext(
            error_type=ErrorType.ENTITY_EXTRACTION,
            severity=ErrorSeverity.ERROR,
            message=f"实体提取失败: {str(e)}",
            component="EntityExtractor",
            timestamp=datetime.now(),
            exception=e,
            context_data={'question': question[:100]}  # 只记录前100个字符
        )
        
        user_message = error_handler.handle(error_ctx)
        raise EntityExtractionException(error_ctx)
```

---

## 2. 改进的配置管理

```python
# improved_config.py

from dataclasses import dataclass, field, asdict
from typing import Optional
import os
import json
from pathlib import Path
from enum import Enum

class Environment(Enum):
    """运行环境"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class LLMConfig:
    """LLM配置"""
    api_key: str
    base_url: str
    model: str = "qwen-plus"
    temperature: float = 0.5
    max_tokens: int = 2048
    timeout: int = 30

@dataclass
class Neo4jConfig:
    """Neo4j配置"""
    uri: str
    username: str
    password: str
    database: str = "neo4j"
    max_connection_pool_size: int = 50
    connection_timeout: int = 30
    max_depth: int = 3

@dataclass
class RAGConfig:
    """RAG配置"""
    chunk_size: int = 800
    chunk_overlap: int = 150
    similarity_threshold: float = 0.3
    top_k: int = 3
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    cache_size: int = 1000
    cache_ttl: int = 3600

@dataclass
class EntityMatchingConfig:
    """实体匹配配置"""
    similarity_threshold: float = 0.85
    use_cache: bool = True
    cache_path: str = "entity_embeddings.pkl"

@dataclass
class QASystemConfig:
    """整体QA系统配置"""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    # 子配置
    llm: LLMConfig = field(default_factory=LLMConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    entity_matching: EntityMatchingConfig = field(default_factory=EntityMatchingConfig)
    
    # 系统配置
    max_conversation_history: int = 10
    log_level: str = "INFO"
    log_file: str = "qa_system.log"
    enable_monitoring: bool = True
    enable_health_check: bool = True
    
    @classmethod
    def from_env_file(cls, env_file: str = ".env") -> 'QASystemConfig':
        """从环境变量文件加载配置"""
        from dotenv import load_dotenv
        
        load_dotenv(env_file)
        
        # 读取环境变量
        environment = Environment(os.getenv('ENVIRONMENT', 'development'))
        debug = os.getenv('DEBUG', 'false').lower() == 'true'
        
        llm_config = LLMConfig(
            api_key=os.getenv('DASHSCOPE_API_KEY') or os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
            model=os.getenv('LLM_MODEL', 'qwen-plus'),
            temperature=float(os.getenv('LLM_TEMPERATURE', 0.5)),
        )
        
        neo4j_config = Neo4jConfig(
            uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            username=os.getenv('NEO4J_USER', 'neo4j'),
            password=os.getenv('NEO4J_PASSWORD', 'neo4j123'),
        )
        
        rag_config = RAGConfig(
            chunk_size=int(os.getenv('RAG_CHUNK_SIZE', 800)),
            similarity_threshold=float(os.getenv('RAG_SIMILARITY_THRESHOLD', 0.3)),
        )
        
        return cls(
            environment=environment,
            debug=debug,
            llm=llm_config,
            neo4j=neo4j_config,
            rag=rag_config,
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
        )
    
    @classmethod
    def from_json_file(cls, json_file: str) -> 'QASystemConfig':
        """从JSON文件加载配置"""
        with open(json_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """转换为字典 (不包含敏感信息)"""
        return {
            'environment': self.environment.value,
            'debug': self.debug,
            'llm': {'model': self.llm.model, 'temperature': self.llm.temperature},
            'neo4j': {'uri': self.neo4j.uri, 'database': self.neo4j.database},
            'rag': {'chunk_size': self.rag.chunk_size, 'top_k': self.rag.top_k},
        }
    
    def validate(self) -> list:
        """验证配置，返回错误列表"""
        errors = []
        
        if not self.llm.api_key:
            errors.append("LLM API密钥未配置")
        
        if not self.neo4j.uri:
            errors.append("Neo4j URI未配置")
        
        if self.entity_matching.similarity_threshold < 0 or self.entity_matching.similarity_threshold > 1:
            errors.append("实体匹配阈值必须在0-1之间")
        
        if self.rag.similarity_threshold < 0 or self.rag.similarity_threshold > 1:
            errors.append("RAG相似度阈值必须在0-1之间")
        
        return errors

# 使用示例
if __name__ == '__main__':
    # 从环境变量加载
    config = QASystemConfig.from_env_file('.env')
    
    # 验证配置
    errors = config.validate()
    if errors:
        for error in errors:
            print(f"配置错误: {error}")
    
    # 打印配置
    print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
```

---

## 3. 改进的性能监控

```python
# improved_monitoring.py

from dataclasses import dataclass, field
from typing import Dict, Callable
import time
import logging
from contextlib import contextmanager
from collections import defaultdict, deque
from functools import wraps

@dataclass
class PerformanceMetrics:
    """性能指标"""
    entity_extraction_time: float = 0.0
    entity_matching_time: float = 0.0
    kg_retrieval_time: float = 0.0
    rag_search_time: float = 0.0
    agent_selection_time: float = 0.0
    llm_response_time: float = 0.0
    total_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'entity_extraction': f"{self.entity_extraction_time:.3f}s",
            'entity_matching': f"{self.entity_matching_time:.3f}s",
            'kg_retrieval': f"{self.kg_retrieval_time:.3f}s",
            'rag_search': f"{self.rag_search_time:.3f}s",
            'agent_selection': f"{self.agent_selection_time:.3f}s",
            'llm_response': f"{self.llm_response_time:.3f}s",
            'total': f"{self.total_time:.3f}s"
        }
    
    def get_slowest_step(self) -> str:
        """获取最慢的步骤"""
        steps = {
            '实体提取': self.entity_extraction_time,
            '实体匹配': self.entity_matching_time,
            'KG检索': self.kg_retrieval_time,
            'RAG搜索': self.rag_search_time,
            '智能体选择': self.agent_selection_time,
            'LLM响应': self.llm_response_time,
        }
        return max(steps, key=steps.get)

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.timings = defaultdict(list)
        self.max_history = 100
    
    @contextmanager
    def measure(self, step_name: str):
        """
        上下文管理器 - 测量代码块执行时间
        
        使用示例:
            with monitor.measure("entity_extraction"):
                entities = extractor.extract(question)
        """
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.timings[step_name].append(elapsed)
            
            # 限制历史记录大小
            if len(self.timings[step_name]) > self.max_history:
                self.timings[step_name].pop(0)
            
            # 如果耗时过长，记录警告
            if elapsed > 5.0:
                self.logger.warning(f"{step_name} 耗时过长: {elapsed:.3f}s")
            else:
                self.logger.debug(f"{step_name} 耗时: {elapsed:.3f}s")
    
    def timing_decorator(self, step_name: str):
        """装饰器 - 自动测量函数执行时间"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.measure(step_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_average_time(self, step_name: str) -> float:
        """获取平均执行时间"""
        times = self.timings.get(step_name, [])
        return sum(times) / len(times) if times else 0.0
    
    def get_statistics(self, step_name: str) -> Dict:
        """获取统计信息"""
        times = self.timings.get(step_name, [])
        if not times:
            return {}
        
        return {
            'count': len(times),
            'min': f"{min(times):.3f}s",
            'max': f"{max(times):.3f}s",
            'avg': f"{sum(times) / len(times):.3f}s",
            'total': f"{sum(times):.3f}s"
        }
    
    def report(self) -> Dict:
        """生成性能报告"""
        report = {}
        for step_name in self.timings.keys():
            report[step_name] = self.get_statistics(step_name)
        return report

# 使用示例
monitor = PerformanceMonitor()

# 方式1: 上下文管理器
def answer_question(qa_system, question):
    metrics = PerformanceMetrics()
    
    with monitor.measure('entity_extraction'):
        entities = qa_system.entity_extractor.extract_entities(question)
    metrics.entity_extraction_time = monitor.timings['entity_extraction'][-1]
    
    with monitor.measure('entity_matching'):
        matched_entities = qa_system.entity_matcher.match_entity(entities[0])
    metrics.entity_matching_time = monitor.timings['entity_matching'][-1]
    
    # ... 其他步骤
    
    return metrics

# 方式2: 装饰器
class ImprovedEntityExtractor:
    def __init__(self, llm, monitor):
        self.llm = llm
        self.monitor = monitor
    
    @property
    def _measure_extraction(self):
        return self.monitor.timing_decorator('entity_extraction')
    
    # 使用装饰器
    def extract_entities(self, question: str):
        # 实现...
        pass
```

---

## 4. 改进的连接池管理

```python
# improved_connection_pool.py

from queue import Queue, Empty
import threading
from typing import Any
import time
import logging

class ConnectionPool:
    """连接池 - 管理数据库连接"""
    
    def __init__(self, connection_factory, max_connections=10, timeout=30):
        """
        初始化连接池
        
        Args:
            connection_factory: 创建连接的函数
            max_connections: 最大连接数
            timeout: 获取连接的超时时间（秒）
        """
        self.connection_factory = connection_factory
        self.max_connections = max_connections
        self.timeout = timeout
        self.pool = Queue(maxsize=max_connections)
        self.lock = threading.Lock()
        self.active_connections = 0
        self.logger = logging.getLogger(__name__)
        
        # 初始化连接池
        self._initialize_pool()
    
    def _initialize_pool(self):
        """初始化连接池"""
        for _ in range(self.max_connections):
            try:
                conn = self.connection_factory()
                self.pool.put(conn, block=False)
            except Exception as e:
                self.logger.error(f"初始化连接失败: {e}")
    
    def get_connection(self) -> Any:
        """获取连接"""
        try:
            conn = self.pool.get(timeout=self.timeout)
            with self.lock:
                self.active_connections += 1
            return conn
        except Empty:
            raise ConnectionPoolException(
                f"在{self.timeout}秒内无法获取可用连接"
            )
    
    def return_connection(self, conn: Any):
        """归还连接"""
        if conn is not None:
            self.pool.put(conn)
            with self.lock:
                self.active_connections -= 1
    
    def close_all(self):
        """关闭所有连接"""
        while not self.pool.empty():
            try:
                conn = self.pool.get_nowait()
                if hasattr(conn, 'close'):
                    conn.close()
            except Empty:
                break
    
    def get_stats(self) -> dict:
        """获取连接池统计信息"""
        return {
            'max_connections': self.max_connections,
            'active_connections': self.active_connections,
            'available_connections': self.pool.qsize(),
        }

class ConnectionPoolException(Exception):
    """连接池异常"""
    pass

# 使用示例
def create_neo4j_connection():
    """创建Neo4j连接"""
    from neo4j import GraphDatabase
    return GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "password")
    )

# 创建连接池
pool = ConnectionPool(
    connection_factory=create_neo4j_connection,
    max_connections=50,
    timeout=30
)

# 使用连接
try:
    conn = pool.get_connection()
    try:
        # 使用连接
        pass
    finally:
        pool.return_connection(conn)
except ConnectionPoolException as e:
    print(f"连接池错误: {e}")
finally:
    pool.close_all()
```

---

## 5. 改进的数据验证

```python
# improved_validation.py

from pydantic import BaseModel, validator, Field
from typing import List, Dict, Optional
from enum import Enum

class EntityType(str, Enum):
    """实体类型枚举"""
    CONCEPT = "概念"
    METHOD = "方法"
    TOOL = "工具"
    MODEL = "模型"
    PRINCIPLE = "原则"
    # ...

class Entity(BaseModel):
    """实体验证模型"""
    name: str = Field(..., min_length=1, max_length=100)
    type: EntityType
    
    @validator('name')
    def name_not_empty(cls, v):
        if not v.strip():
            raise ValueError('实体名称不能为空白')
        return v.strip()

class QuestionInput(BaseModel):
    """用户问题输入验证"""
    text: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None
    
    @validator('text')
    def text_valid(cls, v):
        if not v.strip():
            raise ValueError('问题不能为空')
        if len(v) > 1000:
            raise ValueError('问题过长（最大1000字符）')
        return v.strip()

class AnswerOutput(BaseModel):
    """答案输出模型"""
    answer: str
    agent_used: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    has_kg_context: bool
    doc_results_count: int = Field(..., ge=0)
    execution_time: float = Field(..., ge=0.0)
    timestamp: str

class SearchResult(BaseModel):
    """搜索结果"""
    source: str
    text: str = Field(..., max_length=500)
    similarity: float = Field(..., ge=0.0, le=1.0)

# 使用示例
def validate_and_process_question(question_text: str):
    """验证并处理问题"""
    try:
        # 输入验证
        question_input = QuestionInput(text=question_text)
        
        # 处理经过验证的问题
        result = qa_system.answer_question(question_input.text)
        
        # 输出验证
        answer_output = AnswerOutput(
            answer=result['answer'],
            agent_used=result.get('agent_name', 'unknown'),
            confidence=result.get('confidence', 0.5),
            has_kg_context=len(result.get('kg_context', {}).get('entities', [])) > 0,
            doc_results_count=len(result.get('doc_results', [])),
            execution_time=result.get('execution_time', 0.0),
            timestamp=result.get('timestamp', '')
        )
        
        return answer_output.dict()
        
    except ValueError as e:
        return {'error': str(e), 'status': 'validation_error'}
```

---

## 总结

这些改进涵盖了以下方面：

1. **错误处理**: 从通用异常到具体的异常类型，包含详细的上下文信息
2. **配置管理**: 集中式配置，支持多种加载方式，内置验证
3. **性能监控**: 详细的时间跟踪，统计分析
4. **连接管理**: 连接池，防止资源泄露
5. **数据验证**: 使用Pydantic进行输入/输出验证

这些改进使代码更加**可维护、可调试、可监控**，更符合生产级别的要求。
