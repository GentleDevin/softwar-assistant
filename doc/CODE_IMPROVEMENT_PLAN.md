# 软件工程课程助手 - 代码改进方案

## 项目概述

本项目是一个基于大语言模型与知识图谱的软件工程课程智能问答系统。通过分析项目代码，发现以下几个主要改进方向：

---

## 一、代码质量问题分析

### 1.1 日志系统问题

| 文件 | 问题描述 | 严重程度 |
|------|----------|----------|
| `agents.py` | 仍然使用 `print()` 语句进行调试输出 | 中 |
| `softeng_kg_qa.py` | 日志调用与 `print()` 混用 | 中 |
| `softeng_qa_ui.py` | 使用 `print()` 进行调试 | 中 |

**问题详情**：
- `agents.py` 第594、655、679行仍有 `print()` 语句
- `softeng_qa_ui.py` 第219、223、283、325行使用 `print()`

### 1.2 代码重复问题

| 问题类型 | 位置 | 影响 |
|----------|------|------|
| CSS样式重复 | `softeng_qa_ui.py` 中多处定义相同CSS | 维护困难 |
| 格式化逻辑重复 | `Agent._format_kg_context` 与 `ResponseGenerator._format_kg_context` 重复 | 代码冗余 |
| Neo4jHandler重复 | `softeng_kg_qa.py` 和 `kg_construct.py` 各有一套实现 | 代码冗余 |

### 1.3 配置管理问题

| 文件 | 问题 |
|------|------|
| `softeng_qa_ui.py` | Neo4j密码硬编码 |
| `kg_construct.py` | Neo4j密码硬编码 |
| `neo4j_loader.py` | 默认密码硬编码 |

### 1.4 类型注解完整性

部分函数缺少类型注解，影响代码可维护性和IDE支持。

### 1.5 异常处理改进空间

- 异常捕获过于宽泛（`except Exception`）
- 缺少异常日志记录
- 缺少优雅降级机制

### 1.6 单例模式问题

`RAGManager` 的单例实现缺少线程安全保护的完整测试。

---

## 二、代码改进方案

### 2.1 日志系统优化

**目标**：统一日志系统，移除所有 `print()` 语句

**改进步骤**：

```python
# 在 agents.py 中添加日志配置
import logging

logger = logging.getLogger(__name__)

# 将所有 print() 替换为 logger 调用
# 例如:
# print(f"AgentCoordinator: 初始化了 {len(self.agents)} 个智能体")
# 改为:
logger.info(f"AgentCoordinator: 初始化了 {len(self.agents)} 个智能体")
```

**需要修改的文件**：
- `agents.py` - 替换所有 `print()` 为 `logger.*`
- `softeng_qa_ui.py` - 替换所有 `print()` 为 `logger.*`

### 2.2 代码去重与重构

**方案A：提取公共格式化工具**

创建 `utils/formatters.py`：

```python
# utils/formatters.py
def format_kg_context(kg_context: Dict[str, Any]) -> str:
    """统一格式化知识图谱上下文"""
    if not kg_context or (not kg_context.get("entities") and not kg_context.get("paths")):
        return ""
    
    formatted_parts = []
    if kg_context.get("entities"):
        formatted_parts.append("--- 知识图谱实体与关系 ---")
        for entity_data in kg_context["entities"]:
            entity = entity_data.get("entity", {})
            relationships = entity_data.get("relationships", [])
            
            if not entity or not entity.get("name"):
                continue
            
            entity_name = entity["name"]
            entity_type = entity.get("type", "?")
            formatted_parts.append(f"\n实体: {entity_name} (类型: {entity_type})")
            
            if relationships:
                formatted_parts.append("  关系:")
                for rel in relationships:
                    rel_name = rel.get("rel_name") or rel.get("relationship", "?")
                    if rel.get("direction") == "outgoing":
                        target = rel.get("target", "?")
                        target_type = rel.get("target_type", "?")
                        formatted_parts.append(
                            f"    - {entity_name} --[{rel_name}]--> {target} (类型: {target_type})")
                    elif rel.get("direction") == "incoming":
                        source = rel.get("source", "?")
                        source_type = rel.get("source_type", "?")
                        formatted_parts.append(
                            f"    - {source} (类型: {source_type}) --[{rel_name}]--> {entity_name}")
    
    if kg_context.get("paths"):
        formatted_parts.append("\n--- 知识图谱实体间路径 ---")
        for i, path_data in enumerate(kg_context["paths"]):
            nodes = path_data.get("nodes", [])
            rels = path_data.get("relationships", [])
            
            if not nodes or not rels:
                continue
            
            path_str = f"\n路径 {i + 1}: "
            elements = []
            
            for j, node in enumerate(nodes):
                elements.append(f"({node.get('name', '?')})")
                if j < len(rels):
                    rel = rels[j]
                    rel_name = rel.get('name') or rel.get('type', '?')
                    elements.append(
                        f"--[{rel_name}]-->" if rel.get('source') == node.get('name') else f"<--[{rel_name}]--")
            
            formatted_parts.append(path_str + "".join(elements))
    
    return "\n".join(formatted_parts)


def format_doc_context(doc_results: List[Dict[str, Any]]) -> str:
    """统一格式化文档上下文"""
    if not doc_results:
        return ""
    
    formatted_parts = ["--- 相关文档片段 ---"]
    for i, doc in enumerate(doc_results):
        source = doc.get("source", "?")
        text = doc.get("text", "").strip()
        similarity = doc.get("similarity", 0.0)
        max_len = 300
        truncated = text[:max_len] + ("..." if len(text) > max_len else "")
        formatted_parts.append(f"\n片段 {i + 1} (来源: {source}, Sim: {similarity:.3f}):\n---\n{truncated}\n---")
    
    return "\n".join(formatted_parts)
```

**方案B：统一CSS样式管理**

创建 `ui/styles.py`：

```python
# ui/styles.py

COMMON_CSS = """
/* 通用样式 */
.gr-examples button {
    background-color: #f0f7ff !important; 
    color: #2a81e3 !important;
    border: 1px solid #bde0ff !important; 
    border-radius: 20px !important;
    margin: 5px !important; 
    padding: 8px 18px !important; 
    font-size: 13px !important;
    transition: all 0.2s ease;
}
.gr-examples button:hover { 
    background-color: #e1f0ff !important; 
    box-shadow: 0 2px 5px rgba(42, 129, 227, 0.2); 
}
.styled-table {
    border-collapse: collapse; margin: 10px 0; font-size: 0.9em;
    font-family: sans-serif; min-width: 400px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); width: 100%;
}
.styled-table th {
    background-color: #4a90e2; color: #ffffff; text-align: left;
    padding: 10px 12px; border: 1px solid #ddd;
}
.styled-table td {
    padding: 10px 12px; border: 1px solid #ddd;
}
.styled-table tbody tr:nth-of-type(even) { background-color: #f8f8f8; }
.styled-table tbody tr:last-of-type { border-bottom: 2px solid #4a90e2; }
h4 { margin-top: 15px; margin-bottom: 5px; color: #333; }
h5 { margin-top: 10px; margin-bottom: 5px; color: #555; }
"""

CHAT_CSS = """
/* 聊天界面样式 */
body { background-color: #f8f9fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
h1 { color: #2a81e3; text-align: center; font-size: 28px; margin-bottom: 5px; font-weight: bold; }
h2 { font-size: 18px; margin-top: 0; margin-bottom: 15px; color: #555; text-align: center; font-weight: normal; }
h3 { color: #333; margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 5px;}
h4 { color: #444; margin-top: 15px; margin-bottom: 10px; font-size: 1.1em; }

.container { max-width: 1600px; margin: 20px auto; padding: 0 15px; }

.header { text-align: center; margin-bottom: 30px; }
.upload-section, .qa-section {
    background-color: white; padding: 20px; border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08); margin-bottom: 25px;
}
.chat-section {
    background-color: white; padding: 20px; border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08); margin-bottom: 20px;
}

.message { 
    margin: 8px 0; 
    padding: 8px 12px;
    border-radius: 10px;
}
.user-message {
    text-align: right;
    background-color: #e1f0ff;
    margin-left: 35%;
    display: inline-block;
    max-width: 65%;
}
.assistant-message {
    text-align: left;
    background-color: #f0f0f0;
    margin-right: 15%;
    display: inline-block;
    max-width: 85%;
}

.chatbot {
    max-width: 100% !important;
}
.chatbot .message-wrap {
    padding: 10px 20px !important;
}

.resizable-chat {
    resize: horizontal;
    overflow: auto;
    min-width: 600px;
    max-width: 100%;
}

.example-btn { 
    background-color: #f0f7ff !important; color: #2a81e3 !important;
    border: 1px solid #bde0ff !important; border-radius: 20px !important;
    margin: 5px !important; padding: 8px 18px !important; font-size: 13px !important;
    transition: all 0.2s ease;
}
.example-btn:hover { background-color: #e1f0ff !important; box-shadow: 0 2px 5px rgba(42, 129, 227, 0.2); }
.submit-btn { 
    background-color: #2a81e3 !important; color: white !important; font-weight: bold;
    border-radius: 8px !important; padding: 12px 20px !important; font-size: 15px !important;
    transition: background-color 0.2s ease;
}
.submit-btn:hover { background-color: #1e6ac7 !important; }
.clear-btn { 
    background-color: #dc3545 !important; color: white !important;
    border-radius: 8px !important; padding: 10px 16px !important; font-size: 14px !important;
}
.clear-btn:hover { background-color: #c82333 !important; }
.tab-content { 
    padding: 20px; background-color: #ffffff; border-radius: 0 0 8px 8px;
    border: 1px solid #dee2e6; border-top: none; min-height: 200px;
}
footer { text-align: center; margin-top: 30px; color: #888; font-size: 12px; }
.book-icon { font-size: 36px; color: #2a81e3; margin-right: 10px; vertical-align: middle;}
.gr-label { 
    font-weight: bold !important; color: #555 !important; margin-bottom: 5px !important;
}
.gr-box { border-radius: 8px !important; }
.gr-input { border-radius: 8px !important; }
.gr-button { border-radius: 8px !important; }
.gr-dropdown { border-radius: 8px !important; }
.gr-tabs { border-radius: 8px 8px 0 0 !important; }
"""
```

### 2.3 配置管理改进

**方案：创建统一配置管理模块**

创建 `config/__init__.py`：

```python
# config/__init__.py
import os
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
class AppConfig:
    """应用配置"""
    neo4j: Neo4jConfig
    llm: LLMConfig
    rag: RAGConfig


def load_config() -> AppConfig:
    """从环境变量加载配置"""
    # Neo4j配置
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "")
    
    if not neo4j_password:
        raise ValueError("NEO4J_PASSWORD 环境变量未设置")
    
    # LLM配置
    llm_api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    llm_base_url = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    if not llm_api_key:
        raise ValueError("DASHSCOPE_API_KEY 或 OPENAI_API_KEY 环境变量未设置")
    
    return AppConfig(
        neo4j=Neo4jConfig(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password
        ),
        llm=LLMConfig(
            api_key=llm_api_key,
            base_url=llm_base_url
        ),
        rag=RAGConfig()
    )


# 全局配置实例
config: Optional[AppConfig] = None


def init_config():
    """初始化配置"""
    global config
    config = load_config()


def get_config() -> AppConfig:
    """获取配置"""
    if config is None:
        init_config()
    return config
```

**使用方式**：

```python
# 在需要配置的文件中
from config import get_config

config = get_config()
neo4j_handler = Neo4jHandler(
    config.neo4j.uri,
    config.neo4j.username,
    config.neo4j.password
)
```

### 2.4 异常处理改进

**方案：统一异常处理模式**

```python
# utils/exceptions.py
class QAException(Exception):
    """问答系统基础异常"""
    pass


class Neo4jConnectionError(QAException):
    """Neo4j连接异常"""
    pass


class LLMError(QAException):
    """LLM调用异常"""
    pass


class EntityExtractionError(QAException):
    """实体提取异常"""
    pass


class DocumentProcessingError(QAException):
    """文档处理异常"""
    pass


def handle_exception(func):
    """异常处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Neo4jConnectionError as e:
            logger.error(f"Neo4j连接错误: {e}")
            raise
        except LLMError as e:
            logger.error(f"LLM调用错误: {e}")
            # 可以实现降级策略
            return f"抱歉，服务暂时不可用，请稍后重试。"
        except Exception as e:
            logger.error(f"未知错误: {e}", exc_info=True)
            raise
    return wrapper
```

### 2.5 单元测试增强

**方案：增加测试覆盖率**

```python
# tests/test_agents.py
import unittest
from unittest.mock import Mock, patch, MagicMock
from agents import Agent, AgentCoordinator, ConceptExplanationAgent


class TestAgentProcess(unittest.TestCase):
    """测试智能体处理逻辑"""
    
    @patch('agents.LLMChain')
    def test_agent_process_with_empty_context(self, mock_llm_chain):
        """测试空上下文处理"""
        mock_llm = MagicMock()
        agent = ConceptExplanationAgent(mock_llm)
        
        # Mock chain.invoke 返回预期结果
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"text": "测试回答"}
        mock_llm_chain.return_value = mock_chain
        
        result = agent.process(
            question="测试问题",
            kg_context={},
            entities=[],
            doc_results=[],
            conversation_history=[]
        )
        
        self.assertEqual(result, "测试回答")
        mock_chain.invoke.assert_called_once()
    
    @patch('agents.LLMChain')
    def test_agent_process_with_llm_error(self, mock_llm_chain):
        """测试LLM调用错误处理"""
        mock_llm = MagicMock()
        agent = ConceptExplanationAgent(mock_llm)
        
        # Mock chain.invoke 抛出异常
        mock_llm_chain.side_effect = Exception("LLM服务错误")
        
        result = agent.process(
            question="测试问题",
            kg_context={},
            entities=[],
            doc_results=[],
            conversation_history=[]
        )
        
        # 应该返回错误消息
        self.assertIn("遇到了问题", result)


class TestAgentCoordinator(unittest.TestCase):
    """测试智能体协调器"""
    
    @patch('agents.select_agents_function')
    def test_agent_selection(self, mock_select):
        """测试智能体选择逻辑"""
        mock_llm = MagicMock()
        coordinator = AgentCoordinator(mock_llm)
        
        # Mock 选择结果
        mock_select.return_value = '[{"agent": "概念解释智能体", "relevance": 0.9, "reasoning": "测试"}]'
        
        selected = coordinator.select_agents("什么是敏捷开发？", [])
        
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][0].name, "概念解释智能体")


if __name__ == '__main__':
    unittest.main()
```

### 2.6 性能优化建议

**方案A：异步处理改进**

```python
# async_utils.py
import asyncio
from typing import Callable, Any


async def async_process_agents(agents, question, kg_context, entities, doc_results, history):
    """异步并行处理多个智能体"""
    async def process_agent(agent, relevance):
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            None,
            agent.process,
            question,
            kg_context,
            entities,
            doc_results,
            history
        )
        return {"agent": agent.name, "relevance": relevance, "answer": answer}
    
    tasks = [process_agent(agent, relevance) for agent, relevance in agents]
    return await asyncio.gather(*tasks)
```

**方案B：缓存优化**

```python
# cache_utils.py
from functools import lru_cache
from typing import Any, Callable


class EmbeddingCache:
    """嵌入缓存管理器"""
    
    def __init__(self, maxsize: int = 1000):
        self._cache = {}
        self._maxsize = maxsize
    
    def get(self, key: str) -> Any:
        """获取缓存"""
        return self._cache.get(key)
    
    def set(self, key: str, value: Any):
        """设置缓存"""
        if len(self._cache) >= self._maxsize:
            # 简单的FIFO淘汰策略
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value
    
    def clear(self):
        """清空缓存"""
        self._cache.clear()
    
    def __contains__(self, key: str) -> bool:
        return key in self._cache
```

---

## 三、改进优先级

| 优先级 | 改进项 | 预期收益 | 实施难度 |
|--------|--------|----------|----------|
| P0 | 日志系统统一 | 提高可观测性 | 低 |
| P0 | 配置管理改进 | 消除硬编码，提高安全性 | 低 |
| P1 | 代码去重 | 提高可维护性 | 中 |
| P1 | 异常处理统一 | 提高系统健壮性 | 中 |
| P2 | 单元测试增强 | 提高代码质量 | 中 |
| P2 | 性能优化 | 提高响应速度 | 高 |

---

## 四、实施计划

### 第1周：基础改进

- [ ] 移除所有 `print()` 语句，替换为 `logger` 调用
- [ ] 创建统一配置管理模块
- [ ] 更新所有使用配置的文件

### 第2周：代码重构

- [ ] 创建 `utils/formatters.py` 统一格式化工具
- [ ] 创建 `ui/styles.py` 统一样式管理
- [ ] 清理重复代码

### 第3周：健壮性增强

- [ ] 创建统一异常类
- [ ] 增加单元测试覆盖率
- [ ] 代码审查和测试验证

### 第4周：性能优化（可选）

- [ ] 实现异步智能体处理
- [ ] 优化缓存策略
- [ ] 性能测试和调优

---

## 五、代码质量检查清单

### 完成标准

| 检查项 | 完成状态 |
|--------|----------|
| 所有 `print()` 替换为 `logger` | ☐ |
| 配置从环境变量加载 | ☐ |
| 代码重复率 < 10% | ☐ |
| 异常处理覆盖率 100% | ☐ |
| 单元测试覆盖率 > 60% | ☐ |
| 类型注解完整 | ☐ |

---

## 六、参考文档

1. [Python日志最佳实践](https://docs.python.org/3/howto/logging.html)
2. [LangChain最佳实践](https://python.langchain.com/docs/guides/)
3. [代码质量标准](https://google.github.io/styleguide/pyguide.html)
