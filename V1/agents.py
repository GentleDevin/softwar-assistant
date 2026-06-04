# -*- coding: utf-8 -*-
import json
import logging
import re
from typing import List, Dict, Any, Tuple, Optional

from langchain_classic import LLMChain
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

# 配置日志
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class Agent:
    """所有软件工程智能体的基类"""
    
    def __init__(self, llm: ChatOpenAI, name: str, description: str, system_message: str):
        self.llm = llm
        self.name = name
        self.description = description
        self.system_message = system_message
    
    def process(self, question: str, kg_context: Dict[str, Any], 
                entities: List[Dict[str, str]], doc_results: List[Dict[str, Any]],
                conversation_history: List[Tuple[str, str]] = None) -> str:
        """
        处理问题并生成响应，支持对话历史
        
        Args:
            question: 当前问题
            kg_context: 知识图谱上下文
            entities: 提取的实体
            doc_results: 文档检索结果
            conversation_history: 对话历史（用户问题，系统回答）对列表
        """
        # 准备上下文
        kg_context_str = self._format_kg_context(kg_context)
        doc_context_str = self._format_doc_context(doc_results)
        entities_str = json.dumps(entities, ensure_ascii=False, indent=2)
        history_str = self._format_conversation_history(conversation_history)
        
        # 设置提示模板
        prompt_template = self._get_prompt_template()
        
        # 提供所有必需的变量
        prompt_vars = {
            "question": question,
            "entities": entities_str,
            "kg_context": kg_context_str or "无相关知识图谱信息。",
            "doc_context": doc_context_str or "无相关文档片段。",
            "history": history_str or "这是新的对话"
        }
        
        # 使用LangChain链生成回答
        try:
            chain = LLMChain(
                llm=self.llm,
                prompt=prompt_template
            )
            return chain.invoke(prompt_vars)["text"]
        except Exception as e:
            logger.error(f"{self.name}调用LLM时出错: {e}")
            return f"抱歉，在生成回答时遇到了问题：{e}。"
    
    def _get_prompt_template(self) -> PromptTemplate:
        """获取提示模板 - 子类可以重写以提供自定义模板"""
        template = """
        任务：作为软件工程课程助手，根据【知识图谱信息】、【相关文档片段】和【对话历史】回答【用户问题】。
        【当前问题】: {question}
        【提取实体】: {entities}
        【知识图谱信息】: {kg_context}
        【相关文档片段】: {doc_context}
        【对话历史】: {history}
        
        请生成回答:
        """
        return PromptTemplate(
            template=template,
            input_variables=["question", "entities", "kg_context", "doc_context", "history"]
        )
    
    def _format_kg_context(self, kg_context: Dict[str, Any]) -> str:
        """格式化知识图谱上下文（共享实现）"""
        if not kg_context or (not kg_context.get("entities") and not kg_context.get("paths")): return ""
        formatted_parts = []
        if kg_context.get("entities"):
            formatted_parts.append("--- 知识图谱实体与关系 ---")
            for entity_data in kg_context["entities"]:
                entity = entity_data.get("entity", {}); relationships = entity_data.get("relationships", [])
                if not entity or not entity.get("name"): continue
                entity_name = entity["name"]; entity_type = entity.get("type", "?")
                formatted_parts.append(f"\n实体: {entity_name} (类型: {entity_type})")
                if relationships:
                    formatted_parts.append("  关系:")
                    for rel in relationships:
                        rel_name = rel.get("rel_name") or rel.get("relationship", "?")
                        if rel.get("direction") == "outgoing": target = rel.get("target", "?"); target_type = rel.get("target_type", "?"); formatted_parts.append(f"    - {entity_name} --[{rel_name}]--> {target} (类型: {target_type})")
                        elif rel.get("direction") == "incoming": source = rel.get("source", "?"); source_type = rel.get("source_type", "?"); formatted_parts.append(f"    - {source} (类型: {source_type}) --[{rel_name}]--> {entity_name}")
        if kg_context.get("paths"):
            formatted_parts.append("\n--- 知识图谱实体间路径 ---")
            for i, path_data in enumerate(kg_context["paths"]):
                nodes = path_data.get("nodes", []); rels = path_data.get("relationships", [])
                if not nodes or not rels: continue
                path_str = f"\n路径 {i+1}: "; elements = []
                for j, node in enumerate(nodes):
                    elements.append(f"({node.get('name', '?')})")
                    if j < len(rels): rel = rels[j]; rel_name = rel.get('name') or rel.get('type', '?'); elements.append(f"--[{rel_name}]-->" if rel.get('source') == node.get('name') else f"<--[{rel_name}]--")
                formatted_parts.append(path_str + "".join(elements))
        return "\n".join(formatted_parts)
        
    def _format_doc_context(self, doc_results: List[Dict[str, Any]]) -> str:
        """格式化文档结果（共享实现）"""
        if not doc_results: return ""
        formatted_parts = ["--- 相关文档片段 ---"]
        for i, doc in enumerate(doc_results):
            source = doc.get("source", "?"); text = doc.get("text", "").strip()
            similarity = doc.get("similarity", 0.0); max_len = 300
            truncated = text[:max_len] + ("..." if len(text) > max_len else "")
            formatted_parts.append(f"\n片段 {i+1} (来源: {source}, Sim: {similarity:.3f}):\n---\n{truncated}\n---")
        return "\n".join(formatted_parts)
    
    def _format_conversation_history(self, conversation_history: List[Tuple[str, str]]) -> str:
        """格式化对话历史"""
        if not conversation_history:
            return ""
        
        formatted_parts = []
        for i, (user_q, assistant_a) in enumerate(conversation_history[-5:]):  # 只使用最近5轮对话
            formatted_parts.append(f"用户第{i+1}轮问题: {user_q}")
            formatted_parts.append(f"助手第{i+1}轮回答: {assistant_a[:200]}...")  # 限制长度
            if i < len(conversation_history) - 1:
                formatted_parts.append("")  # 添加空行分隔
        
        return "\n".join(formatted_parts)

    def as_tool(self) -> Tool:
        """将智能体转换为LangChain工具"""
        def tool_func(input_str: str) -> str:
            try:
                # 尝试解析JSON输入
                input_data = json.loads(input_str)
                return self.process(
                    question=input_data.get("question", ""),
                    kg_context=input_data.get("kg_context", {}),
                    entities=input_data.get("entities", []),
                    doc_results=input_data.get("doc_results", []),
                    conversation_history=input_data.get("conversation_history", [])
                )
            except json.JSONDecodeError:
                # 如果不是有效的JSON，则假设它只是一个问题
                return self.process(
                    question=input_str,
                    kg_context={},
                    entities=[],
                    doc_results=[],
                    conversation_history=[]
                )
        
        return Tool(
            name=self.name,
            description=self.description,
            func=tool_func
        )

# 专业化的智能体实现

class ConceptExplanationAgent(Agent):
    """专门解释软件工程概念的智能体"""
    
    def __init__(self, llm: ChatOpenAI):
        super().__init__(
            llm,
            "概念解释智能体",
            "清晰全面地解释软件工程概念、术语和方法论",
            "你是软件工程课程助手中的概念解释专家，专长于清晰、准确地解释软件工程概念、术语和方法论。你的解释既权威又易于理解，适合不同层次的学生。你会根据对话历史提供连贯的解释。"
        )
    
    def _get_prompt_template(self) -> PromptTemplate:
        template = """
        任务：作为软件工程课程助手的概念解释专家，根据【知识图谱信息】、【相关文档片段】和【对话历史】解释【用户问题】中的软件工程概念。
        【当前问题】: {question}
        【提取实体】: {entities}
        【知识图谱信息】: {kg_context}
        【相关文档片段】: {doc_context}
        【对话历史】: {history}
        
        回答要求:
        1. 清晰解释概念的定义和核心特点
        2. 提供概念的背景和用途
        3. 如有必要，提供具体示例来帮助理解
        4. 使用层次化结构组织解释（从基础到高级）
        5. 适当引用软件工程权威标准或最佳实践
        6. 如果概念与其他概念密切相关，简要说明关系
        7. 考虑对话上下文，避免重复已解释过的内容，或进行更深入的拓展
        
        请生成解释:
        """
        return PromptTemplate(
            template=template,
            input_variables=["question", "entities", "kg_context", "doc_context", "history"]
        )


class RequirementsAnalysisAgent(Agent):
    """专门处理需求工程和分析的智能体"""
    
    def __init__(self, llm: ChatOpenAI):
        super().__init__(
            llm,
            "需求分析智能体",
            "协助需求工程、需求获取和需求分析",
            "你是软件工程课程助手中的需求分析专家，擅长需求工程各个方面，包括需求获取、分析、规格说明和管理。你提供的建议遵循业界最佳实践和标准，帮助学生理解并应用需求工程技术。你会根据对话历史提供连贯的指导。"
        )
    
    def _get_prompt_template(self) -> PromptTemplate:
        template = """
        任务：作为软件工程课程助手的需求分析专家，根据【知识图谱信息】、【相关文档片段】和【对话历史】回答【用户问题】中关于需求工程的问题。
        【当前问题】: {question}
        【提取实体】: {entities}
        【知识图谱信息】: {kg_context}
        【相关文档片段】: {doc_context}
        【对话历史】: {history}
        
        回答要求:
        1. 提供关于需求工程流程、技术或最佳实践的专业指导
        2. 如需解释需求相关概念，提供清晰定义并说明在软件开发中的作用
        3. 如问题涉及需求文档或规格说明，提供标准格式和关键内容建议
        4. 如问题涉及需求获取技术，详细解释方法的优缺点和适用场景
        5. 如需提供模板，给出结构化的示例
        6. 关注需求的质量属性（如明确性、可验证性、必要性、一致性等）
        7. 基于对话历史提供连贯的指导，避免重复或建立在之前讨论基础上深入探讨
        
        请生成回答:
        """
        return PromptTemplate(
            template=template,
            input_variables=["question", "entities", "kg_context", "doc_context", "history"]
        )


class SoftwareDesignAgent(Agent):
    """专门处理软件架构和设计的智能体"""
    
    def __init__(self, llm: ChatOpenAI):
        super().__init__(
            llm,
            "软件设计智能体",
            "协助软件架构、设计模式和建模",
            "你是软件工程课程助手中的软件设计专家，精通软件架构、设计模式、建模技术和设计原则。你帮助学生理解如何设计可维护、可扩展和高质量的软件系统。你会根据对话历史提供连续的设计指导。"
        )
    
    def _get_prompt_template(self) -> PromptTemplate:
        template = """
        任务：作为软件工程课程助手的软件设计专家，根据【知识图谱信息】、【相关文档片段】和【对话历史】回答【用户问题】中关于软件设计和架构的问题。
        【当前问题】: {question}
        【提取实体】: {entities}
        【知识图谱信息】: {kg_context}
        【相关文档片段】: {doc_context}
        【对话历史】: {history}
        
        回答要求:
        1. 如涉及设计模式，清晰解释其结构、用途、优缺点和实现方式
        2. 如涉及架构风格，解释其特点、适用场景和示例
        3. 如涉及UML或其他建模语言，提供规范的表示法和实例
        4. 解释设计原则（如SOLID、DRY、KISS等）并说明如何应用
        5. 提供设计决策的权衡分析，解释不同方案的优缺点
        6. 如适用，提供简洁的代码示例或伪代码说明设计实现
        7. 基于对话历史保持讨论连贯性，深入探讨用户感兴趣的设计方面
        
        请生成回答:
        """
        return PromptTemplate(
            template=template,
            input_variables=["question", "entities", "kg_context", "doc_context", "history"]
        )


class SoftwareTestingAgent(Agent):
    """专门处理软件测试和质量保证的智能体"""
    
    def __init__(self, llm: ChatOpenAI):
        super().__init__(
            llm,
            "软件测试智能体",
            "专注于测试策略、测试用例和质量保证",
            "你是软件工程课程助手中的软件测试专家，精通各种测试方法、测试类型、测试工具和质量保证流程。你帮助学生理解如何设计和执行有效的测试策略，确保软件质量。你会根据对话历史提供深入的测试指导。"
        )
    
    def _get_prompt_template(self) -> PromptTemplate:
        template = """
        任务：作为软件工程课程助手的软件测试专家，根据【知识图谱信息】、【相关文档片段】和【对话历史】回答【用户问题】中关于软件测试和质量保证的问题。
        【当前问题】: {question}
        【提取实体】: {entities}
        【知识图谱信息】: {kg_context}
        【相关文档片段】: {doc_context}
        【对话历史】: {history}
        
        回答要求:
        1. 如涉及测试方法，解释其流程、技术、优势以及适用场景
        2. 如涉及测试类型（单元测试、集成测试等），说明其目的、范围和执行方式
        3. 如涉及测试工具，提供其用途和基本使用指南
        4. 如涉及测试计划或测试案例设计，提供结构化指导和示例
        5. 如涉及缺陷管理，解释流程和最佳实践
        6. 关注测试的质量标准和度量指标
        7. 根据对话历史，逐步深入测试主题或拓展相关测试知识
        
        请生成回答:
        """
        return PromptTemplate(
            template=template,
            input_variables=["question", "entities", "kg_context", "doc_context", "history"]
        )


class ProjectManagementAgent(Agent):
    """专门处理软件项目管理和方法论的智能体"""
    
    def __init__(self, llm: ChatOpenAI):
        super().__init__(
            llm,
            "项目管理智能体",
            "协助软件开发生命周期、方法论和团队管理",
            "你是软件工程课程助手中的项目管理专家，精通软件开发生命周期、开发方法论、团队管理和项目规划。你帮助学生理解如何有效管理软件项目，应用合适的方法论，并解决项目管理中的挑战。你会基于对话历史提供更深入的项目管理指导。"
        )
    
    def _get_prompt_template(self) -> PromptTemplate:
        template = """
        任务：作为软件工程课程助手的项目管理专家，根据【知识图谱信息】、【相关文档片段】和【对话历史】回答【用户问题】中关于软件项目管理的问题。
        【当前问题】: {question}
        【提取实体】: {entities}
        【知识图谱信息】: {kg_context}
        【相关文档片段】: {doc_context}
        【对话历史】: {history}
        
        回答要求:
        1. 如涉及开发方法论，解释其原则、实践、优缺点和适用场景
        2. 如涉及项目计划，提供结构化的规划方法和关键要素
        3. 如涉及团队管理，提供角色定义和协作最佳实践
        4. 如涉及风险管理，解释识别、评估和缓解策略
        5. 如涉及项目监控和度量，说明关键指标和评估方法
        6. 如问题包含方法论比较，提供客观的对比分析
        7. 基于之前的讨论，深入探讨用户关心的项目管理方面
        
        请生成回答:
        """
        return PromptTemplate(
            template=template,
            input_variables=["question", "entities", "kg_context", "doc_context", "history"]
        )


class CodeImplementationAgent(Agent):
    """专门处理编码实践和实现的智能体"""
    
    def __init__(self, llm: ChatOpenAI):
        super().__init__(
            llm,
            "代码实现智能体",
            "协助编码实践、算法和实现细节",
            "你是软件工程课程助手中的代码实现专家，精通编程语言、算法、数据结构和编码最佳实践。你帮助学生实现高质量、可维护的代码，并理解软件工程中的编程实践。你会基于对话历史提供连续的代码指导。"
        )
    
    def _get_prompt_template(self) -> PromptTemplate:
        template = """
        任务：作为软件工程课程助手的代码实现专家，根据【知识图谱信息】、【相关文档片段】和【对话历史】回答【用户问题】中关于代码实现和编程实践的问题。
        【当前问题】: {question}
        【提取实体】: {entities}
        【知识图谱信息】: {kg_context}
        【相关文档片段】: {doc_context}
        【对话历史】: {history}
        
        回答要求:
        1. 如涉及编程语言或框架，提供准确的语法和用法示例
        2. 如涉及算法或数据结构，解释原理并提供简洁实现
        3. 如涉及编码规范或最佳实践，提供清晰指导和示例
        4. 如涉及API设计，解释设计原则和接口示例
        5. 如需提供代码，确保代码清晰、高效且符合软件工程规范
        6. 解释代码的质量特性（如可读性、可维护性、性能等）
        7. 根据对话历史，逐步完善代码示例或深入探讨编程技术
        
        请生成回答:
        """
        return PromptTemplate(
            template=template,
            input_variables=["question", "entities", "kg_context", "doc_context", "history"]
        )


class SoftwareEthicsAgent(Agent):
    """专门处理软件伦理和职业道德的智能体"""
    
    def __init__(self, llm: ChatOpenAI):
        super().__init__(
            llm,
            "软件伦理智能体",
            "处理软件伦理、职业道德和社会影响相关问题",
            "你是软件工程课程助手中的软件伦理专家，关注软件开发和应用中的伦理考量、社会责任和职业道德。你帮助学生理解和应对软件系统带来的伦理挑战和社会影响。你会基于对话历史深入探讨伦理问题。"
        )
    
    def _get_prompt_template(self) -> PromptTemplate:
        template = """
        任务：作为软件工程课程助手的软件伦理专家，根据【知识图谱信息】、【相关文档片段】和【对话历史】回答【用户问题】中关于软件伦理和职业道德的问题。
        【当前问题】: {question}
        【提取实体】: {entities}
        【知识图谱信息】: {kg_context}
        【相关文档片段】: {doc_context}
        【对话历史】: {history}
        
        回答要求:
        1. 如涉及伦理原则，解释其内涵和在软件开发中的应用
        2. 如涉及隐私或安全问题，提供保护用户和数据的最佳实践
        3. 如涉及偏见和公平性，解释如何设计包容性软件系统
        4. 如涉及法律和知识产权，提供合规建议和注意事项
        5. 如涉及社会影响，分析软件系统可能带来的广泛后果
        6. 在回答中保持平衡，考虑多方视角和利益相关者
        7. 基于对话历史，深入探讨伦理问题的复杂性和解决方案
        
        请生成回答:
        """
        return PromptTemplate(
            template=template,
            input_variables=["question", "entities", "kg_context", "doc_context", "history"]
        )


def select_agents_function(query: str, entities: List[Dict[str, str]], llm: ChatOpenAI, agents: List["Agent"]) -> str:
    """分析问题并选择合适的智能体"""
    if entities is None:
        entities = []
        
    entity_names = [entity.get("name", "") for entity in entities if entity.get("name")]
    entity_types = [entity.get("type", "") for entity in entities if entity.get("type")]
    
    agent_descriptions = [{"name": agent.name, "description": agent.description} for agent in agents]
    agent_json = json.dumps(agent_descriptions, ensure_ascii=False, indent=2)
    
    prompt = f"""
    任务：分析用户问题，选择最合适的一个或多个软件工程智能体来回答此问题。

    用户问题: {query}

    提取的实体:
    - 名称: {', '.join(entity_names) if entity_names else '无'}
    - 类型: {', '.join(entity_types) if entity_types else '无'}

    可用的智能体:
    {agent_json}

    请分析问题的软件工程领域和复杂度，然后选择一个或多个最合适的智能体来协作回答。
    
    输出格式(JSON):
    [
        {{"agent": "智能体名称", "relevance": 0.1-1.0之间的浮点数, "reasoning": "选择此智能体的理由"}},
        ...
    ]

    注意:
    1. relevance分数范围为0.1-1.0，表示智能体对回答此问题的相关性和重要性
    2. 只选择真正相关的智能体，相关性低于0.5的不要包含
    3. 对于复杂或跨领域问题，可以选择多个智能体
    4. 对于简单或单领域问题，可以只选择一个最合适的智能体
    """
    
    messages = [
        SystemMessage(content="你是一个高级软件工程分析器，专注于理解问题的领域和复杂度，以便分配最合适的专家智能体来回答。"),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        
        try:
            # 尝试解析JSON
            selected_agents_data = json.loads(content)
        except json.JSONDecodeError:
            # 如果解析失败，尝试从文本中提取JSON部分
            json_match = re.search(r'\[\s*{.*}\s*\]', content, re.DOTALL)
            if json_match:
                try:
                    selected_agents_data = json.loads(json_match.group(0))
                except Exception as inner_e:
                    logger.error(f"无法从LLM响应中提取有效的JSON: {inner_e}")
                    selected_agents_data = []
            else:
                selected_agents_data = []
        
        # 过滤并返回选择结果
        valid_selections = []
        for selection in selected_agents_data:
            agent_name = selection.get("agent")
            relevance = float(selection.get("relevance", 0))
            reasoning = selection.get("reasoning", "")
            
            if relevance >= 0.5:
                valid_selections.append({
                    "agent": agent_name,
                    "relevance": relevance,
                    "reasoning": reasoning
                })
        
        # 如果没有选择，返回默认
        if not valid_selections:
            valid_selections = [{
                "agent": "概念解释智能体",
                "relevance": 1.0,
                "reasoning": "未找到合适的智能体，使用概念解释智能体作为默认"
            }]
            
        return json.dumps(valid_selections, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"智能体选择出错: {e}")
        return json.dumps([{
            "agent": "概念解释智能体", 
            "relevance": 1.0,
            "reasoning": "选择过程出错，使用默认智能体"
        }], ensure_ascii=False)


def synthesize_answers_function(answers_json: str, query: str, llm: ChatOpenAI) -> str:
    """合成多个智能体的答案"""
    try:
        answers = json.loads(answers_json)
        
        if len(answers) == 1:
            return answers[0]["answer"]
        
        answers_text = []
        for i, answer_data in enumerate(answers):
            agent_name = answer_data.get("agent", f"智能体{i+1}")
            relevance = answer_data.get("relevance", 1.0)
            answer = answer_data.get("answer", "")
            
            # 截断过长的答案
            max_answer_len = 1000
            truncated_answer = answer[:max_answer_len] + ("..." if len(answer) > max_answer_len else "")
            answers_text.append(f"【智能体{i+1}】{agent_name} (相关性: {relevance:.2f}):\n{truncated_answer}\n")
        
        synthesis_prompt = f"""
        用户问题: {query}

        以下是来自多个软件工程专家智能体的回答:
        
        {"".join(answers_text)}
        
        请基于以上专家回答，综合出一个全面、连贯且无冗余的最终答案。
        注意:
        1. 合并共同点，保留各个答案的独特见解
        2. 优先考虑相关性较高的智能体的观点
        3. 解决可能的矛盾，给出一致的回答
        4. 保持专业且易于理解的语言风格
        5. 在最终答案中不要引用或提及智能体，直接以软件工程助手的身份回答

        最终综合答案:
        """
        
        messages = [
            SystemMessage(content="你是一个高级内容综合器，擅长整合多个专家的见解成一个连贯、全面的回答。"),
            HumanMessage(content=synthesis_prompt)
        ]
        
        response = llm.invoke(messages)
        return response.content.strip()
        
    except Exception as e:
        logger.error(f"答案合成出错: {e}")
        
        # 如果解析失败但字符串看起来像单个答案，则直接返回
        if isinstance(answers_json, str) and len(answers_json) > 100 and not answers_json.startswith('['):
            return answers_json
            
        return "抱歉，在合成答案时遇到了技术问题。请再次尝试您的问题。"


class AgentCoordinator:
    """智能体协调器，使用LLM选择多个智能体来协作处理问题"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.agents = []
        self._initialize_agents()
        
    def _initialize_agents(self):
        """创建并注册所有智能体"""
        self.agents = [
            ConceptExplanationAgent(self.llm),
            RequirementsAnalysisAgent(self.llm),
            SoftwareDesignAgent(self.llm),
            SoftwareTestingAgent(self.llm),
            ProjectManagementAgent(self.llm),
            CodeImplementationAgent(self.llm),
            SoftwareEthicsAgent(self.llm)
        ]
        logger.info(f"AgentCoordinator: 初始化了 {len(self.agents)} 个智能体")
    
    def get_agent_by_name(self, name: str) -> Optional[Agent]:
        """根据名称获取智能体"""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None
    
    def select_agents(self, question: str, entities: List[Dict[str, str]]) -> List[Tuple[Agent, float]]:
        """使用LLM评估并选择合适的智能体处理问题"""
        selection_json = select_agents_function(question, entities, self.llm, self.agents)
        
        try:
            selected_data = json.loads(selection_json)
            selected_agents = []
            
            for item in selected_data:
                agent_name = item.get("agent")
                relevance = item.get("relevance", 0.0)
                agent = self.get_agent_by_name(agent_name)
                
                if agent and relevance >= 0.5:
                    selected_agents.append((agent, relevance))
            
            # 如果没有选择任何智能体，选择概念解释智能体
            if not selected_agents:
                default_agent = self.get_agent_by_name("概念解释智能体") or self.agents[0]
                selected_agents.append((default_agent, 1.0))
            
            # 按相关性排序
            selected_agents.sort(key=lambda x: x[1], reverse=True)
            return selected_agents
            
        except Exception as e:
            logger.error(f"解析智能体选择结果时出错: {e}")
            # 出错时选择第一个智能体（概念解释智能体）
            return [(self.agents[0], 1.0)]
    
    def process_question(self, question: str, kg_context: Dict[str, Any], 
                         entities: List[Dict[str, str]], doc_results: List[Dict[str, Any]],
                         conversation_history: List[Tuple[str, str]] = None) -> Tuple[str, str]:
        """
        处理问题，使用多个智能体
        返回: (回答, 智能体名称)
        """
        try:
            # 选择智能体
            selected_agents = self.select_agents(question, entities)
            
            # 收集智能体回答
            if len(selected_agents) == 1:
                # 单个智能体处理
                agent, relevance = selected_agents[0]
                answer = agent.process(question, kg_context, entities, doc_results, conversation_history)
                agent_names = agent.name
                return answer, agent_names
            
            # 多个智能体处理
            answers_data = []
            for agent, relevance in selected_agents:
                logger.info(f"智能体 {agent.name} 正在处理问题 (相关性: {relevance:.2f})...")
                answer = agent.process(question, kg_context, entities, doc_results, conversation_history)
                answers_data.append({
                    "agent": agent.name,
                    "relevance": relevance,
                    "answer": answer
                })
            
            # 合成答案
            answers_json = json.dumps(answers_data, ensure_ascii=False)
            final_answer = synthesize_answers_function(answers_json, question, self.llm)
            
            # 创建联合智能体名称
            primary_agent = selected_agents[0][0].name
            secondary_agents = [a[0].name for a in selected_agents[1:]]
            
            if secondary_agents:
                combined_name = f"{primary_agent} + {', '.join(secondary_agents)}"
            else:
                combined_name = primary_agent
                
            return final_answer, combined_name
            
        except Exception as e:
            logger.error(f"智能体协调处理出错: {e}")
            # 出错时使用概念解释智能体作为应急方案
            fallback_agent = self.get_agent_by_name("概念解释智能体") or self.agents[0]
            fallback_answer = fallback_agent.process(question, kg_context, entities, doc_results, conversation_history)
            return fallback_answer, f"{fallback_agent.name} (应急方案)"