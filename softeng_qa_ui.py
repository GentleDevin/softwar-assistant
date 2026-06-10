# -*- coding: utf-8 -*-
import os
import time
import traceback
from typing import List

# 设置环境变量解决OpenMP重复初始化问题 - 必须在导入其他库之前设置
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import gradio as gr

# 导入问答系统和RAG相关函数
from softeng_kg_qa import (
    initialize_qa_system,  # 导入初始化函数
    get_qa_system_instance,  # 导入获取实例函数
    process_uploaded_files,  # 导入处理文件函数
)

# 导入统一配置模块和格式化工具
from config import get_neo4j_config, get_log_config
from utils import setup_logger
from utils.formatters import format_triples_as_html as format_kg_triples_html

# 配置日志
log_config = get_log_config()
logger = setup_logger(__name__, log_file=log_config.log_file, log_level="DEBUG")

# 从配置模块获取 Neo4j 配置
neo4j_config = get_neo4j_config()
NEO4J_URI = neo4j_config.uri
NEO4J_USERNAME = neo4j_config.username
NEO4J_PASSWORD = neo4j_config.password

# --- 初始化 QA 系统 ---
try:
    logger.info("Initializing QA System for Gradio UI...")
    initialize_qa_system(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    logger.info("QA System initialized successfully for UI.")
except Exception as e:
    logger.error(f"FATAL ERROR: Failed to initialize QA System for UI: {e}")
    QA_SYSTEM_INIT_ERROR = f"QA系统初始化失败: {e}"


# --- 全局状态变量 ---
enable_web_search = gr.State(False)
enable_table_output = gr.State(False)
enable_multi_hop = gr.State(False)
enable_search_progress = gr.State(False)


# --- Gradio UI 函数 ---

def format_triples_as_html() -> str:
    """将知识图谱三元组格式化为HTML表格展示"""
    try:
        qa_system = get_qa_system_instance()
        kg_context = qa_system.current_kg_context
    except RuntimeError:
        return "<p>错误：QA 系统未初始化。</p>"
    except AttributeError:
        return "<p>错误：无法从 QA 系统获取知识图谱上下文。</p>"
    return format_kg_triples_html(kg_context)


def handle_file_upload(files):
    """处理文件上传"""
    try:
        if not files:
            return "请选择至少一个文件上传"
        logger.info(f"UI: Processing upload of {len(files)} files...")
        result = process_uploaded_files(files)
        logger.info("UI: File processing finished.")
        return result
    except Exception as e:
        error_msg = f"文件上传处理错误: {str(e)}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        return error_msg


def format_agent_info_html(agent_name: str) -> str:
    """将智能体信息格式化为HTML"""
    if not agent_name:
        return "<p>未识别使用的智能体</p>"
    
    agent_icons = {
        "概念解释智能体": "📚",
        "需求分析智能体": "📋",
        "软件设计智能体": "🏗️",
        "软件测试智能体": "🧪",
        "项目管理智能体": "📊",
        "代码实现智能体": "💻",
        "软件伦理智能体": "⚖️",
        "回退响应生成器": "🤖"
    }
    
    agent_descriptions = {
        "概念解释智能体": "专注于清晰解释软件工程概念、术语和方法论",
        "需求分析智能体": "专注于需求获取、分析、规格说明和需求管理",
        "软件设计智能体": "专注于软件架构、设计模式和系统建模",
        "软件测试智能体": "专注于测试方法、测试类型和质量保证",
        "项目管理智能体": "专注于软件开发生命周期、方法论和团队管理",
        "代码实现智能体": "专注于编程语言、算法和编码最佳实践",
        "软件伦理智能体": "专注于软件伦理、职业道德和社会影响",
        "回退响应生成器": "基于通用软件工程知识提供回答"
    }
    
    icon = agent_icons.get(agent_name, "🔍")
    description = agent_descriptions.get(agent_name, "提供软件工程相关回答")
    
    return f"""<div style="background: rgba(30,41,59,0.9); border-left: 4px solid #8B5CF6; padding: 8px 14px; margin: 6px 0 10px 0; border-radius: 0 8px 8px 0; border-top: 1px solid rgba(139,92,246,0.2); border-bottom: 1px solid rgba(139,92,246,0.2); border-right: 1px solid rgba(139,92,246,0.1);">
<span style="font-size: 1.1em; vertical-align: middle;">{icon}</span>&nbsp;<strong style="color: #A78BFA; font-size: 13px; vertical-align: middle;">{agent_name}</strong>
<div style="color: #94A3B8; font-size: 11.5px; margin-top: 3px; line-height: 1.5;">{description}</div>
</div>"""


def answer_question_ui(question: str, chat_history, 
                       web_search_enabled, table_output_enabled, 
                       multi_hop_enabled, search_progress_enabled):
    """处理问题并更新对话历史，支持多种功能开关"""
    global QA_SYSTEM_INIT_ERROR
    
    # 检查系统初始化状态
    if 'QA_SYSTEM_INIT_ERROR' in globals() and QA_SYSTEM_INIT_ERROR:
        error_msg = f"系统错误：{QA_SYSTEM_INIT_ERROR}"
        yield chat_history, error_msg, error_msg, "", ""
        return

    # 检查输入问题
    if not question or question.strip() == "":
        yield chat_history, "", "", "", ""
        return

    try:
        qa_system = get_qa_system_instance()

        logger.info(f"UI: Answering question: '{question}'")
        logger.info(f"UI: Web search enabled: {web_search_enabled}")
        logger.info(f"UI: Table output enabled: {table_output_enabled}")
        logger.info(f"UI: Multi-hop enabled: {multi_hop_enabled}")
        logger.info(f"UI: Search progress enabled: {search_progress_enabled}")

        start_time = time.time()
        progress_messages = []
        steps = ["正在分析问题...", "正在提取实体...", "正在检索知识图谱...", 
                 "正在搜索文档...", "正在生成回答..."]
        step_start_times = []  # 记录每个步骤的开始时间
        
        # 创建一个线程来执行问答处理
        import threading
        import queue
        
        result_queue = queue.Queue()
        
        def do_work():
            try:
                response = qa_system.answer_question(
                    question,
                    web_search=web_search_enabled,
                    table_output=table_output_enabled,
                    multi_hop=multi_hop_enabled
                )
                
                triples_html = format_triples_as_html()
                doc_results_raw = qa_system.current_doc_results_raw
                rag_manager = qa_system.rag_manager
                search_output_html = rag_manager.format_search_results_as_html(doc_results_raw)
                
                result_queue.put({
                    'success': True,
                    'response': response,
                    'triples': triples_html,
                    'docs': search_output_html
                })
            except Exception as e:
                result_queue.put({
                    'success': False,
                    'error': str(e)
                })
        
        # 启动工作线程
        thread = threading.Thread(target=do_work)
        thread.daemon = True
        thread.start()
        
        # 在等待结果的同时更新进度显示
        current_step = 0
        max_steps = len(steps)
        
        while thread.is_alive():
            elapsed = time.time() - start_time
            
            # 更新进度步骤
            progress_html = ""
            if search_progress_enabled and current_step < max_steps:
                step_to_show = min(current_step, max_steps - 1)
                step_start_times.append(time.time())
                # 立即显示步骤和时间（从0开始）
                progress_messages.append(f"{steps[step_to_show]} (0.00s)")
                progress_messages = progress_messages[-10:]  # 最多保留10条
                progress_html = "<br>".join(progress_messages)
                current_step += 1
            elif search_progress_enabled and progress_messages:
                # 更新所有步骤的耗时显示
                updated_messages = []
                for i, msg in enumerate(progress_messages):
                    if i < len(step_start_times) - 1:
                        # 已完成的步骤，显示最终耗时
                        step_elapsed = step_start_times[i+1] - step_start_times[i]
                        updated_messages.append(f"{steps[i]} ({step_elapsed:.2f}s)")
                    elif i == len(step_start_times) - 1:
                        # 当前正在执行的步骤，显示实时耗时
                        step_elapsed = time.time() - step_start_times[i]
                        updated_messages.append(f"{steps[i]} ({step_elapsed:.2f}s)")
                    else:
                        updated_messages.append(msg)
                progress_html = "<br>".join(updated_messages)
            
            # 更新 status_display（时间进度）和 progress_output（步骤信息）
            status = f"processing | {elapsed:.1f}s"
            yield chat_history, "", "", status, progress_html
            
            # 等待一小段时间再检查
            time.sleep(0.3)
        
        # 获取处理结果
        result = result_queue.get(timeout=5)
        
        if not result['success']:
            error_msg = f"处理问题时发生错误: {result['error']}"
            logger.error(error_msg)
            yield chat_history, f"<p>处理时出错: {result['error']}</p>", f"<p>处理时出错: {result['error']}</p>", error_msg, ""
            return
        
        total_time = time.time() - start_time
        
        response = result['response']
        triples_html = result['triples']
        search_output_html = result['docs']
        
        # 解析响应
        if isinstance(response, dict):
            answer = response.get("answer", "未能获取答案")
            agent_name = response.get("agent_name", "未知智能体")
        else:
            answer = response
            agent_name = "未知智能体"

        agent_info_html = format_agent_info_html(agent_name)
        
        # 如果启用表格输出，尝试格式化答案为表格
        if table_output_enabled and isinstance(response, dict):
            answer = format_as_table(answer)
        
        formatted_answer = f"{agent_info_html}\n\n{answer}"

        # 更新对话历史
        try:
            new_history = chat_history if chat_history else []
            new_history = new_history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": formatted_answer}
            ]
        except Exception as e:
            logger.warning(f"Failed to format chat history: {e}")
            new_history = chat_history if chat_history else []
            new_history = new_history + [[question, formatted_answer]]

        final_status = f"响应完成 | 总耗时: {total_time:.2f}秒"
        
        # 保留检索进展信息，不清空
        if search_progress_enabled and progress_messages:
            # 计算最终的步骤耗时
            final_progress_messages = []
            for i, msg in enumerate(progress_messages):
                if i < len(step_start_times) - 1:
                    step_elapsed = step_start_times[i+1] - step_start_times[i]
                    final_progress_messages.append(f"{steps[i]} ({step_elapsed:.2f}s)")
                elif i == len(step_start_times) - 1:
                    # 最后一个步骤的耗时（到完成）
                    step_elapsed = time.time() - step_start_times[i]
                    final_progress_messages.append(f"{steps[i]} ({step_elapsed:.2f}s)")
            progress_html = "<br>".join(final_progress_messages)
        else:
            progress_html = ""
        
        logger.info(f"UI: Answer and contexts retrieved. 耗时: {total_time:.3f}s")
        yield new_history, triples_html, search_output_html, final_status, progress_html

    except RuntimeError as e:
        error_msg = f"系统错误: {str(e)}"
        logger.error(error_msg)
        yield chat_history, "<p>QA系统未初始化</p>", "<p>QA系统未初始化</p>", error_msg, ""
    except Exception as e:
        error_msg = f"处理问题时发生意外错误: {str(e)}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        yield chat_history, f"<p>处理时出错: {str(e)}</p>", f"<p>处理时出错: {str(e)}</p>", error_msg, ""


def format_as_table(answer: str) -> str:
    """尝试将答案格式化为表格形式"""
    try:
        import re
        
        # 检查是否有列表形式的内容
        lines = answer.split('\n')
        table_rows = []
        
        for line in lines:
            # 匹配带有序号的列表项
            match = re.match(r'^\s*\d+[\.\、]\s*(.*)', line)
            if match:
                table_rows.append(match.group(1))
        
        if len(table_rows) >= 2:
            table_html = "<table style='width:100%; border-collapse:collapse; margin:10px 0; font-size:13px;'>"
            table_html += "<thead><tr><th style='border:1px solid rgba(139,92,246,0.2); padding:8px 12px; background:rgba(139,92,246,0.15); text-align:left; color:#A78BFA; font-weight:600;'>序号</th><th style='border:1px solid rgba(139,92,246,0.2); padding:8px 12px; background:rgba(139,92,246,0.15); text-align:left; color:#A78BFA; font-weight:600;'>内容</th></tr></thead>"
            table_html += "<tbody>"
            for i, row in enumerate(table_rows, 1):
                bg = "rgba(30,41,59,0.5)" if i % 2 == 0 else "transparent"
                table_html += f"<tr style='background:{bg};'><td style='border:1px solid rgba(51,65,85,0.5); padding:8px 12px; color:#94A3B8; width:48px;'>{i}</td><td style='border:1px solid rgba(51,65,85,0.5); padding:8px 12px; color:#E2E8F0;'>{row}</td></tr>"
            table_html += "</tbody></table>"
            return table_html
        
        return answer
    except Exception as e:
        logger.error(f"表格格式化失败: {e}")
        return answer


def clear_conversation(chat_history) -> List:
    """清除对话历史"""
    try:
        qa_system = get_qa_system_instance()
        qa_system.clear_conversation_history()
        logger.info("UI: Conversation history cleared.")
        return []
    except Exception as e:
        logger.error(f"清除对话历史时出错: {e}")
        return chat_history


def clear_input():
    """清空输入框"""
    return ""


def clear_status_display():
    """清空状态显示"""
    return "", ""


def handle_web_search_toggle(value):
    """处理联网搜索开关"""
    logger.info(f"联网搜索功能已{'启用' if value else '禁用'}")


def handle_table_output_toggle(value):
    """处理表格输出开关"""
    logger.info(f"表格输出功能已{'启用' if value else '禁用'}")


def handle_multi_hop_toggle(value):
    """处理多跳推理开关"""
    logger.info(f"多跳推理功能已{'启用' if value else '禁用'}")


def handle_search_progress_toggle(value):
    """处理检索进展开关"""
    logger.info(f"检索进展显示已{'启用' if value else '禁用'}")


# 自定义CSS样式 - 深色科技风（参考softeng.html设计，强制暗色模式，不随Mac系统主题变化）
custom_css = """
/* ============================================================
   强制暗色模式 - 不随 macOS 系统主题变化
   ============================================================ */
:root {
    color-scheme: dark !important;
    --primary: #8B5CF6;
    --primary-light: #A78BFA;
    --primary-dark: #7C3AED;
    --dark-bg: #0F172A;
    --card-bg: #1E293B;
    --accent-red: #EF4444;
    --neutral-gray: #334155;
    --border-color: rgba(139, 92, 246, 0.2);
    --border-subtle: rgba(51, 65, 85, 0.5);
    --text-primary: #F1F5F9;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    --radius: 12px;
    --radius-sm: 8px;
    --shadow-card: 0 4px 24px rgba(0, 0, 0, 0.4);
    --shadow-glow: 0 0 16px rgba(139, 92, 246, 0.3);
    --transition: all 0.25s ease;
}

/* ── 全局基础 ── */
*, *::before, *::after { box-sizing: border-box; }

html, body {
    background-color: #0F172A !important;
    color: #F1F5F9 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    -webkit-font-smoothing: antialiased;
    min-height: 100vh;
}

/* ── 滚动条 ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #1E293B; border-radius: 10px; }
::-webkit-scrollbar-thumb { background: #475569; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #8B5CF6; }

/* ── Gradio 根容器强制暗色 ── */
.gradio-container,
.gradio-container-outer,
.wrap,
#root,
.app {
    background: #0F172A !important;
    color: #F1F5F9 !important;
    min-height: 100vh !important;
}

/* 清除默认白底 */
.gr-box,
.gr-form,
.gr-panel,
.panel,
.container,
.block,
.block.padded,
.block.label-inner,
.wrap.svelte-xgdnc5,
.overflow-hidden {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #F1F5F9 !important;
}

/* ── 顶部导航栏 ── */
.header-section {
    background: linear-gradient(135deg, #7C3AED 0%, #8B5CF6 50%, #A78BFA 100%) !important;
    color: #fff !important;
    border-radius: var(--radius) !important;
    padding: 14px 24px !important;
    margin-bottom: 16px !important;
    box-shadow: var(--shadow-glow) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    border: 1px solid rgba(167, 139, 250, 0.3) !important;
}
.header-section h1 {
    color: #fff !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    margin: 0 !important;
    letter-spacing: -0.3px;
}
.header-section h2 {
    color: rgba(255,255,255,0.85) !important;
    font-size: 12px !important;
    font-weight: 400 !important;
    margin: 0 !important;
    letter-spacing: 0.2px;
}

/* ── 玻璃卡片 ── */
.card {
    background: linear-gradient(135deg, rgba(30,41,59,0.96), rgba(15,23,42,0.98)) !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--border-color) !important;
    box-shadow: var(--shadow-card) !important;
    padding: 16px !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
}

/* ── 卡片头部 ── */
.card-header {
    display: flex;
    align-items: center;
    padding: 0 0 12px 0;
    margin-bottom: 12px;
    border-bottom: 1px solid var(--border-color);
}
.card-header h3 {
    margin: 0 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #A78BFA !important;
    border-bottom: none !important;
    padding-bottom: 0 !important;
}

/* ── 设置项行 ── */
.setting-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 4px;
    border-bottom: 1px solid var(--border-subtle);
    transition: background 0.2s ease;
    border-radius: 6px;
}
.setting-item:last-child { border-bottom: none; }
.setting-item:hover { background: rgba(139, 92, 246, 0.08); padding-left: 8px; }

.setting-content { flex: 1; display: flex; flex-direction: column; gap: 2px; }
.setting-title {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 13.5px;
    font-weight: 500;
    color: #F1F5F9;
}
.setting-title span { font-size: 15px; line-height: 1; }
.setting-desc {
    font-size: 11px;
    color: #64748B;
    line-height: 1.4;
    padding-left: 0;
}

/* ── Checkbox 样式 ── */
.gr-checkbox,
.gr-checkbox-wrap,
.checkbox-wrap,
[data-testid="checkbox"] {
    margin: 0 !important;
    flex-shrink: 0;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 22px !important;
    height: 22px !important;
}
.gr-checkbox label,
.gr-checkbox-wrap label,
.checkbox-wrap label,
[data-testid="checkbox"] label,
.svelte-1gfwr6n,
.label-wrap,
.label-text {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    height: 0 !important;
    width: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
}
input[type="checkbox"] {
    width: 16px !important;
    height: 16px !important;
    cursor: pointer !important;
    accent-color: #8B5CF6 !important;
    border-radius: 4px !important;
    margin: 0 !important;
    flex-shrink: 0 !important;
}

/* ── 标签页 ── */
.tabs > .tab-nav {
    background: rgba(15,23,42,0.8) !important;
    border-bottom: 1px solid var(--border-color) !important;
    padding: 0 4px !important;
}
.tabs > .tab-nav button {
    color: #94A3B8 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important;
    transition: var(--transition) !important;
    background: transparent !important;
    border-radius: 0 !important;
}
.tabs > .tab-nav button:hover { color: #A78BFA !important; }
.tabs > .tab-nav button.selected {
    color: #A78BFA !important;
    border-bottom-color: #8B5CF6 !important;
    background: transparent !important;
}

/* ── 输入框 ── */
textarea,
input[type="text"],
.gr-text-input,
.gr-input {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(139, 92, 246, 0.25) !important;
    border-radius: var(--radius-sm) !important;
    color: #F1F5F9 !important;
    padding: 10px 14px !important;
    font-size: 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    caret-color: #A78BFA !important;
}
textarea:focus,
input[type="text"]:focus {
    border-color: #8B5CF6 !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.18) !important;
    outline: none !important;
}
textarea::placeholder,
input[type="text"]::placeholder { color: #475569 !important; }

/* ── 主按钮（提交） ── */
.btn-primary,
button.primary,
.gr-button.primary {
    background: linear-gradient(135deg, #7C3AED 0%, #8B5CF6 60%, #A78BFA 100%) !important;
    color: #fff !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    border-radius: var(--radius-sm) !important;
    height: 42px !important;
    min-width: 110px !important;
    box-shadow: 0 4px 16px rgba(139, 92, 246, 0.3) !important;
    transition: var(--transition) !important;
    letter-spacing: 0.2px;
}
.btn-primary:hover,
button.primary:hover,
.gr-button.primary:hover {
    box-shadow: 0 6px 22px rgba(139, 92, 246, 0.5) !important;
    transform: translateY(-2px) !important;
    filter: brightness(1.1) !important;
}

/* ── 次级按钮 ── */
.btn-secondary,
.btn-secondary:not(.primary) {
    background: rgba(51, 65, 85, 0.6) !important;
    color: #CBD5E1 !important;
    border: 1px solid rgba(51, 65, 85, 0.8) !important;
    border-radius: var(--radius-sm) !important;
    height: 40px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: var(--transition) !important;
}
.btn-secondary:hover {
    background: rgba(139, 92, 246, 0.18) !important;
    border-color: rgba(139, 92, 246, 0.5) !important;
    color: #A78BFA !important;
    transform: translateY(-1px) !important;
}

/* ── 危险按钮（清空对话） ── */
.btn-danger {
    background: linear-gradient(135deg, #B91C1C 0%, #EF4444 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    height: 40px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: var(--transition) !important;
    box-shadow: 0 2px 8px rgba(239,68,68,0.2) !important;
}
.btn-danger:hover {
    box-shadow: 0 4px 16px rgba(239, 68, 68, 0.4) !important;
    transform: translateY(-1px) !important;
    filter: brightness(1.1) !important;
}

/* ── 状态显示 ── */
.status-display p,
.status-display {
    color: #A78BFA !important;
    font-size: 12.5px !important;
    font-family: 'SF Mono', 'JetBrains Mono', 'Fira Code', Monaco, monospace !important;
    background: rgba(139, 92, 246, 0.08) !important;
    border: 1px solid rgba(139, 92, 246, 0.2) !important;
    border-radius: var(--radius-sm) !important;
    padding: 8px 14px !important;
    margin: 0 !important;
}

/* ── 进度显示 ── */
.progress-display {
    background: rgba(0, 0, 0, 0.25) !important;
    border-radius: var(--radius-sm) !important;
    padding: 10px 12px !important;
    font-size: 12px !important;
    color: #94A3B8 !important;
    font-family: 'SF Mono', Monaco, monospace !important;
    max-height: 140px !important;
    overflow-y: auto !important;
    border: 1px solid var(--border-color) !important;
    line-height: 1.7 !important;
}

/* ── 示例问题卡片 ── */
.examples-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.96), rgba(15,23,42,0.98)) !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--border-color) !important;
    box-shadow: var(--shadow-card) !important;
    padding: 16px !important;
    margin-top: 12px !important;
}
.examples-card h3 {
    margin: 0 0 10px 0 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #A78BFA !important;
    border-bottom: none !important;
    padding-bottom: 0 !important;
}
.examples-card table,
.examples-card .examples {
    background: transparent !important;
    border: none !important;
}
.examples-card button,
.examples-card .gr-button {
    background: rgba(51, 65, 85, 0.45) !important;
    color: #94A3B8 !important;
    border: 1px solid rgba(139, 92, 246, 0.15) !important;
    border-radius: 20px !important;
    font-size: 12px !important;
    font-weight: 400 !important;
    padding: 5px 14px !important;
    margin: 3px !important;
    transition: var(--transition) !important;
    cursor: pointer !important;
    white-space: nowrap !important;
}
.examples-card button:hover,
.examples-card .gr-button:hover {
    background: rgba(139, 92, 246, 0.22) !important;
    color: #A78BFA !important;
    border-color: rgba(139, 92, 246, 0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── 聊天气泡 ── */
.message-wrap,
.chatbot,
.chat-history {
    background: transparent !important;
    border: none !important;
}
.chatbot .message {
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-size: 14px !important;
    line-height: 1.65 !important;
}
.chatbot .message.bot,
.bot-message,
.assistant-message {
    background: rgba(30, 41, 59, 0.85) !important;
    border: 1px solid rgba(139, 92, 246, 0.15) !important;
    color: #E2E8F0 !important;
}
.chatbot .message.user,
.user-message {
    background: linear-gradient(135deg, #7C3AED 0%, #8B5CF6 100%) !important;
    color: #fff !important;
    border: none !important;
}

/* ── 文件上传 ── */
.gr-file,
.file-preview,
[data-testid="file"] {
    background: rgba(30, 41, 59, 0.5) !important;
    border: 2px dashed rgba(139, 92, 246, 0.35) !important;
    border-radius: var(--radius) !important;
    transition: var(--transition) !important;
    color: #94A3B8 !important;
}
.gr-file:hover,
.file-preview:hover {
    border-color: #8B5CF6 !important;
    background: rgba(139, 92, 246, 0.08) !important;
}

/* ── Textbox 标签、边框 ── */
.label-wrap span,
.block > label > span:first-child {
    color: #94A3B8 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}
.block.gr-textbox,
.block textarea {
    background: rgba(30, 41, 59, 0.8) !important;
    border-color: rgba(139, 92, 246, 0.25) !important;
    color: #F1F5F9 !important;
}

/* ── Markdown / HTML 内容区 ── */
.prose, .md {
    color: #CBD5E1 !important;
}
.prose h1, .prose h2, .prose h3, .prose h4 { color: #A78BFA !important; }
.prose a { color: #8B5CF6 !important; }
.prose code {
    background: rgba(139, 92, 246, 0.12) !important;
    color: #C4B5FD !important;
    border-radius: 4px;
    padding: 1px 5px;
}
.prose pre {
    background: rgba(0,0,0,0.3) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-sm) !important;
}

/* ── h3/h4 标题 ── */
h3 {
    color: #A78BFA;
    font-size: 14px;
    font-weight: 600;
    margin: 0 0 10px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-color);
}
h4 { color: #F1F5F9; font-size: 13px; font-weight: 500; margin: 10px 0 6px 0; }

/* ── 页脚 ── */
footer.svelte-1rjryqp,
.footer {
    display: none !important;
}
"""


def create_ui():
    with gr.Blocks() as app:
        gr.HTML("""
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
          /* 额外强制覆盖：防止 macOS 系统主题渗透 */
          @media (prefers-color-scheme: light) {
            .gradio-container, body, html { background: #0F172A !important; color: #F1F5F9 !important; }
          }
        </style>
        <div class="main-container">
            <div class="header-section">
                <div style="display:flex;align-items:center;gap:12px;">
                    <h1 style="margin:0;font-size:20px;font-weight:700;color:#fff;">🎓 软件工程课程助手</h1>
                </div>
                <h2 style="margin:0;font-size:12px;color:rgba(255,255,255,0.8);">基于大语言模型与知识图谱的智能学习辅助系统</h2>
            </div>
        """)
        
        # 主标签页
        with gr.Tabs():
            # 对话交互标签页
            with gr.TabItem("对话交互"):
                with gr.Row():
                    # 左侧：对话设置
                    with gr.Column(scale=1, min_width=280):
                        # 对话设置卡片 - 合并所有设置项
                        with gr.Group(elem_classes=["card", "settings-card"]):
                            gr.HTML("""<div class="card-header"><h3>⚙️ 对话设置</h3></div>""")
                            
                            # 启用联网搜索
                            with gr.Row(elem_classes=["setting-item"]):
                                web_search_toggle = gr.Checkbox(value=False, label="", show_label=False)
                                gr.HTML("""<div class="setting-content">
                                    <div class="setting-title"><span>🌐</span> 启用联网搜索</div>
                                    <div class="setting-desc">联网获取最新动态</div>
                                </div>""")
                            
                            # 表格格式输出
                            with gr.Row(elem_classes=["setting-item"]):
                                table_output_toggle = gr.Checkbox(value=False, label="", show_label=False)
                                gr.HTML("""<div class="setting-content">
                                    <div class="setting-title"><span>📊</span> 表格格式输出</div>
                                    <div class="setting-desc">使用Markdown表格展示结构化回答</div>
                                </div>""")
                            
                            # 启用多跳推理
                            with gr.Row(elem_classes=["setting-item"]):
                                multi_hop_toggle = gr.Checkbox(value=False, label="", show_label=False)
                                gr.HTML("""<div class="setting-content">
                                    <div class="setting-title"><span>🔗</span> 启用多跳推理</div>
                                    <div class="setting-desc">使用高级多跳推理机制（较慢但更全面）</div>
                                </div>""")
                            
                            # 显示检索进展
                            with gr.Row(elem_classes=["setting-item"]):
                                search_progress_toggle = gr.Checkbox(value=True, label="", show_label=False)
                                gr.HTML("""<div class="setting-content">
                                    <div class="setting-title"><span>📈</span> 显示检索进展</div>
                                    <div class="setting-desc">实时显示检索和处理进度</div>
                                </div>""")
                        
                        # 检索进展显示区域
                        with gr.Group(elem_classes=["card"]):
                            gr.HTML("<h4>🔄 检索进展</h4>")
                            progress_output = gr.HTML(
                                value="<div class='progress-display' style='color:#475569;'>⏳ 等待输入...</div>",
                                elem_classes=["progress-display"]
                            )
                
                    # 右侧：对话历史和输入区域
                    with gr.Column(scale=3):
                        # 知识图谱三元组显示
                        with gr.Group(elem_classes=["card"]):
                            gr.HTML("<h3>📚 知识图谱</h3>")
                            kg_triples_output = gr.HTML(value="<div style='color:#475569;font-size:13px;padding:8px 0;'>⏳ 等待问题输入...</div>")
                        
                        # 对话历史
                        with gr.Group(elem_classes=["card"]):
                            gr.HTML("<h3>💬 对话历史</h3>")
                            chat_display = gr.Chatbot(
                                label="",
                                height=450,
                                show_label=False,
                                render_markdown=True,
                                elem_classes=["chat-history"]
                            )
                        
                        # 状态显示
                        status_display = gr.Markdown("", elem_classes=["status-display"])
                        
                        # 输入区域
                        with gr.Row():
                            question_input = gr.Textbox(
                                placeholder="输入软件工程相关问题",
                                label="",
                                lines=1,
                                show_label=False,
                                scale=4,
                                elem_classes=["gr-input"]
                            )
                            submit_button = gr.Button("提交问题", variant="primary", elem_classes=["btn-primary"])
                        
                        # 操作按钮
                        with gr.Row():
                            clear_button = gr.Button("清空输入", elem_classes=["btn-secondary"])
                            clear_history_button = gr.Button("清空对话历史", elem_classes=["btn-danger"])
                        
                        # 示例问题
                        with gr.Group(elem_classes=["examples-card"]):
                            gr.HTML("<h3>💡 快速提问示例</h3>")
                            example_questions = [
                                "函数式编程的概念是什么？", 
                                "敏捷开发的核心原则是什么？",
                                "软件测试与质量保证有什么区别？", 
                                "UML图有哪些类型？",
                                "设计模式的作用是什么？", 
                                "如何进行软件架构评估？",
                                "什么是微服务架构？",
                                "SOLID原则包括哪些内容？"
                            ]
                            gr.Examples(
                                examples=example_questions, 
                                inputs=question_input, 
                                label=""
                            )
            
            # 知识库管理标签页
            with gr.TabItem("知识库管理"):
                with gr.Group(elem_classes=["card"]):
                    gr.HTML("<h3>📁 上传课程资料</h3>")
                    file_upload = gr.File(
                        file_types=[".txt", ".pdf"],
                        file_count="multiple",
                        label="选择或拖拽文件 (支持TXT/PDF)"
                    )
                    upload_button = gr.Button("处理上传的文件", variant="primary", elem_classes=["btn-primary"])
                    upload_status = gr.Textbox(label="文件处理状态", lines=8, interactive=False)
        
        # 文档检索结果显示组件（隐藏，仅用于接收回答问题时的文档搜索结果）
        docs_output = gr.HTML(visible=False)
        
        gr.HTML("""
        </div>
        <footer style="text-align:center;margin:24px 0 12px 0;color:#334155;font-size:11.5px;letter-spacing:0.3px;">
            Powered by LLM + Knowledge Graph + RAG &nbsp;|&nbsp; 软件工程课程专用学习助手
        </footer>
        """)

        # --- 事件绑定 ---
        
        # 文件上传
        upload_button.click(
            fn=handle_file_upload,
            inputs=[file_upload],
            outputs=[upload_status]
        )
        
        # 开关状态绑定
        web_search_toggle.change(
            fn=handle_web_search_toggle,
            inputs=[web_search_toggle],
            outputs=[]
        )
        
        table_output_toggle.change(
            fn=handle_table_output_toggle,
            inputs=[table_output_toggle],
            outputs=[]
        )
        
        multi_hop_toggle.change(
            fn=handle_multi_hop_toggle,
            inputs=[multi_hop_toggle],
            outputs=[]
        )
        
        search_progress_toggle.change(
            fn=handle_search_progress_toggle,
            inputs=[search_progress_toggle],
            outputs=[]
        )
        
        # 提交问题
        submit_button.click(
            fn=answer_question_ui,
            inputs=[question_input, chat_display, 
                    web_search_toggle, table_output_toggle, 
                    multi_hop_toggle, search_progress_toggle],
            outputs=[chat_display, kg_triples_output, docs_output, status_display, progress_output]
        ).then(
            fn=clear_input,
            inputs=None,
            outputs=[question_input]
        )
        
        question_input.submit(
            fn=answer_question_ui,
            inputs=[question_input, chat_display, 
                    web_search_toggle, table_output_toggle, 
                    multi_hop_toggle, search_progress_toggle],
            outputs=[chat_display, kg_triples_output, docs_output, status_display, progress_output]
        ).then(
            fn=clear_input,
            inputs=None,
            outputs=[question_input]
        )
        
        # 清空输入
        clear_button.click(
            fn=clear_input,
            inputs=None,
            outputs=[question_input]
        )
        
        # 清空对话历史
        clear_history_button.click(
            fn=clear_conversation,
            inputs=[chat_display],
            outputs=[chat_display]
        ).then(
            fn=clear_status_display,
            inputs=None,
            outputs=[status_display, progress_output]
        )
    
    return app


if __name__ == "__main__":
    if 'QA_SYSTEM_INIT_ERROR' in globals() and QA_SYSTEM_INIT_ERROR:
        logger.error(f"!!! Gradio UI cannot start due to QA System initialization error: {QA_SYSTEM_INIT_ERROR} !!!")
        with gr.Blocks() as error_app:
            gr.Markdown(f"# 系统启动失败\n\n无法初始化问答系统，请检查配置和后台服务。\n\n**错误信息:**\n```\n{QA_SYSTEM_INIT_ERROR}\n```")
        logger.info("\nLaunching error display UI...")
        error_app.launch(share=True)
    else:
        logger.info("\nLaunching Gradio UI...")
        app = create_ui()
        app.queue()

        # 构建主题：hue/size 在 __init__，设计 token 通过 .set() 设置
        # 注意：gr.themes.Base() 不接受 token 参数，必须用 .set() 方法
        _dark_theme = gr.themes.Base(
            primary_hue=gr.themes.Color(
                c50="#F5F3FF",
                c100="#EDE9FE",
                c200="#DDD6FE",
                c300="#C4B5FD",
                c400="#A78BFA",
                c500="#8B5CF6",
                c600="#7C3AED",
                c700="#6D28D9",
                c800="#5B21B6",
                c900="#4C1D95",
                c950="#2E1065",
            ),
            neutral_hue=gr.themes.Color(
                c50="#F8FAFC",
                c100="#F1F5F9",
                c200="#E2E8F0",
                c300="#CBD5E1",
                c400="#94A3B8",
                c500="#64748B",
                c600="#475569",
                c700="#334155",
                c800="#1E293B",
                c900="#0F172A",
                c950="#020617",
            ),
            spacing_size=gr.themes.sizes.spacing_md,
            radius_size=gr.themes.sizes.radius_md,
        ).set(
            # 强制暗色背景 token，防止 macOS Light Mode 渗透
            body_background_fill="#0F172A",
            body_background_fill_dark="#0F172A",
            body_text_color="#F1F5F9",
            body_text_color_dark="#F1F5F9",
            body_text_color_subdued="#94A3B8",
            body_text_color_subdued_dark="#94A3B8",
            background_fill_primary="#0F172A",
            background_fill_primary_dark="#0F172A",
            background_fill_secondary="#1E293B",
            background_fill_secondary_dark="#1E293B",
            border_color_accent="#8B5CF6",
            border_color_accent_dark="#8B5CF6",
            border_color_primary="rgba(139,92,246,0.25)",
            border_color_primary_dark="rgba(139,92,246,0.25)",
            color_accent="#8B5CF6",
            color_accent_soft="rgba(139,92,246,0.15)",
            color_accent_soft_dark="rgba(139,92,246,0.15)",
            link_text_color="#A78BFA",
            link_text_color_dark="#A78BFA",
            link_text_color_active="#C4B5FD",
            link_text_color_active_dark="#C4B5FD",
            link_text_color_hover="#C4B5FD",
            link_text_color_hover_dark="#C4B5FD",
            link_text_color_visited="#8B5CF6",
            link_text_color_visited_dark="#8B5CF6",
            block_background_fill="#1E293B",
            block_background_fill_dark="#1E293B",
            block_border_color="rgba(139,92,246,0.2)",
            block_border_color_dark="rgba(139,92,246,0.2)",
            block_label_background_fill="#1E293B",
            block_label_background_fill_dark="#1E293B",
            block_label_text_color="#94A3B8",
            block_label_text_color_dark="#94A3B8",
            block_title_text_color="#A78BFA",
            block_title_text_color_dark="#A78BFA",
            input_background_fill="rgba(30,41,59,0.8)",
            input_background_fill_dark="rgba(30,41,59,0.8)",
            input_border_color="rgba(139,92,246,0.25)",
            input_border_color_dark="rgba(139,92,246,0.25)",
            input_border_color_focus="#8B5CF6",
            input_border_color_focus_dark="#8B5CF6",
            input_placeholder_color="#475569",
            input_placeholder_color_dark="#475569",
            checkbox_background_color="#1E293B",
            checkbox_background_color_dark="#1E293B",
            checkbox_border_color="rgba(139,92,246,0.4)",
            checkbox_border_color_dark="rgba(139,92,246,0.4)",
            checkbox_label_background_fill="transparent",
            checkbox_label_background_fill_dark="transparent",
            table_even_background_fill="rgba(30,41,59,0.5)",
            table_even_background_fill_dark="rgba(30,41,59,0.5)",
            table_odd_background_fill="#1E293B",
            table_odd_background_fill_dark="#1E293B",
            stat_background_fill="rgba(139,92,246,0.1)",
            stat_background_fill_dark="rgba(139,92,246,0.1)",
            panel_background_fill="#1E293B",
            panel_background_fill_dark="#1E293B",
            panel_border_color="rgba(139,92,246,0.2)",
            panel_border_color_dark="rgba(139,92,246,0.2)",
            button_primary_background_fill="linear-gradient(135deg, #7C3AED, #8B5CF6)",
            button_primary_background_fill_dark="linear-gradient(135deg, #7C3AED, #8B5CF6)",
            button_primary_background_fill_hover="linear-gradient(135deg, #6D28D9, #7C3AED)",
            button_primary_background_fill_hover_dark="linear-gradient(135deg, #6D28D9, #7C3AED)",
            button_primary_text_color="#ffffff",
            button_primary_text_color_dark="#ffffff",
            button_secondary_background_fill="rgba(51,65,85,0.6)",
            button_secondary_background_fill_dark="rgba(51,65,85,0.6)",
            button_secondary_background_fill_hover="rgba(139,92,246,0.2)",
            button_secondary_background_fill_hover_dark="rgba(139,92,246,0.2)",
            button_secondary_text_color="#CBD5E1",
            button_secondary_text_color_dark="#CBD5E1",
            button_cancel_background_fill="linear-gradient(135deg, #B91C1C, #EF4444)",
            button_cancel_background_fill_dark="linear-gradient(135deg, #B91C1C, #EF4444)",
            button_cancel_text_color="#ffffff",
            button_cancel_text_color_dark="#ffffff",
        )

        app.launch(
            share=True,
            theme=_dark_theme,
            css=custom_css
        )
