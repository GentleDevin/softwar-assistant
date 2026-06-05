# -*- coding: utf-8 -*-
import logging
import os
import traceback
from typing import List, Dict, Any, Tuple

import gradio as gr

# 导入问答系统和RAG相关函数
from softeng_kg_qa import (
    initialize_qa_system,  # 导入初始化函数
    get_qa_system_instance,  # 导入获取实例函数
    process_uploaded_files,  # 导入处理文件函数
    # 导入独立的文档搜索函数 (给Tab用)
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
# 在Gradio应用启动前尝试初始化QA系统
# 如果初始化失败，Gradio可能无法启动或功能不全
try:
    logger.info("Initializing QA System for Gradio UI...")
    initialize_qa_system(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    logger.info("QA System initialized successfully for UI.")
except Exception as e:
    logger.error(f"FATAL ERROR: Failed to initialize QA System for UI: {e}")
    QA_SYSTEM_INIT_ERROR = f"QA系统初始化失败: {e}"


# --- Gradio UI 函数 ---

def format_triples_as_html(context: Dict[str, Any]) -> str:
    """将知识图谱三元组格式化为HTML表格展示（使用统一的格式化工具）"""
    # 注意：这个函数现在依赖于 answer_question 函数更新 QA system 实例的 current_kg_context
    try:
        qa_system = get_qa_system_instance()
        kg_context = qa_system.current_kg_context  # 从实例获取最新的 KG 上下文
    except RuntimeError:
        return "<p>错误：QA 系统未初始化。</p>"
    except AttributeError:
        return "<p>错误：无法从 QA 系统获取知识图谱上下文。</p>"

    # 使用统一的格式化工具
    return format_kg_triples_html(kg_context)


def handle_file_upload(files):
    """处理文件上传，直接调用导出的 process_uploaded_files 函数"""
    # 这个函数不需要 QA system 实例
    try:
        if not files:
            return "请选择至少一个文件上传"

        logger.info(f"UI: Processing upload of {len(files)} files...")
        # 调用导出的函数
        result = process_uploaded_files(files)
        logger.info("UI: File processing finished.")
        return result
    except Exception as e:
        error_msg = f"文件上传处理错误: {str(e)}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        return error_msg

# 新增函数：格式化智能体信息为HTML
def format_agent_info_html(agent_name: str) -> str:
    """将智能体信息格式化为HTML"""
    if not agent_name:
        return "<p>未识别使用的智能体</p>"
    
    # 智能体图标映射
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
    
    # 智能体描述映射
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
    
    return f"""<div style="background-color: #f5f9ff; border-left: 4px solid #2a81e3; padding: 8px 12px; margin: 8px 0; border-radius: 0 8px 8px 0;">
<span style="font-size: 1.2em;">{icon}</span> <strong style="color: #2a81e3;">{agent_name}</strong>
<div style="color: #555; font-size: 0.9em; margin-top: 4px;">{description}</div>
</div>"""

# answer_question_ui 函数以处理多轮对话
def answer_question_ui(question: str, chat_history) -> Tuple[Any, str, str, str]:
    """处理问题并更新对话历史"""
    global QA_SYSTEM_INIT_ERROR # 引用全局错误状态
    if 'QA_SYSTEM_INIT_ERROR' in globals() and QA_SYSTEM_INIT_ERROR:
            error_msg = f"系统错误：{QA_SYSTEM_INIT_ERROR}"
            return chat_history, error_msg, error_msg, ""

    try:
        # 获取 QA 系统实例
        qa_system = get_qa_system_instance()

        if not question or question.strip() == "":
            return chat_history, "", "", ""

        logger.info(f"UI: Answering question: '{question}'")

        # --- 调用核心问答函数 ---
        response = qa_system.answer_question(question)
        
        # 从响应中提取答案和智能体名称
        if isinstance(response, dict):
            answer = response.get("answer", "未能获取答案")
            agent_name = response.get("agent_name", "未知智能体")
        else:
            # 兼容旧格式（如果 answer_question 没有更新）
            answer = response
            agent_name = "未知智能体"

        # --- 获取并格式化知识图谱结果 ---
        # format_triples_as_html 会从 qa_system 获取最新上下文
        triples_html = format_triples_as_html({}) # 传递空字典，让函数内部获取

        # --- 获取并格式化文档检索结果 ---
        doc_results_raw = qa_system.current_doc_results_raw
        rag_manager = qa_system.rag_manager
        search_output_html = rag_manager.format_search_results_as_html(doc_results_raw)

        # 创建智能体信息HTML
        agent_info_html = format_agent_info_html(agent_name)
        
        # 结合智能体信息和答案
        formatted_answer = f"{agent_info_html}\n\n{answer}"

        # 更新对话历史 - 根据Gradio版本使用不同格式
        try:
            # 尝试使用新格式（带role和content的字典）
            new_history = chat_history if chat_history else []
            new_history = new_history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": formatted_answer}
            ]
        except:
            # 如果新格式失败，使用旧格式（tuples）
            new_history = chat_history if chat_history else []
            new_history = new_history + [[question, formatted_answer]]

        logger.info("UI: Answer and contexts retrieved.")
        return new_history, triples_html, search_output_html, ""  # 增加第四个返回值（空字符串）

    except RuntimeError as e: # QA 系统未初始化
            error_msg = f"系统错误: {str(e)}"
            logger.error(error_msg)
            return chat_history, "<p>QA系统未初始化</p>", "<p>QA系统未初始化</p>", error_msg
    except Exception as e:
        error_msg = f"处理问题时发生意外错误: {str(e)}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        return chat_history, f"<p>处理时出错: {e}</p>", f"<p>处理时出错: {e}</p>", error_msg

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

# 自定义CSS样式
custom_css = """
body { background-color: #f8f9fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
h1 { color: #2a81e3; text-align: center; font-size: 28px; margin-bottom: 5px; font-weight: bold; }
h2 { font-size: 18px; margin-top: 0; margin-bottom: 15px; color: #555; text-align: center; font-weight: normal; }
h3 { color: #333; margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 5px;}
h4 { color: #444; margin-top: 15px; margin-bottom: 10px; font-size: 1.1em; }

/* 增加容器最大宽度 */
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

/* 改进聊天消息样式，减少空白 */
.message { 
    margin: 8px 0; 
    padding: 8px 12px;
    border-radius: 10px;
}
.user-message {
    text-align: right;
    background-color: #e1f0ff;
    margin-left: 35%;  /* 减少左侧空白 */
    display: inline-block;
    max-width: 65%;
}
.assistant-message {
    text-align: left;
    background-color: #f0f0f0;
    margin-right: 15%;  /* 减少右侧空白 */
    display: inline-block;
    max-width: 85%;
}

/* 优化聊天框样式 */
.chatbot {
    max-width: 100% !important;
}
.chatbot .message-wrap {
    padding: 10px 20px !important;
}

/* 添加可调节的对话区域 */
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

# 创建Gradio界面
def create_ui():


    with gr.Blocks() as app:
        with gr.Column(elem_classes=["container"]):
            # 标题区域
            with gr.Row():
                gr.HTML("""
                <div class="header">
                    <h1><span class="book-icon">🎓</span> 软件工程课程助手</h1>
                    <h2>基于大语言模型与知识图谱的智能学习辅助系统</h2>
                </div>
                """)

            # 主要布局：调整比例，让右侧更宽
            with gr.Row(variant="panel"):
                # 左侧上传部分 - 减小scale（使用整数）
                with gr.Column(scale=1, min_width=300, elem_classes=["upload-section"]):
                    gr.HTML("<h3>📥 上传课程资料</h3>")
                    file_upload = gr.File(
                        file_types=[".txt", ".pdf"],
                        file_count="multiple",
                        label="选择或拖拽文件 (支持TXT/PDF)"
                    )
                    upload_button = gr.Button("处理上传的文件", variant="primary")
                    upload_status = gr.Textbox(label="文件处理状态", lines=8, interactive=False, 
                                            placeholder="上传并处理文件后，这里会显示状态...")

                # 右侧问答和对话部分 - 增大scale（使用整数）并调整min_width
                with gr.Column(scale=3, min_width=800, elem_classes=["resizable-chat"]):
                    # 对话聊天部分
                    with gr.Column(elem_classes=["chat-section"]):
                        gr.HTML("<h3>💬 课程对话</h3>")
                        
                        # 聊天记录显示区域 - 增加高度
                        chat_display = gr.Chatbot(
                            label="对话历史",
                            height=500,  # 增加高度
                            show_label=True,
                            render_markdown=True,
                            elem_classes=["chatbot"]  # 添加自定义类
                        )
                        
                        # 输入和按钮区域
                        with gr.Row():
                            question_input = gr.Textbox(
                                placeholder="输入您的问题，例如: 什么是敏捷开发？",
                                label="提问",
                                lines=1,
                                show_label=False,
                                scale=4
                            )
                            submit_button = gr.Button("发送", variant="primary", scale=1)
                        
                        # 操作按钮行
                        with gr.Row():
                            clear_button = gr.Button("清除对话历史", variant="secondary", elem_classes=["clear-btn"])
                            status_output = gr.Textbox(label="状态信息", visible=False)
                    
                    # 详细信息标签页
                    gr.HTML("<h3>🔍 详细信息</h3>")
                    with gr.Tabs():
                        with gr.TabItem("知识图谱"):
                            triples_output = gr.HTML(label="检索到的知识图谱信息")
                        with gr.TabItem("相关文档"):
                            search_output_html = gr.HTML(label="检索到的文档信息")
                    
                    # 示例问题区域
                    gr.HTML("<h4>快速提问示例:</h4>")
                    example_questions = [
                        "函数式编程的概念是什么？", "敏捷开发的核心原则是什么？",
                        "软件测试与质量保证有什么区别？", "UML图有哪些类型？",
                        "设计模式的作用是什么？", "如何进行软件架构评估？"
                    ]
                    gr.Examples(
                        examples=example_questions, 
                        inputs=question_input, 
                        label=""
                    )

            # 页脚
            gr.HTML("""
            <footer>
            Powered by LLM + Knowledge Graph + RAG | 软件工程课程专用学习助手
            </footer>
            """)

        # --- 事件绑定 ---

        # 文件上传按钮点击事件
        upload_button.click(
            fn=handle_file_upload,
            inputs=[file_upload],
            outputs=[upload_status]
        )

        # 提交问题按钮点击事件
        submit_button.click(
            fn=answer_question_ui,
            inputs=[question_input, chat_display],
            outputs=[chat_display, triples_output, search_output_html, status_output]
        ).then(
            # 清空输入框
            lambda: "",
            None,
            question_input
        )

        # 问题输入框回车事件 (等同于点击提交)
        question_input.submit(
            fn=answer_question_ui,
            inputs=[question_input, chat_display],
            outputs=[chat_display, triples_output, search_output_html, status_output]
        ).then(
            # 清空输入框
            lambda: "",
            None,
            question_input
        )
        
        # 清除对话历史按钮
        clear_button.click(
            fn=clear_conversation,
            inputs=[chat_display],
            outputs=[chat_display]
        )

    return app

# 创建并启动界面
if __name__ == "__main__":
    # 检查 QA 系统是否成功初始化
    if 'QA_SYSTEM_INIT_ERROR' in globals() and QA_SYSTEM_INIT_ERROR:
            logger.error(f"!!! Gradio UI cannot start due to QA System initialization error: {QA_SYSTEM_INIT_ERROR} !!!")
            with gr.Blocks() as error_app:
                gr.Markdown(f"# 系统启动失败\n\n无法初始化问答系统，请检查配置和后台服务。\n\n**错误信息:**\n```\n{QA_SYSTEM_INIT_ERROR}\n```")
            logger.info("\nLaunching error display UI...")
            error_app.launch(share=True)
    else:
            logger.info("\nLaunching Gradio UI...")
            app = create_ui()
            app.launch(
                share=True,
                css=custom_css,
                theme=gr.themes.Soft()
            ) # share=True 会创建公网链接