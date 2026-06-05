# -*- coding: utf-8 -*-
"""
格式化工具函数
提供知识图谱上下文、文档上下文、对话历史的统一格式化
"""

import html
from typing import List, Dict, Any, Tuple


def format_kg_context(kg_context: Dict[str, Any]) -> str:
    """
    统一格式化知识图谱上下文
    
    Args:
        kg_context: 知识图谱上下文字典，包含 entities 和 paths
        
    Returns:
        格式化后的字符串
    """
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
    """
    统一格式化文档上下文
    
    Args:
        doc_results: 文档检索结果列表
        
    Returns:
        格式化后的字符串
    """
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


def format_conversation_history(conversation_history: List[Tuple[str, str]]) -> str:
    """
    格式化对话历史
    
    Args:
        conversation_history: 对话历史列表，每个元素是 (用户问题, 系统回答) 元组
        
    Returns:
        格式化后的字符串
    """
    if not conversation_history:
        return ""
    
    formatted_parts = []
    # 只使用最近5轮对话
    recent_history = conversation_history[-5:]
    
    for i, (user_q, assistant_a) in enumerate(recent_history):
        formatted_parts.append(f"用户第{i+1}轮问题: {user_q}")
        # 限制回答长度
        truncated_answer = assistant_a[:200] + ("..." if len(assistant_a) > 200 else "")
        formatted_parts.append(f"助手第{i+1}轮回答: {truncated_answer}")
        if i < len(recent_history) - 1:
            formatted_parts.append("")  # 添加空行分隔
    
    return "\n".join(formatted_parts)


def format_triples_as_html(kg_context: Dict[str, Any]) -> str:
    """
    将知识图谱三元组格式化为HTML表格展示
    
    Args:
        kg_context: 知识图谱上下文字典
        
    Returns:
        HTML格式字符串
    """
    if not kg_context or (not kg_context.get("entities") and not kg_context.get("paths")):
        return "<p>未检索到相关的知识图谱信息。</p>"
    
    html_parts = []
    css_added = False
    
    # 处理实体及其关系
    if kg_context.get("entities"):
        html_parts.append("<h4>知识图谱实体与关系：</h4>")
        for entity_data in kg_context["entities"]:
            entity = entity_data.get("entity", {})
            relationships = entity_data.get("relationships", [])
            
            if not entity or not entity.get("name"):
                continue
            
            entity_name = entity.get("name", "未知")
            entity_type = entity.get("type", "未知")
            
            # 实体信息标题
            html_parts.append(f"<h5>实体: {entity_name} (类型: {entity_type})</h5>")
            
            # 构建关系表格
            if relationships:
                if not css_added:
                    css = """
                    <style>
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
                    </style>
                    """
                    html_parts.insert(0, css)
                    css_added = True
                
                html_parts.append("<table class='styled-table'>")
                html_parts.append("<tr><th>方向</th><th>关系</th><th>相关实体</th><th>类型</th></tr>")
                
                for rel in relationships:
                    direction_arrow = ""
                    rel_name = rel.get("rel_name") or rel.get("relationship", "未知关系")
                    other_entity_name = ""
                    other_entity_type = ""
                    
                    if rel.get("direction") == "outgoing":
                        direction_arrow = f"{entity_name} →"
                        other_entity_name = rel.get("target", "未知")
                        other_entity_type = rel.get("target_type", "未知")
                    elif rel.get("direction") == "incoming":
                        direction_arrow = f"→ {entity_name}"
                        other_entity_name = rel.get("source", "未知")
                        other_entity_type = rel.get("source_type", "未知")
                    else:
                        direction_arrow = "?"
                    
                    html_parts.append(f"<tr><td>{direction_arrow}</td><td>{rel_name}</td><td>{other_entity_name}</td><td>{other_entity_type}</td></tr>")
                
                html_parts.append("</table>")
            else:
                html_parts.append("<p style='font-size:0.9em; color:#777;'>（未找到该实体的直接关系）</p>")
            html_parts.append("<br>")
    
    # 处理实体间路径
    if kg_context.get("paths"):
        html_parts.append("<h4>知识图谱实体间路径：</h4>")
        if not css_added:
            css = """
            <style>
            h4 { margin-top: 15px; margin-bottom: 5px; color: #333; }
            h5 { margin-top: 10px; margin-bottom: 5px; color: #555; }
            </style>
            """
            html_parts.insert(0, css)
            css_added = True
        
        for i, path_data in enumerate(kg_context["paths"]):
            nodes = path_data.get("nodes", [])
            relationships = path_data.get("relationships", [])
            if not nodes or not relationships:
                continue
            
            html_parts.append(f"<h5>路径 {i+1}:</h5>")
            path_str_elements = []
            for j, node in enumerate(nodes):
                node_name = node.get('name', '?')
                node_type = node.get('type', '?')
                path_str_elements.append(f"({node_name}<small>:{node_type}</small>)")
                if j < len(relationships):
                    rel = relationships[j]
                    rel_name = rel.get('name') or rel.get('type', 'RELATED_TO')
                    if rel.get('source') == nodes[j].get('name'):
                        path_str_elements.append(f"-[<small>{rel_name}</small>]-&gt;")
                    else:
                        path_str_elements.append(f"&lt;-[<small>{rel_name}</small>]-")
            
            html_parts.append(f"<p style='font-family: monospace; font-size: 0.9em;'>{''.join(path_str_elements)}</p>")
            html_parts.append("<br>")
    
    if not html_parts:
        return "<p>未能格式化知识图谱信息。</p>"
    
    return "\n".join(html_parts)


def format_search_results_as_html(results: List[Dict[str, Any]], knowledge_base_loaded: bool = True) -> str:
    """
    将文档搜索结果格式化为HTML
    
    Args:
        results: 文档检索结果列表
        knowledge_base_loaded: 知识库是否已加载
        
    Returns:
        HTML格式字符串
    """
    if not results:
        if not knowledge_base_loaded:
            return "<p>⚠️ 知识库尚未加载或加载失败。请先上传有效的知识库文档。</p>"
        else:
            return "<p>✅ 知识库已加载，但在文档中未找到与查询语义相关的段落 (阈值 > 0.3)。</p>"
    
    html_result = """
    <style>
    .search-results {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin: 10px 0;
    }
    .search-results h4 {
        color: #333;
        margin-bottom: 15px;
        font-size: 1.1em;
        font-weight: 600;
    }
    .result-item {
        background-color: #ffffff;
        border: 1px solid #eef2f6;
        border-left: 4px solid #2a81e3;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    }
    .result-item h5 {
        color: #2a81e3;
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 1em;
    }
    .result-text {
        color: #555;
        line-height: 1.6;
        font-size: 0.95em;
    }
    </style>
    """
    
    html_result += "<div class='search-results'>"
    html_result += f"<h4>在知识库文档中找到 {len(results)} 个相关段落：</h4>"
    
    for i, res in enumerate(results):
        source_escaped = html.escape(res.get('source', '未知来源'))
        text_escaped = html.escape(res.get('text', ''))
        similarity = res.get('similarity', 0.0)
        html_result += f"""
        <div class='result-item'>
            <h5>相关段落 {i + 1} (来自: {source_escaped}, 相似度: {similarity:.3f})</h5>
            <div class='result-text'>{text_escaped}</div>
        </div>"""
    
    html_result += "</div>"
    return html_result
