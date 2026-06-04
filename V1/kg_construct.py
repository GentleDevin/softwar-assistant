import torch
import json
import logging
import re
import time
import os
import gc
from tqdm import tqdm
from neo4j import GraphDatabase
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Any, Tuple
from modelscope import AutoModelForCausalLM, AutoTokenizer

# 导入统一配置模块
from config import get_neo4j_config, get_log_config
from utils import setup_logger

# 配置日志
log_config = get_log_config()
logger = setup_logger(__name__, log_file=log_config.log_file, log_level=log_config.log_level)

# 定义函数编程的实体类型
ENTITY_TYPES = [
    "概念", "方法", "工具",
    "模型", "原则", "阶段",
    "角色", "工件", "技术",
    "框架", "模式", "流程",
    "标准", "实践", "语言"
]

# 定义函数编程的关系类型
RELATIONSHIP_TYPES = [
    "属于", "包含", "使用",
    "定义", "实现", "创建",
    "前置", "后置", "依赖",
    "派生", "应用于", "结合",
    "基于", "替代", "优化",
    "参与", "生成", "验证",
    "遵循", "扩展", "关联"
]

class QwenFunctionalProgrammingKGBuilder:
    """函数编程知识图谱构建类"""
    
    def __init__(self, model_name="/workspace/model/Qwen2.5-7B-Instruct"):
        """初始化知识图谱构建器"""
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def validate_triple_structure(self, triple):
        """
        验证三元组的结构是否有效
        
        Args:
            triple: 要验证的三元组
            
        Returns:
            如果三元组结构有效则为True，否则为False
        """
        # 检查必要的键是否存在
        if not isinstance(triple, dict):
            return False
            
        if "subject" not in triple or "predicate" not in triple or "object" not in triple:
            return False
            
        # 检查subject是否有效
        if not isinstance(triple["subject"], dict):
            return False
        if "name" not in triple["subject"] or "type" not in triple["subject"]:
            return False
            
        # 检查object是否有效
        if not isinstance(triple["object"], dict):
            return False
        if "name" not in triple["object"] or "type" not in triple["object"]:
            return False
            
        return True
        
    def initialize_model(self):
        """加载和初始化Qwen模型"""
        logger.info(f"正在初始化模型: {self.model_name}")
        
        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 使用ModelScope加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        logger.info("模型初始化成功")
        
    def extract_triples(self, text: str) -> List[Dict[str, Any]]:
        """从文本中提取函数编程知识三元组"""
        if not self.model or not self.tokenizer:
            raise ValueError("模型未初始化，请先调用initialize_model()")
        
        # 初始化时清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 创建三元组提取提示
        triple_prompt = f"""
        请从以下函数编程文本中提取知识图谱三元组(主语-谓语-宾语)。

        对于主语和宾语实体，确保它们属于以下实体类型之一并指定类型:
        {', '.join(ENTITY_TYPES)}

        关系类型应属于以下之一:
        {', '.join(RELATIONSHIP_TYPES)}

        输出格式:
        仅返回具有以下确切结构的有效JSON数组:
        [
        {{
            "subject": {{
            "name": "实体名称",
            "type": "实体类型(从列表中选择)"
            }},
            "predicate": "关系(从列表中选择)",
            "object": {{
            "name": "实体名称",
            "type": "实体类型(从列表中选择)"
            }}
        }}
        ]

        文本: {text}

        重要: 输出必须仅是格式正确的JSON数组，前后没有其他文本。不要包含解释、markdown格式或代码块。
        """
        
        # 创建三元组提取消息
        messages = [
            {"role": "system", "content": "你是一个专门从函数编程文献中提取知识三元组的AI助手。你只输出干净的JSON格式的三元组，不包含任何解释性文本。"},
            {"role": "user", "content": triple_prompt},
        ]
        
        # 提取三元组，使用更保守的参数
        try:
            # 使用Qwen的接口生成响应
            chat_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 准备模型输入
            model_inputs = self.tokenizer([chat_text], return_tensors="pt").to(self.model.device)
            
            # 显示当前内存使用情况
            if torch.cuda.is_available():
                logger.debug(f"模型输入前GPU内存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            
            # 生成响应
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=2048,  # 减小最大token数
                do_sample=False       # 确定性生成
            )
            
            # 显示当前内存使用情况
            if torch.cuda.is_available():
                logger.debug(f"模型生成后GPU内存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            
            # 提取生成的部分
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            # 解码响应
            response_content = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 尝试从响应中提取JSON
            triples = []
            if response_content:
                # 首先去除任何潜在的markdown代码块
                cleaned_response = re.sub(r'```json\s*|\s*```', '', response_content)
                cleaned_response = re.sub(r'```\s*|\s*```', '', cleaned_response)
                
                # 尝试查找并提取JSON数组
                json_match = re.search(r'\[\s*\{.*\}\s*\]', cleaned_response, re.DOTALL)
                if json_match:
                    try:
                        triples = json.loads(json_match.group(0))
                        logger.info(f"成功解析JSON，找到{len(triples)}个三元组")
                    except json.JSONDecodeError as e:
                        logger.error(f"找到类JSON结构但无法解析: {e}")
                        logger.error(f"提取的JSON: {json_match.group(0)[:200]}...")
                else:
                    # 尝试解析整个清理后的响应
                    try:
                        triples = json.loads(cleaned_response)
                        logger.info(f"成功解析整个响应，找到{len(triples)}个三元组")
                    except json.JSONDecodeError:
                        logger.error(f"无法将响应解析为JSON")
                        logger.error(f"清理后的响应: {cleaned_response[:200]}...")
            
            # 验证三元组结构
            valid_triples = []
            invalid_count = 0
            for i, trip in enumerate(triples):
                if self.validate_triple_structure(trip):
                    valid_triples.append(trip)
                else:
                    invalid_count += 1
                    if invalid_count <= 3:  # 只显示前3个无效三元组
                        logger.warning(f"警告: 索引 {i} 处的三元组结构无效: {trip}")
                    elif invalid_count == 4:
                        logger.warning("更多无效三元组被跳过...")
            
            logger.info(f"提取总结: 找到 {len(triples)} 个三元组，其中 {len(valid_triples)} 个结构有效")
            
            # 释放内存
            del model_inputs, generated_ids, response_content
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return valid_triples
                
        except Exception as e:
            logger.error(f"三元组提取过程中出错: {e}")
            # 释放内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return []
    
    def process_text(self, text: str) -> List[Dict[str, Any]]:
        """处理文本以提取三元组"""
        # 提取三元组
        logger.info("正在提取三元组...")
        triples = self.extract_triples(text)
        logger.info(f"找到{len(triples)}个三元组")
        
        return triples
    
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 50) -> List[str]:
        """
        将长文本分割成小块，保留一定的重叠以维持上下文
        
        Args:
            text: 要分块的文本
            chunk_size: 每个块的最大字符数
            overlap: 块之间重叠的字符数
            
        Returns:
            文本块列表
        """
        chunks = []
        start = 0
        
        while start < len(text):
            # 计算当前块的结束位置
            end = min(start + chunk_size, len(text))
            
            # 如果不是最后一块，尝试在句子边界上切分
            if end < len(text):
                # 向后查找最近的句号、问号或感叹号后的位置
                sentence_end = -1
                for i in range(end, max(start, end - 200), -1):
                    if text[i-1] in "。？！.?!":
                        sentence_end = i
                        break
                
                if sentence_end > 0:
                    end = sentence_end
            
            # 添加当前块
            chunks.append(text[start:end])
            
            # 更新下一块的起始位置，考虑重叠
            start = max(start, end - overlap)
        
        return chunks
    
    def process_single_file(self, file_path: str, output_json_path: str, chunk_size: int = 800, overlap: int = 50, max_chunks: int = 10):
        """
        处理单个文本文件以提取三元组并保存结果到JSON文件
        
        Args:
            file_path: 输入文本文件的路径
            output_json_path: 保存提取的三元组的JSON文件路径
            chunk_size: 每个文本块的最大字符数
            overlap: 块之间重叠的字符数
            max_chunks: 一次处理的最大块数
            
        Returns:
            包含三元组和处理统计信息的字典
        """
        # 首先清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 如果模型尚未初始化，初始化模型
        if not self.model or not self.tokenizer:
            self.initialize_model()
        
        # 读取文本文件 - 改成分块读取处理
        logger.info(f"正在读取文本文件: {file_path}")
        all_triples = []
        
        try:
            # 读取并立即分块，避免整个文件都加载到内存
            with open(file_path, 'r', encoding='utf-8') as f:
                # 获取文件总大小
                f.seek(0, 2)  # 移动到文件末尾
                file_size = f.tell()  # 获取文件大小
                f.seek(0)  # 回到文件开头
                
                logger.info(f"文件大小: {file_size} 字节")
                
                # 分块读取文件
                text_buffer = ""
                chunks = []
                chunk_count = 0
                
                while True:
                    # 读取一部分文本
                    chunk = f.read(chunk_size * 2)  # 一次读取两个块大小
                    if not chunk:
                        break
                        
                    # 添加到缓冲区
                    text_buffer += chunk
                    
                    # 当缓冲区足够大时，提取一个块
                    while len(text_buffer) >= chunk_size:
                        # 截取一个块
                        current_chunk = text_buffer[:chunk_size]
                        
                        # 尝试在句子边界处切分
                        sentence_end = -1
                        for i in range(min(chunk_size, len(current_chunk)), max(0, chunk_size - 200), -1):
                            if i < len(current_chunk) and current_chunk[i-1] in "。？！.?!":
                                sentence_end = i
                                break
                        
                        if sentence_end > 0:
                            current_chunk = text_buffer[:sentence_end]
                        else:
                            current_chunk = text_buffer[:chunk_size]
                        
                        # 添加到块列表
                        chunks.append(current_chunk)
                        chunk_count += 1
                        
                        # 移动缓冲区，保留重叠部分
                        overlap_size = min(overlap, len(current_chunk))
                        text_buffer = text_buffer[len(current_chunk) - overlap_size:]
                        
                        # 如果达到最大块数，停止
                        if chunk_count >= max_chunks:
                            break
                    
                    # 如果达到最大块数，停止
                    if chunk_count >= max_chunks:
                        break
                
                # 处理剩余的文本
                if text_buffer and chunk_count < max_chunks:
                    chunks.append(text_buffer)
                    chunk_count += 1
                
                logger.info(f"文本已分割为 {len(chunks)} 个块")
        except Exception as e:
            logger.error(f"读取或分块文件时出错: {e}")
            return {"error": str(e)}
        
        # 显示内存使用情况
        logger.info("初始内存状态:")
        if torch.cuda.is_available():
            logger.debug(f"GPU内存分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            logger.debug(f"GPU内存缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        # 限制处理的块数
        chunks_to_process = min(len(chunks), max_chunks)
        logger.info(f"将处理前 {chunks_to_process} 个块")
        
        # 处理每个块并收集三元组
        for i, chunk in enumerate(tqdm(chunks[:chunks_to_process], desc="处理文本块")):
            logger.info(f"\n处理块 {i+1}/{chunks_to_process}, 大小: {len(chunk)} 字符")
            
            try:
                # 处理当前块
                triples = self.process_text(chunk)
                
                # 添加到总集合
                all_triples.extend(triples)
                
                # 保存中间结果
                if (i+1) % 2 == 0 or i+1 == chunks_to_process:  # 每处理2个块保存一次
                    # 创建中间备份文件
                    output_dir = os.path.dirname(output_json_path)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                        
                    with open(output_json_path + f".part{i+1}", "w", encoding="utf-8") as out_f:
                        json.dump(all_triples, out_f, indent=2, ensure_ascii=False)
                    logger.info(f"已保存中间结果到{output_json_path}.part{i+1}")
                
                # 显示内存使用情况
                if torch.cuda.is_available():
                    logger.debug(f"处理块 {i+1} 后GPU内存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                
                # 释放内存
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"处理块 {i+1} 时出错: {e}")
                # 记录错误但继续处理下一个块
                continue
            
            # 避免模型过载
            if i < chunks_to_process - 1:
                logger.info("休息3秒...")
                time.sleep(3)
        
        # 保存最终结果
        output_dir = os.path.dirname(output_json_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
                
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(all_triples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"提取完成。处理了{chunks_to_process}个文本块，找到{len(all_triples)}个三元组")
        logger.info(f"结果已保存到{output_json_path}")
        
        return {
            "triples": all_triples,
            "processed_chunks": chunks_to_process,
            "total_triples": len(all_triples)
        }
    
    def deduplicate_triples(self, triples):
        """
        去除重复的三元组
        
        Args:
            triples: 三元组列表
            
        Returns:
            去重后的三元组列表
        """
        # 创建用于去重的集合
        unique_dict = {}
        invalid_count = 0
        
        for i, triple in enumerate(triples):
            # 验证三元组结构
            if not self.validate_triple_structure(triple):
                invalid_count += 1
                if invalid_count <= 5:  # 只显示前5个无效三元组，避免日志过多
                    logger.warning(f"警告: 跳过索引 {i} 处的无效三元组: {triple}")
                elif invalid_count == 6:
                    logger.warning("更多无效三元组被跳过...")
                continue
            
            # 创建三元组的唯一键
            key = (
                triple["subject"]["name"],
                triple["subject"]["type"],
                triple["predicate"],
                triple["object"]["name"],
                triple["object"]["type"]
            )
            
            # 如果这个键不存在，添加它
            if key not in unique_dict:
                # 复制三元组并添加
                unique_dict[key] = triple.copy()
                
                # 如果有多个来源，记录在sources字段中
                if "source" in triple:
                    unique_dict[key]["sources"] = [triple["source"]]
                    # 保留但重命名原始source字段
                    unique_dict[key]["primary_source"] = triple["source"]
            else:
                # 如果三元组已存在，更新来源信息
                if "source" in triple:
                    if "sources" in unique_dict[key]:
                        if triple["source"] not in unique_dict[key]["sources"]:
                            unique_dict[key]["sources"].append(triple["source"])
                    else:
                        # 如果之前没有sources字段，创建它
                        source1 = unique_dict[key].get("primary_source")
                        if source1:
                            unique_dict[key]["sources"] = [source1, triple["source"]]
                        else:
                            unique_dict[key]["sources"] = [triple["source"]]
                            unique_dict[key]["primary_source"] = triple["source"]
        
        # 转换回列表
        unique_triples = list(unique_dict.values())
        
        logger.info(f"去重过程：处理了 {len(triples)} 个三元组，跳过了 {invalid_count} 个无效三元组，剩余 {len(unique_triples)} 个唯一三元组")
        
        return unique_triples
    
    def create_visualization(self, triples, output_file=None, max_nodes=100):
        """创建知识图谱的可视化"""
        # 如果有太多三元组，抽样一个子集进行可视化
        if len(triples) > max_nodes * 2:
            logger.info(f"三元组太多，无法可视化。抽样{max_nodes*2}个三元组...")
            import random
            random.seed(42)  # 为了可重现性
            triples_sample = random.sample(triples, max_nodes*2)
        else:
            triples_sample = triples
            
        G = nx.DiGraph()
        
        # 添加节点和边
        node_count = 0
        for triple in triples_sample:
            if node_count >= max_nodes:
                break
            
            # 验证三元组结构
            if not self.validate_triple_structure(triple):
                continue
                
            subject = triple["subject"]["name"]
            subject_type = triple["subject"]["type"]
            predicate = triple["predicate"]
            obj = triple["object"]["name"]
            object_type = triple["object"]["type"]
            
            # 添加主语和宾语节点（如果不存在）
            if not G.has_node(subject):
                G.add_node(subject, type=subject_type)
                node_count += 1
            if not G.has_node(obj) and node_count < max_nodes:
                G.add_node(obj, type=object_type)
                node_count += 1
                
            # 添加关系边
            if G.has_node(subject) and G.has_node(obj):
                # 如果有来源信息，添加到边属性
                edge_attrs = {"label": predicate}
                if "sources" in triple:
                    edge_attrs["sources"] = ", ".join(triple["sources"])
                elif "source" in triple:
                    edge_attrs["source"] = triple["source"]
                    
                G.add_edge(subject, obj, **edge_attrs)
        
        # 检查是否有节点可视化
        if len(G.nodes()) == 0:
            logger.warning("没有节点可视化。")
            return
            
        # 创建可视化
        plt.figure(figsize=(16, 12))
        
        # 定位节点
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        
        # 绘制节点
        node_types = [G.nodes[node]["type"] for node in G.nodes()]
        unique_types = list(set(node_types))
        type_color_map = {t: i for i, t in enumerate(unique_types)}
        node_colors = [type_color_map[G.nodes[node]["type"]] for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, alpha=0.8, cmap=plt.cm.tab20)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, arrows=True, arrowsize=15)
        
        # 绘制标签
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_family="SimHei")
        
        # 绘制边标签
        edge_labels = {(u, v): G[u][v]["label"] for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_family="SimHei")
        
        # 添加图例
        handles = []
        labels = []
        for t, i in type_color_map.items():
            color = plt.cm.tab20(i)
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10))
            labels.append(t)
            
        plt.legend(handles, labels, loc='upper right', fontsize=10)
        
        plt.title(f"函数编程知识图谱（{len(G.nodes())}个节点样本）", fontproperties="SimHei")
        plt.axis("off")
        
        # 保存或显示可视化
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            logger.info(f"可视化已保存到{output_file}")
        else:
            plt.show()
        
        plt.close()


class Neo4jHandler:
    """Neo4j数据库操作处理器"""
    
    def __init__(self, uri, username, password):
        """初始化Neo4j处理器"""
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        
    def connect(self):
        """连接到Neo4j数据库"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # 测试连接
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                result.single()
            logger.info("成功连接到Neo4j数据库")
        except Exception as e:
            logger.error(f"连接Neo4j失败: {e}")
            raise
        
    def close(self):
        """关闭Neo4j连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")
            
    def clear_database(self):
        """从数据库中删除所有节点和关系"""
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("数据库已清空")
        except Exception as e:
            logger.error(f"清空数据库时出错: {e}")
            
    def create_constraints(self):
        """为实体类型创建约束"""
        try:
            with self.driver.session() as session:
                # 为每个实体类型创建约束
                for entity_type in ENTITY_TYPES:
                    # 净化实体类型以适应Neo4j
                    clean_type = re.sub(r'[^\w\u4e00-\u9fa5]', '', entity_type)
                    try:
                        # 创建约束（语法取决于Neo4j版本）
                        try:
                            # Neo4j 4.x+
                            session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{clean_type}`) REQUIRE n.name IS UNIQUE")
                        except:
                            try:
                                # Neo4j 3.x
                                session.run(f"CREATE CONSTRAINT ON (n:`{clean_type}`) ASSERT n.name IS UNIQUE")
                            except:
                                logger.warning(f"无法为{clean_type}创建约束，跳过")
                    except Exception as e:
                        logger.error(f"为{clean_type}创建约束时出错: {e}")
                logger.info("已为所有实体类型创建约束")
        except Exception as e:
            logger.error(f"设置约束时出错: {e}")
            
    def add_triple(self, triple):
        """向数据库添加三元组"""
        try:
            with self.driver.session() as session:
                # 提取三元组信息
                subject_name = triple["subject"]["name"]
                subject_type = triple["subject"]["type"]
                predicate = triple["predicate"]
                object_name = triple["object"]["name"]
                object_type = triple["object"]["type"]
                
                # 净化类型以适应Neo4j
                clean_subject_type = re.sub(r'[^\w\u4e00-\u9fa5]', '', subject_type)
                clean_object_type = re.sub(r'[^\w\u4e00-\u9fa5]', '', object_type)
                clean_predicate = re.sub(r'[^\w\u4e00-\u9fa5]', '_', predicate)
                
                # 准备来源属性
                source_props = ""
                if "sources" in triple:
                    source_list = triple["sources"]
                    source_props = ", r.sources = $sources"
                elif "source" in triple:
                    source_list = [triple["source"]]
                    source_props = ", r.source = $source"
                else:
                    source_list = None
                
                # 创建三元组
                query = f"""
                MERGE (s:`{clean_subject_type}` {{name: $subject_name}})
                SET s.type = $subject_type
                MERGE (o:`{clean_object_type}` {{name: $object_name}})
                SET o.type = $object_type
                MERGE (s)-[r:`{clean_predicate}`]->(o)
                SET r.name = $predicate{source_props}
                RETURN s, r, o
                """
                
                # 准备查询参数
                params = {
                    "subject_name": subject_name,
                    "subject_type": subject_type,
                    "object_name": object_name,
                    "object_type": object_type,
                    "predicate": predicate
                }
                
                # 添加来源参数
                if "sources" in triple:
                    params["sources"] = source_list
                elif "source" in triple and source_list:
                    params["source"] = source_list[0]
                
                result = session.run(query, **params)
                
                return result.single() is not None
        except Exception as e:
            logger.error(f"添加三元组时出错: {e}")
            logger.error(f"有问题的三元组: {triple}")
            return False
            
    def add_triples_batch(self, triples, batch_size=100):
        """批量向数据库添加三元组"""
        successful = 0
        total = len(triples)
        
        # 添加三元组结构验证
        valid_triples = []
        invalid_count = 0
        
        # 创建验证函数
        def validate_triple_structure(triple):
            if not isinstance(triple, dict):
                return False
                
            if "subject" not in triple or "predicate" not in triple or "object" not in triple:
                return False
                
            if not isinstance(triple["subject"], dict):
                return False
            if "name" not in triple["subject"] or "type" not in triple["subject"]:
                return False
                
            if not isinstance(triple["object"], dict):
                return False
            if "name" not in triple["object"] or "type" not in triple["object"]:
                return False
                
            return True
        
        # 验证三元组
        for i, triple in enumerate(triples):
            if validate_triple_structure(triple):
                valid_triples.append(triple)
            else:
                invalid_count += 1
                if invalid_count <= 3:
                    logger.warning(f"警告: 跳过无效三元组 {i}: {triple}")
                elif invalid_count == 4:
                    logger.warning("更多无效三元组被跳过...")
        
        logger.info(f"验证总结: 总共 {len(triples)} 个三元组，其中 {len(valid_triples)} 个有效， {invalid_count} 个无效")
        
        # 使用有效的三元组进行批处理
        for i in range(0, len(valid_triples), batch_size):
            batch = valid_triples[i:i+batch_size]
            try:
                with self.driver.session() as session:
                    with session.begin_transaction() as tx:
                        for triple in batch:
                            # 提取三元组信息
                            subject_name = triple["subject"]["name"]
                            subject_type = triple["subject"]["type"]
                            predicate = triple["predicate"]
                            object_name = triple["object"]["name"]
                            object_type = triple["object"]["type"]
                            
                            # 净化类型以适应Neo4j
                            clean_subject_type = re.sub(r'[^\w\u4e00-\u9fa5]', '', subject_type)
                            clean_object_type = re.sub(r'[^\w\u4e00-\u9fa5]', '', object_type)
                            clean_predicate = re.sub(r'[^\w\u4e00-\u9fa5]', '_', predicate)
                            
                            # 准备来源属性
                            source_props = ""
                            if "sources" in triple:
                                source_list = triple["sources"]
                                source_props = ", r.sources = $sources"
                            elif "source" in triple:
                                source_list = [triple["source"]]
                                source_props = ", r.source = $source"
                            else:
                                source_list = None
                            
                            # 创建三元组
                            query = f"""
                            MERGE (s:`{clean_subject_type}` {{name: $subject_name}})
                            SET s.type = $subject_type
                            MERGE (o:`{clean_object_type}` {{name: $object_name}})
                            SET o.type = $object_type
                            MERGE (s)-[r:`{clean_predicate}`]->(o)
                            SET r.name = $predicate{source_props}
                            """
                            
                            # 准备查询参数
                            params = {
                                "subject_name": subject_name,
                                "subject_type": subject_type,
                                "object_name": object_name,
                                "object_type": object_type,
                                "predicate": predicate
                            }
                            
                            # 添加来源参数
                            if "sources" in triple:
                                params["sources"] = source_list
                            elif "source" in triple and source_list:
                                params["source"] = source_list[0]
                            
                            tx.run(query, **params)
                        
                        # 提交事务
                        tx.commit()
                        successful += len(batch)
                
                logger.info(f"已添加 {i+len(batch)}/{len(valid_triples)} 个三元组")
                
            except Exception as e:
                logger.error(f"添加三元组批次 {i//batch_size + 1} 时出错: {e}")
        
        return successful
            
    def get_statistics(self):
        """获取知识图谱的统计信息"""
        try:
            stats = {}
            
            with self.driver.session() as session:
                # 获取节点数
                node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
                stats["node_count"] = node_count
                
                # 获取关系数
                rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
                stats["relationship_count"] = rel_count
                
                # 获取节点类型分布
                node_types_result = session.run(
                    "MATCH (n) RETURN DISTINCT labels(n)[0] as type, count(*) as count ORDER BY count DESC"
                )
                stats["node_types"] = [(record["type"], record["count"]) for record in node_types_result]
                
                # 获取关系类型分布
                rel_types_result = session.run(
                    "MATCH ()-[r]->() RETURN DISTINCT type(r) as type, count(*) as count ORDER BY count DESC"
                )
                stats["relationship_types"] = [(record["type"], record["count"]) for record in rel_types_result]
                
                return stats
        except Exception as e:
            logger.error(f"获取统计信息时出错: {e}")
            return {"error": str(e)}


def load_triples_to_neo4j(triples_json_path: str, neo4j_uri: str, neo4j_username: str, neo4j_password: str, 
                         clear_db: bool = True, batch_size: int = 100):
    """
    将三元组从JSON文件加载到Neo4j
    
    Args:
        triples_json_path: 包含三元组的JSON文件路径
        neo4j_uri: Neo4j数据库的URI
        neo4j_username: Neo4j用户名
        neo4j_password: Neo4j密码
        clear_db: 是否在加载三元组前清空数据库
        batch_size: 批处理大小
    
    Returns:
        包含已加载知识图谱统计信息的字典
    """
    # 从JSON加载三元组
    try:
        with open(triples_json_path, 'r', encoding='utf-8') as f:
            triples = json.load(f)
            
        logger.info(f"已从{triples_json_path}加载{len(triples)}个三元组")
    except Exception as e:
        logger.error(f"从{triples_json_path}加载三元组时出错: {e}")
        return {"error": str(e)}
    
    # 初始化Neo4j处理器
    neo4j_handler = Neo4jHandler(neo4j_uri, neo4j_username, neo4j_password)
    neo4j_handler.connect()
    
    # 如果请求，清空数据库
    if clear_db:
        neo4j_handler.clear_database()
    
    # 创建约束
    neo4j_handler.create_constraints()
    
    # 将三元组添加到数据库
    logger.info("正在将三元组批量添加到Neo4j...")
    successful_triples = neo4j_handler.add_triples_batch(triples, batch_size=batch_size)
    
    # 获取统计信息
    stats = neo4j_handler.get_statistics()
    stats["successful_triples"] = successful_triples
    stats["total_triples"] = len(triples)
    
    # 关闭Neo4j连接
    neo4j_handler.close()
    
    logger.info(f"\n成功添加{successful_triples}个三元组（共{len(triples)}个）到Neo4j")
    logger.info(f"Neo4j现在包含{stats['node_count']}个节点和{stats['relationship_count']}个关系")
    
    return stats


def visualize_triples(triples_json_path: str, output_file: str = None, max_nodes: int = 100):
    """
    从三元组JSON文件创建可视化
    
    Args:
        triples_json_path: 包含三元组的JSON文件路径
        output_file: 保存可视化图像的路径
        max_nodes: 可视化中包含的最大节点数
    """
    # 从JSON加载三元组
    try:
        with open(triples_json_path, 'r', encoding='utf-8') as f:
            triples = json.load(f)
            
        logger.info(f"已从{triples_json_path}加载{len(triples)}个三元组")
    except Exception as e:
        logger.error(f"从{triples_json_path}加载三元组时出错: {e}")
        return
    
    # 创建知识图谱构建器实例，仅用于可视化
    kg_builder = QwenFunctionalProgrammingKGBuilder()
    
    # 生成可视化
    kg_builder.create_visualization(triples, output_file, max_nodes)


# 主函数，用于运行代码
def main():
    """展示知识图谱构建过程的主函数"""
    
    # Neo4j数据库配置 - 优先从配置模块获取
    neo4j_config = get_neo4j_config()
    NEO4J_URI = neo4j_config.uri
    NEO4J_USERNAME = neo4j_config.username
    NEO4J_PASSWORD = neo4j_config.password
    
    # 创建输出目录
    output_dir = "functional_programming_kg_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理单个文件
    input_file = "函数编程.txt"
    output_json_path = os.path.join(output_dir, "functional_programming_triples.json")
    
    try:
        # 步骤1：处理文本文件，提取三元组
        logger.info("\n=== 步骤1：从文本中提取三元组 ===")
        print("\n=== 步骤1：从文本中提取三元组 ===")
        kg_builder = QwenFunctionalProgrammingKGBuilder()
        
        result = kg_builder.process_single_file(
            file_path=input_file,
            output_json_path=output_json_path,
            chunk_size=800,        # 每个文本块的大小
            overlap=50,           # 文本块之间的重叠
            max_chunks=10         # 处理的最大块数
        )
        
        # 去重三元组
        if "triples" in result:
            unique_triples = kg_builder.deduplicate_triples(result["triples"])
            
            # 保存去重后的三元组
            unique_output_path = os.path.splitext(output_json_path)[0] + "_unique.json"
            with open(unique_output_path, "w", encoding="utf-8") as f:
                json.dump(unique_triples, f, indent=2, ensure_ascii=False)
            
            logger.info(f"总共提取了 {len(result['triples'])} 个三元组，去重后有 {len(unique_triples)} 个")
            logger.info(f"去重结果已保存到 {unique_output_path}")
            print(f"总共提取了 {len(result['triples'])} 个三元组，去重后有 {len(unique_triples)} 个")
            print(f"去重结果已保存到 {unique_output_path}")
        
        # 步骤2：将三元组加载到Neo4j（可选）
        logger.info("\n=== 步骤2：将三元组加载到Neo4j（可选） ===")
        print("\n=== 步骤2：将三元组加载到Neo4j（可选） ===")
        
        import sys
        if sys.stdin.isatty():
            response = input("是否要将三元组加载到Neo4j？(y/n): ").strip().lower()
        else:
            logger.info("检测到非交互式终端，默认选择将三元组加载到Neo4j。")
            print("检测到非交互式终端，默认选择将三元组加载到Neo4j。")
            response = 'y'
            
        if response == 'y':
            # 使用去重后的三元组文件
            stats = load_triples_to_neo4j(
                triples_json_path=unique_output_path,
                neo4j_uri=NEO4J_URI,
                neo4j_username=NEO4J_USERNAME,
                neo4j_password=NEO4J_PASSWORD,
                clear_db=True,
                batch_size=100
            )
            
            # 报告统计信息
            logger.info("\n=== Neo4j知识图谱统计信息 ===")
            logger.info(f"总节点数: {stats['node_count']}")
            logger.info(f"总关系数: {stats['relationship_count']}")
            
            print("\n=== Neo4j知识图谱统计信息 ===")
            print(f"总节点数: {stats['node_count']}")
            print(f"总关系数: {stats['relationship_count']}")
            
            logger.info("\n节点类型:")
            print("\n节点类型:")
            for node_type, count in stats['node_types']:
                logger.info(f"  {node_type}: {count}")
                print(f"  {node_type}: {count}")
                
            logger.info("\n关系类型:")
            print("\n关系类型:")
            for rel_type, count in stats['relationship_types']:
                logger.info(f"  {rel_type}: {count}")
                print(f"  {rel_type}: {count}")
        
        # 步骤3：创建可视化
        logger.info("\n=== 步骤3：创建知识图谱可视化 ===")
        print("\n=== 步骤3：创建知识图谱可视化 ===")
        visualization_path = os.path.join(output_dir, "函数编程知识图谱可视化.png")
        visualize_triples(unique_output_path, visualization_path)
        
        logger.info(f"\n流程完成。结果保存到{output_dir}")
        print(f"\n流程完成。结果保存到{output_dir}")
        
    except Exception as e:
        logger.error(f"执行过程中出错: {e}")
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()