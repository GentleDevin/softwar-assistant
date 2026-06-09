# 软件工程课程助手

基于大语言模型、知识图谱和 RAG 的智能学习辅助系统。

## 功能特性

- 知识图谱检索：基于 Neo4j 的知识图谱查询
- 文档检索：基于 FAISS 的本地文档向量化检索
- 多智能体系统：针对不同类型问题的专业智能体
- 响应时间统计：显示每个查询的处理时间
- 日志管理：按天和按大小轮转的日志系统

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env` 文件并填入您的实际配置：

```bash
cp .env.example .env
# 编辑 .env 文件
```

### 3. 启动 Neo4j（如需要）

确保 Neo4j 数据库已启动并配置正确。

### 4. 启动系统

```bash
python softeng_qa_ui.py
```

然后在浏览器中访问显示的 URL（通常是 http://localhost:7860）。

## 项目结构

```
├── agents/                # 智能体模块
├── config/                # 配置模块
├── core/                  # 核心功能模块
│   ├── config.py         # 配置管理
│   ├── error_handling.py # 错误处理
│   └── monitoring.py     # 性能监控
├── models/               # 数据模型
├── utils/                # 工具函数
│   └── logging.py       # 日志管理
├── faiss_cache/         # FAISS 向量缓存
├── logs/                # 日志文件
├── softeng_kg_qa.py     # 核心问答系统
└── softeng_qa_ui.py     # Gradio 界面
```

## 日志系统

日志文件存储在 `logs/` 目录下：
- 命名格式：`app-YYYY-MM-DD.log`（未超过大小限制）
- 超过大小限制：`app-YYYY-MM-DD-01.log`、`app-YYYY-MM-DD-02.log` 等

## 性能优化

- 实体提取：关键词快速匹配优先，LLM 作为后备
- 嵌入缓存：实体嵌入和文档嵌入都有缓存机制
- 知识图谱查询：连接池管理

## License

本项目仅供学习使用。
