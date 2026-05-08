# ChatDoc 智能体问答系统

一个基于 RAG（检索增强生成）技术的智能文档问答系统。

## ✨ 功能特性

- ✅ **文档上传**: 支持 TXT、PDF、Word 等多种格式
- ✅ **智能分块**: 自动按语义分割文档
- ✅ **向量化**: 使用 sentence-transformers 进行语义嵌入
- ✅ **语义检索**: ChromaDB 向量数据库（或 NumPy 备用）
- ✅ **智能问答**: 支持 OpenAI GPT、Ollama 本地模型
- ✅ **文档摘要**: 自动生成文档概要
- ✅ **文档翻译**: 多语言翻译支持
- ✅ **引用溯源**: 回答时显示原文引用

## 🚀 快速开始

### 方式1: 使用启动脚本（推荐）

```bash
chmod +x run_chatdoc.sh
./run_chatdoc.sh
```

### 方式2: 手动运行

```bash
# 激活虚拟环境
source .venv/bin/activate

# 检查依赖
python check_dependencies.py

# 运行完整版（需要安装依赖）
python chatdoc_real_agent.py

# 或运行简化版（无需依赖）
python chatdoc_simple.py
```

## 📦 安装依赖

### 核心依赖（必需）

```bash
pip install sentence-transformers numpy chromadb -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 可选依赖

```bash
# PDF 支持
pip install PyPDF2

# Word 支持
pip install python-docx

# OpenAI LLM
pip install openai
```

## 💡 使用示例

### 1. 加载文档

```python
chatdoc = ChatDocAgent(
    embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
    llm_type="mock",
    use_chroma=True
)

doc_id = chatdoc.load_document("/path/to/document.txt")
```

### 2. 智能问答

```python
answer = chatdoc.ask_question("文档的主要内容是什么？", top_k=5)
print(answer.answer)
print(f"置信度: {answer.confidence}")
```

### 3. 生成摘要

```python
summary = chatdoc.summarize_document(doc_id)
print(summary)
```

### 4. 文档翻译

```python
translation = chatdoc.translate_document(doc_id, target_lang="英文")
print(translation)
```

## 🏗️ 技术架构

```
文档上传 → 文本解析 → 智能分块 → 向量化 → ChromaDB索引
                                              ↓
用户提问 → 问题向量化 → 相似度检索 → 构建上下文 → LLM生成 → 返回答案
```

### 核心技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 向量化 | sentence-transformers | 多语言语义嵌入模型 |
| 向量数据库 | ChromaDB | 轻量级向量检索引擎 |
| LLM | OpenAI / Ollama / Mock | 回答生成 |
| 文档解析 | PyPDF2 / python-docx | 多格式支持 |

## 📊 性能对比

| 方案 | 检索速度 | 安装难度 | 准确度 |
|------|---------|---------|--------|
| ChromaDB | ⭐⭐⭐⭐ 快 | ⭐⭐⭐⭐⭐ 简单 | ⭐⭐⭐⭐⭐ |
| NumPy | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐⭐ 简单 | ⭐⭐⭐⭐⭐ |
| FAISS | ⭐⭐⭐⭐⭐ 最快 | ❌ 编译困难 | ⭐⭐⭐⭐⭐ |

## 🔧 配置选项

### LLM 类型

```python
# 模拟模式（无需 API Key）
chatdoc = ChatDocAgent(llm_type="mock")

# OpenAI GPT
chatdoc = ChatDocAgent(
    llm_type="openai",
    openai_api_key="sk-your-key"
)

# Ollama 本地模型
chatdoc = ChatDocAgent(
    llm_type="ollama",
    ollama_base_url="http://localhost:11434"
)
```

### 向量化模型

```python
# 中文优化模型
chatdoc = ChatDocAgent(embedding_model="paraphrase-multilingual-MiniLM-L12-v2")

# 英文优化模型
chatdoc = ChatDocAgent(embedding_model="all-MiniLM-L6-v2")
```

## 📝 项目结构

```
FutureTradingAgent/
├── chatdoc_real_agent.py      # 完整版 ChatDoc（真正的 RAG）
├── chatdoc_simple.py          # 简化版 ChatDoc（无需依赖）
├── chatdoc_lingma_demo.py     # 原理演示版
├── check_dependencies.py      # 依赖检查工具
├── requirements.txt           # 依赖列表
├── install_with_mirror.sh     # 安装脚本（国内镜像）
├── run_chatdoc.sh            # 快速启动脚本
└── README_CHATDOC.md         # 本文档
```

## ❓ 常见问题

### Q1: 为什么搜索结果不准确？
**A:** 确保已安装 `sentence-transformers`，否则使用的是随机向量。

### Q2: ChromaDB 初始化失败怎么办？
**A:** 系统会自动降级到 NumPy，功能完整但速度稍慢。

### Q3: 如何支持 PDF/Word？
**A:** 安装对应库：`pip install PyPDF2 python-docx`

### Q4: 如何使用自己的 API Key？
**A:** 
```python
chatdoc = ChatDocAgent(
    llm_type="openai",
    openai_api_key="sk-your-key"
)
```

## 🎯 应用场景

- 📚 **学术研究**: 快速阅读论文，提取关键信息
- ⚖️ **法律文档**: 合同审查，条款查询
- 💻 **技术文档**: API 文档查询，代码说明
- 📊 **财务报告**: 数据分析，趋势总结
- 🏢 **知识库**: 企业内部文档问答系统

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**开发时间**: 2026-05  
**Python 版本**: 3.13+  
**最后更新**: 2026-05-06
