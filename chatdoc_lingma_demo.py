# 介绍chatDoc并给出示例Demo
"""
ChatDoc - 基于 RAG 的智能文档问答系统

ChatDoc 是一种 RAG（检索增强生成）应用，核心功能包括：
1. 上传文档（PDF、Word、Excel 等格式）
2. 对文档内容进行智能问答
3. 支持多文档管理和跨文档检索
4. 引用原文片段进行回答，确保答案可追溯

技术架构：
- 文档解析：提取文本、表格、图像内容
- 向量化：将文档内容转换为向量嵌入
- 检索：基于相似度搜索相关文档片段
- 生成：使用 LLM 生成带引用的回答

本示例演示如何使用 ChatDoc 构建文档问答系统。
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Document:
    """文档数据类"""
    doc_id: str
    file_name: str
    file_path: str
    content: str
    metadata: Dict = field(default_factory=dict)
    chunks: List[str] = field(default_factory=list)  # 文档分块


@dataclass
class Answer:
    """问答结果数据类"""
    question: str
    answer: str
    references: List[Dict] = field(default_factory=list)  # 引用来源
    doc_id: str = ""
    confidence: float = 0.0


class ChatDocSystem:
    """
    ChatDoc 文档问答系统

    基于 RAG 技术实现文档的智能问答功能
    """

    def __init__(self, api_key: str = ""):
        """
        初始化 ChatDoc 系统

        Args:
            api_key: LLM API 密钥（可选）
        """
        self.documents: Dict[str, Document] = {}
        self.chat_history: List[Dict] = []
        self.api_key = api_key
        self.vector_store = {}  # 模拟向量存储

    def upload_document(self, file_path: str, metadata: Optional[Dict] = None) -> str:
        """
        上传并解析文档
                
        支持的格式：
        - PDF: 学术论文、报告、书籍
        - Word (.docx): 文档、合同、说明
        - Excel (.xlsx): 表格数据、财务报表
        - TXT: 纯文本文件
                
        Args:
            file_path: 文档文件路径
            metadata: 文档元数据（标签、分类等）
                    
        Returns:
            文档 ID
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 检查文件格式
        allowed_extensions = ['.pdf', '.docx', '.xlsx', '.txt', '.doc', '.xls']
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in allowed_extensions:
            raise ValueError(f"不支持的文件格式: {file_ext}")

        # 读取文档内容
        content = self._parse_document(file_path)

        # 生成文档ID
        doc_id = f"doc_{len(self.documents) + 1:03d}"

        # 文档分块（用于向量检索）
        chunks = self._split_into_chunks(content, chunk_size=500)

        # 创建文档对象
        document = Document(
            doc_id=doc_id,
            file_name=os.path.basename(file_path),
            file_path=file_path,
            content=content,
            metadata=metadata or {},
            chunks=chunks
        )

        # 存储文档
        self.documents[doc_id] = document

        # 构建向量索引（简化示例）
        self._build_vector_index(doc_id, chunks)

        print(f"✓ 文档上传成功")
        print(f"  文档ID: {doc_id}")
        print(f"  文件名: {document.file_name}")
        print(f"  内容长度: {len(content)} 字符")
        print(f"  分块数量: {len(chunks)}")

        return doc_id

    def ask_question(self, question: str, doc_ids: Optional[List[str]] = None) -> Answer:
        """
        对文档进行问答
                
        RAG 流程：
        1. 检索：从文档中检索与问题相关的片段
        2. 增强：将检索结果作为上下文
        3. 生成：LLM 基于上下文生成回答
                
        Args:
            question: 用户问题
            doc_ids: 指定查询的文档ID列表（None表示所有文档）
                    
        Returns:
            Answer 对象，包含回答和引用来源
        """
        if not self.documents:
            raise ValueError("请先上传文档")

        # 确定查询范围
        target_docs = doc_ids if doc_ids else list(self.documents.keys())

        # 步骤1: 检索相关文档片段
        relevant_chunks = self._retrieve_relevant_chunks(question, target_docs)

        if not relevant_chunks:
            return Answer(
                question=question,
                answer="抱歉，未在文档中找到相关信息。",
                references=[],
                confidence=0.0
            )

        # 步骤2: 构建上下文
        context = self._build_context(relevant_chunks)

        # 步骤3: 生成回答（模拟 LLM 调用）
        answer_text = self._generate_answer(question, context)

        # 步骤4: 提取引用来源
        references = self._extract_references(relevant_chunks)

        # 创建回答对象
        answer = Answer(
            question=question,
            answer=answer_text,
            references=references,
            doc_id=relevant_chunks[0]['doc_id'],
            confidence=0.85  # 模拟置信度
        )

        # 记录对话历史
        self.chat_history.append({
            'question': question,
            'answer': answer,
            'timestamp': self._get_timestamp()
        })

        return answer

    def list_documents(self) -> List[Dict]:
        """
        列出所有已上传的文档

        Returns:
            文档信息列表
        """
        doc_list = []
        for doc_id, doc in self.documents.items():
            doc_list.append({
                'doc_id': doc_id,
                'file_name': doc.file_name,
                'content_length': len(doc.content),
                'chunk_count': len(doc.chunks),
                'metadata': doc.metadata
            })
        return doc_list

    def delete_document(self, doc_id: str) -> bool:
        """
        删除指定文档

        Args:
            doc_id: 文档ID

        Returns:
            是否删除成功
        """
        if doc_id in self.documents:
            del self.documents[doc_id]
            # 清理向量索引
            if doc_id in self.vector_store:
                del self.vector_store[doc_id]
            print(f"✓ 文档 {doc_id} 已删除")
            return True
        return False

    def get_chat_history(self, limit: int = 10) -> List[Dict]:
        """
        获取对话历史

        Args:
            limit: 返回最近的 N 条记录

        Returns:
            对话历史列表
        """
        return self.chat_history[-limit:]

    def summarize_document(self, doc_id: str) -> str:
        """
        生成文档摘要

        Args:
            doc_id: 文档ID

        Returns:
            文档摘要文本
        """
        if doc_id not in self.documents:
            raise ValueError(f"文档 {doc_id} 不存在")

        doc = self.documents[doc_id]

        # 模拟文档摘要生成
        summary = f"""
【文档摘要】
文件名: {doc.file_name}
总长度: {len(doc.content)} 字符
分块数: {len(doc.chunks)}

这是一个示例摘要。在实际应用中，ChatDoc 会：
1. 分析文档结构和关键内容
2. 提取主要观点和结论
3. 生成简洁的摘要文本
4. 保留重要数据和事实

元数据: {doc.metadata}
        """
        return summary

    # ==================== 内部方法 ====================

    def _parse_document(self, file_path: str) -> str:
        """
        解析文档内容

        实际应用中会使用相应的库：
        - PDF: PyPDF2, pdfplumber, PyMuPDF
        - Word: python-docx
        - Excel: openpyxl, pandas
        - TXT: 直接读取
        """
        file_ext = os.path.splitext(file_path)[1].lower()

        try:
            if file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif file_ext == '.pdf':
                # 实际使用: import PyPDF2
                return f"[PDF文档内容] {os.path.basename(file_path)}"
            elif file_ext in ['.docx', '.doc']:
                # 实际使用: from docx import Document
                return f"[Word文档内容] {os.path.basename(file_path)}"
            elif file_ext in ['.xlsx', '.xls']:
                # 实际使用: import pandas as pd
                return f"[Excel表格内容] {os.path.basename(file_path)}"
            else:
                return ""
        except Exception as e:
            print(f"文档解析错误: {e}")
            return ""

    def _split_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        将文档内容分块

        Args:
            text: 文档文本
            chunk_size: 每块大小（字符数）

        Returns:
            文本块列表
        """
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)

        # 如果没有内容，创建一个默认块
        if not chunks:
            chunks.append("文档内容为空")

        return chunks

    def _build_vector_index(self, doc_id: str, chunks: List[str]):
        """
        构建向量索引

        实际应用中会使用：
        - OpenAI Embeddings
        - HuggingFace Transformers
        - 向量数据库: ChromaDB, FAISS, Pinecone
        """
        # 简化示例：存储文本块的索引
        self.vector_store[doc_id] = [
            {'chunk_id': i, 'text': chunk}
            for i, chunk in enumerate(chunks)
        ]

    def _retrieve_relevant_chunks(self, question: str, doc_ids: List[str], top_k: int = 3) -> List[Dict]:
        """
        检索与问题相关的文档片段

        实际应用中会使用向量相似度搜索：
        - 计算问题向量与文档片段的余弦相似度
        - 返回最相似的 top_k 个片段
        """
        relevant_chunks = []

        for doc_id in doc_ids:
            if doc_id in self.vector_store:
                chunks = self.vector_store[doc_id]
                # 简化示例：返回前几个块
                for chunk in chunks[:top_k]:
                    relevant_chunks.append({
                        'doc_id': doc_id,
                        'chunk_id': chunk['chunk_id'],
                        'text': chunk['text'],
                        'score': 0.9  # 模拟相似度分数
                    })

        # 按相似度排序
        relevant_chunks.sort(key=lambda x: x['score'], reverse=True)

        return relevant_chunks[:top_k]

    def _build_context(self, relevant_chunks: List[Dict]) -> str:
        """
        构建上下文文本

        Args:
            relevant_chunks: 相关文档片段

        Returns:
            拼接后的上下文字本
        """
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            context_parts.append(f"[片段{i}] {chunk['text']}")

        return "\n\n".join(context_parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """
        生成回答

        实际应用中会调用 LLM API：
        - OpenAI GPT-4 / GPT-3.5
        - Anthropic Claude
        - 本地模型: Llama, Qwen 等
        """
        answer = f"""
基于文档内容，关于"{question}"的回答：

在实际的 ChatDoc 系统中，这里会：
1. 将问题和上下文发送给 LLM
2. LLM 基于提供的文档内容生成准确回答
3. 确保回答有文档依据，避免幻觉
4. 保持回答简洁、清晰、有用

当前检索到的上下文长度: {len(context)} 字符
        """
        return answer

    def _extract_references(self, relevant_chunks: List[Dict]) -> List[Dict]:
        """
        提取引用来源

        Args:
            relevant_chunks: 相关文档片段

        Returns:
            引用列表，包含文档ID、片段位置等信息
        """
        references = []
        for chunk in relevant_chunks:
            references.append({
                'doc_id': chunk['doc_id'],
                'chunk_id': chunk['chunk_id'],
                'text_preview': chunk['text'][:100] + "...",
                'relevance_score': chunk['score']
            })
        return references

    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """主函数 - 演示 ChatDoc 的完整使用流程"""

    print("=" * 70)
    print("ChatDoc - 基于 RAG 的智能文档问答系统演示")
    print("=" * 70)

    # 创建 ChatDoc 系统实例
    chatdoc = ChatDocSystem()

    # ========== 示例1: 上传文档 ==========
    print("\n【示例1】上传文档")
    print("-" * 70)
    
    doc_id = None
    try:
        # 上传测试文档（使用 TXT 格式）
        test_file = '/Users/anjuke/PycharmProjects/FutureTradingAgent/main.py'
        
        # 创建临时测试文件（TXT 格式）
        test_content = """
期货交易代理系统 v1.0

项目简介：
这是一个基于 Python 的智能化期货交易代理系统，旨在帮助期货散户
实现自动化交易策略执行和风险管理。

核心功能：
1. 数据获取与分析
   - 实时行情数据接入
   - 历史数据回测
   - 技术指标计算
   
2. 交易策略执行
   - 双均线策略（Dual MA）
   - 趋势跟踪策略
   - 均值回归策略
   
3. 风险管理
   - 仓位控制
   - 止损止盈
   - 最大回撤监控
   
4. 回测验证
   - 历史数据回测
   - 策略性能评估
   - 参数优化

技术栈：
- Python 3.13
- TBQuant3 框架
- Pandas 数据分析
- NumPy 数值计算

使用说明：
运行 main.py 启动交易系统
配置文件位于 config.yaml
日志文件保存在 logs/ 目录
        """
        test_file = '/tmp/test_trading_doc.txt'
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        doc_id = chatdoc.upload_document(test_file, metadata={
            'type': 'documentation',
            'version': '1.0'
        })
        
    except Exception as e:
        print(f"⚠ 文档上传失败: {e}")
        print("  将使用模拟数据继续演示...")
        # 创建模拟文档
        doc_id = "doc_simulated"
        from dataclasses import replace
        simulated_doc = Document(
            doc_id=doc_id,
            file_name="simulated_doc.txt",
            file_path="/tmp/simulated.txt",
            content="这是模拟的文档内容，用于演示。",
            metadata={'type': 'simulated'},
            chunks=["这是模拟的文档内容，用于演示。"]
        )
        chatdoc.documents[doc_id] = simulated_doc

    # ========== 示例2: 查看文档列表 ==========
    print("\n【示例2】查看已上传的文档")
    print("-" * 70)
        
    documents = chatdoc.list_documents()
    if documents:
        for doc in documents:
            print(f"文档ID: {doc['doc_id']}")
            print(f"  文件名: {doc['file_name']}")
            print(f"  内容长度: {doc['content_length']} 字符")
            print(f"  分块数: {doc['chunk_count']}")
            print(f"  元数据: {doc['metadata']}")
            print()
    else:
        print("  (暂无文档)")
        print()

    # ========== 示例3: 文档摘要 ==========
    print("\n【示例3】生成文档摘要")
    print("-" * 70)
        
    if doc_id and doc_id in chatdoc.documents:
        summary = chatdoc.summarize_document(doc_id)
        print(summary)
    else:
        print("  (跳过：没有可用的文档)")

    # ========== 示例4: 智能问答 ==========
    print("\n【示例4】对文档进行问答")
    print("-" * 70)
        
    questions = [
        "这个交易系统有哪些核心功能？",
        "使用了哪些技术栈？",
        "如何进行风险管理？"
    ]
        
    if doc_id and doc_id in chatdoc.documents:
        for question in questions:
            print(f"\n问: {question}")
            print("-" * 70)
                
            answer = chatdoc.ask_question(question, doc_ids=[doc_id])
                
            print(f"答: {answer.answer}")
            print(f"\n置信度: {answer.confidence:.2%}")
            print(f"引用来源 ({len(answer.references)} 处):")
            for ref in answer.references:
                print(f"  - 文档: {ref['doc_id']}, 片段: {ref['chunk_id']}")
                print(f"    预览: {ref['text_preview']}")
                print(f"    相关性: {ref['relevance_score']:.2%}")
            print()
    else:
        print("  (跳过：没有可用的文档)")

    # ========== 示例5: 查看对话历史 ==========
    print("\n【示例5】对话历史")
    print("-" * 70)

    history = chatdoc.get_chat_history(limit=5)
    for i, record in enumerate(history, 1):
        print(f"[{i}] {record['timestamp']}")
        print(f"    问: {record['question'][:50]}...")
        print(f"    答: {record['answer'].answer[:50]}...")
        print()

    # ========== 示例6: 多文档管理 ==========
    print("\n【示例6】多文档管理演示")
    print("-" * 70)

    print("当前文档数量:", len(chatdoc.documents))
    print("可以执行的操作:")
    print("  - 继续上传更多文档")
    print("  - 跨文档问答: chatdoc.ask_question(question)")
    print("  - 删除文档: chatdoc.delete_document(doc_id)")
    print("  - 指定文档查询: chatdoc.ask_question(q, doc_ids=['doc_001'])")

    # ========== 总结 ==========
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)

    print("\nChatDoc 的核心优势：")
    print("✓ 支持多种文档格式（PDF、Word、Excel、TXT）")
    print("✓ 基于 RAG 技术，回答有据可依")
    print("✓ 引用原文片段，可追溯验证")
    print("✓ 支持多文档管理和跨文档检索")
    print("✓ 智能分块和向量检索，提高准确性")
    print("✓ 保留对话历史，便于回顾")

    print("\n实际应用场景：")
    print("  • 学术研究：快速阅读论文，提取关键信息")
    print("  • 法律文档：合同审查，条款查询")
    print("  • 技术文档：API 文档查询，代码说明")
    print("  • 财务报告：数据分析，趋势总结")
    print("  • 知识库：企业内部文档问答系统")


if __name__ == '__main__':
    main()
