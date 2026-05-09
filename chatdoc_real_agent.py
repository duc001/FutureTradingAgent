"""
ChatDoc 智能体问答系统 - 基于 RAG 的文档智能问答

功能特性：
1. 文档解析：支持 TXT、PDF、Word 等格式
2. 智能分块：基于语义边界切割，保留上下文连贯性
3. 向量化检索：使用 sentence-transformers 编码，结合 ChromaDB 高效检索
4. 智能问答：支持 DeepSeek、OpenAI、Ollama 等多种 LLM
5. 引用溯源：精准标注每个答案的文档来源

依赖安装：
    pip install sentence-transformers chromadb openai numpy

使用示例：
    from chatdoc_real_agent import ChatDocAgent, ChatDocConfig
    
    config = ChatDocConfig(llm_type="deepseek")
    chatdoc = ChatDocAgent(config)
    chatdoc.load_document("文档.txt")
    answer = chatdoc.ask_question("文档主要内容是什么？")
    print(answer.answer)
"""

import os
import sys
import gc
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

# ==================== 全局配置 ====================

# DeepSeek API 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-205ee372c5df491f8050324eb697e504")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL_NAME = "deepseek-v4-flash"  # 快速响应模式
TEMPERATURE = 0.7
REQUEST_TIMEOUT = 120  # 请求超时（秒）
MAX_RETRIES = 3  # 最大重试次数

# ==================== 依赖检测 ====================
# 静默检测各依赖库是否安装，避免启动时报错

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("[ChatDoc] 警告: sentence-transformers 未安装，向量化功能不可用")

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    print("[ChatDoc] 提示: chromadb 未安装，将使用 NumPy 替代")

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("[ChatDoc] 警告: openai 未安装，LLM 调用不可用")


@dataclass
class ChatDocConfig:
    """ChatDoc 配置参数"""
    # 向量化模型配置
    embedding_model: str = "all-MiniLM-L6-v2"  # 轻量高效的文本编码模型
    use_chroma: bool = True  # 是否使用 ChromaDB 向量数据库
    
    # LLM 配置
    llm_type: str = "deepseek"  # LLM 类型: 'mock' | 'deepseek' | 'openai' | 'ollama'
    temperature: float = 0.3  # 生成温度，越低越准确（建议 0.1-0.5）
    max_tokens: int = 1000  # 最大生成 token 数
    request_timeout: int = 120  # LLM 请求超时（秒）
    max_retries: int = 3  # LLM 调用失败重试次数
    
    # 检索配置
    top_k: int = 5  # 检索返回的最相关片段数量
    similarity_threshold: float = 0.3  # 相似度阈值，低于此值的结果会被过滤


@dataclass
class DocumentChunk:
    """文档分块：文档被切割后的最小检索单元"""
    chunk_id: str  # 分块唯一标识，格式: doc_001_chunk_001
    doc_id: str  # 所属文档 ID
    text: str  # 分块文本内容
    embedding: Optional[np.ndarray] = None  # 向量化表示
    metadata: Dict = field(default_factory=dict)  # 元数据（如文件名、页码等）


@dataclass
class SearchResult:
    """检索结果：包含分块及其与查询的相似度"""
    chunk: DocumentChunk  # 匹配到的文档分块
    similarity_score: float  # 相似度得分 (0-1)
    rank: int  # 排名（1为最相关）


@dataclass
class Answer:
    """问答结果：包含回答内容及相关引用"""
    question: str  # 用户问题
    answer: str  # 生成的回答
    references: List[SearchResult] = field(default_factory=list)  # 引用的文档片段
    confidence: float = 0.0  # 回答置信度 (0-1)
    model_used: str = ""  # 使用的 LLM 模型


class ChatDocAgent:
    """
    ChatDoc 智能体 - 基于 RAG 的文档问答系统
    
    完整工作流程：
    ┌─────────────┐    ┌───────────┐    ┌────────────┐    ┌─────────┐
    │  加载文档   │ -> │  文本分块  │ -> │   向量化   │ -> │ 构建索引 │
    └─────────────┘    └───────────┘    └────────────┘    └─────────┘
                                                                │
                                                                v
    ┌─────────────┐    ┌───────────┐    ┌────────────┐    ┌─────────┐
    │  返回结果   │ <- │ 生成回答  │ <- │  语义检索  │ <- │ 接收问题 │
    └─────────────┘    └───────────┘    └────────────┘    └─────────┘
    """
    
    def __init__(self, config: Optional[ChatDocConfig] = None):
        """
        初始化 ChatDoc 智能体
        
        Args:
            config: ChatDoc 配置对象，None 时使用默认配置
        """
        # 应用配置（默认或自定义）
        self.config = config if config else ChatDocConfig()
        
        # 文档存储
        self.documents: Dict[str, List[DocumentChunk]] = {}  # doc_id -> 分块列表
        self.all_chunks: List[DocumentChunk] = []  # 所有文档的分块
        self.chat_history: List[Dict] = []  # 对话历史
        
        # ChromaDB 配置
        self.use_chroma = self.config.use_chroma and HAS_CHROMA
        self.chroma_client = None
        self.chroma_collection = None
        
        # 初始化向量化模型
        self._init_embedding_model()
        
        # 初始化 LLM
        self.llm_type = self.config.llm_type
        self._init_llm()
        
        # 初始化 ChromaDB
        if self.use_chroma:
            self._init_chromadb()
    
    def _init_embedding_model(self):
        """初始化文本向量化模型"""
        self.embedding_model = None
        self.embedding_dim = 384  # 默认维度
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                print(f"[ChatDoc] 加载向量化模型: {self.config.embedding_model}")
                self.embedding_model = SentenceTransformer(
                    self.config.embedding_model,
                    device='cpu'
                )
                # 获取向量维度
                self.embedding_dim = getattr(
                    self.embedding_model, 
                    'get_embedding_dimension', 
                    lambda: self.embedding_model.get_sentence_embedding_dimension
                )()
                print(f"[ChatDoc] 向量化模型加载成功 (维度: {self.embedding_dim})")
            except Exception as e:
                print(f"[ChatDoc] 向量化模型加载失败: {e}")
                print("[ChatDoc] 将使用随机向量替代，检索结果可能不准确")
    
    def _init_llm(self):
        """
        初始化 LLM 客户端
        
        根据配置初始化对应 LLM 的 API 客户端，
        若初始化失败则降级到 mock 模式（基于关键词的简单匹配）
        """
        if self.llm_type == 'deepseek':
            if HAS_OPENAI and DEEPSEEK_API_KEY:
                try:
                    self.openai_client = OpenAI(
                        api_key=DEEPSEEK_API_KEY,
                        base_url=DEEPSEEK_BASE_URL,
                        timeout=REQUEST_TIMEOUT,
                        max_retries=MAX_RETRIES
                    )
                    print(f"[ChatDoc] DeepSeek LLM 初始化成功 (模型: {DEEPSEEK_MODEL_NAME})")
                except Exception as e:
                    print(f"[ChatDoc] DeepSeek 初始化失败: {e}")
                    print("[ChatDoc] 降级到 Mock 模式")
                    self.llm_type = 'mock'
            else:
                print("[ChatDoc] DeepSeek API Key 未配置，降级到 Mock 模式")
                self.llm_type = 'mock'
                
        elif self.llm_type == 'openai':
            if HAS_OPENAI:
                openai_key = os.getenv("OPENAI_API_KEY", "")
                if openai_key:
                    self.openai_client = OpenAI(api_key=openai_key)
                    print("[ChatDoc] OpenAI LLM 初始化成功")
                else:
                    print("[ChatDoc] OpenAI API Key 未配置，降级到 Mock 模式")
                    self.llm_type = 'mock'
            else:
                print("[ChatDoc] OpenAI SDK 未安装，降级到 Mock 模式")
                self.llm_type = 'mock'
                
        elif self.llm_type == 'ollama':
            self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            print(f"[ChatDoc] Ollama LLM 配置完成 (地址: {self.ollama_base_url})")
            
        else:
            print("[ChatDoc] 使用 Mock 模式（基于关键词匹配，结果可能不准确）")
    
    def _init_chromadb(self):
        """初始化 ChromaDB 向量数据库"""
        try:
            print("[ChatDoc] 初始化 ChromaDB 向量数据库...")
            self.chroma_client = chromadb.Client(Settings(
                anonymized_telemetry=False,
                allow_reset=True
            ))
            
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="chatdoc_collection",
                metadata={"description": "ChatDoc document collection"}
            )
            print("[ChatDoc] ChromaDB 初始化成功")
        except Exception as e:
            print(f"[ChatDoc] ChromaDB 初始化失败: {type(e).__name__}")
            print("[ChatDoc] 将使用 NumPy 进行向量检索")
            self.use_chroma = False
        
    def load_document(self, file_path: str, metadata: Optional[Dict] = None) -> str:
        """
        加载并处理文档
        
        执行流程：
        1. 解析文档内容
        2. 智能分块
        3. 向量化编码
        4. 存入向量数据库
        
        Args:
            file_path: 文档路径（支持 .txt, .pdf, .docx）
            metadata: 文档元数据（如 {"source": "工作日志"}）
            
        Returns:
            文档 ID（格式：doc_001）
            
        Raises:
            FileNotFoundError: 文件不存在
            Exception: 向量化失败
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        print(f"\n[文档加载] {os.path.basename(file_path)}")
        
        # 1. 解析文档内容
        content = self._parse_document(file_path)
        if not content:
            raise ValueError("文档内容为空或解析失败")
        print(f"[文档加载] 解析完成，字符数: {len(content):,}")
        
        # 2. 生成文档 ID
        doc_id = f"doc_{len(self.documents) + 1:03d}"
        
        # 3. 智能分块
        chunks = self._split_text(content, chunk_size=800, overlap=100)
        print(f"[文档加载] 分块完成，块数: {len(chunks)}")
        
        # 4. 创建分块对象
        doc_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk = DocumentChunk(
                chunk_id=f"{doc_id}_chunk_{i:03d}",
                doc_id=doc_id,
                text=chunk_text,
                metadata={**(metadata or {}), 'file_name': os.path.basename(file_path)}
            )
            doc_chunks.append(chunk)
        
        # 5. 向量化编码
        if self.embedding_model:
            try:
                batch_size = 16
                all_embeddings = []
                total_batches = (len(doc_chunks) + batch_size - 1) // batch_size
                
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(doc_chunks))
                    batch = doc_chunks[start_idx:end_idx]
                    batch_texts = [c.text for c in batch]
                    
                    batch_embeddings = self.embedding_model.encode(
                        batch_texts,
                        show_progress_bar=False,
                        batch_size=batch_size,
                        normalize_embeddings=True
                    )
                    all_embeddings.extend(batch_embeddings)
                
                # 分配向量到各分块
                for chunk, emb in zip(doc_chunks, all_embeddings):
                    chunk.embedding = emb
                
                # 释放临时内存
                del all_embeddings
                gc.collect()
                print(f"[文档加载] 向量化完成 (模型: {self.config.embedding_model})")
                
            except Exception as e:
                print(f"[文档加载] 向量化失败: {e}")
                raise
        
        # 6. 存储到内存
        self.documents[doc_id] = doc_chunks
        self.all_chunks.extend(doc_chunks)
        
        # 7. 存储到 ChromaDB
        if self.use_chroma and self.embedding_model:
            try:
                batch_size = 50
                for i in range(0, len(doc_chunks), batch_size):
                    batch = doc_chunks[i:i + batch_size]
                    
                    self.chroma_collection.add(
                        ids=[c.chunk_id for c in batch],
                        documents=[c.text for c in batch],
                        embeddings=[c.embedding.tolist() for c in batch],
                        metadatas=[
                            {'doc_id': c.doc_id, 'chunk_id': c.chunk_id, 
                             'file_name': os.path.basename(file_path)}
                            for c in batch
                        ]
                    )
                print(f"[文档加载] ChromaDB 索引完成")
                gc.collect()
            except Exception as e:
                print(f"[文档加载] ChromaDB 存储失败: {type(e).__name__}")
        
        print(f"[文档加载] 加载成功 (doc_id: {doc_id})")
        return doc_id
    
    def ask_question(self, question: str, doc_ids: Optional[List[str]] = None, 
                    top_k: Optional[int] = None, similarity_threshold: Optional[float] = None) -> Answer:
        """
        对文档进行智能问答
        
        执行流程：
        1. 查询扩展（增加同义词检索）
        2. 向量化检索
        3. 结果去重排序
        4. LLM 生成回答
        
        Args:
            question: 用户问题
            doc_ids: 指定文档ID列表（None表示搜索所有文档）
            top_k: 返回最相关的 K 个片段（None使用配置值）
            similarity_threshold: 相似度阈值（None使用配置值）
            
        Returns:
            Answer 对象：包含回答、引用来源、置信度
        """
        if not self.all_chunks:
            raise ValueError("请先加载文档")
        
        # 应用配置参数
        top_k = top_k if top_k is not None else self.config.top_k
        similarity_threshold = similarity_threshold if similarity_threshold is not None else self.config.similarity_threshold
        
        print(f"\n[智能问答] 问题: {question}")
        print(f"[智能问答] 检索配置: top_k={top_k}, 阈值={similarity_threshold}")
        
        # 1. 查询扩展
        expanded_queries = self._expand_query(question)
        if len(expanded_queries) > 1:
            print(f"[智能问答] 扩展查询: {expanded_queries}")
        
        # 2. 多查询检索并合并
        all_results = []
        seen_chunk_ids = set()
        
        for query in expanded_queries:
            query_embedding = self._embed_text(query)
            results = self._search_similar(query_embedding, top_k=top_k * 2, doc_ids=doc_ids)
            
            # 去重
            for r in results:
                if r.chunk.chunk_id not in seen_chunk_ids:
                    all_results.append(r)
                    seen_chunk_ids.add(r.chunk.chunk_id)
        
        # 3. 排序和过滤
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # 过滤低相似度结果
        filtered_results = [
            r for r in all_results 
            if r.similarity_score >= similarity_threshold
        ]
        
        search_results = filtered_results[:top_k] if filtered_results else all_results[:top_k]
        print(f"[智能问答] 检索到 {len(search_results)} 个相关片段")
        
        if not search_results:
            return Answer(
                question=question,
                answer="抱歉，未在文档中找到相关信息。请尝试换一种问法。",
                references=[],
                confidence=0.0,
                model_used=self.llm_type
            )
        
        # 4. 构建上下文并生成回答
        context = self._build_context(search_results)
        print(f"[智能问答] 使用模型: {self.llm_type}")
        answer_text = self._generate_answer(question, context)
        
        # 5. 计算置信度
        base_confidence = search_results[0].similarity_score if search_results else 0.0
        coverage_bonus = min(len(search_results) / 5, 0.2)
        confidence = min(base_confidence + coverage_bonus, 1.0)
        
        # 6. 创建结果
        answer = Answer(
            question=question,
            answer=answer_text,
            references=search_results,
            confidence=confidence,
            model_used=self.llm_type
        )
        
        # 7. 记录历史
        self.chat_history.append({
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        print(f"[智能问答] 完成 (置信度: {confidence:.0%})")
        return answer
    
    def _expand_query(self, question: str) -> List[str]:
        """
        查询扩展：增加同义词和关键词检索，提高召回率
        
        策略：
        - 专业术语同义词替换（如：期货 -> Futures）
        - 疑问句关键词提取（如：有哪些 -> 提取核心名词）
        
        Args:
            question: 原始问题
            
        Returns:
            扩展后的查询列表（最多3个）
        """
        queries = [question]
        
        if len(question) < 5:
            return queries
        
        # 专业术语同义词映射表
        term_mappings = {
            '中收': ['中收', '中间收入', '手续费'],
            '签约': ['签约', '开通', '绑定'],
            '期货': ['期货', 'Futures'],
            '策略': ['策略', 'strategy', '方案'],
            '回测': ['回测', 'backtest', '测试'],
            '止损': ['止损', 'stop loss'],
            '风控': ['风控', '风险控制'],
        }
        
        # 检测并扩展专业术语
        for term, synonyms in term_mappings.items():
            if term in question:
                for syn in synonyms:
                    if syn not in question:
                        new_query = question.replace(term, syn)
                        if new_query not in queries:
                            queries.append(new_query)
        
        # 疑问句关键词提取
        if any(kw in question for kw in ['哪些', '什么', '有没有', '列出']):
            keywords = [w for w in question if len(w) >= 2]
            if keywords:
                queries.append(' '.join(keywords[:3]))
        
        return queries[:3]
    
    def summarize_document(self, doc_id: str, max_length: int = 500) -> str:
        """
        生成文档摘要
        
        使用 LLM 总结文档的核心内容，摘要长度可自定义。
        
        Args:
            doc_id: 文档ID
            max_length: 摘要最大字符数
            
        Returns:
            生成的摘要文本
        """
        if doc_id not in self.documents:
            raise ValueError(f"文档 {doc_id} 不存在")
        
        chunks = self.documents[doc_id]
        
        # 提取关键片段（首尾各3个分块，保留核心信息）
        key_chunks = chunks[:3] + chunks[-3:] if len(chunks) > 6 else chunks
        context = "\n\n".join([c.text for c in key_chunks])
        
        prompt = f"请为以下文档生成一个简洁的摘要（{max_length}字以内）：\n\n{context}\n\n摘要："
        summary = self._generate_answer("生成文档摘要", prompt)
        
        return summary
    
    def translate_document(self, doc_id: str, target_lang: str = "中文") -> str:
        """
        翻译文档内容
        
        将文档内容翻译为指定语言（默认中文）。
        注意：仅翻译前5000字符以避免超长。
        
        Args:
            doc_id: 文档ID
            target_lang: 目标语言（如 "英文"、"日文"）
            
        Returns:
            翻译后的文本
        """
        if doc_id not in self.documents:
            raise ValueError(f"文档 {doc_id} 不存在")
        
        chunks = self.documents[doc_id]
        sample_text = "\n".join([c.text for c in chunks[:10]])[:5000]
        
        prompt = f"请将以下内容翻译成{target_lang}：\n\n{sample_text}\n\n翻译："
        translation = self._generate_answer("翻译文档", prompt)
        
        return translation
    
    def list_documents(self) -> List[Dict]:
        """
        列出已加载的所有文档
        
        Returns:
            文档信息列表，每项包含 doc_id、chunk_count、total_chars
        """
        return [
            {
                'doc_id': doc_id,
                'chunk_count': len(chunks),
                'total_chars': sum(len(c.text) for c in chunks)
            }
            for doc_id, chunks in self.documents.items()
        ]
    
    def get_chat_history(self) -> List[Dict]:
        """获取对话历史记录"""
        return self.chat_history
    
    def clear_history(self):
        """清空对话历史"""
        self.chat_history.clear()
        print("[ChatDoc] 对话历史已清空")
    
    # ==================== 内部方法 ====================
    
    def _parse_document(self, file_path: str) -> str:
        """
        解析文档内容
        
        支持格式：
        - .txt: 纯文本
        - .pdf: PDF 文档（需安装 PyPDF2）
        - .docx/.doc: Word 文档（需安装 python-docx）
        
        Args:
            file_path: 文档路径
            
        Returns:
            解析后的文本内容，失败返回空字符串
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
                    
            elif ext == '.pdf':
                try:
                    import PyPDF2
                    reader = PyPDF2.PdfReader(file_path)
                    return "".join(page.extract_text() for page in reader.pages)
                except ImportError:
                    print("[文档解析] PDF 需要安装 PyPDF2: pip install PyPDF2")
                    return ""
                    
            elif ext in ['.docx', '.doc']:
                try:
                    from docx import Document
                    return "\n".join(para.text for para in Document(file_path).paragraphs)
                except ImportError:
                    print("[文档解析] Word 需要安装 python-docx: pip install python-docx")
                    return ""
                    
            else:
                print(f"[文档解析] 不支持的文件格式: {ext}")
                return ""
                
        except Exception as e:
            print(f"[文档解析] 解析失败: {e}")
            return ""
    
    def _split_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """
        智能文本分块
        
        在句子边界处切割，保留语义完整性。
        使用滑动窗口 + 重叠机制确保上下文连贯。
        
        Args:
            text: 待分块文本
            chunk_size: 每块目标字符数
            overlap: 相邻块的重叠字符数
            
        Returns:
            分块后的文本列表
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            
            # 在句子边界处切割
            if end < text_len:
                best_split = end
                search_start = max(start, end - chunk_size // 2)
                
                # 优先在段落/句子边界切割
                for sep in ['。\n', '\n\n', '。', '！', '？', '\n']:
                    pos = text.rfind(sep, search_start, end)
                    if pos != -1 and pos > best_split - chunk_size // 2:
                        best_split = pos + len(sep)
                        break
                
                end = best_split
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # 滑动窗口
            start = end - overlap
            if start <= 0 or start >= end:
                start = end
            
            # 安全限制
            if len(chunks) > text_len:
                break
        
        return chunks if chunks else [text]
    
    def _embed_text(self, text: str) -> np.ndarray:
        """
        文本向量化编码
        
        Args:
            text: 待编码文本
            
        Returns:
            文本向量（numpy 数组）
        """
        if self.embedding_model:
            return self.embedding_model.encode(text)
        else:
            # 无向量化模型时返回随机向量
            return np.random.rand(self.embedding_dim).astype('float32')
    
    def _search_similar(self, query_embedding: np.ndarray, top_k: int = 5, 
                       doc_ids: Optional[List[str]] = None) -> List[SearchResult]:
        """
        语义相似度检索
        
        优先使用 ChromaDB 加速检索，失败时降级为 NumPy 计算。
        
        Args:
            query_embedding: 查询向量
            top_k: 返回数量上限
            doc_ids: 限定文档 ID 列表
            
        Returns:
            检索结果列表，按相似度降序排列
        """
        if not self.all_chunks:
            return []
        
        # 策略1: ChromaDB 检索
        if self.use_chroma and self.chroma_collection:
            try:
                where_filter = {"doc_id": {"$in": doc_ids}} if doc_ids else None
                
                results = self.chroma_collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k * 2,
                    where=where_filter
                )
                
                search_results = []
                if results['ids'] and results['ids'][0]:
                    for i, (chunk_id, distance, metadata, doc_text) in enumerate(zip(
                        results['ids'][0],
                        results['distances'][0],
                        results['metadatas'][0],
                        results['documents'][0]
                    )):
                        chunk = next((c for c in self.all_chunks if c.chunk_id == chunk_id), None)
                        if chunk:
                            # 距离转相似度 (L2 距离)
                            similarity = 1.0 / (1.0 + distance)
                            search_results.append(SearchResult(
                                chunk=chunk,
                                similarity_score=similarity,
                                rank=i + 1
                            ))
                
                return search_results[:top_k]
            except Exception as e:
                print(f"[检索] ChromaDB 失败，降级到 NumPy: {e}")
        
        # 策略2: NumPy 余弦相似度计算
        filtered_chunks = [
            c for c in self.all_chunks 
            if c.doc_id in (doc_ids or [c.doc_id]) and c.embedding is not None
        ]
        
        if not filtered_chunks:
            return []
        
        query_vec = np.array(query_embedding).reshape(1, -1)
        similarities = []
        
        for chunk in filtered_chunks:
            chunk_vec = np.array(chunk.embedding).reshape(1, -1)
            
            dot_product = np.dot(query_vec, chunk_vec.T)[0][0]
            norm_query = np.linalg.norm(query_vec)
            norm_chunk = np.linalg.norm(chunk_vec)
            
            if norm_query > 0 and norm_chunk > 0:
                similarity = dot_product / (norm_query * norm_chunk)
            else:
                similarity = 0.0
            
            similarities.append((chunk, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [
            SearchResult(chunk=chunk, similarity_score=float(sim), rank=i+1)
            for i, (chunk, sim) in enumerate(similarities[:top_k])
        ]
    
    def _build_context(self, search_results: List[SearchResult]) -> str:
        """
        构建检索上下文
        
        将多个检索片段整合为 LLM 可读的上下文，
        附带来源标注便于答案溯源。
        
        Args:
            search_results: 检索结果列表
            
        Returns:
            格式化的上下文字符串
        """
        if not search_results:
            return ""
        
        context_parts = []
        sorted_results = sorted(search_results, key=lambda x: x.similarity_score, reverse=True)
        
        for idx, result in enumerate(sorted_results, 1):
            doc_info = result.chunk.metadata.get('file_name', '未知')
            chunk_text = result.chunk.text.strip()
            
            # 来源标注格式：【来源1】相关度:85% | 文件:xxx.txt
            context_parts.append(
                f"【来源{idx}】相关度:{result.similarity_score:.0%} | {doc_info}\n"
                f"{chunk_text}"
            )
        
        return "\n---\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """
        根据 LLM 类型分发到对应生成方法
        
        Args:
            question: 用户问题
            context: 检索到的上下文
            
        Returns:
            生成的回答文本
        """
        if self.llm_type == 'openai' and HAS_OPENAI:
            return self._generate_with_openai(question, context)
        elif self.llm_type == 'deepseek' and HAS_OPENAI:
            return self._generate_with_deepseek(question, context)
        elif self.llm_type == 'ollama':
            return self._generate_with_ollama(question, context)
        else:
            return self._generate_mock_answer(question, context)
    
    def _generate_with_deepseek(self, question: str, context: str) -> str:
        """
        使用 DeepSeek 生成回答
        
        包含对话历史上下文，支持多轮对话。
        """
        # 构建对话历史
        history_messages = []
        for hist in self.chat_history[-3:]:
            history_messages.append({"role": "user", "content": hist['question']})
            history_messages.append({
                "role": "assistant", 
                "content": hist['answer'].answer if hasattr(hist['answer'], 'answer') else str(hist['answer'])
            })
        
        system_prompt = """你是一个专业的文档问答助手，专门根据提供的文档内容回答用户问题。

【核心原则】
1. **严格基于文档回答**：只使用文档中明确包含的信息，不要推理或编造
2. **原文引用**：尽量使用文档中的原话，避免过度概括
3. **如实告知**：如果文档中没有相关信息，直接说"未找到"而不是猜测

【回答格式】
1. 首先**直接回答问题**（一句话概括）
2. 然后**详细说明**（从文档中提取的具体信息）
3. 最后**标注来源**（如：来自【来源1】）

【注意事项】
- 问题中的专业术语（期货、中收、策略等）保持原样
- 如果多段文档都相关，都应该引用
- 回答要准确、完整、有条理"""

        user_prompt = f"""【文档内容】
{context}

【当前问题】
{question}

请仔细阅读文档内容，然后回答问题。"""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history_messages)
        messages.append({"role": "user", "content": user_prompt})
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self.openai_client.chat.completions.create(
                    model=DEEPSEEK_MODEL_NAME,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=self.config.max_tokens,
                    extra_body={"thinking": {"type": "disabled"}}
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"[LLM] DeepSeek 失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}")
                    time.sleep(2 ** attempt)
                else:
                    return f"[错误] DeepSeek 调用失败: {e}"
    
    def _generate_with_openai(self, question: str, context: str) -> str:
        """使用 OpenAI GPT 生成回答"""
        system_prompt = """你是一个专业的文档问答助手，擅长从提供的上下文中提取准确信息并回答问题。

请严格遵守以下规则：
1. **仅基于提供的上下文**回答问题，不要编造或使用外部知识
2. 如果上下文中没有相关信息，明确说明"根据提供的文档，未找到相关信息"
3. 回答要**条理清晰、结构完整**，使用分点列举
4. **引用来源**：在回答末尾注明参考了哪些片段
5. 保持回答**简洁但完整**

回答格式：
- 先直接回答问题
- 然后列出详细信息
- 最后注明参考来源"""
        
        user_prompt = f"请基于以下文档内容回答问题：\n\n{context}\n\n问题：{question}\n\n请给出准确、完整的回答："
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"[LLM] OpenAI 失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}")
                    time.sleep(2 ** attempt)
                else:
                    return f"[错误] OpenAI 调用失败: {e}"
    
    def _generate_with_ollama(self, question: str, context: str) -> str:
        """使用 Ollama 本地模型生成回答"""
        try:
            import requests
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": "llama2",
                    "prompt": f"上下文：\n{context}\n\n问题：{question}\n\n回答：",
                    "stream": False
                }
            )
            return response.json().get('response', '[错误] Ollama 无响应')
        except Exception as e:
            return f"[错误] Ollama 调用失败: {e}"
    
    def _generate_mock_answer(self, question: str, context: str) -> str:
        """
        Mock 模式回答（基于关键词匹配）
        
        当没有可用 LLM 时使用此模式。
        通过关键词匹配返回相关内容，简单但不够智能。
        """
        keywords = [kw for kw in ['中收', '会员', '资金', '签约', '退款', '回调', '需求', 
                                   '期货', '交易', '策略', '唐奇安', 'ADX',
                                   '小米', '天下信用', '及未', '蓝海', '梅州'] 
                   if kw in question or kw in context]
        
        if not keywords:
            return f"根据文档内容，关于\"{question}\"的相关信息：\n\n{context[:300]}...\n\n（以上是从文档中提取的相关片段）"
        
        relevant_sentences = [
            s.strip() for s in context.split('\n')
            if any(kw in s for kw in keywords) and len(s.strip()) > 10
        ]
        
        if relevant_sentences:
            answer = f"根据文档内容，关于\"{question}\"找到以下相关信息：\n\n"
            for i, sent in enumerate(relevant_sentences[:5], 1):
                answer += f"{i}. {sent}\n"
            answer += f"\n（共找到 {len(relevant_sentences)} 处相关内容）"
            return answer
        else:
            return f"根据文档内容，关于\"{question}\"的相关信息：\n\n{context[:300]}...\n\n（以上是从文档中提取的相关片段）"


def main():
    """演示 ChatDoc 智能体的完整使用流程"""
    
    print("\n" + "="*60)
    print("  ChatDoc 智能体问答系统演示")
    print("="*60 + "\n")
    
    # 1. 依赖检查
    if not HAS_SENTENCE_TRANSFORMERS:
        print("[错误] sentence-transformers 未安装")
        print("  安装命令: pip install sentence-transformers chromadb openai")
        return
    
    # 2. 创建智能体
    try:
        config = ChatDocConfig(
            embedding_model="all-MiniLM-L6-v2",
            llm_type="deepseek" if DEEPSEEK_API_KEY else "mock",
            use_chroma=True,
            temperature=0.3,
            max_tokens=1000,
            request_timeout=REQUEST_TIMEOUT,
            max_retries=MAX_RETRIES,
            top_k=5,
            similarity_threshold=0.3
        )
        chatdoc = ChatDocAgent(config=config)
    except Exception as e:
        print(f"[错误] 初始化失败: {e}")
        return
    
    # 3. 加载文档
    file_path = "/Users/anjuke/duchuan/工作区/2026/2602.txt"
    
    if not os.path.exists(file_path):
        print(f"[错误] 文件不存在: {file_path}")
        return
    
    try:
        doc_id = chatdoc.load_document(file_path)
    except Exception as e:
        print(f"[错误] 文档加载失败: {e}")
        return
    
    # 4. 显示文档信息
    for doc in chatdoc.list_documents():
        print(f"\n[文档] {doc['doc_id']} | 块数: {doc['chunk_count']} | 字符: {doc['total_chars']:,}")
    
    # 5. 问答演示
    questions = [
        "2月份有哪些中收需求？",
        "提到了哪些技术问题？",
        "有什么关于期货交易的策略？"
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"问: {question}")
        print("="*60)
        
        try:
            answer = chatdoc.ask_question(question, top_k=5, similarity_threshold=0.3)
            print(f"\n答:\n{answer.answer}\n")
            print(f"置信度: {answer.confidence:.0%}")
            
            if answer.references:
                print(f"\n引用来源 ({len(answer.references)} 处):")
                for ref in answer.references[:3]:
                    preview = ref.chunk.text[:60].replace('\n', ' ')
                    print(f"  [{ref.rank}] {ref.similarity_score:.0%} | {preview}...")
        except Exception as e:
            print(f"[错误] 问答失败: {e}")
    
    # 6. 文档摘要
    try:
        print(f"\n{'='*60}")
        print("文档摘要")
        print("="*60)
        summary = chatdoc.summarize_document(doc_id, max_length=300)
        print(summary)
    except Exception as e:
        print(f"[错误] 摘要生成失败: {e}")
    
    print("\n" + "="*60)
    print("  演示完成！")
    print("="*60 + "\n")


def create_chatdoc_with_deepseek() -> ChatDocAgent:
    """
    创建使用 DeepSeek 的 ChatDoc 智能体（便捷函数）
    
    Returns:
        配置好的 ChatDocAgent 实例
        
    使用示例:
        >>> chatdoc = create_chatdoc_with_deepseek()
        >>> doc_id = chatdoc.load_document("文档.txt")
        >>> answer = chatdoc.ask_question("问题")
        >>> print(answer.answer)
    """
    if not DEEPSEEK_API_KEY:
        raise ValueError(
            "未设置 DEEPSEEK_API_KEY 环境变量\n"
            "请运行: export DEEPSEEK_API_KEY='your-api-key'"
        )
    
    config = ChatDocConfig(
        embedding_model="all-MiniLM-L6-v2",
        llm_type="deepseek",
        use_chroma=True,
        temperature=0.3,
        max_tokens=1000,
        request_timeout=REQUEST_TIMEOUT,
        max_retries=MAX_RETRIES,
        top_k=5,
        similarity_threshold=0.3
    )
    
    return ChatDocAgent(config=config)


def create_chatdoc_mock() -> ChatDocAgent:
    """
    创建 Mock 模式的 ChatDoc 智能体（无需 API Key）
    
    Returns:
        配置好的 ChatDocAgent 实例
        
    使用示例:
        >>> chatdoc = create_chatdoc_mock()
        >>> doc_id = chatdoc.load_document("文档.txt")
        >>> answer = chatdoc.ask_question("问题")
    """
    config = ChatDocConfig(
        embedding_model="all-MiniLM-L6-v2",
        llm_type="mock",
        use_chroma=True,
        temperature=0.3,
        max_tokens=1000,
        top_k=5,
        similarity_threshold=0.3
    )
    
    return ChatDocAgent(config=config)


if __name__ == '__main__':
    main()
