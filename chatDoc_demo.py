
import os
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# 使用 python-docx 读取 Word 文档
from docx import Document

# 使用 PyPDF2 读取 PDF 文档
from PyPDF2 import PdfReader

# 使用 LangChain 的 UnstructuredExcelLoader 读取 Excel
from langchain_community.document_loaders import UnstructuredExcelLoader

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-205ee372c5df491f8050324eb697e504")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL_NAME = "deepseek-v4-flash"  # 快速响应模式
TEMPERATURE = 0.7
REQUEST_TIMEOUT = 120  # 请求超时（秒）
MAX_RETRIES = 3  # 最大重试次数
FILE_NAME = "2026-test.docx"
PDF_FILE = "2026-test.pdf"
EXCEL_FILE = "跟踪止损策略回测数据_202605.xlsx"


class ChatDoc:
    """统一文档读取器 - 支持 Word、PDF、Excel"""
    
    # 文件扩展名映射到对应的读取方法
    READERS = {
        '.docx': '_read_word',
        '.pdf': '_read_pdf',
        '.xlsx': '_read_excel',
        '.xls': '_read_excel'
    }
    
    def __init__(self, file_name=None):
        """初始化
        
        Args:
            file_name: 文档路径
        """
        self.file_name = file_name if file_name else FILE_NAME
        
        if not os.path.exists(self.file_name):
            raise FileNotFoundError(f"文件不存在: {self.file_name}")
        
        # 检测文件类型
        ext = os.path.splitext(self.file_name)[1].lower()
        if ext not in self.READERS:
            raise ValueError(f"不支持的文件格式: {ext}\n支持的格式: {list(self.READERS.keys())}")
        
        self.reader_method = self.READERS[ext]
    
    def get_file_content(self):
        """获取文档内容（用于 RAG 处理）
        
        Returns:
            str: 文档的纯文本内容
        """
        # 根据映射调用对应的读取方法
        reader_func = getattr(self, self.reader_method)
        return reader_func()
    
    def _read_word(self):
        """读取 Word 文档"""
        doc = Document(self.file_name)
        return "\n".join([para.text for para in doc.paragraphs])
    
    def _read_pdf(self):
        """读取 PDF 文档"""
        reader = PdfReader(self.file_name)
        text_parts = []
        
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        
        return "\n\n--- 分页 ---\n\n".join(text_parts)
    
    def _read_excel(self):
        """读取 Excel 文档"""
        loader = UnstructuredExcelLoader(self.file_name)
        documents = loader.load()
        
        text_parts = []
        for doc in documents:
            if hasattr(doc, 'page_content') and doc.page_content:
                text_parts.append(doc.page_content)
        
        return "\n".join(text_parts)

# 示例：测试不同文件类型
print("="*70)
print("ChatDoc 统一文档读取器")
print("="*70)

# 测试1: Word 文档
print("\n【测试1】Word 文档")
print("-"*70)
try:
    chatdoc = ChatDoc(FILE_NAME)
    text = chatdoc.get_file_content()
    print(f"读取方法: {chatdoc.reader_method}")
    print(f"内容长度: {len(text)} 字符")
    print(f"\n内容预览（前300字符）:\n{text[:300]}...")
except Exception as e:
    print(f"读取失败: {e}")

# 测试2: PDF 文档
print("\n【测试2】PDF 文档")
print("-"*70)
try:
    chatdoc = ChatDoc(PDF_FILE)
    text = chatdoc.get_file_content()
    print(f"读取方法: {chatdoc.reader_method}")
    print(f"内容长度: {len(text)} 字符")
    print(f"\n内容预览（前300字符）:\n{text[:300]}...")
except Exception as e:
    print(f"读取失败: {e}")

# 测试3: Excel 文档
print("\n【测试3】Excel 文档")
print("-"*70)
try:
    chatdoc = ChatDoc(EXCEL_FILE)
    text = chatdoc.get_file_content()
    print(f"读取方法: {chatdoc.reader_method}")
    print(f"内容长度: {len(text)} 字符")
    print(f"\n内容预览（前300字符）:\n{text[:300]}...")
except Exception as e:
    print(f"读取失败: {e}")