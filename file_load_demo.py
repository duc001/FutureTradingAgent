"""
LangChain 文档加载器完整示例
演示如何使用 LangChain 加载各种类型的文件
"""
import os
from typing import List
from langchain_core.documents import Document


# ==================== 1. 文本文件加载器 ====================

def load_text_file(file_path: str = "README.md") -> List[Document]:
    """加载纯文本文件（.txt, .md, .log 等）"""
    from langchain_community.document_loaders import TextLoader
    
    print(f"\n{'='*60}")
    print(f"加载文本文件: {file_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return []
    
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    
    print(f"加载成功: {len(documents)} 个文档")
    if documents:
        print(f"内容预览: {documents[0].page_content[:200]}...")
        print(f"元数据: {documents[0].metadata}")
    
    return documents


# ==================== 2. Word 文档加载器 ====================

def load_word_document(file_path: str = "2026-test.docx") -> List[Document]:
    """加载 Word 文档（.docx）"""
    from langchain_community.document_loaders import Docx2txtLoader
    
    print(f"\n{'='*60}")
    print(f"加载 Word 文档: {file_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return []
    
    # 方式1: 使用 Docx2txtLoader（轻量级）
    loader = Docx2txtLoader(file_path)
    documents = loader.load()
    
    print(f"加载成功: {len(documents)} 个文档")
    if documents:
        preview = documents[0].page_content.replace('\n', ' ')[:200]
        print(f"内容预览: {preview}...")
        print(f"元数据: {documents[0].metadata}")
    
    return documents


def load_word_with_unstructured(file_path: str = "2026-test.docx") -> List[Document]:
    """使用 Unstructured 加载 Word（保留更多格式信息）"""
    try:
        from langchain_community.document_loaders import UnstructuredWordDocumentLoader
        
        print(f"\n{'='*60}")
        print(f"使用 Unstructured 加载 Word: {file_path}")
        print(f"{'='*60}")
        
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return []
        
        loader = UnstructuredWordDocumentLoader(file_path)
        documents = loader.load()
        
        print(f"加载成功: {len(documents)} 个文档")
        if documents:
            preview = documents[0].page_content.replace('\n', ' ')[:200]
            print(f"内容预览: {preview}...")
        
        return documents
    except ImportError:
        print("需要安装: pip install unstructured python-docx")
        return []


# ==================== 3. PDF 文档加载器 ====================

def load_pdf_with_pypdf(file_path: str = "2026-test.pdf") -> List[Document]:
    """使用 PyPDF 加载 PDF 文件"""
    from langchain_community.document_loaders import PyPDFLoader
    
    print(f"\n{'='*60}")
    print(f"使用 PyPDF 加载 PDF: {file_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return []
    
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    print(f"加载成功: {len(documents)} 页")
    for i, doc in enumerate(documents[:3]):  # 只显示前3页
        preview = doc.page_content.replace('\n', ' ')[:150]
        print(f"  第{i+1}页: {preview}...")
        print(f"  元数据: {doc.metadata}")
    
    return documents


def load_pdf_with_pymupdf(file_path: str = "2026-test.pdf") -> List[Document]:
    """使用 PyMuPDF 加载 PDF（更快）"""
    try:
        from langchain_community.document_loaders import PyMuPDFLoader
        
        print(f"\n{'='*60}")
        print(f"使用 PyMuPDF 加载 PDF: {file_path}")
        print(f"{'='*60}")
        
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return []
        
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        print(f"加载成功: {len(documents)} 页")
        if documents:
            preview = documents[0].page_content.replace('\n', ' ')[:200]
            print(f"首页预览: {preview}...")
        
        return documents
    except ImportError:
        print("需要安装: pip install pymupdf")
        return []


# ==================== 4. Excel 文件加载器 ====================

def load_excel_file(file_path: str = "跟踪止损策略回测数据_202605.xlsx") -> List[Document]:
    """加载 Excel 文件（.xlsx, .xls）"""
    try:
        from langchain_community.document_loaders import UnstructuredExcelLoader
        
        print(f"\n{'='*60}")
        print(f"加载 Excel 文件: {file_path}")
        print(f"{'='*60}")
        
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return []
        
        # mode="elements": 按单元格元素分割
        # mode="single": 整个表格作为一个文档
        loader = UnstructuredExcelLoader(file_path, mode="elements")
        documents = loader.load()
        
        print(f"加载成功: {len(documents)} 个文档块")
        for i, doc in enumerate(documents[:3]):  # 只显示前3个
            preview = doc.page_content.replace('\n', ' ')[:150]
            print(f"  [{i+1}] {preview}...")
            print(f"      元数据: {doc.metadata}")
        
        return documents
    except ImportError:
        print("需要安装: pip install unstructured openpyxl")
        return []


# ==================== 5. CSV 文件加载器 ====================

def load_csv_file(file_path: str = "data.csv") -> List[Document]:
    """加载 CSV 文件"""
    from langchain_community.document_loaders import CSVLoader
    
    print(f"\n{'='*60}")
    print(f"加载 CSV 文件: {file_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return []
    
    loader = CSVLoader(file_path)
    documents = loader.load()
    
    print(f"加载成功: {len(documents)} 行数据")
    for i, doc in enumerate(documents[:3]):
        preview = doc.page_content.replace('\n', ' ')[:150]
        print(f"  [{i+1}] {preview}...")
    
    return documents


# ==================== 6. JSON 文件加载器 ====================

def load_json_file(file_path: str = "a.json") -> List[Document]:
    """加载 JSON 文件"""
    try:
        from langchain_community.document_loaders import JSONLoader
        
        print(f"\n{'='*60}")
        print(f"加载 JSON 文件: {file_path}")
        print(f"{'='*60}")
        
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return []
        
        # jq_schema 用于指定要提取的 JSON 路径
        # "." 表示根节点，".messages[]" 表示提取 messages 数组
        loader = JSONLoader(
            file_path=file_path,
            jq_schema=".",
            text_content=False
        )
        documents = loader.load()
        
        print(f"加载成功: {len(documents)} 个文档")
        for i, doc in enumerate(documents[:3]):
            preview = doc.page_content.replace('\n', ' ')[:150]
            print(f"  [{i+1}] {preview}...")
        
        return documents
    except ImportError:
        print("需要安装: pip install jq")
        return []


# ==================== 7. 目录批量加载器 ====================

def load_directory(dir_path: str = "./", pattern: str = "**/*.py") -> List[Document]:
    """批量加载目录中的文件"""
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    
    print(f"\n{'='*60}")
    print(f"批量加载目录: {dir_path}")
    print(f"文件模式: {pattern}")
    print(f"{'='*60}")
    
    if not os.path.exists(dir_path):
        print(f"目录不存在: {dir_path}")
        return []
    
    loader = DirectoryLoader(
        path=dir_path,
        glob=pattern,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    
    print(f"加载成功: {len(documents)} 个文件")
    
    # 统计文件类型
    file_types = {}
    for doc in documents:
        source = doc.metadata.get('source', '')
        ext = os.path.splitext(source)[1]
        file_types[ext] = file_types.get(ext, 0) + 1
    
    print(f"文件类型分布: {file_types}")
    
    return documents


# ==================== 8. 统一文档加载器（推荐）====================

class UniversalDocumentLoader:
    """统一文档加载器 - 根据文件扩展名自动选择加载器"""
    
    # 文件扩展名映射到加载器类
    LOADERS = {
        '.txt': ('TextLoader', {}),
        '.md': ('TextLoader', {}),
        '.log': ('TextLoader', {}),
        '.docx': ('Docx2txtLoader', {}),
        '.pdf': ('PyPDFLoader', {}),
        '.xlsx': ('UnstructuredExcelLoader', {'mode': 'elements'}),
        '.xls': ('UnstructuredExcelLoader', {'mode': 'elements'}),
        '.csv': ('CSVLoader', {}),
    }
    
    @staticmethod
    def load(file_path: str) -> List[Document]:
        """根据文件扩展名自动选择加载器并加载"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext not in UniversalDocumentLoader.LOADERS:
            raise ValueError(f"不支持的文件格式: {ext}\n支持的格式: {list(UniversalDocumentLoader.LOADERS.keys())}")
        
        loader_name, loader_kwargs = UniversalDocumentLoader.LOADERS[ext]
        
        # 动态导入并实例化加载器
        if loader_name == 'TextLoader':
            from langchain_community.document_loaders import TextLoader
            LoaderClass = TextLoader
        elif loader_name == 'Docx2txtLoader':
            from langchain_community.document_loaders import Docx2txtLoader
            LoaderClass = Docx2txtLoader
        elif loader_name == 'PyPDFLoader':
            from langchain_community.document_loaders import PyPDFLoader
            LoaderClass = PyPDFLoader
        elif loader_name == 'UnstructuredExcelLoader':
            try:
                from langchain_community.document_loaders import UnstructuredExcelLoader
                LoaderClass = UnstructuredExcelLoader
            except ImportError:
                raise ImportError("需要安装: pip install unstructured openpyxl")
        elif loader_name == 'CSVLoader':
            from langchain_community.document_loaders import CSVLoader
            LoaderClass = CSVLoader
        else:
            raise ValueError(f"未知的加载器: {loader_name}")
        
        # 创建加载器实例
        loader = LoaderClass(file_path, **loader_kwargs)
        
        print(f"\n使用 {loader_name} 加载: {file_path}")
        documents = loader.load()
        print(f"加载成功: {len(documents)} 个文档")
        
        return documents


# ==================== 9. 结合文本分割器使用 ====================

def load_and_split(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """加载文档并进行智能分割"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    print(f"\n{'='*60}")
    print(f"加载并分割文档: {file_path}")
    print(f"块大小: {chunk_size}, 重叠: {chunk_overlap}")
    print(f"{'='*60}")
    
    # 1. 加载文档
    documents = UniversalDocumentLoader.load(file_path)
    
    if not documents:
        print("未加载到文档")
        return []
    
    # 2. 创建文本分割器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        is_separator_regex=True,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
    )
    
    # 3. 分割文档
    split_docs = splitter.split_documents(documents)
    print(f"分割结果: {len(documents)} 个文档 -> {len(split_docs)} 个块")
    
    # 显示前几个块的统计信息
    for i, doc in enumerate(split_docs[:3]):
        print(f"\n块 {i+1}:")
        print(f"  长度: {len(doc.page_content)} 字符")
        print(f"  预览: {doc.page_content[:100]}...")
        print(f"  元数据: {doc.metadata}")
    
    return split_docs


# ==================== 主函数 - 演示所有加载器 ====================

def main():
    """演示各种文档加载器"""
    print("\n" + "="*60)
    print("LangChain 文档加载器完整示例")
    print("="*60)
    
    # 测试文件列表
    test_files = {
        "text": "README.md",
        "word": "2026-test.docx",
        "pdf": "2026-test.pdf",
        "excel": "跟踪止损策略回测数据_202605.xlsx",
        "json": "a.json",
    }
    
    # 1. 加载文本文件
    if os.path.exists(test_files["text"]):
        load_text_file(test_files["text"])
    
    # 2. 加载 Word 文档
    if os.path.exists(test_files["word"]):
        load_word_document(test_files["word"])
    
    # 3. 加载 PDF 文档
    if os.path.exists(test_files["pdf"]):
        load_pdf_with_pypdf(test_files["pdf"])
    
    # 4. 加载 Excel 文件
    if os.path.exists(test_files["excel"]):
        load_excel_file(test_files["excel"])
    
    # 5. 加载 JSON 文件
    if os.path.exists(test_files["json"]):
        load_json_file(test_files["json"])
    
    # 6. 批量加载目录
    print(f"\n{'='*60}")
    print("批量加载 Python 文件")
    print(f"{'='*60}")
    load_directory("./", pattern="*.py")
    
    # 7. 使用统一加载器
    print(f"\n{'='*60}")
    print("使用统一加载器")
    print(f"{'='*60}")
    for file_type, file_path in test_files.items():
        if os.path.exists(file_path):
            try:
                docs = UniversalDocumentLoader.load(file_path)
                print(f"✓ {file_type}: 加载了 {len(docs)} 个文档\n")
            except Exception as e:
                print(f"✗ {file_type}: {e}\n")
    
    # 8. 加载并分割
    if os.path.exists(test_files["word"]):
        load_and_split(test_files["word"], chunk_size=300, chunk_overlap=50)


if __name__ == "__main__":
    main()
