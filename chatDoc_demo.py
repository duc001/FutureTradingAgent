"""
ChatDoc - 文档智能问答系统（纯 Python 实现）
=============================================

核心功能：
1. 上传文档（PDF、Word、Excel、TXT 等）
2. 对文档内容进行自然语言问答
3. 支持多文档管理
4. 引用原文进行回答（可溯源）
5. 支持表格数据的理解

应用场景：
- 策略文档问答（如 TBQuant 使用手册）
- 回测报告分析
- 交易规则查询
- 持仓数据解读
"""

import os
import re
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field


# ========== 配置 ==========
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-your-api-key")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


# ========== LLM 客户端 ==========

class LLMClient:
    """LLM 调用客户端"""

    def __init__(
        self,
        api_key: str = DEEPSEEK_API_KEY,
        base_url: str = DEEPSEEK_BASE_URL,
        model: str = "deepseek-chat"
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/') + '/v1'
        self.model = model
        self.conversation_history: List[Dict] = []

    def chat(
        self,
        prompt: str,
        system: str = "你是一个有用的AI助手。",
        temperature: float = 0.7
    ) -> str:
        """发送对话请求"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = [{"role": "system", "content": system}]
        messages.extend(self.conversation_history[-10:])
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            assistant_msg = result["choices"][0]["message"]["content"]

            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": assistant_msg})

            return assistant_msg

        except requests.exceptions.RequestException as e:
            return f"请求失败: {str(e)}"

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []


# ========== 文档加载器 ==========

class DocumentLoaders:
    """文档加载器集合"""

    @staticmethod
    def load_text(file_path: str) -> str:
        """加载文本文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def load_pdf(file_path: str) -> str:
        """加载 PDF 文件"""
        try:
            from pypdf import PdfReader
        except ImportError:
            return "[错误] 请安装 pypdf: pip install pypdf"

        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() for page in reader.pages)

    @staticmethod
    def load_docx(file_path: str) -> str:
        """加载 Word 文档"""
        try:
            import docx
        except ImportError:
            return "[错误] 请安装 python-docx: pip install python-docx"

        doc = docx.Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)

    @staticmethod
    def load_excel(file_path: str) -> str:
        """加载 Excel 文件"""
        try:
            import pandas as pd
        except ImportError:
            return "[错误] 请安装 pandas: pip install pandas"

        excel_file = pd.ExcelFile(file_path)
        parts = []
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            parts.append(f"=== Sheet: {sheet_name} ===\n{df.to_string()}")
        return "\n\n".join(parts)

    @staticmethod
    def load_csv(file_path: str) -> str:
        """加载 CSV 文件"""
        try:
            import pandas as pd
        except ImportError:
            return "[错误] 请安装 pandas: pip install pandas"

        df = pd.read_csv(file_path)
        return df.to_string()

    @classmethod
    def load(cls, file_path: str) -> str:
        """根据文件类型自动选择加载器"""
        path = Path(file_path)
        suffix = path.suffix.lower()

        loaders = {
            '.txt': cls.load_text,
            '.md': cls.load_text,
            '.pdf': cls.load_pdf,
            '.docx': cls.load_docx,
            '.doc': cls.load_docx,
            '.xlsx': cls.load_excel,
            '.xls': cls.load_excel,
            '.csv': cls.load_csv,
        }

        if suffix not in loaders:
            return f"[错误] 不支持的文件类型: {suffix}"

        return loaders[suffix](str(path))


# ========== 向量存储（简化版）==========

class VectorStore:
    """
    简化的向量存储 - 基于关键词匹配
    无需安装 Chroma 等向量数据库
    """

    def __init__(self):
        self.documents: List[Dict] = []

    def add_text(self, text: str, source: str = "unknown", metadata: Dict = None):
        """添加文本"""
        words = set(re.findall(r'[\w]+', text.lower()))
        self.documents.append({
            "content": text,
            "source": source,
            "metadata": metadata or {},
            "words": words
        })

    def add_document(self, file_path: str, metadata: Dict = None):
        """添加文档"""
        content = DocumentLoaders.load(file_path)
        self.add_text(
            text=content,
            source=Path(file_path).name,
            metadata=metadata
        )

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """搜索相关文档（基于关键词重叠度）"""
        query_words = set(re.findall(r'[\w]+', query.lower()))
        results = []

        for doc in self.documents:
            common_words = query_words & doc["words"]
            if common_words:
                score = len(common_words) / max(len(query_words), 1)
                results.append({
                    "content": doc["content"],
                    "source": doc["source"],
                    "score": score,
                    "common_words": list(common_words)[:10]
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def clear(self):
        """清空存储"""
        self.documents = []


# ========== 数据类 ==========

@dataclass
class ChatMessage:
    """聊天消息"""
    role: str
    content: str
    sources: Optional[List[Dict]] = None


# ========== ChatDoc 核心类 ==========

class ChatDoc:
    """
    ChatDoc - 文档智能问答系统

    使用步骤：
    1. 初始化 ChatDoc 实例
    2. 添加文档 (add_document) 或文本 (add_text)
    3. 问答 (ask)
    4. 可选：查看历史 (get_history)
    """

    def __init__(
        self,
        api_key: str = DEEPSEEK_API_KEY,
        base_url: str = DEEPSEEK_BASE_URL,
        model: str = "deepseek-chat"
    ):
        self.llm = LLMClient(api_key, base_url, model)
        self.vectorstore = VectorStore()
        self.history: List[ChatMessage] = []

    def add_text(self, text: str, source: str = "unknown"):
        """添加文本知识"""
        self.vectorstore.add_text(text, source)

    def add_document(self, file_path: str, metadata: Dict = None):
        """添加文档"""
        self.vectorstore.add_document(file_path, metadata)

    def ask(
        self,
        question: str,
        top_k: int = 3,
        return_sources: bool = True
    ) -> ChatMessage:
        """问答"""
        docs = self.vectorstore.search(question, top_k)

        if docs:
            context_parts = []
            for i, doc in enumerate(docs, 1):
                context_parts.append(
                    f"【参考 {i}】来源: {doc['source']}\n{doc['content'][:500]}"
                )
            context = "\n\n".join(context_parts)
        else:
            context = "（知识库为空，请先添加文档或文本）"

        prompt = f"""你是一个专业的文档问答助手。请根据以下参考文档回答用户问题。

参考文档：
---
{context}
---

用户问题：{question}

要求：
1. 如果参考文档中有相关信息，请基于文档回答
2. 如果文档中没有足够信息，请说明并给出通用建议
3. 回答要专业、准确
4. 适当引用文档内容
5. 用【1】、【2】等标注参考来源
"""

        answer = self.llm.chat(prompt, system="你是一个专业的文档问答助手。")

        sources = None
        if return_sources and docs:
            sources = [
                {"content": doc["content"][:200] + "...", "source": doc["source"]}
                for doc in docs
            ]

        user_msg = ChatMessage(role="user", content=question)
        assistant_msg = ChatMessage(role="assistant", content=answer, sources=sources)

        self.history.extend([user_msg, assistant_msg])
        return assistant_msg

    def get_history(self) -> List[ChatMessage]:
        """获取对话历史"""
        return self.history

    def clear_history(self):
        """清空对话历史"""
        self.history = []
        self.llm.clear_history()


# ========== 示例演示 ==========

def demo_basic_usage():
    """基础用法演示"""
    print("=" * 60)
    print("ChatDoc 基础用法演示")
    print("=" * 60)

    chatdoc = ChatDoc(api_key=DEEPSEEK_API_KEY)

    # 添加文档（可选）
    # chatdoc.add_document("TBQuant使用手册.pdf")
    # chatdoc.add_document("策略设计指南.docx")

    # 问答
    question = "双均线策略的参数如何选择？"
    print(f"\n问题: {question}")

    response = chatdoc.ask(question)
    print(f"\n回答: {response.content}")

    if response.sources:
        print("\n--- 参考来源 ---")
        for i, src in enumerate(response.sources, 1):
            print(f"{i}. {src['source']}: {src['content'][:100]}...")


def demo_trading_scenario():
    """期货交易场景演示"""
    print("\n" + "=" * 60)
    print("期货交易场景演示")
    print("=" * 60)

    chatdoc = ChatDoc(api_key=DEEPSEEK_API_KEY)

    # 添加策略知识
    chatdoc.add_text("""
    双均线策略要点：
    1. 快速均线周期：一般 5-20
    2. 慢速均线周期：一般 20-60
    3. 金叉买入，死叉卖出
    4. 适用于趋势明显的品种
    """, source="策略笔记")

    chatdoc.add_text("""
    螺纹钢期货特点：
    1. 交易所：上海期货交易所 (SHFE)
    2. 交易单位：10吨/手
    3. 波动较大，适合趋势策略
    4. 主力合约流动性好
    """, source="品种介绍")

    chatdoc.add_text("""
    TBQuant 基本语法：
    Params
        Integer fastLength(5);
        Integer slowLength(20);
    Vars
        Series<Numeric> fastMa;
        Series<Numeric> slowMa;
    Events
        OnInit() { ... }
        OnBar() { ... }
    """, source="TBQuant语法")

    # 问题示例
    questions = [
        "TBQuant 如何设置自动换月？",
        "双均线策略参数有什么讲究？",
        "回测最大回撤 20% 正常吗？"
    ]

    for q in questions:
        print(f"\n问题: {q}")
        print("-" * 40)
        response = chatdoc.ask(q)
        print(f"回答: {response.content}")


def demo_backtest_analysis():
    """回测报告分析演示"""
    print("\n" + "=" * 60)
    print("回测报告分析演示")
    print("=" * 60)

    chatdoc = ChatDoc(api_key=DEEPSEEK_API_KEY)

    # 添加回测 Excel 报告
    # chatdoc.add_document("回测结果.xlsx")

    # 分析问题
    questions = [
        "哪个品种收益最高？",
        "哪些品种夏普比率超过 1？",
        "有哪些品种最大回撤超过 15%？"
    ]

    for q in questions:
        print(f"\n问题: {q}")
        response = chatdoc.ask(q)
        print(f"分析: {response.content}")


# ========== 主函数 ==========

if __name__ == "__main__":
    print("ChatDoc 示例 Demo")
    print("=" * 60)

    # 运行演示
    demo_basic_usage()
    # demo_trading_scenario()
    # demo_backtest_analysis()

    print("\n" + "=" * 60)
    print("演示结束")
    print("=" * 60)
