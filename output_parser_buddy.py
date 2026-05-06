"""
ChatOpenAI 输出解析器介绍和 使用示例

输出解析器（Output Parser）的作用：
1. 解析 LLM 原始输出为结构化数据
2. 指导 LLM 生成指定格式的内容
3. 验证输出是否符合预期格式

LangChain 内置输出解析器：
| 解析器 | 用途 |
|--------|------|
| StrOutputParser | 简单字符串解析 |
| JsonOutputParser | JSON 对象解析 |
| PydanticOutputParser | Pydantic 模型解析（类型安全）|
| CommaSeparatedListOutputParser | 逗号分隔列表 |
| XMLOutputParser | XML 格式解析 |
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    CommaSeparatedListOutputParser,
    PydanticOutputParser,
)
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, field_validator

# 加载环境变量
load_dotenv()

# 初始化 ChatOpenAI
chat_llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.7,
)


# ============================================================
# 示例 1：StrOutputParser - 最简单的字符串解析
# ============================================================
def demo_str_parser():
    """直接获取 LLM 输出，不做额外处理"""
    print("\n" + "=" * 60)
    print("示例 1: StrOutputParser - 字符串解析")
    print("=" * 60)

    parser = StrOutputParser()

    # 使用管道符连接 LLM 和解析器
    chain = chat_llm | parser

    # 单次调用
    result = chain.invoke("什么是 Python？用一句话回答")
    print(f"输出类型: {type(result)}")
    print(f"输出内容: {result}")

    return result


# ============================================================
# 示例 2：JsonOutputParser - JSON 格式解析
# ============================================================
def demo_json_parser():
    """将 LLM 输出解析为 Python 字典/列表"""
    print("\n" + "=" * 60)
    print("示例 2: JsonOutputParser - JSON 解析")
    print("=" * 60)

    parser = JsonOutputParser()

    # 通过 PromptTemplate 指导 LLM 输出 JSON 格式
    prompt = PromptTemplate.from_template(
        "请介绍一下 {city} 这个城市，"
        "用 JSON 格式返回，包含以下字段：name(城市名), country(国家), description(简介)"
    )

    chain = prompt | chat_llm | parser
    result = chain.invoke({"city": "东京"})

    print(f"输出类型: {type(result)}")
    print(f"输出内容: {result}")
    print(f"城市: {result.get('name')}, 国家: {result.get('country')}")

    return result


# ============================================================
# 示例 3：CommaSeparatedListOutputParser - 列表解析
# ============================================================
def demo_list_parser():
    """将逗号分隔的文本转为 Python 列表"""
    print("\n" + "=" * 60)
    print("示例 3: CommaSeparatedListOutputParser - 列表解析")
    print("=" * 60)

    parser = CommaSeparatedListOutputParser()

    prompt = PromptTemplate.from_template(
        "列出 {topic} 的 {num} 个特点，用逗号分隔"
    )

    chain = prompt | chat_llm | parser
    result = chain.invoke({"topic": "人工智能", "num": 4})

    print(f"输出类型: {type(result)}")
    print(f"输出内容: {result}")

    return result


# ============================================================
# 示例 4：PydanticOutputParser - Pydantic 模型解析（推荐）
# ============================================================
def demo_pydantic_parser():
    """使用 Pydantic 模型定义输出结构，类型安全"""
    print("\n" + "=" * 60)
    print("示例 4: PydanticOutputParser - Pydantic 模型解析（推荐）")
    print("=" * 60)

    # 定义 Pydantic 模型
    class BookInfo(BaseModel):
        """图书信息模型"""
        title: str = Field(description="书名")
        author: str = Field(description="作者")
        year: int = Field(description="出版年份")
        genres: list[str] = Field(description="类型标签列表")
        rating: float = Field(description="评分，0-10分")

        @field_validator('year')
        @classmethod
        def validate_year(cls, v):
            if v < 1900 or v > 2030:
                raise ValueError('年份必须在 1900-2030 之间')
            return v

        @field_validator('rating')
        @classmethod
        def validate_rating(cls, v):
            if v < 0 or v > 10:
                raise ValueError('评分必须在 0-10 之间')
            return v

    # 创建解析器，使用 PydanticOutputParser 返回 Pydantic 对象
    parser = PydanticOutputParser(pydantic_object=BookInfo)

    # 构建提示词，partial_variables 作为构造函数参数传入
    prompt = PromptTemplate.from_template(
        "请为书籍《{book_name}》生成信息。"
        "\n\n{format_instructions}",
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | chat_llm | parser
    result = chain.invoke({"book_name": "百年孤独"})

    print(f"输出类型: {type(result)}")
    print(f"书名: {result.title}")
    print(f"作者: {result.author}")
    print(f"年份: {result.year}")
    print(f"类型: {result.genres}")
    print(f"评分: {result.rating}/10")

    return result


# ============================================================
# 示例 5：流式输出 + 解析器
# ============================================================
def demo_streaming_parser():
    """流式输出结合解析器"""
    print("\n" + "=" * 60)
    print("示例 5: 流式输出 + StrOutputParser")
    print("=" * 60)

    parser = StrOutputParser()
    chain = chat_llm | parser

    print("流式输出内容: ", end="", flush=True)
    for chunk in chain.stream("简单介绍一下什么是机器学习"):
        print(chunk, end="", flush=True)
    print()


# ============================================================
# 示例 6：批量处理 + 解析器
# ============================================================
def demo_batch_parser():
    """批量处理多个请求"""
    print("\n" + "=" * 60)
    print("示例 6: 批量处理 + JsonOutputParser")
    print("=" * 60)

    parser = JsonOutputParser()

    prompt = PromptTemplate.from_template(
        "用 JSON 格式返回水果 {fruit} 的信息，包含 name 和 color 字段"
    )

    chain = prompt | chat_llm | parser

    # 批量处理
    results = chain.batch([
        {"fruit": "苹果"},
        {"fruit": "香蕉"},
        {"fruit": "橙子"},
    ])

    for r in results:
        print(f"  - {r.get('name')}: {r.get('color')}")

    return results


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("ChatOpenAI 输出解析器完整示例")
    print("=" * 60)

    try:
        # 运行所有示例
        demo_str_parser()
        demo_json_parser()
        demo_list_parser()
        demo_pydantic_parser()
        demo_streaming_parser()
        demo_batch_parser()

        print("\n" + "=" * 60)
        print("所有示例执行完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n执行出错: {e}")
        raise


if __name__ == "__main__":
    main()
