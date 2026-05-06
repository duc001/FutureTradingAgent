"""
ChatOpenAI 输出解析器（Output Parser）介绍和使用示例
======================================================

输出解析器的作用：
1. 将 LLM 的原始文本输出转换为结构化数据
2. 确保输出符合预期格式
3. 自动验证和纠错
4. 简化后续数据处理

常用解析器类型：
- StrOutputParser: 字符串输出（默认）
- JsonOutputParser: JSON 格式解析
- PydanticOutputParser: Pydantic 模型解析
- CSVOutputParser: CSV 格式解析
- DatetimeOutputParser: 日期时间解析
- EnumOutputParser: 枚举值解析
"""
import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate

print("="*70)
print("ChatOpenAI 输出解析器完整示例")
print("="*70)

# 创建 ChatOpenAI 实例
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com",
    temperature=0,
)


# ==================== 1. StrOutputParser（字符串解析器）====================
print("\n" + "="*70)
print("1. StrOutputParser - 字符串解析器（最基础）")
print("="*70)
print("说明：将 AIMessage 对象转换为纯字符串")

# 创建解析器
str_parser = StrOutputParser()

# 使用方式1：单独使用
prompt = ChatPromptTemplate.from_template("请用一句话介绍{name}")
chain = prompt | llm | str_parser

result = chain.invoke({"name": "Python"})
print(f"\n结果类型: {type(result)}")
print(f"结果内容: {result}")

# 使用方式2：直接使用（invoke 返回的就是 AIMessage）
response = llm.invoke("用三个字形容AI")
parsed = str_parser.invoke(response)
print(f"\n原始响应类型: {type(response)}")
print(f"解析后类型: {type(parsed)}")
print(f"解析结果: {parsed}")


# ==================== 2. JsonOutputParser（JSON 解析器）====================
print("\n\n" + "="*70)
print("2. JsonOutputParser - JSON 解析器")
print("="*70)
print("说明：将 LLM 输出的 JSON 字符串解析为 Python 字典/列表")

json_parser = JsonOutputParser()

# 示例1：简单 JSON
print("\n【示例1】简单 JSON 对象")
json_prompt = ChatPromptTemplate.from_template("""
请分析以下产品评论，并以 JSON 格式返回：
评论："{review}"

要求返回格式：
{{
    "sentiment": "positive/negative/neutral",
    "score": 1-10的评分,
    "keywords": ["关键词1", "关键词2"]
}}
""")

chain = json_prompt | llm | json_parser
result = chain.invoke({
    "review": "这款手机非常好用，电池续航长，拍照清晰，就是价格有点贵"
})

print(f"结果类型: {type(result)}")
print(f"情感倾向: {result['sentiment']}")
print(f"评分: {result['score']}")
print(f"关键词: {result['keywords']}")

# 示例2：JSON 数组
print("\n【示例2】JSON 数组")
array_prompt = ChatPromptTemplate.from_template("""
请列出3个流行的Python Web框架，以JSON数组格式返回：
[
    {{"name": "框架名", "feature": "特点"}}
]
""")

chain = array_prompt | llm | json_parser
result = chain.invoke({})
print(f"结果类型: {type(result)}")
print(f"框架列表:")
for item in result:
    print(f"  - {item['name']}: {item['feature']}")


# ==================== 3. PydanticOutputParser（Pydantic 模型解析器）====================
print("\n\n" + "="*70)
print("3. PydanticOutputParser - Pydantic 模型解析器（推荐）")
print("="*70)
print("说明：将 LLM 输出解析为强类型的 Pydantic 模型对象")

# 定义 Pydantic 模型
class PersonInfo(BaseModel):
    """人物信息模型"""
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    occupation: str = Field(description="职业")
    skills: List[str] = Field(description="技能列表")
    bio: Optional[str] = Field(None, description="个人简介")

class ProductReview(BaseModel):
    """产品评价模型"""
    product_name: str = Field(description="产品名称")
    rating: int = Field(description="评分，1-5星", ge=1, le=5)
    pros: List[str] = Field(description="优点列表")
    cons: List[str] = Field(description="缺点列表")
    recommendation: bool = Field(description="是否推荐购买")

# 创建解析器
pydantic_parser = PydanticOutputParser(pydantic_object=PersonInfo)

# 获取格式化指令
format_instructions = pydantic_parser.get_format_instructions()
print(f"\n格式化指令:\n{format_instructions[:200]}...\n")

# 使用示例
person_prompt = ChatPromptTemplate.from_template("""
请根据以下描述提取人物信息：
"{description}"

{format_instructions}
""")

chain = person_prompt | llm | pydantic_parser
result = chain.invoke({
    "description": "张三，28岁，是一名资深Python开发工程师，擅长Django、FastAPI和机器学习，有5年工作经验",
    "format_instructions": format_instructions
})

print(f"结果类型: {type(result)}")
print(f"姓名: {result.name}")
print(f"年龄: {result.age}")
print(f"职业: {result.occupation}")
print(f"技能: {result.skills}")
print(f"简介: {result.bio}")

# 验证数据类型
print(f"\n✅ 类型验证:")
print(f"  age 是 int: {isinstance(result.age, int)}")
print(f"  skills 是 list: {isinstance(result.skills, list)}")


# ==================== 4. 复杂嵌套模型示例 ====================
print("\n\n" + "="*70)
print("4. 复杂嵌套 Pydantic 模型示例")
print("="*70)

# 定义嵌套模型
class Address(BaseModel):
    """地址信息"""
    city: str = Field(description="城市")
    district: str = Field(description="区域")
    street: str = Field(description="街道")

class ContactInfo(BaseModel):
    """联系方式"""
    email: str = Field(description="邮箱")
    phone: str = Field(description="电话")

class Employee(BaseModel):
    """员工完整信息"""
    name: str = Field(description="姓名")
    department: str = Field(description="部门")
    position: str = Field(description="职位")
    address: Address = Field(description="地址")
    contact: ContactInfo = Field(description="联系方式")
    projects: List[str] = Field(description="参与的项目")

# 创建解析器
employee_parser = PydanticOutputParser(pydantic_object=Employee)
format_instructions = employee_parser.get_format_instructions()

employee_prompt = ChatPromptTemplate.from_template("""
请根据以下描述提取员工信息：
"李四在技术部担任高级前端工程师，住在北京市朝阳区建国路100号，
邮箱是lisi@example.com，电话13800138000，正在参与电商平台重构和移动端APP开发项目"

{format_instructions}
""")

chain = employee_prompt | llm | employee_parser
result = chain.invoke({"format_instructions": format_instructions})

print(f"\n员工信息:")
print(f"  姓名: {result.name}")
print(f"  部门: {result.department}")
print(f"  职位: {result.position}")
print(f"  城市: {result.address.city}")
print(f"  区域: {result.address.district}")
print(f"  邮箱: {result.contact.email}")
print(f"  电话: {result.contact.phone}")
print(f"  项目: {', '.join(result.projects)}")


# ==================== 5. 手动 CSV 解析（替代方案）====================
print("\n\n" + "="*70)
print("5. 手动 CSV 解析（替代方案）")
print("="*70)
print("说明：CsvOutputParser 已移除，可手动解析逗号分隔文本")

# 新版 LangChain 中 CsvOutputParser 已移除
# 可以使用简单的字符串分割实现类似功能
def parse_csv(text: str) -> List[str]:
    """手动解析 CSV 格式文本"""
    # 去除空白并按逗号分割
    items = [item.strip() for item in text.split(',') if item.strip()]
    return items

csv_prompt = ChatPromptTemplate.from_template("""
请列出5个中国一线城市，用逗号分隔：
""")

chain = csv_prompt | llm | StrOutputParser()
result_text = chain.invoke({})
result = parse_csv(result_text)

print(f"\n原始输出: {result_text}")
print(f"解析结果类型: {type(result)}")
print(f"城市列表: {result}")
print(f"第一个城市: {result[0] if result else 'N/A'}")

print("\n💡 提示: 如需更复杂的 CSV 解析，可使用 Python 标准库:")
print("  import csv")
print("  import io")
print("  reader = csv.reader(io.StringIO(csv_text))")


# ==================== 6. 自定义输出解析器 ====================
print("\n\n" + "="*70)
print("6. 自定义输出解析器")
print("="*70)
print("说明：继承 BaseOutputParser 创建自定义解析逻辑")

from langchain_core.output_parsers import BaseOutputParser

class BooleanOutputParser(BaseOutputParser[bool]):
    """自定义布尔值解析器"""
    
    true_values: List[str] = ["是", "yes", "true", "y", "t"]
    false_values: List[str] = ["否", "no", "false", "n", "f"]
    
    def parse(self, text: str) -> bool:
        """解析文本为布尔值"""
        text = text.strip().lower()
        
        if text in self.true_values:
            return True
        elif text in self.false_values:
            return False
        else:
            # 尝试从文本中提取关键词
            if any(word in text for word in self.true_values):
                return True
            elif any(word in text for word in self.false_values):
                return False
            else:
                raise ValueError(f"无法解析为布尔值: {text}")

# 使用自定义解析器
bool_parser = BooleanOutputParser()

bool_prompt = ChatPromptTemplate.from_template("""
问题：Python是一门编程语言吗？
请回答"是"或"否"：
""")

chain = bool_prompt | llm | bool_parser
result = chain.invoke({})
print(f"\n问题: Python是一门编程语言吗？")
print(f"LLM原始回答: {llm.invoke(bool_prompt.format()).content}")
print(f"解析结果: {result}")
print(f"结果类型: {type(result)}")


# ==================== 7. 带重试的解析器（手动实现）====================
print("\n\n" + "="*70)
print("7. 带重试机制的解析器（手动实现）")
print("="*70)
print("说明：RetryWithErrorOutputParser 已移除，可手动实现重试逻辑")

# 新版 LangChain 中 RetryWithErrorOutputParser 已移除
# 可以手动实现重试逻辑
def parse_with_retry(parser, chain, inputs, max_retries=2):
    """
    带重试机制的解析函数
    
    Args:
        parser: 解析器实例
        chain: 调用链
        inputs: 输入参数
        max_retries: 最大重试次数
    
    Returns:
        解析结果
    """
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                print(f"  🔄 第 {attempt} 次重试...")
            result = chain.invoke(inputs)
            return result
        except Exception as e:
            last_error = e
            print(f"  ⚠️ 第 {attempt + 1} 次尝试失败: {str(e)[:80]}")
            if attempt < max_retries:
                # 可以在这里添加修正逻辑
                continue
    
    raise last_error

strict_prompt = ChatPromptTemplate.from_template("""
请提取以下文本中的人物信息（必须包含姓名、年龄、职业）：
"{text}"

{format_instructions}
""")

try:
    chain = strict_prompt | llm | pydantic_parser
    result = parse_with_retry(
        parser=pydantic_parser,
        chain=chain,
        inputs={
            "text": "王五是一位优秀的程序员",
            "format_instructions": pydantic_parser.get_format_instructions()
        },
        max_retries=2
    )
    print(f"\n✅ 解析成功:")
    print(f"  姓名: {result.name}")
    print(f"  年龄: {result.age}")
    print(f"  职业: {result.occupation}")
except Exception as e:
    print(f"\n❌ 解析失败: {e}")

print("\n💡 提示: 如需更强大的重试机制，可使用:")
print("  - tenacity 库: pip install tenacity")
print("  - 自定义装饰器实现指数退避重试")


# ==================== 8. 实际应用场景 ====================
print("\n\n" + "="*70)
print("8. 实际应用场景示例")
print("="*70)

# 场景1：API 响应解析
print("\n【场景1】API 响应解析")
class APIResponse(BaseModel):
    status: str = Field(description="状态码")
    message: str = Field(description="响应消息")
    data: Optional[dict] = Field(None, description="数据内容")

api_parser = PydanticOutputParser(pydantic_object=APIResponse)
api_prompt = ChatPromptTemplate.from_template("""
模拟一个用户登录成功的API响应，返回JSON格式：
用户名：admin
登录时间：当前时间

{format_instructions}
""")

chain = api_prompt | llm | api_parser
result = chain.invoke({"format_instructions": api_parser.get_format_instructions()})
print(f"状态: {result.status}")
print(f"消息: {result.message}")
print(f"数据: {result.data}")

# 场景2：数据提取
print("\n【场景2】从非结构化文本提取数据")
class InvoiceInfo(BaseModel):
    """发票信息"""
    invoice_number: str = Field(description="发票号码")
    amount: float = Field(description="金额")
    date: str = Field(description="日期")
    vendor: str = Field(description="供应商")

invoice_parser = PydanticOutputParser(pydantic_object=InvoiceInfo)
invoice_text = """
这是一张购物发票，发票号为INV-2024-001234，
购买日期是2024年1月15日，供应商是京东电子商城，
总金额为1599.99元。
"""

invoice_prompt = ChatPromptTemplate.from_template("""
从以下文本中提取发票信息：
{text}

{format_instructions}
""")

chain = invoice_prompt | llm | invoice_parser
result = chain.invoke({
    "text": invoice_text,
    "format_instructions": invoice_parser.get_format_instructions()
})
print(f"发票号: {result.invoice_number}")
print(f"金额: ¥{result.amount}")
print(f"日期: {result.date}")
print(f"供应商: {result.vendor}")


# ==================== 9. 解析器对比总结 ====================
print("\n\n" + "="*70)
print("📊 输出解析器对比总结")
print("="*70)
print("""
┌──────────────────────┬──────────────┬──────────────┬─────────────────┐
│ 解析器类型            │ 返回类型      │ 适用场景      │ 难度             │
├──────────────────────┼──────────────┼──────────────┼─────────────────┤
│ StrOutputParser      │ str          │ 普通文本      │ ⭐ 简单          │
├──────────────────────┼──────────────┼──────────────┼─────────────────┤
│ JsonOutputParser     │ dict/list    │ JSON数据      │ ⭐⭐ 中等        │
├──────────────────────┼──────────────┼──────────────┼─────────────────┤
│ PydanticOutputParser │ BaseModel    │ 结构化数据    │ ⭐⭐⭐ 较复杂    │
├──────────────────────┼──────────────┼──────────────┼─────────────────┤
│ 手动解析（CSV等）     │ 自定义       │ 简单格式      │ ⭐⭐ 中等        │
├──────────────────────┼──────────────┼──────────────┼─────────────────┤
│ 自定义解析器          │ 任意类型      │ 特殊需求      │ ⭐⭐⭐⭐ 复杂   │
└──────────────────────┴──────────────┴──────────────┴─────────────────┘

💡 选择建议：
1. 只需要文本 → StrOutputParser
2. 简单JSON → JsonOutputParser
3. 复杂结构化数据 → PydanticOutputParser（推荐）
4. 简单列表数据 → 手动解析（split等）
5. 特殊格式 → 自定义解析器
6. CsvOutputParser 已在最新版移除，可用手动解析替代

✅ 最佳实践：
- 优先使用 PydanticOutputParser，类型安全且有验证
- 添加重试机制提高鲁棒性
- 在 prompt 中明确说明期望的输出格式
- 使用 get_format_instructions() 提供格式指导
""")

print("\n" + "="*70)
print("演示完成！")
print("="*70)
