from langchain_core.prompts import StringPromptTemplate
from langchain_openai import ChatOpenAI
import os
import inspect


# 简单的helloworld函数
def hello_word(abc):
    print("this is hello world")
    return abc

PROMPT = """
你是一个非常有经验和天赋的程序员，现在给你如下函数名称，你会按照如下格式，输出这段代码的名称,源代码，中文解释。
函数名称：{function_name}
源代码：{source_code}
中文解释：{chinese_explain}"""

import inspect
def get_function_info(function_name):
    return inspect.getsource(function_name)

class CustomPrompt(StringPromptTemplate):
    def format(self, **kwargs) -> str:
        source_code = get_function_info(kwargs["function_name"])
        prompt = PROMPT.format(
            function_name=kwargs["function_name"].__name__,
            source_code=source_code,
            chinese_explain=hello_word(kwargs["function_name"])
        )
        return prompt

a = CustomPrompt(input_variables=["function_name"])
pm = a.format(function_name=hello_word)
print(pm)

# ==================== 方案1：使用原生 OpenAI SDK ====================
print("\n" + "="*60)
print("方案1: 使用原生 OpenAI SDK")
print("="*60)

from openai import OpenAI

# 创建客户端（注意：OpenAI 客户端不接受 temperature 等参数）
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com",
)

# 调用 API（temperature、model 等参数在 create() 中指定）
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "user", "content": pm}
    ],
    temperature=0,
    timeout=120,
    extra_body={"thinking": {"type": "disabled"}}
)

print("\n✅ OpenAI SDK 回答:")
print("-"*60)
print(response.choices[0].message.content)

# ==================== 方案2：使用 LangChain ChatOpenAI ====================
print("\n\n" + "="*60)
print("方案2: 使用 LangChain ChatOpenAI")
print("="*60)

# 使用 LangChain 的 ChatOpenAI（推荐）
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com",
    temperature=0,
    timeout=120,
    max_retries=3,
    extra_body={"thinking": {"type": "disabled"}}
)

# 调用模型
response2 = llm.invoke(pm)
print("\n✅ LangChain ChatOpenAI 回答:")
print("-"*60)
print(response2.content)

# ==================== 两种方法对比总结 ====================
print("\n\n" + "="*60)
print("📊 两种方法对比")
print("="*60)
print("""
┌─────────────────┬──────────────────────┬──────────────────────┐
│     特性         │   OpenAI SDK         │  LangChain           │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 导入方式         │ from openai          │ from langchain_      │
│                 │   import OpenAI      │   openai import      │
│                 │                      │   ChatOpenAI         │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 初始化参数       │ api_key, base_url    │ model, api_key,      │
│                 │                      │ base_url,            │
│                 │                      │ temperature...       │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 调用方法         │ client.chat.         │ llm.invoke(prompt)   │
│                 │ completions.create() │                      │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 消息格式         │ 手动构建 messages    │ 自动处理字符串/      │
│                 │ 列表                 │ 消息对象             │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 返回值提取       │ response.choices[0]  │ response.content     │
│                 │ .message.content     │                      │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 适用场景         │ 简单API调用          │ 复杂应用、Agent、    │
│                 │                      │ Chain、记忆管理      │
└─────────────────┴──────────────────────┴──────────────────────┘

💡 建议：
   • 快速原型/简单调用 → 使用 OpenAI SDK
   • 生产环境/复杂应用 → 使用 LangChain
   • 需要与 Prompt/Agent 集成 → 必须用 LangChain
""")
