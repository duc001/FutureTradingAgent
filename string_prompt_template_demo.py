"""
展示 StringPromptTemplate 的正确用法
对比：普通类 vs 继承 StringPromptTemplate
"""
from langchain_core.prompts import StringPromptTemplate
from langchain_openai import ChatOpenAI
import os
import inspect


# ==================== 示例函数 ====================
def hello_word(abc):
    """简单的测试函数"""
    print("this is hello world")
    return abc


PROMPT_TEMPLATE = """
你是一个非常有经验和天赋的程序员，现在给你如下函数名称，你会按照如下格式，输出这段代码的名称、源代码、中文解释。
函数名称：{function_name}
源代码：{source_code}
中文解释：{code_explanation}
"""


def get_function_source(func):
    """获取函数源代码"""
    return inspect.getsource(func)


# ==================== 方式1：普通类（不推荐）====================
print("="*60)
print("方式1: 普通 Python 类")
print("="*60)

class SimplePrompt:
    """普通的 Python 类，没有继承 LangChain 基类"""
    
    def __init__(self):
        pass
    
    def generate(self, func):
        """生成提示词"""
        source_code = get_function_source(func)
        prompt = PROMPT_TEMPLATE.format(
            function_name=func.__name__,
            source_code=source_code,
            code_explanation="这是一个测试函数"
        )
        return prompt

# 使用
simple_prompt = SimplePrompt()
pm1 = simple_prompt.generate(hello_word)
print(pm1)

# ❌ 问题：不能直接与 LangChain 组件集成
# llm.invoke(simple_prompt)  # 这会报错！


# ==================== 方式2：继承 StringPromptTemplate（推荐）====================
print("\n\n" + "="*60)
print("方式2: 继承 StringPromptTemplate")
print("="*60)

class CustomPrompt(StringPromptTemplate):
    """
    自定义 Prompt 模板，继承 LangChain 基类
    
    优势：
    1. 可以与 LLM、Chain、Agent 无缝集成
    2. 支持 .invoke()、.batch() 等标准接口
    3. 符合 LangChain 的 Runnable 协议
    """
    
    def format(self, **kwargs) -> str:
        """
        重写 format 方法，实现自定义逻辑
        
        Args:
            **kwargs: 包含 function_name 等变量
            
        Returns:
            str: 格式化后的提示词字符串
        """
        func = kwargs["function_name"]
        source_code = get_function_source(func)
        
        prompt = PROMPT_TEMPLATE.format(
            function_name=func.__name__,
            source_code=source_code,
            code_explanation="这是一个测试函数，用于演示 StringPromptTemplate 的用法"
        )
        return prompt


# 创建实例（必须指定 input_variables）
custom_prompt = CustomPrompt(input_variables=["function_name"])

# ✅ 方式 A：直接调用 format（和普通类一样）
pm2 = custom_prompt.format(function_name=hello_word)
print("直接调用 format():")
print(pm2)

# ✅ 方式 B：使用 invoke（LangChain 标准接口）
print("\n使用 invoke() 方法:")
pm3 = custom_prompt.invoke({"function_name": hello_word})
print(pm3)

# ✅ 方式 C：与 LLM 直接集成（这才是真正的价值！）
print("\n\n" + "="*60)
print("方式2的优势：与 LLM 无缝集成")
print("="*60)

try:
    llm = ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com",
        temperature=0,
        extra_body={"thinking": {"type": "disabled"}}
    )
    
    # 🎯 关键：可以直接将 CustomPrompt 传给 LLM！
    print("\n调用 LLM...")
    response = llm.invoke(custom_prompt.invoke({"function_name": hello_word}))
    
    print("\n✅ AI 回答:")
    print("-"*60)
    print(response.content)
    
except Exception as e:
    print(f"\n⚠️  LLM 调用失败（可能未配置 API Key）: {e}")


# ==================== 总结对比 ====================
print("\n\n" + "="*60)
print("📊 两种方式对比")
print("="*60)
print("""
┌─────────────────┬──────────────────────┬──────────────────────┐
│     特性         │   普通类              │  StringPromptTemplate│
├─────────────────┼──────────────────────┼──────────────────────┤
│ 实现难度         │ ⭐ 简单               │ ⭐⭐ 需要理解基类     │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 独立性           │ ✅ 完全独立           │ ❌ 依赖 LangChain    │
├─────────────────┼──────────────────────┼──────────────────────┤
│ LangChain 集成   │ ❌ 不能直接使用       │ ✅ 完美集成          │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 标准接口         │ ❌ 自定义方法名       │ ✅ invoke/batch等    │
├─────────────────┼──────────────────────┼──────────────────────┤
│ Chain/Agent      │ ❌ 需要额外转换       │ ✅ 直接使用          │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 适用场景         │ 独立脚本、简单应用    │ LangChain 生态系统   │
└─────────────────┴──────────────────────┴──────────────────────┘

💡 结论：
   • 如果只是生成字符串 → 普通类就够了
   • 如果要构建 LangChain 应用 → 必须继承 StringPromptTemplate
   • 本示例中，方式2的真正价值体现在与 LLM 的集成上
""")
