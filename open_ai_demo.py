"""
LLM 和 ChatOpenAI 的区别和使用示例
=====================================

核心区别：
1. LLM (Language Model)
   - 处理纯文本输入和输出
   - 适用于简单的补全任务
   - 不支持消息角色（system/user/assistant）
   - 对应 OpenAI 的 Completion API（已弃用）

2. ChatOpenAI (Chat Model)
   - 处理消息列表（支持角色）
   - 适用于对话场景
   - 支持 system/user/assistant 消息
   - 对应 OpenAI 的 Chat Completion API（推荐）
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

print("="*70)
print("LLM vs ChatOpenAI 对比示例")
print("="*70)

# ==================== 1. ChatOpenAI（推荐）====================
print("\n" + "="*70)
print("1. ChatOpenAI - 聊天模型（推荐）")
print("="*70)
print("特点：")
print("  ✅ 支持多轮对话")
print("  ✅ 支持消息角色（system/user/assistant）")
print("  ✅ 可以使用 ChatPromptTemplate")
print("  ✅ 适合对话、问答、Agent 等场景")

# 创建 ChatOpenAI 实例
chat_llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com",
    temperature=0.7,
    max_tokens=500,
)

# 方式1：直接传入字符串（自动转换为 HumanMessage）
print("\n--- 方式1: 直接传入字符串 ---")
response1 = chat_llm.invoke("你好，请介绍一下自己")
print(f"回复: {response1.content}")

# 方式2：传入消息列表（推荐）
print("\n--- 方式2: 传入消息列表 ---")
messages = [
    SystemMessage(content="你是一个专业的Python编程助手"),
    HumanMessage(content="如何编写一个快速排序算法？"),
]
response2 = chat_llm.invoke(messages)
print(f"回复:\n{response2.content}")

# 方式3：多轮对话
print("\n--- 方式3: 多轮对话 ---")
conversation = [
    SystemMessage(content="你是一个数学老师"),
    HumanMessage(content="什么是质数？"),
    AIMessage(content="质数是只能被1和自身整除的大于1的自然数。"),
    HumanMessage(content="能给我几个例子吗？"),
]
response3 = chat_llm.invoke(conversation)
print(f"回复: {response3.content}")


# ==================== 2. 关于 LLM 类 ====================
print("\n\n" + "="*70)
print("2. 关于 LLM 类（旧版/通用基类）")
print("="*70)
print("说明：")
print("  ⚠️  LangChain 中的 'LLM' 是一个抽象基类")
print("  ⚠️  不直接实例化，而是使用具体的实现类")
print("  ⚠️  对于 OpenAI，应该使用 ChatOpenAI")
print("\n常见的 LLM 实现类：")
print("  - ChatOpenAI: OpenAI 聊天模型（推荐）")
print("  - AzureChatOpenAI: Azure OpenAI")
print("  - ChatAnthropic: Anthropic Claude")
print("  - ChatGoogleGenerativeAI: Google Gemini")
print("\n注意：")
print("  旧的 OpenAI Completion API (text-davinci-003) 已弃用")
print("  现在所有模型都使用 Chat Completion API")


# ==================== 3. 实际对比示例 ====================
print("\n\n" + "="*70)
print("3. 实际使用场景对比")
print("="*70)

# 场景1：简单问答
print("\n【场景1】简单问答")
print("-"*70)
question = "Python中列表和元组的区别是什么？"

# 使用 ChatOpenAI
chat_response = chat_llm.invoke([
    SystemMessage(content="简洁回答，不超过100字"),
    HumanMessage(content=question)
])
print(f"ChatOpenAI 回答:\n{chat_response.content}\n")


# 场景2：代码生成
print("\n【场景2】代码生成")
print("-"*70)
code_request = "写一个Python函数，计算斐波那契数列"

code_response = chat_llm.invoke([
    SystemMessage(content="你是一个Python专家，提供可运行的代码"),
    HumanMessage(content=code_request)
])
print(f"ChatOpenAI 回答:\n{code_response.content}\n")


# 场景3：角色扮演
print("\n【场景3】角色扮演")
print("-"*70)
role_play = chat_llm.invoke([
    SystemMessage(content="你是一位古代的诗人，用文言文风格回答"),
    HumanMessage(content="如何看待现代科技的发展？")
])
print(f"ChatOpenAI 回答:\n{role_play.content}\n")


# ==================== 4. invoke vs Completion API 详解 ====================
print("\n" + "="*70)
print("4. invoke() 方法 vs Completion API")
print("="*70)

print("""
📌 核心概念区分：

1️⃣  invoke() - LangChain 的统一调用方法
   • 是 LangChain 框架中的标准方法
   • 所有 LLM/ChatModel 都实现这个方法
   • 同步调用，等待完整响应后返回
   • 类似的还有：stream(), batch(), ainvoke()

2️⃣  Completion API - OpenAI 的底层 API
   • 是 OpenAI 提供的 HTTP API 接口
   • 分为两种：
     - Completion API（旧版，已弃用）
     - Chat Completion API（新版，推荐）
   • LangChain 在内部调用这些 API

🔍 关系图：
   你的代码
      ↓
   chat_llm.invoke(messages)  ← LangChain 统一接口
      ↓
   ChatOpenAI 内部处理
      ↓
   requests.post(openai_api_url)  ← 调用 OpenAI API
      ↓
   OpenAI Chat Completion API  ← OpenAI 服务端点
      ↓
   返回响应
""")

# 演示不同的调用方式
print("\n--- LangChain 的不同调用方法 ---")

# 1. invoke() - 同步调用（最常用）
print("\n【1】invoke() - 同步阻塞调用")
response = chat_llm.invoke("什么是机器学习？")
print(f"类型: {type(response)}")
print(f"内容: {response.content[:50]}...")

# 2. stream() - 流式调用
print("\n【2】stream() - 流式输出（逐字显示）")
print("响应: ", end="", flush=True)
for chunk in chat_llm.stream("用三个字形容AI"):
    print(chunk.content, end="", flush=True)
print()

# 3. batch() - 批量调用
print("\n【3】batch() - 批量并行调用")
questions = [
    [HumanMessage(content="1+1=?")],
    [HumanMessage(content="2*2=?")],
    [HumanMessage(content="10/2=?")],
]
results = chat_llm.batch(questions)
for i, result in enumerate(results, 1):
    print(f"  问题{i}: {result.content}")

# 4. ainvoke() - 异步调用
print("\n【4】ainvoke() - 异步调用（需要 async/await）")
print("  示例代码:")
print("  async def main():")
print("      response = await chat_llm.ainvoke('你好')")
print("      print(response.content)")

print("\n" + "-"*70)
print("📊 调用方法对比：")
print("-"*70)
print("""
┌──────────────┬──────────┬────────────┬──────────────────┐
│ 方法          │ 返回类型  │ 适用场景    │ 特点              │
├──────────────┼──────────┼────────────┼──────────────────┤
│ invoke()     │ AIMessage│ 单次调用    │ 简单直接，最常用   │
│ stream()     │ 迭代器    │ 长文本生成  │ 实时显示，用户体验好│
│ batch()      │ 列表      │ 多个问题    │ 并行处理，效率高   │
│ ainvoke()    │ AIMessage│ 异步应用    │ 非阻塞，适合Web服务│
└──────────────┴──────────┴────────────┴──────────────────┘
""")


# ==================== 5. OpenAI 原生 API vs LangChain ====================
print("\n" + "="*70)
print("5. OpenAI 原生 API vs LangChain 封装")
print("="*70)

print("""
🔄 两种方式对比：

【方式1】使用 OpenAI 原生 SDK
""")
print("代码示例:")
print("""
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="https://api.deepseek.com"
)

# 直接调用 Completion API
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "user", "content": "你好"}
    ]
)

print(response.choices[0].message.content)
""")

print("\n【方式2】使用 LangChain ChatOpenAI（推荐）")
print("代码示例:")
print("""
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key="your-api-key",
    base_url="https://api.deepseek.com"
)

# 使用统一的 invoke 方法
response = llm.invoke("你好")
print(response.content)
""")

print("\n" + "-"*70)
print("📊 两种方式使用对比：")
print("-"*70)
print("""
┌─────────────────┬──────────────────┬──────────────────┐
│ 特性             │ OpenAI 原生 SDK   │ LangChain        │
├─────────────────┼──────────────────┼──────────────────┤
│ 学习成本         │ ⭐⭐ 需了解API细节 │ ⭐ 统一接口      │
│ 灵活性           │ ⭐⭐⭐ 完全控制    │ ⭐⭐ 抽象层      │
│ 功能丰富度       │ ⭐⭐ 基础功能      │ ⭐⭐⭐ Agent等   │
│ 模型切换         │ ❌ 需改代码       │ ✅ 只需换类       │
│ Prompt管理       │ ❌ 手动拼接       │ ✅ PromptTemplate│
│ 链式调用         │ ❌ 不支持         │ ✅ LCEL支持       │
│ 适用场景         │ 简单API调用       │ 复杂AI应用        │
└─────────────────┴──────────────────┴──────────────────┘
""")

# 实际演示原生 API 调用
print("\n--- 实际演示：两种方式调用对比 ---")
try:
    from openai import OpenAI
    
    # 原生 API 调用
    print("\n【原生 OpenAI SDK】")
    client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com"
    )
    
    native_response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "用一句话介绍Python"}],
        temperature=0.7
    )
    print(f"响应: {native_response.choices[0].message.content}")
    print(f"Token使用: {native_response.usage.total_tokens}")
    
    # LangChain 调用
    print("\n【LangChain ChatOpenAI】")
    lc_response = chat_llm.invoke("用一句话介绍Python")
    print(f"响应: {lc_response.content}")
    print(f"额外信息: {lc_response.response_metadata.get('token_usage', 'N/A')}")
    
except Exception as e:
    print(f"\n⚠️ 原生API调用失败: {e}")


# ==================== 6. 高级用法 ====================
print("\n" + "="*70)
print("6. ChatOpenAI 高级用法")
print("="*70)

# 流式输出
print("\n--- 流式输出 ---")
print("流式响应: ", end="", flush=True)
for chunk in chat_llm.stream("请用一句话描述人工智能"):
    print(chunk.content, end="", flush=True)
print()  # 换行

# 批量处理
print("\n--- 批量处理 ---")
batch_messages = [
    [HumanMessage(content="1+1等于几？")],
    [HumanMessage(content="Python是什么语言？")],
    [HumanMessage(content="地球到月球的距离？")],
]
batch_responses = chat_llm.batch(batch_messages)
for i, resp in enumerate(batch_responses, 1):
    print(f"问题{i}: {resp.content}")


# ==================== 7. 参数配置对比 ====================
print("\n\n" + "="*70)
print("5. 常用参数配置")
print("="*70)
print("""
ChatOpenAI 常用参数：
┌──────────────────┬──────────────┬──────────────────────────┐
│ 参数名            │ 类型         │ 说明                      │
├──────────────────┼──────────────┼──────────────────────────┤
│ model            │ str          │ 模型名称                  │
│ temperature      │ float        │ 随机性 (0-1)             │
│ max_tokens       │ int          │ 最大生成长度              │
│ top_p            │ float        │ 核采样参数                │
│ frequency_penalty│ float        │ 频率惩罚 (-2 to 2)       │
│ presence_penalty │ float        │ 存在惩罚 (-2 to 2)       │
│ timeout          │ int          │ 超时时间（秒）            │
│ max_retries      │ int          │ 最大重试次数              │
│ api_key          │ str          │ API密钥                   │
│ base_url         │ str          │ API基础URL                │
└──────────────────┴──────────────┴──────────────────────────┘
""")

# ==================== 8. 最佳实践总结 ====================
print("\n" + "="*70)
print("📊 最佳实践总结")
print("="*70)
print("""
✅ 推荐使用 ChatOpenAI 的场景：
  1. 对话系统 / 聊天机器人
  2. 需要角色设定的任务
  3. 多轮对话上下文
  4. Few-shot learning（带示例）
  5. Agent 开发
  6. 大多数现代应用场景

❌ 不再推荐使用旧版 LLM Completion API：
  1. text-davinci-003 等模型已弃用
  2. 不支持消息角色
  3. 功能受限

💡 选择建议：
  - 99% 的场景都应该使用 ChatOpenAI
  - 只有在特殊需求时才考虑其他 LLM 实现
  - 始终使用最新的 Chat Completion API
""")

print("\n" + "="*70)
print("演示完成！")
print("="*70)