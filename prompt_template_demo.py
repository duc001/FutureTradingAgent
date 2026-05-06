from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# from langchain_core.prompts import ChatMessagePromptTemplate  # 已弃用，不推荐使用
# 1. PromptTemplate模板
prompt = PromptTemplate.from_template("你是一个{name}， 帮我起一个具有{country}特色{sex}名字")
#print(prompt.format(name="算命大师", country="中国", sex="男性"))

# 2. ChatPromptTemplate模板
chart_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个算命大师，你的名字叫{name}"),
        ("human", "你好{name}"),
        ("ai", "请问你有事吗"),
        ("human", "请帮我起一个具有{country}特色{sex}名字")
    ]
)
#print(chart_prompt.format(name="陈大师", country="中国", sex="男性"))

# 3. 混合使用：消息对象 + 模板字符串
# 注意：只有字符串/元组格式支持变量替换，消息对象是静态的
chart_prompt2 = ChatPromptTemplate.from_messages([
    ("system", "你是一个算命大师，你的名字叫{name}"),  # ✅ 支持变量
    HumanMessage(content="请帮我起个名字"),  # ❌ 静态内容，不支持变量
    AIMessage(content="好的，请问你想要什么风格的名字？"),  # ❌ 静态内容
    ("human", "请帮我起一个具有{country}特色{sex}名字")  # ✅ 支持变量
])

# 格式化消息（只替换模板中的变量）
prompt3 = chart_prompt2.format_messages(name="陈大师", country="中国", sex="男性")

print("格式化后的消息列表:")
for i, msg in enumerate(prompt3):
    print(f"[{i}] {type(msg).__name__}: {msg.content}")

# ==================== 4. ChatMessagePromptTemplate 示例（已弃用，仅供参考）====================
# 注意：这个类已经弃用，推荐使用 ChatPromptTemplate
print("\n" + "="*60)
print("ChatMessagePromptTemplate 示例（旧版API，不推荐）")
print("="*60)

try:
    from langchain_core.prompts import ChatMessagePromptTemplate
    
    # 创建单个消息模板
    human_template = ChatMessagePromptTemplate.from_template(
        role="human",
        template="请帮我起一个具有{country}特色的{sex}名字"
    )
    
    # 格式化
    formatted_msg = human_template.format(country="中国", sex="男性")
    print(f"格式化结果: {formatted_msg.content}")
    print(f"消息类型: {type(formatted_msg).__name__}")
except ImportError:
    print("ChatMessagePromptTemplate 在当前版本中不可用（已弃用）")
    print("请使用 ChatPromptTemplate 替代")