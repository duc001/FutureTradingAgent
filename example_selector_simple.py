"""
Example Selector 快速入门示例
简单易懂的 Few-Shot Learning 演示
"""
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_openai import ChatOpenAI
import os

print("="*60)
print("Example Selector 快速入门")
print("="*60)

# ==================== 第1步: 准备示例数据 ====================
examples = [
    {"word": "happy", "translation": "开心的"},
    {"word": "sad", "translation": "悲伤的"},
    {"word": "angry", "translation": "愤怒的"},
    {"word": "excited", "translation": "兴奋的"},
    {"word": "tired", "translation": "疲惫的"},
]

# ==================== 第2步: 定义示例模板 ====================
example_template = """单词: {word}
翻译: {translation}"""

example_prompt = PromptTemplate.from_template(example_template)

# ==================== 第3步: 创建示例选择器 ====================
selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=150,  # 限制总长度
)

# ==================== 第4步: 构建 Few-Shot Prompt ====================
few_shot_prompt = FewShotPromptTemplate(
    example_selector=selector,
    example_prompt=example_prompt,
    prefix="请将英文单词翻译成中文:",
    suffix="单词: {word}\n翻译:",
    input_variables=["word"],
)

# ==================== 第5步: 查看生成的提示词 ====================
test_word = "delighted"
prompt = few_shot_prompt.format(word=test_word)

print(f"\n测试单词: {test_word}")
print("\n生成的提示词:")
print("-"*60)
print(prompt)

# ==================== 第6步: 调用 LLM ====================
print("\n" + "="*60)
print("调用 LLM 获取翻译结果")
print("="*60)

try:
    llm = ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com",
        temperature=0,
    )
    
    response = llm.invoke(prompt)
    print(f"\n{test_word} -> {response.content.strip()}")
    
except Exception as e:
    print(f"\n⚠️ 调用失败: {e}")


# ==================== 对比: 不使用示例选择器 ====================
print("\n\n" + "="*60)
print("对比: 手动构建 Few-Shot Prompt（不使用选择器）")
print("="*60)

manual_prompt = """请将英文单词翻译成中文:

单词: happy
翻译: 开心的

单词: sad
翻译: 悲伤的

单词: angry
翻译: 愤怒的

单词: delighted
翻译:"""

print(manual_prompt)
print("\n❌ 缺点:")
print("  - 所有示例都包含，浪费Token")
print("  - 无法根据输入动态调整")
print("  - 示例过多可能超出上下文限制")

print("\n✅ 使用 Example Selector 的优点:")
print("  - 智能选择最相关的示例")
print("  - 控制Token用量")
print("  - 自动适应不同长度的输入")
