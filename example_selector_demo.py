"""
LangChain Example Selector 完整示例
展示各种示例选择器的使用方法
"""
import os
# 设置 tiktoken 缓存目录，避免重复下载
os.environ.setdefault('TIKTOKEN_CACHE_DIR', './tiktoken_cache')

from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import (
    LengthBasedExampleSelector,
    MaxMarginalRelevanceExampleSelector,
)
from langchain_openai import ChatOpenAI

# ==================== 定义示例数据集 ====================
print("="*70)
print("LangChain Example Selector 完整示例")
print("="*70)

# 定义一些翻译示例
examples = [
    {"input": "happy", "output": "开心的"},
    {"input": "sad", "output": "悲伤的"},
    {"input": "excited", "output": "兴奋的"},
    {"input": "angry", "output": "愤怒的"},
    {"input": "tired", "output": "疲惫的"},
    {"input": "confused", "output": "困惑的"},
    {"input": "surprised", "output": "惊讶的"},
    {"input": "disappointed", "output": "失望的"},
    {"input": "grateful", "output": "感激的"},
    {"input": "anxious", "output": "焦虑的"},
    {"input": "proud", "output": "自豪的"},
    {"input": "jealous", "output": "嫉妒的"},
]

# 定义示例模板
example_template = """
英文: {input}
中文: {output}
"""

example_prompt = PromptTemplate.from_template(example_template)


# ==================== 1. LengthBasedExampleSelector ====================
print("\n" + "="*70)
print("1. LengthBasedExampleSelector - 基于长度选择")
print("="*70)
print("说明: 根据提示词的总长度限制，动态选择尽可能多的示例")

length_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=200,  # 最大长度（字符数）
)

# 测试不同输入
test_inputs = ["happy", "I am feeling very happy and excited today"]

for test_input in test_inputs:
    selected_examples = length_selector.select_examples({"input": test_input})
    print(f"\n输入: '{test_input}'")
    print(f"选择的示例数量: {len(selected_examples)}")
    for i, ex in enumerate(selected_examples, 1):
        print(f"  示例{i}: {ex['input']} -> {ex['output']}")


# ==================== 2. SemanticSimilarityExampleSelector ====================
print("\n\n" + "="*70)
print("2. SemanticSimilarityExampleSelector - 基于语义相似度")
print("="*70)
print("说明: 使用向量嵌入计算相似度，选择最相关的示例")
print("注意: 需要安装 chromadb 或 faiss-cpu")

try:
    from langchain_core.example_selectors import SemanticSimilarityExampleSelector
    from langchain_openai import OpenAIEmbeddings
    
    # 尝试使用 Chroma 作为向量存储
    try:
        from langchain_chroma import Chroma
        vectorstore_cls = Chroma
        print("✅ 使用 Chroma 向量存储")
    except ImportError:
        try:
            from langchain_community.vectorstores import FAISS
            vectorstore_cls = FAISS
            print("✅ 使用 FAISS 向量存储")
        except ImportError:
            print("⚠️ 未找到向量存储库，跳过此示例")
            print("安装方法: pip install langchain-chroma 或 pip install faiss-cpu")
            raise ImportError("需要安装向量存储库")
    
    # 创建语义相似度选择器
    semantic_selector = SemanticSimilarityExampleSelector.from_examples(
        examples=examples,
        embeddings=OpenAIEmbeddings(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com/v1"
        ),
        vectorstore_cls=vectorstore_cls,  # 必需参数：向量存储类
        k=3,  # 选择最相似的3个示例
    )
    
    # 测试
    test_word = "joyful"
    selected = semantic_selector.select_examples({"input": test_word})
    print(f"\n输入: '{test_word}'")
    print(f"选择的最相似示例:")
    for i, ex in enumerate(selected, 1):
        print(f"  示例{i}: {ex['input']} -> {ex['output']}")
        
except (ImportError, Exception) as e:
    print(f"\n⚠️ 跳过 SemanticSimilarityExampleSelector 示例")
    print(f"原因: {str(e)[:100]}")
    print("\n💡 解决方案:")
    print("   1. 运行 download_tiktoken.py 下载编码文件")
    print("   2. 使用代理或稳定网络")
    print("   3. 继续使用其他示例选择器")


# ==================== 3. MaxMarginalRelevanceExampleSelector ====================
print("\n\n" + "="*70)
print("3. MaxMarginalRelevanceExampleSelector - MMR选择")
print("="*70)
print("说明: 平衡相关性和多样性，避免选择过于相似的示例")

try:
    from langchain_openai import OpenAIEmbeddings
    
    # 尝试使用 Chroma 作为向量存储
    try:
        from langchain_chroma import Chroma
        vectorstore_cls = Chroma
    except ImportError:
        try:
            from langchain_community.vectorstores import FAISS
            vectorstore_cls = FAISS
        except ImportError:
            print("⚠️ 未找到向量存储库，跳过此示例")
            raise ImportError("需要安装向量存储库")
    
    mmr_selector = MaxMarginalRelevanceExampleSelector.from_examples(
        examples=examples,
        embeddings=OpenAIEmbeddings(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com/v1"
        ),
        vectorstore_cls=vectorstore_cls,  # 必需参数：向量存储类
        k=4,  # 选择4个示例
        fetch_k=8,  # 先获取8个候选，再从中选择4个
    )
    
    test_word = "emotional"
    selected = mmr_selector.select_examples({"input": test_word})
    print(f"\n输入: '{test_word}'")
    print(f"选择的示例（兼顾相关性和多样性）:")
    for i, ex in enumerate(selected, 1):
        print(f"  示例{i}: {ex['input']} -> {ex['output']}")
        
except (ImportError, Exception) as e:
    print(f"\n⚠️ 跳过 MaxMarginalRelevanceExampleSelector 示例")
    print(f"原因: {str(e)[:100]}")


# ==================== 4. FixedExampleSelector (替代方案) ====================
print("\n\n" + "="*70)
print("4. FixedExampleSelector - 固定示例选择")
print("="*70)
print("说明: 始终选择固定的前N个示例，简单直接")

# 新版 LangChain 中 NGramOverlapExampleSelector 已移除
# 可以使用简单的切片方式实现类似功能
def fixed_example_selector(examples, k=3):
    """固定选择前k个示例"""
    return examples[:k]

test_word = "glad"
selected = fixed_example_selector(examples, k=3)
print(f"\n输入: '{test_word}'")
print(f"选择的固定示例:")
for i, ex in enumerate(selected, 1):
    print(f"  示例{i}: {ex['input']} -> {ex['output']}")

print("\n💡 提示: 如需更高级的选择器，可考虑:")
print("  - LengthBasedExampleSelector (基于长度)")
print("  - SemanticSimilarityExampleSelector (基于语义相似度)")
print("  - MaxMarginalRelevanceExampleSelector (MMR)")


# ==================== 5. 构建 Few-Shot Prompt ====================
print("\n\n" + "="*70)
print("5. 构建完整的 Few-Shot Prompt")
print("="*70)

# 使用长度选择器创建 few-shot 提示词
few_shot_prompt = FewShotPromptTemplate(
    example_selector=length_selector,
    example_prompt=example_prompt,
    prefix="你是一个英中翻译助手。请参考以下示例进行翻译:",
    suffix="英文: {input}\n中文:",
    input_variables=["input"],
)

# 测试
test_input = "delighted"
formatted_prompt = few_shot_prompt.format(input=test_input)
print(f"\n生成的完整提示词:")
print("-"*70)
print(formatted_prompt)


# ==================== 6. 实际调用 LLM ====================
print("\n\n" + "="*70)
print("6. 实际调用 LLM 进行翻译")
print("="*70)

try:
    llm = ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com",
        temperature=0,
    )
    
    test_words = ["thrilled", "melancholy", "furious"]
    
    for word in test_words:
        prompt = few_shot_prompt.format(input=word)
        response = llm.invoke(prompt)
        print(f"\n{word} -> {response.content.strip()}")
        
except Exception as e:
    print(f"\n⚠️ 调用失败: {e}")
    print("请确保设置了 DEEPSEEK_API_KEY 环境变量")


# ==================== 总结对比 ====================
print("\n\n" + "="*70)
print("📊 示例选择器对比总结")
print("="*70)
print("""
┌──────────────────────────────┬──────────────┬──────────┬───────────────┐
│ 选择器类型                    │ 是否需要Embed │ 速度     │ 适用场景       │
├──────────────────────────────┼──────────────┼──────────┼───────────────┤
│ LengthBased                  │ ❌           │ ⚡⚡⚡   │ 控制Token用量  │
├──────────────────────────────┼──────────────┼──────────┼───────────────┤
│ SemanticSimilarity           │ ✅           │ ⚡⚡     │ 精确匹配       │
├──────────────────────────────┼──────────────┼──────────┼───────────────┤
│ MaxMarginalRelevance         │ ✅           │ ⚡       │ 多样性+相关性  │
├──────────────────────────────┼──────────────┼──────────┼───────────────┤
│ Fixed (自定义)               │ ❌           │ ⚡⚡⚡   │ 简单场景       │
└──────────────────────────────┴──────────────┴──────────┴───────────────┘

💡 选择建议:
1. 预算有限/追求速度 → LengthBased 或 Fixed
2. 精度要求高 → SemanticSimilarity
3. 需要多样化示例 → MaxMarginalRelevance
4. NGramOverlap 已在最新版移除，可用其他选择器替代
""")
