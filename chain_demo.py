"""
LangChain Chain 完整教程 - LCEL 语法详解
=========================================
本文件展示了 LangChain Expression Language (LCEL) 的所有核心用法
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda, RunnableBranch

# ============================================================
# 基础配置
# ============================================================
llm = ChatOpenAI(model="deepseek-v4-flash", api_key="sk-205ee372c5df491f8050324eb697e504", base_url="https://api.deepseek.com")
parser = StrOutputParser()

# ============================================================
# 模式 1：最基础的链 - prompt | llm | parser
# ============================================================
def demo_basic_chain():
    """最简单的三步链：提示词 -> LLM -> 输出解析"""
    prompt = ChatPromptTemplate.from_template("把以下中文翻译成英文：{text}")
    chain = prompt | llm | parser

    result = chain.invoke({"text": "你好，世界"})
    print("模式1结果:", result)

# ============================================================
# 模式 2：保留输入并添加字段 - RunnablePassthrough.assign()
# ============================================================
def demo_passthrough_assign():
    """
    场景：翻译一篇文章，同时生成摘要
    技巧：RunnablePassthrough.assign() 保留原始输入，添加新字段
    """
    translate_prompt = ChatPromptTemplate.from_template(
        "翻译成英文：{text}"
    )
    summary_prompt = ChatPromptTemplate.from_template(
        "用一句话总结：{text}"
    )
    combine_prompt = ChatPromptTemplate.from_template(
        "原文：{text}\n翻译：{translation}\n摘要：{summary}"
    )

    # 翻译链
    translate_chain = translate_prompt | llm | parser
    # 摘要链
    summary_chain = summary_prompt | llm | parser

    # 组合：保留 text，输入 text 生成 translation 和 summary，再合并
    chain = RunnablePassthrough.assign(
        translation=translate_chain,
        summary=summary_chain
    ) | combine_prompt | llm | parser

    result = chain.invoke({"text": "人工智能正在改变我们的生活方式"})
    print("模式2结果:", result)

# ============================================================
# 模式 3：多步链 - 用 lambda 转换输出
# ============================================================
def demo_multi_step_chain():
    """
    场景：先生成大纲，再根据大纲写文章
    技巧：用 lambda 将前一步输出转换为下一步需要的输入格式
    """
    outline_prompt = ChatPromptTemplate.from_template(
        "为以下主题生成3点大纲：{topic}"
    )
    article_prompt = ChatPromptTemplate.from_template(
        "根据以下大纲写一篇文章：\n{outline}"
    )

    # 生成大纲 -> 转换格式 -> 写文章
    chain = (
        outline_prompt
        | llm
        | parser
        | (lambda outline: {"outline": outline})  # 转换为 dict 格式
        | article_prompt
        | llm
        | parser
    )

    result = chain.invoke({"topic": "量子计算的未来"})
    print("模式3结果:", result)

# ============================================================
# 模式 4：并行链 - RunnableParallel
# ============================================================
def demo_parallel_chain():
    """
    场景：同时翻译成多种语言并统计字符数
    技巧：RunnableParallel 并行执行多个链，结果合并为字典
    """
    en_prompt = ChatPromptTemplate.from_template("翻译成英文：{text}")
    jp_prompt = ChatPromptTemplate.from_template("翻译成日文：{text}")
    kr_prompt = ChatPromptTemplate.from_template("翻译成韩文：{text}")

    en_chain = en_prompt | llm | parser
    jp_chain = jp_prompt | llm | parser
    kr_chain = kr_prompt | llm | parser

    # 并行执行
    parallel_chain = RunnableParallel(
        english=en_chain,
        japanese=jp_chain,
        korean=kr_chain,
        char_count=RunnableLambda(lambda x: len(x["text"]))
    )

    result = parallel_chain.invoke({"text": "你好"})
    print("模式4结果:", result)
    # {'english': 'Hello', 'japanese': 'こんにちは', 'korean': '안녕하세요', 'char_count': 2}

# ============================================================
# 模式 5：链的组合复用 - 给链起名字
# ============================================================
def demo_chain_composition():
    """
    技巧：把常用的链封装成函数，方便复用
    """
    def make_translate_chain(target_lang: str):
        """工厂函数：创建翻译链"""
        prompt = ChatPromptTemplate.from_template(
            f"把以下中文翻译成{target_lang}：{{text}}"
        )
        return prompt | llm | parser

    def make_summary_chain(lang: str = "中文"):
        """工厂函数：创建摘要链"""
        prompt = ChatPromptTemplate.from_template(
            f"用{lang}总结要点（不超过50字）：{{text}}"
        )
        return prompt | llm | parser

    # 创建具体链
    translate_en = make_translate_chain("英语")
    translate_jp = make_translate_chain("日语")
    summarize = make_summary_chain()

    # 组合成完整工作流
    workflow = (
        RunnableParallel(
            translation_en=translate_en,
            translation_jp=translate_jp,
        )
        | RunnableLambda(lambda x: {
            "text": x["translation_en"],
            "original": "你好世界"
        })
        | make_summary_chain("英文")
    )

    result = workflow.invoke({"text": "你好世界"})
    print("模式5结果:", result)

# ============================================================
# 模式 6：条件路由 - RunnableBranch
# ============================================================
def demo_branch():
    """
    场景：根据语言选择不同的处理流程
    技巧：RunnableBranch 根据条件选择不同分支
    """
    detect_prompt = ChatPromptTemplate.from_template(
        "判断以下文本的语言，只输出语言名称（如：中文、英文、日文）：{text}"
    )
    en_prompt = ChatPromptTemplate.from_template(
        "分析这篇英文文章的核心观点：{text}"
    )
    zh_prompt = ChatPromptTemplate.from_template(
        "分析这篇中文文章的核心观点：{text}"
    )
    other_prompt = ChatPromptTemplate.from_template(
        "用中文总结：{text}"
    )

    # 检测语言
    detect_chain = detect_prompt | llm | parser

    # 条件分支
    branch = RunnableBranch(
        (lambda x: "中文" in x["lang"], zh_prompt | llm | parser),
        (lambda x: "英文" in x["lang"], en_prompt | llm | parser),
        other_prompt | llm | parser,  # 默认分支
    )

    # 完整链：检测语言 -> 根据语言路由 -> 执行对应链
    chain = (
        {"text": lambda x: x["text"], "lang": detect_chain}
        | branch
    )

    result = chain.invoke({"text": "机器学习是人工智能的分支"})
    print("模式6结果:", result)

# ============================================================
# 模式 7：实用链 - 代码审查工具
# ============================================================
def demo_code_review_chain():
    """
    完整示例：代码审查工具
    1. 并行：原始代码 + 代码审查
    2. 合并：审查结果 + 原代码 -> 生成修复建议
    """
    review_prompt = ChatPromptTemplate.from_template(
        "审查以下Python代码，指出问题并给出改进建议：\n```{code}```"
    )
    fix_prompt = ChatPromptTemplate.from_template(
        "原始代码：\n{code}\n\n审查意见：\n{review}\n\n根据审查意见生成修复后的代码："
    )

    # 代码审查链
    review_chain = review_prompt | llm | parser

    # 组合：原始代码透传 + 并行审查 -> 合并生成修复
    chain = RunnableParallel(
        code=RunnablePassthrough(),  # 透传原始代码
        review=review_chain,
    ) | fix_prompt | llm | parser

    code = '''
def calculate(x,y):
    result = x+y
    print result
    return result
'''

    result = chain.invoke({"code": code})
    print("模式7结果:", result)

# ============================================================
# 实用技巧：调试、批处理、流式输出
# ============================================================
def demo_debug_and_batch():
    """调试方法和批处理"""
    prompt = ChatPromptTemplate.from_template("把 {word} 翻译成英文")
    chain = prompt | llm | parser

    # 1. 调试：查看中间步骤
    print("--- 调试模式 ---")
    from langchain_core.tracers import ConsoleCallbackHandler

    # 单次调用带回调
    chain.invoke({"word": "学习"}, config={"callbacks": [ConsoleCallbackHandler()]})

    # 2. 批处理：一次处理多个输入
    print("\n--- 批处理模式 ---")
    inputs = [{"word": "你好"}, {"word": "再见"}, {"word": "谢谢"}]
    results = chain.batch(inputs)
    for r in results:
        print(r)

    # 3. 流式输出：逐词展示
    print("\n--- 流式输出 ---")
    for token in chain.stream({"word": "人工智能"}):
        print(token, end="", flush=True)
    print()

# ============================================================
# 运行所有示例
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("LangChain Chain 完整示例")
    print("=" * 50)

    # 取消注释运行对应示例
    demo_basic_chain()
    demo_passthrough_assign()
    demo_multi_step_chain()
    demo_parallel_chain()
    demo_chain_composition()
    demo_branch()
    demo_code_review_chain()
    demo_debug_and_batch()
