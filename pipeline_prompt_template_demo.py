"""
Pipeline Prompt 示例
展示如何组合多个提示词模板，构建复杂的提示词流水线
注意：新版 LangChain 中 PipelinePromptTemplate 已被移除，改用手动组合
"""
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os


# ==================== 定义子模板 ====================
print("="*70)
print("PipelinePromptTemplate 示例")
print("="*70)

# 模板1：角色设定
role_template = PromptTemplate.from_template(
    "你是一个{role}专家。"
)

# 模板2：任务描述
task_template = PromptTemplate.from_template(
    "你的任务是{task}。"
)

# 模板3：输出格式要求
format_template = PromptTemplate.from_template(
    "请使用{format_style}格式输出，保持{tone}的语气。"
)

# 模板4：用户问题
question_template = PromptTemplate.from_template(
    "用户问题：{question}"
)


# ==================== 创建 Pipeline（手动组合）====================
print("\n1. 创建 Pipeline（手动组合方式）")
print("-"*70)

# 定义子模板字典
pipelines = {
    "role": role_template,
    "task": task_template,
    "format": format_template,
    "question": question_template,
}

# 定义组合函数
def build_prompt(pipeline_dict, **kwargs):
    """
    手动组合多个 Prompt 模板
    
    Args:
        pipeline_dict: 字典，key为名称，value为PromptTemplate对象
        **kwargs: 所有需要填充的变量
    
    Returns:
        str: 组合后的完整提示词
    """
    parts = []
    for name, template in pipeline_dict.items():
        # 提取该模板需要的变量
        template_vars = {k: v for k, v in kwargs.items() if k in template.input_variables}
        if template_vars:
            parts.append(template.format(**template_vars))
    return "\n".join(parts)

print(f"✅ Pipeline 创建成功")
print(f"   子模板数量: {len(pipelines)}")
print(f"   子模板列表: {', '.join(pipelines.keys())}")


# ==================== 使用 Pipeline ====================
print("\n2. 使用 Pipeline 生成提示词")
print("-"*70)

# 填充所有变量
result = build_prompt(
    pipelines,
    role="你是一个Python编程专家。",
    task="你的任务是解释代码并提供优化建议。",
    format_style="Markdown",
    tone="专业但易懂",
    question="请解释列表推导式的用法"
)

print("生成的完整提示词:")
print("="*70)
print(result)
print("="*70)


# ==================== 复用 Pipeline（不同场景）====================
print("\n3. 复用 Pipeline - 不同场景")
print("-"*70)

scenarios = [
    {
        "role": "数据分析师",
        "task": "分析销售数据趋势",
        "format_style": "表格",
        "tone": "严谨客观",
        "question": "Q3季度销售额下降的原因是什么？"
    },
    {
        "role": "翻译专家",
        "task": "将中文翻译成英文",
        "format_style": "对照文本",
        "tone": "自然流畅",
        "question": "请将'人工智能改变世界'翻译成英文"
    },
    {
        "role": "创意写作助手",
        "task": "创作短篇故事开头",
        "format_style": "叙事文体",
        "tone": "生动有趣",
        "question": "写一个关于时间旅行的故事开头"
    }
]

for i, scenario in enumerate(scenarios, 1):
    print(f"\n场景 {i}: {scenario['role']}")
    print("-"*70)
    result = build_prompt(pipelines, **scenario)
    print(result)


# ==================== 更复杂的 Pipeline 示例 ====================
print("\n\n" + "="*70)
print("4. 复杂 Pipeline - 代码审查助手")
print("="*70)

# 子模板1：系统角色
system_role = PromptTemplate.from_template(
    "你是一位资深的{language}代码审查员。"
)

# 子模板2：审查标准
review_standards = PromptTemplate.from_template(
    """审查标准：
1. 代码规范性：{code_style}
2. 性能优化：{performance_focus}
3. 安全性：{security_level}"""
)

# 子模板3：输出要求
output_requirements = PromptTemplate.from_template(
    """输出格式：
- 问题列表（按严重程度排序）
- 修复建议
- 改进后的代码示例"""
)

# 子模板4：待审查代码
code_to_review = PromptTemplate.from_template(
    "待审查代码：\n```{language}\n{code}\n```"
)

# 组装 Pipeline（使用字典管理）
code_review_pipelines = {
    "system_role": system_role,
    "review_standards": review_standards,
    "output_requirements": output_requirements,
    "code_to_review": code_to_review,
}

# 使用代码审查 Pipeline
code_review_prompt = build_prompt(
    code_review_pipelines,
    language="Python",
    code_style="PEP 8",
    performance_focus="时间复杂度和空间复杂度",
    security_level="高（检查SQL注入、XSS等）",
    code="""def get_user_data(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = db.execute(query)
    return result"""
)

print("生成的代码审查提示词:")
print("="*70)
print(code_review_prompt)
print("="*70)


# ==================== 与 LLM 集成 ====================
print("\n\n" + "="*70)
print("5. 与 LLM 集成 - 实际调用")
print("="*70)

try:
    # 创建 LLM
    llm = ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com",
        temperature=0.7,
        extra_body={"thinking": {"type": "disabled"}}
    )
    
    # 使用第一个场景的提示词
    simple_prompt = build_prompt(
        pipelines,
        role="Python编程",
        task="解释代码概念",
        format_style="简洁明了",
        tone="友好",
        question="用户问题：什么是装饰器？"
    )
    
    print("\n调用 LLM...")
    response = llm.invoke(simple_prompt)
    
    print("\n✅ AI 回答:")
    print("-"*70)
    print(response.content)
    
except Exception as e:
    print(f"\n⚠️  LLM 调用失败（可能未配置 API Key）: {e}")


# ==================== 总结 ====================
print("\n\n" + "="*70)
print("📚 PipelinePromptTemplate 核心要点")
print("="*70)
print("""
✅ 优势：
1. 模块化：将复杂提示词拆分为多个子模板
2. 可复用：子模板可以在不同 Pipeline 中重复使用
3. 易维护：修改某个部分不影响其他部分
4. 灵活性：动态组合不同的子模板

✅ 实现方式（新版 LangChain）：
1. 定义多个子模板（PromptTemplate）
2. 使用字典管理子模板集合
3. 编写 build_prompt() 函数手动组合
4. 调用时传入所有需要的变量

✅ 适用场景：
• 多步骤的复杂提示词
• 需要动态切换部分内容
• 团队协作维护大型提示词库
• A/B 测试不同的提示词组合

❌ 注意事项：
• 确保所有子模板的变量都被正确填充
• 变量名不能冲突
• 最终输出必须包含所有子模板的内容
• PipelinePromptTemplate 已弃用，改用手动组合
""")