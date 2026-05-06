"""
SystemMessage 高级用法示例
展示系统消息的各种复杂应用场景
"""
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import os

# ==================== 1. 基础系统消息 ====================
print("=" * 60)
print("1. 基础系统消息 - 设定角色")
print("=" * 60)

basic_system = SystemMessage(
    content="你是一个专业的Python程序员，擅长编写简洁高效的代码。"
)
print(f"内容: {basic_system.content}\n")


# ==================== 2. 多段落系统消息 ====================
print("=" * 60)
print("2. 多段落系统消息 - 复杂指令")
print("=" * 60)

complex_system = SystemMessage(
    content="""你是一个资深数据分析师，请遵循以下规则：

【职责范围】
- 分析数据趋势和模式
- 提供可视化建议
- 解释统计结果

【输出格式】
1. 先给出结论
2. 再提供详细分析
3. 最后给出建议

【注意事项】
- 使用通俗易懂的语言
- 避免过度技术化
- 重点突出关键发现"""
)
print(f"内容长度: {len(complex_system.content)} 字符")
print(f"前100字符: {complex_system.content[:100]}...\n")


# ==================== 3. 动态系统消息（带变量）====================
print("=" * 60)
print("3. 动态系统消息 - 模板化")
print("=" * 60)

# 方式1：使用 f-string 预填充
role = "金融分析师"
tone = "专业且谨慎"
dynamic_system_1 = SystemMessage(
    content=f"你是一位{role}，回答问题时保持{tone}的语气。"
)
print(f"动态内容: {dynamic_system_1.content}\n")

# 方式2：在 ChatPromptTemplate 中使用
prompt_with_system = ChatPromptTemplate.from_messages([
    ("system", "你是{role}专家，用{style}风格回答"),
    ("human", "{question}")
])

formatted = prompt_with_system.format_messages(
    role="医疗",
    style="严谨科学",
    question="什么是糖尿病？"
)
print(f"系统消息: {formatted[0].content}")
print(f"用户消息: {formatted[1].content}\n")


# ==================== 4. 带上下文历史的系统消息 ====================
print("=" * 60)
print("4. 结合历史对话的系统消息")
print("=" * 60)

# 构建完整对话上下文
conversation_context = [
    SystemMessage(content="你是一个旅行规划助手"),
    HumanMessage(content="我想去日本旅游"),
    AIMessage(content="好的！请问您计划什么时候去？预算大概多少？"),
    HumanMessage(content="下个月，预算2万元"),
]

# 添加新的系统指令（可以覆盖或补充之前的设定）
updated_system = SystemMessage(
    content="基于用户预算，优先推荐性价比高的行程。"
)

full_conversation = conversation_context + [updated_system]
print(f"对话轮数: {len(full_conversation)}")
for i, msg in enumerate(full_conversation):
    print(f"  [{i}] {type(msg).__name__}: {msg.content[:50]}...")
print()


# ==================== 5. 条件化系统消息 ====================
print("=" * 60)
print("5. 条件化系统消息 - 根据场景切换")
print("=" * 60)

def get_system_message(scenario: str) -> SystemMessage:
    """根据不同场景返回不同的系统消息"""
    
    scenarios = {
        "coding": """你是一个编程助手。
- 提供可运行的代码示例
- 添加详细注释
- 说明时间/空间复杂度""",
        
        "writing": """你是一个创意写作助手。
- 使用生动的语言
- 注重情感表达
- 保持故事连贯性""",
        
        "translation": """你是一个专业翻译。
- 保持原意准确
- 符合目标语言习惯
- 标注文化差异"""
    }
    
    content = scenarios.get(scenario, "你是一个通用助手")
    return SystemMessage(content=content)

# 测试不同场景
for scenario in ["coding", "writing", "translation"]:
    sys_msg = get_system_message(scenario)
    print(f"[{scenario}] 指令长度: {len(sys_msg.content)} 字符")
print()


# ==================== 6. 多层级系统消息 ====================
print("=" * 60)
print("6. 多层级系统消息 - 优先级控制")
print("=" * 60)

# 基础角色定义（低优先级）
base_role = SystemMessage(content="你是一个AI助手")

# 领域专业知识（中优先级）
domain_knowledge = SystemMessage(
    content="你精通机器学习和深度学习，熟悉PyTorch和TensorFlow。"
)

# 当前任务指令（高优先级，放在最后）
current_task = SystemMessage(
    content="现在请解释Transformer架构的工作原理，使用类比帮助理解。"
)

# 组合使用（后面的会覆盖前面的某些行为）
layered_messages = [base_role, domain_knowledge, current_task]
print("多层级系统消息组合:")
for i, msg in enumerate(layered_messages, 1):
    print(f"  层级{i}: {msg.content[:60]}...")
print()


# ==================== 7. 系统消息 + Few-Shot 示例 ====================
print("=" * 60)
print("7. 系统消息 + Few-Shot 学习示例")
print("=" * 60)

few_shot_messages = [
    SystemMessage(content="请将用户输入转换为正式商务邮件格式。"),
    
    # 示例1
    HumanMessage(content="开会推迟到明天"),
    AIMessage(content="尊敬的各位同事：\n\n原定于今天的会议将推迟至明天举行，具体时间另行通知。\n\n感谢您的理解与配合。"),
    
    # 示例2
    HumanMessage(content="项目延期了"),
    AIMessage(content="尊敬的项目相关方：\n\n由于技术调整需要，项目交付时间将适当延后。我们将尽快提供更新后的时间表。\n\n对此带来的不便深表歉意。"),
    
    # 实际请求
    HumanMessage(content="客户要求降价"),
]

print(f"Few-shot 示例数量: {(len(few_shot_messages) - 1) // 2}")
print(f"最后一条是: {few_shot_messages[-1].content}\n")


# ==================== 8. 系统消息的最佳实践 ====================
print("=" * 60)
print("8. SystemMessage 最佳实践总结")
print("=" * 60)

best_practices = """
✅ 推荐做法：
1. 清晰明确：使用结构化格式（标题、列表）
2. 具体指令：说明期望的输出格式和风格
3. 设定边界：明确什么能做、什么不能做
4. 提供示例：Few-shot 提升效果
5. 分层设计：基础角色 + 领域知识 + 当前任务

❌ 避免做法：
1. 过于冗长：超过1000字可能效果下降
2. 矛盾指令：不要互相冲突的要求
3. 模糊表述：避免"可能"、"也许"等词
4. 重复设置：多个SystemMessage可能混淆
5. 忽略上下文：要考虑对话历史
"""
print(best_practices)


# ==================== 9. 实际应用：构建完整的 Agent 提示词 ====================
print("=" * 60)
print("9. 实战：完整的 Agent 系统提示词")
print("=" * 60)

agent_system_prompt = SystemMessage(
    content="""你是一个智能数据分析助手，具备以下能力：

【核心功能】
1. 数据清洗和预处理
2. 统计分析和可视化
3. 机器学习模型建议
4. 业务洞察提取

【工作流程】
第1步：理解用户需求和问题背景
第2步：分析可用数据和特征
第3步：选择合适的分析方法
第4步：执行分析并解释结果
第5步：提供可操作的建议

【输出规范】
- 使用 Markdown 格式
- 关键结论加粗显示
- 代码块标明编程语言
- 图表描述清晰易懂

【交互风格】
- 主动澄清模糊需求
- 分步骤解释复杂概念
- 平衡专业性和可读性
- 适时提供额外建议

【限制条件】
- 不生成有害或偏见内容
- 不确定时明确说明
- 保护数据隐私和安全
- 遵守法律法规要求"""
)

print(f"Agent 系统提示词长度: {len(agent_system_prompt.content)} 字符")
print(f"字数统计: {len(agent_system_prompt.content.split())} 个单词")
print(f"\n前200字符预览:\n{agent_system_prompt.content[:200]}...\n")


# ==================== 10. 动态构建系统消息 ====================
print("=" * 60)
print("10. 动态构建系统消息 - 工厂模式")
print("=" * 60)

class SystemMessageBuilder:
    """系统消息构建器"""
    
    def __init__(self, base_role: str = "AI助手"):
        self.components = [f"你是{base_role}。"]
    
    def add_expertise(self, domain: str) -> 'SystemMessageBuilder':
        """添加专业领域"""
        self.components.append(f"你精通{domain}。")
        return self
    
    def set_tone(self, tone: str) -> 'SystemMessageBuilder':
        """设定语气风格"""
        self.components.append(f"回答时保持{tone}的语气。")
        return self
    
    def add_format_rule(self, rule: str) -> 'SystemMessageBuilder':
        """添加输出格式规则"""
        self.components.append(f"输出格式要求：{rule}")
        return self
    
    def add_constraint(self, constraint: str) -> 'SystemMessageBuilder':
        """添加约束条件"""
        self.components.append(f"注意：{constraint}")
        return self
    
    def build(self) -> SystemMessage:
        """构建最终的 SystemMessage"""
        content = "\n\n".join(self.components)
        return SystemMessage(content=content)

# 使用构建器创建定制化的系统消息
custom_system = (
    SystemMessageBuilder(base_role="数据科学顾问")
    .add_expertise("统计学和机器学习")
    .set_tone("专业但易懂")
    .add_format_rule("使用Markdown，代码加注释")
    .add_constraint("不编造数据，不确定时说明")
    .build()
)

print("动态构建的系统消息:")
print(custom_system.content)
print()


print("=" * 60)
print("所有示例完成！")
print("=" * 60)
