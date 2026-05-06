import akshare as ak
import pandas as pd
from datetime import datetime, timedelta

# 1. 使用 LangChain 1.0 推荐的导入方式
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# ==================== 配置你的 DeepSeek API 密钥 ====================
# ⚠️ 请务必将下面的值替换为你自己的真实信息
DEEPSEEK_API_KEY = "sk-205ee372c5df491f8050324eb697e504"  # 替换为你的 DeepSeek API Key
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
# ================================================================

# 初始化大语言模型
# 使用 DeepSeek 的模型名称 "deepseek-chat" 和官方 API Base URL
model = ChatOpenAI(
    model="deepseek-chat",  # DeepSeek 模型名称
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.7,
)


# ==================== 2. 定义工具 ====================
@tool
def get_contracts_list() -> str:
    """获取所有国内期货合约的基本信息（代码、名称、交易所）"""
    try:
        df = ak.futures_contract_info()
        # 选择关键列，避免返回过多
        df = df[['symbol', 'name', 'exchange']]
        return df.to_string()
    except Exception as e:
        return f"获取合约列表失败: {str(e)}"

@tool
def get_future_klines(symbol_prefix: str = "CU", days: int = 30) -> str:
    """获取指定期货品种（上海期货交易所）的历史日线数据。
    输入品种前缀如 'CU' 代表铜, 'RB' 代表螺纹钢, 'AU' 代表黄金。
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        date_list = [(start_date + timedelta(days=i)).strftime("%Y%m%d")
                     for i in range((end_date - start_date).days + 1)]
        all_data = []
        for date in date_list:
            df_day = ak.get_shfe_daily(date=date)
            if not df_day.empty:
                # 筛选以指定前缀开头的合约，如 'CU2501'
                filtered = df_day[df_day['symbol'].str.startswith(symbol_prefix.upper())]
                if not filtered.empty:
                    all_data.append(filtered)
        if not all_data:
            return f"未找到品种 {symbol_prefix} 在指定日期范围内的数据"
        df_result = pd.concat(all_data, ignore_index=True)
        df_result = df_result.sort_values('date')
        return df_result.tail(days).to_string()
    except Exception as e:
        return f"获取行情数据失败: {str(e)}"

tools = [get_contracts_list, get_future_klines]

# ==================== 3. 使用 create_agent 创建并执行 Agent ====================
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="你是一位专业的期货量化分析师。你的任务是使用提供的工具来帮助用户查询和分析国内期货数据。",
)

if __name__ == "__main__":
    question = "请帮我看看最近30天的铜期货（代码CU）行情怎么样，简单分析一下。"
    print(f"你的问题: {question}")

    # invoke 方法执行 agent 并返回最终结果
    response = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )

    # 输出 Agent 的最终回答
    print(f"\nAI Agent的回答: {response['messages'][-1].content}")