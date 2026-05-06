"""
期货数据获取 Agent
使用 LangChain + DeepSeek API 分析期货市场数据
"""
import os
from datetime import datetime, timedelta
from typing import List

import akshare as ak
import pandas as pd
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# 设置 akshare 全局超时
ak.DEFAULT_REQUEST_TIMEOUT = 30

# ==================== 配置常量 ====================

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-v4-flash"
TEMPERATURE = 0.7
REQUEST_TIMEOUT = 120
MAX_RETRIES = 3


# ==================== LLM 初始化 ====================

def init_llm() -> ChatOpenAI:
    """
    初始化 DeepSeek 大语言模型
    
    Returns:
        ChatOpenAI: 配置好的 DeepSeek 模型实例
        
    Raises:
        ValueError: 当未找到 API Key 时抛出异常
    """
    if not DEEPSEEK_API_KEY:
        raise ValueError(
            "未找到 DeepSeek API Key。请设置环境变量 DEEPSEEK_API_KEY\n"
            "Windows PowerShell: $env:DEEPSEEK_API_KEY='your-api-key'\n"
            "Windows CMD: set DEEPSEEK_API_KEY=your-api-key\n"
            "Linux/Mac: export DEEPSEEK_API_KEY='your-api-key'"
        )
    
    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=TEMPERATURE,
        timeout=REQUEST_TIMEOUT,
        max_retries=MAX_RETRIES,
        extra_body={
            "thinking": {"type": "disabled"}  # 禁用思考模式，使用快速响应模式
        }
    )


# ==================== 工具函数 ====================

@tool
def get_contracts_list() -> str:
    """
    获取所有国内期货合约的基本信息（代码、名称、交易所）
    
    Returns:
        str: 格式化的合约列表字符串，包含 symbol、name、exchange 字段
    """
    try:
        df = ak.futures_contract_info()
        df = df[['symbol', 'name', 'exchange']]
        return df.to_string(index=False)
    except Exception as e:
        return f"获取合约列表失败: {str(e)}"


@tool
def get_future_klines(symbol_prefix: str = "CU", days: int = 30) -> str:
    """
    获取指定期货品种的历史日线数据。
    
    Args:
        symbol_prefix: 品种前缀，如 'CU'(铜), 'RB'(螺纹钢), 'AU'(黄金)
        days: 获取最近多少天的数据，默认30天
    
    Returns:
        str: 格式化后的行情数据字符串，最多返回50条记录
    """
    try:
        end_date = datetime.now()
        all_data: List[pd.DataFrame] = []
        
        # 逐日获取数据并过滤指定品种
        for i in range(days):
            date_str = (end_date - timedelta(days=i)).strftime("%Y%m%d")
            try:
                df_day = ak.get_shfe_daily(date=date_str)
                if not df_day.empty:
                    filtered = df_day[df_day['symbol'].str.startswith(symbol_prefix.upper())]
                    if not filtered.empty:
                        all_data.append(filtered)
            except Exception:
                continue
        
        if not all_data:
            return f"未找到品种 {symbol_prefix} 在指定日期范围内的数据"
        
        # 合并、去重、排序
        df_result = pd.concat(all_data, ignore_index=True)
        df_result = df_result.drop_duplicates(subset=['symbol', 'date'])
        df_result = df_result.sort_values(['symbol', 'date'])
        
        return df_result.tail(50).to_string(index=False)
    except Exception as e:
        return f"获取行情数据失败: {str(e)}"


# ==================== Agent 核心逻辑 ====================

def create_futures_agent(llm: ChatOpenAI) :
    """
    创建期货分析 Agent
    
    Args:
        llm: 已初始化的语言模型实例
        
    Returns:
        Agent 执行器
    """
    tools = [get_contracts_list, get_future_klines]
    return create_agent(llm, tools)


def run_agent(question: str) -> str:
    """
    运行 Agent 并回答用户问题
    
    Args:
        question: 用户提出的问题
        
    Returns:
        str: Agent 的回答内容
    """
    # 初始化模型和 Agent
    llm = init_llm()
    agent = create_futures_agent(llm)
    
    # 打印问题标题
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"{'='*60}\n")
    
    # 调用 Agent（LangGraph 方式）
    messages = [{"role": "user", "content": question}]
    response = agent.invoke({"messages": messages})
    
    # 提取最后的助手回复
    return response['messages'][-1].content


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    # 示例问题：分析铜期货行情
    question = "请帮我看看最近30天的铜期货（代码CU）行情怎么样，简单分析一下。"
    
    try:
        answer = run_agent(question)
        print(f"\n{'='*60}")
        print(f"AI Agent 回答:")
        print(f"{'='*60}")
        print(answer)
    except Exception as e:
        print(f"\n错误: {str(e)}")
