import akshare as ak
import pandas as pd
from datetime import datetime, timedelta

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# ==================== 1. 获取上海期货交易所某一天的日线数据 ====================
print("--- 获取上海期货交易所(SHFE) 2025-01-02 日线数据 ---")
try:
    df_shfe = ak.get_shfe_daily(date="20250102")
    if not df_shfe.empty:
        print(f"成功获取 {len(df_shfe)} 条合约数据")
        print("前5条数据:")
        print(df_shfe.head())
    else:
        print("当日无数据")
except Exception as e:
    print(f"get_shfe_daily 出错: {e}")

# ==================== 2. 获取某合约的历史数据（需遍历日期） ====================
# 示例：获取螺纹钢主力合约（假设代码 RB8888）的历史数据需要逐日请求
# 这里展示获取铜主力合约 CU8888 最近30天的数据
print("\n--- 获取铜主力合约(CU8888) 最近30天日线数据 ---")
try:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    date_list = [(start_date + timedelta(days=i)).strftime("%Y%m%d") for i in range((end_date - start_date).days + 1)]

    all_data = []
    for date in date_list:
        df_day = ak.get_shfe_daily(date=date)
        if not df_day.empty:
            # 筛选出铜合约（symbol包含'CU'）
            cu_data = df_day[df_day['symbol'].str.startswith('CU')]
            if not cu_data.empty:
                all_data.append(cu_data)

    if all_data:
        df_result = pd.concat(all_data, ignore_index=True)
        print(f"成功获取 {len(df_result)} 条数据")
        print(df_result.tail())
    else:
        print("未获取到铜合约数据")
except Exception as e:
    print(f"批量获取出错: {e}")

# ==================== 3. 获取期货合约基本信息（辅助查询代码） ====================
print("\n--- 获取期货合约基本信息（用于查询品种代码）---")
try:
    df_info = ak.futures_contract_info()
    # 只显示部分列
    print(df_info[['symbol', 'name', 'exchange']].head(10))
except Exception as e:
    print(f"futures_contract_info 出错: {e}")