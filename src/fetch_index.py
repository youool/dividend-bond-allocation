"""
第 2 关 · 步骤 3：拉指数数据，做长周期版（甲方案的 A 面）

为什么需要这一套：
  红利低波 ETF（512890）2019-01-18 才上市，可交易版只能从 2019 年起，
  覆盖不到 2015 股灾和 2018 熊市。要补这两个极端样本，只能用指数。

关键要求：必须是「全收益」指数。
  红利资产的收益里有一大块是分红，价格指数不含分红，
  用它跑长周期版会系统性低估红利策略——这不是精度问题，是结论会反。
  本脚本取的是 H20269（中证红利低波动全收益）和 H00300（沪深300 全收益），
  数据来自中证指数官网，不是被限流的东财接口。

债与货币不需要指数：
  511010、511880 的历史都能回溯到 2013，已覆盖 2015 年起的全部区间，
  直接复用 ETF 数据即可，也避免"指数 vs ETF"两套口径混用。
"""

import os

import akshare as ak
import pandas as pd

OUT_DIR = "data/index"

TARGETS = [
    ("H20269", "红利低波_全收益"),
    ("H00300", "沪深300_全收益"),
]


def fetch_index(symbol):
    # 从 2013 开始拉：120 日均线需要预热，2015 年才能算出正常信号
    df = ak.index_hist_csindex(symbol=symbol, start_date="20130101", end_date="20261231")
    if df is None or df.empty:
        raise RuntimeError("返回空数据")

    need = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close"}
    df = df.rename(columns=need)
    missing = [c for c in need.values() if c not in df.columns]
    if missing:
        raise RuntimeError(f"缺少列 {missing}，实际列 {list(df.columns)}")

    df = df[["date", "open", "high", "low", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"{'指数代码':<10}{'名称':<18}{'行数':>6}  {'起始':<12}{'结束':<12}")
    print("-" * 66)

    for symbol, name in TARGETS:
        try:
            df = fetch_index(symbol)
            path = os.path.join(OUT_DIR, f"{name}.csv")
            df.to_csv(path, index=False, encoding="utf-8")
            print(f"{symbol:<10}{name:<18}{len(df):>6}  {df['date'].iloc[0]:<12}{df['date'].iloc[-1]:<12}")
        except Exception as e:
            print(f"{symbol:<10}{name:<18}{'失败':>6}  {str(e)[:50]}")

    print("-" * 66)
    print("长周期版数据构成：")
    print("  红利低波 → H20269 全收益指数")
    print("  沪深300  → H00300 全收益指数")
    print("  国债     → data/511010.csv（ETF，2013 起，已覆盖 2015）")
    print("  货币     → data/511880.csv（ETF，2013 起，已覆盖 2015）")
    print()
    print("提醒：报告里必须写明长周期版的两条腿用了指数（不含费用与跟踪误差），")
    print("      可交易版才是你能真正拿到手的收益。")


if __name__ == "__main__":
    main()
