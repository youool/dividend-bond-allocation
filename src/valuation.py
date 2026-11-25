"""
第 3 关 · 步骤 1：取沪深300 的 PE-TTM 历史，并算每个月的估值分位

为什么用月度数据：
  策略本来就是每月最后一个交易日判断一次，月度 PE 和它天然对齐，
  不需要为了"精确到日"引入日频数据再降采样。

数据来源与口径：
  akshare 的 stock_index_pe_lg（乐咕）提供 2005 年至今的**月度**滚动市盈率，
  列名「滚动市盈率」= PE-TTM（用最近四个季度盈利），比静态 PE 更贴近实际。

分位怎么算：
  对每个月末 t，取 t 之前最近 10 年（120 个月）的 PE 样本，
  算「有多少比例的月份 PE 比现在更低」。
  分位越低 = 现在越便宜。
  不用全历史（2005 起）是因为 2007 年那种极端估值会把基准拉歪。

不做这一步会怎样：
  没有分位，"便宜的适合加仓"就只是一句感觉，没法写成能跑的条件。
"""

import os

import akshare as ak
import pandas as pd

OUT = "data/pe_300.csv"
WINDOW_MONTHS = 120  # 近 10 年


def fetch_pe():
    df = ak.stock_index_pe_lg(symbol="沪深300")
    if df is None or df.empty:
        raise RuntimeError("PE 数据为空")

    df = df[["日期", "滚动市盈率"]].copy()
    df.columns = ["date", "pe_ttm"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["pe_ttm"])
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return df


def add_percentile(df):
    """对每个月末，算它在过去 120 个月里的分位。"""
    pcts = []
    for i in range(len(df)):
        window = df["pe_ttm"].iloc[max(0, i - WINDOW_MONTHS + 1): i + 1]
        cur = df["pe_ttm"].iloc[i]
        pcts.append(round(float((window < cur).mean()), 4))
    df["pe_pct"] = pcts
    df["window_n"] = [min(i + 1, WINDOW_MONTHS) for i in range(len(df))]
    return df


def main():
    os.makedirs("data", exist_ok=True)
    df = fetch_pe()
    df = add_percentile(df)
    df.to_csv(OUT, index=False, encoding="utf-8")

    print(f"已保存 {OUT}")
    print(f"共 {len(df)} 期月度数据，{df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")
    print()

    print("最近 12 期：")
    print(f"{'日期':<12}{'PE-TTM':>9}{'分位':>9}{'样本月数':>9}")
    print("-" * 40)
    for _, r in df.tail(12).iterrows():
        print(f"{r['date'].date()!s:<12}{r['pe_ttm']:>9.2f}{r['pe_pct']:>9.2%}{r['window_n']:>9}")

    print()
    latest = df.iloc[-1]
    pct = latest["pe_pct"]
    if pct < 0.30:
        band, pos = "低于 30%", "80%"
    elif pct <= 0.70:
        band, pos = "30%–70%", "60%"
    else:
        band, pos = "高于 70%", "30%"
    print(f"当前状态：PE {latest['pe_ttm']:.2f}，处于近 10 年 {pct:.1%} 分位（{band}）→ 基础权益仓位 {pos}")

    # 分位分布的体检：三档各占多少个月，太偏就说明阈值定得不合理
    print()
    print("历史上三档各占多少个月：")
    lo = (df["pe_pct"] < 0.30).sum()
    mid = ((df["pe_pct"] >= 0.30) & (df["pe_pct"] <= 0.70)).sum()
    hi = (df["pe_pct"] > 0.70).sum()
    total = len(df)
    print(f"  <30%：{lo:>4} 个月（{lo/total:.0%}）")
    print(f"  30–70%：{mid:>4} 个月（{mid/total:.0%}）")
    print(f"  >70%：{hi:>4} 个月（{hi/total:.0%}）")
    print("  （三档大致均分才合理；某一档几乎不出现，说明阈值要调）")


if __name__ == "__main__":
    main()
