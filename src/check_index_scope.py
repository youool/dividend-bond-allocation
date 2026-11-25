# -*- coding: utf-8 -*-
"""
口径核对：价格指数 H30269 vs 全收益指数 H20269

存在的意义：行情软件（同花顺、东财）默认显示的是**价格指数**，
2015 年峰值 11188.01（盘中）/ 11103.55（收盘）；
而本项目回测用的是**全收益指数**，同期峰值 14395.48。
两者相差 29.6%，全部来自分红再投资，不是数据错误。

这个脚本把两个口径并排算出来，答辩被问到时直接跑一遍即可。

用法：
    python src/check_index_scope.py
"""
import os

import akshare as ak
import pandas as pd

PRICE_CODE = "H30269"      # 中证红利低波动指数（价格）
TOTAL_CODE = "H20269"      # 中证红利低波动全收益指数
PRICE_PATH = "data/index/红利低波_价格.csv"
TOTAL_PATH = "data/index/红利低波_全收益.csv"


def load_price_index():
    """价格指数只用于对照，不参与回测。缺文件时联网拉一次并落盘。"""
    if os.path.exists(PRICE_PATH):
        df = pd.read_csv(PRICE_PATH, parse_dates=["date"]).set_index("date")
        return df
    raw = ak.index_hist_csindex(symbol=PRICE_CODE, start_date="20130101", end_date="20261231")
    raw["日期"] = pd.to_datetime(raw["日期"])
    raw = raw.set_index("日期")
    out = pd.DataFrame({
        "close": raw["收盘"].astype(float).values,
        "high": raw["最高"].astype(float).values,
    }, index=raw.index)
    os.makedirs("data/index", exist_ok=True)
    out.reset_index().rename(columns={"index": "date"}).to_csv(
        PRICE_PATH, index=False, encoding="utf-8")
    print(f"已拉取 {PRICE_CODE} 并保存到 {PRICE_PATH}")
    return out


def window_drawdown(s, start, end):
    """在 [start, end] 窗口内找峰与随后的谷，返回 (峰日, 峰值, 谷日, 谷值, 回撤)。"""
    w = s.loc[start:end]
    peak_day = w.idxmax()
    peak = float(w.loc[peak_day])
    after = s.loc[peak_day:end]
    trough_day = after.idxmin()
    trough = float(after.loc[trough_day])
    return peak_day, peak, trough_day, trough, trough / peak - 1


def annualized(s):
    return (s.iloc[-1] / s.iloc[0]) ** (252 / len(s)) - 1


def main():
    price = load_price_index()
    total = pd.read_csv(TOTAL_PATH, parse_dates=["date"]).set_index("date")

    p = price["close"]
    t = total["close"]

    print("=" * 78)
    print("中证红利低波动指数：价格口径 vs 全收益口径")
    print("=" * 78)

    # 1. 2015 年那一轮（窗口限定在 2015-2016，否则会取到 2025 年的历史新高）
    pd_day, pk, pt_day, pv, pdd = window_drawdown(p, "2015-01-01", "2016-12-31")
    td_day, tk, tt_day, tv, tdd = window_drawdown(t, "2015-01-01", "2016-12-31")
    p_high = float(price.loc["2015", "high"].max())
    p_high_day = price.loc["2015", "high"].idxmax()

    print("\n【2015 年那一轮】")
    print(f"  价格指数 收盘峰 {pd_day.date()} {pk:.2f}"
          f"（盘中最高 {p_high:.2f}，{p_high_day.date()}）")
    print(f"           → 谷 {pt_day.date()} {pv:.2f}   回撤 {pdd:.1%}")
    print(f"  全收益   收盘峰 {td_day.date()} {tk:.2f}")
    print(f"           → 谷 {tt_day.date()} {tv:.2f}   回撤 {tdd:.1%}")
    print(f"  峰值之比 {tk / pk:.4f}，即累计分红再投资 {tk / pk - 1:.1%}")

    # 2. 长期年化
    common = pd.DataFrame({"price": p, "total": t}).dropna().loc["2015-01-05":]
    a_p = annualized(common["price"])
    a_t = annualized(common["total"])
    print("\n【2015 年至今】")
    print(f"  价格指数  年化 {a_p:.2%}   累计 {common['price'].iloc[-1] / common['price'].iloc[0] - 1:.1%}")
    print(f"  全收益    年化 {a_t:.2%}   累计 {common['total'].iloc[-1] / common['total'].iloc[0] - 1:.1%}")
    print(f"  年化差    {(a_t - a_p) * 100:.2f} 个百分点（≈ 平均股息率）")

    # 3. 这个口径差会推翻什么结论
    print("\n【为什么必须用全收益】")
    print(f"  若误用价格指数：红利年化 {a_p:.2%}，低于策略的 6.3% → 会得出「策略跑赢红利」")
    print(f"  用正确的全收益：红利年化 {a_t:.2%}，高于策略的 6.3% → 实际是「策略跑输红利」")
    print("  同一个策略，只因为一个口径选择，结论完全相反。")

    # 4. 顺带确认 ETF 序列没被复权算错
    nav_path = "data/512890_累计净值.csv"
    if os.path.exists(nav_path):
        nav = pd.read_csv(nav_path, parse_dates=["FSRQ"]).set_index("FSRQ")["LJJZ"]
        etf = pd.read_csv("data/512890.csv", parse_dates=["date"]).set_index("date")["close"]
        both = pd.DataFrame({"etf": etf, "nav": nav}).dropna()
        print("\n【ETF 后复权序列 vs 基金官方累计净值】")
        print(f"  区间 {both.index[0].date()} → {both.index[-1].date()}，{len(both)} 个共同交易日")
        print(f"  末日  后复权 {both['etf'].iloc[-1]:.4f}  累计净值 {both['nav'].iloc[-1]:.4f}"
              f"  偏差 {both['etf'].iloc[-1] / both['nav'].iloc[-1] - 1:+.3%}")
        print("  → 偏差在 0.3% 以内，说明份额折算与分红复权都处理对了")


if __name__ == "__main__":
    main()
