"""
第 5 关 · 对照实验五：把择时幅度关小，还会不会伤到自己

前三轮实验给出的结论是「大范围择时是负贡献」：仓位在 0–80% 之间摆动，
把长期持有红利的过程反复打断，最后不如什么都不做。

但这不等于「任何择时都无用」。有没有可能，问题出在**幅度太大**？
本实验把择时幅度压到 ±10%：红利比例只在 30% / 40% / 50% 三档之间动，
其余全给国债，不做趋势判断。三组对照：

  C0  固定 40% 红利 + 60% 国债（不判断）
  C1  估值微调：分位 <30% → 50%；30–70% → 40%；>70% → 30%
  C2  估值微调 + 反向（>70% → 50%，<30% → 30%）—— 故意做反，当安慰剂检验

  C2 的用处：如果 C2 和最差的那组差不多，说明「改比例」这件事本身就在主导结果，
  不是估值在起作用。安慰剂组是判断信号真假最省事的办法。

判读标准：
  C1 要在**样本外**同时满足：年化不低于 C0 的 90%、回撤不明显变大（≤1.5 倍）。
  达不到就不该留在策略里——不如老老实实写固定比例。

输出：
  output/exp/core_summary.csv
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt  # noqa: E402
import signals as sg  # noqa: E402

EXP_DIR = "output/exp"
DIVIDEND_PATH = {"long": "data/index/红利低波_全收益.csv", "trade": "data/512890.csv"}
WCOLS = ["w_dividend", "w_broad", "w_bond", "w_cash"]


def price_pct_series(ver):
    """红利低波自身价格分位：过去 5 年里有多少交易日比今天便宜。"""
    s = sg.load_close(DIVIDEND_PATH[ver])
    pct = s.rolling(1260, min_periods=252).apply(lambda w: float((w[:-1] < w[-1]).mean()), raw=True)
    return pct.dropna()


def make_table(dates, pct_series, mode):
    """按 mode 生成红利比例序列，其余给国债。"""
    aligned = pct_series.reindex(pct_series.index.union(pd.DatetimeIndex(dates))).ffill()
    aligned = aligned.reindex(pd.DatetimeIndex(dates))

    rows = []
    for d, p in zip(pd.DatetimeIndex(dates), aligned.values):
        if mode == "fixed40":
            r = 0.40
        elif np.isnan(p):
            r = 0.40          # 分位还没算出来的月份按中间档处理
        elif mode == "tilt":
            r = 0.50 if p < 0.30 else (0.40 if p <= 0.70 else 0.30)
        else:  # placebo：故意做反
            r = 0.30 if p < 0.30 else (0.40 if p <= 0.70 else 0.50)
        rows.append({"date": d, "w_dividend": r, "w_broad": 0.0,
                     "w_bond": 1.0 - r, "w_cash": 0.0})
    return pd.DataFrame(rows)


def main():
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs("charts", exist_ok=True)
    rows = []

    for ver in ["long", "trade"]:
        cfg = bt.VERSIONS[ver]
        ins_end = pd.Timestamp(cfg["insample_end"])
        dates = pd.read_csv(f"output/weights_{ver}.csv", parse_dates=["date"])["date"].values
        pct = price_pct_series(ver)

        for mode, name in [("fixed40", "C0 固定40%红利"), ("tilt", "C1 估值微调30/40/50"),
                           ("placebo", "C2 反向微调（安慰剂）")]:
            w = make_table(dates, pct, mode)
            op, cl = bt.load_prices(cfg["prices"])
            nav, fee = bt.simulate(w, op, cl, cfg["exec_price"])
            ins, out = nav[nav.index <= ins_end], nav[nav.index > ins_end]
            mi, mo = bt.metrics(ins), bt.metrics(out)
            mf = bt.metrics(nav)
            rows.append({"版本": ver, "方案": name,
                         "红利均值": float(w["w_dividend"].mean()),
                         "内_年化": mi["年化收益率"], "内_回撤": mi["最大回撤"], "内_夏普": mi["夏普比率"],
                         "外_年化": mo["年化收益率"], "外_回撤": mo["最大回撤"], "外_夏普": mo["夏普比率"],
                         "全_年化": mf["年化收益率"], "全_回撤": mf["最大回撤"], "全_夏普": mf["夏普比率"],
                         "手续费": fee})

    df = pd.DataFrame(rows)
    df.to_csv(f"{EXP_DIR}/core_summary.csv", index=False, encoding="utf-8")

    for ver in ["long", "trade"]:
        print()
        print("=" * 104)
        print(f"{bt.VERSIONS[ver]['label']}：小幅择时 vs 固定比例 vs 安慰剂")
        print("=" * 104)
        print(f"{'方案':<24}{'红利均值':>8}{'内年化':>9}{'内回撤':>9}{'内夏普':>8}"
              f"{'外年化':>9}{'外回撤':>9}{'外夏普':>8}{'全年化':>9}")
        print("-" * 104)
        sub = df[df["版本"] == ver]
        for _, r in sub.iterrows():
            print(f"{r['方案']:<24}{r['红利均值']:>8.0%}{r['内_年化']:>9.1%}{r['内_回撤']:>9.1%}"
                  f"{r['内_夏普']:>8.2f}{r['外_年化']:>9.1%}{r['外_回撤']:>9.1%}"
                  f"{r['外_夏普']:>8.2f}{r['全_年化']:>9.1%}")

        c0 = sub[sub["方案"].str.startswith("C0")].iloc[0]
        c1 = sub[sub["方案"].str.startswith("C1")].iloc[0]
        c2 = sub[sub["方案"].str.startswith("C2")].iloc[0]
        print()
        print(f"样本外年化：C0 {c0['外_年化']:.1%}　C1 {c1['外_年化']:.1%}　C2 {c2['外_年化']:.1%}")
        print(f"样本外夏普：C0 {c0['外_夏普']:.2f}　C1 {c1['外_夏普']:.2f}　C2 {c2['外_夏普']:.2f}")
        verdict = "C1 优于 C0" if c1["外_夏普"] > c0["外_夏普"] else "C1 不如 C0"
        print(f"结论：{verdict}；C2（反向）外夏普 {c2['外_夏普']:.2f}，"
              f"{'低于 C1，说明估值方向有信息量' if c2['外_夏普'] < c1['外_夏普'] else '不低于 C1，说明方向没有信息量'}")

    # 和原策略放在一起看
    print()
    print("=" * 104)
    print("汇总：原择时策略 vs 现在的候选（同口径，样本外）")
    print("=" * 104)
    for ver in ["long", "trade"]:
        s0 = pd.read_csv(f"output/exp/structure_summary.csv")
        base = s0[(s0["版本"] == ver) & (s0["方案"] == "S0_基线_估值加趋势") & (s0["区间"] == "样本外")].iloc[0]
        c0 = df[(df["版本"] == ver) & (df["方案"].str.startswith("C0"))].iloc[0]
        c1 = df[(df["版本"] == ver) & (df["方案"].str.startswith("C1"))].iloc[0]
        print(f"{bt.VERSIONS[ver]['label']}")
        print(f"  原策略（估值+趋势）   外年化 {base['年化收益率']:>6.1%}　外回撤 {base['最大回撤']:>7.1%}　外夏普 {base['夏普比率']:>5.2f}")
        print(f"  C0 固定 40% 红利      外年化 {c0['外_年化']:>6.1%}　外回撤 {c0['外_回撤']:>7.1%}　外夏普 {c0['外_夏普']:>5.2f}")
        print(f"  C1 估值微调 30/40/50  外年化 {c1['外_年化']:>6.1%}　外回撤 {c1['外_回撤']:>7.1%}　外夏普 {c1['外_夏普']:>5.2f}")


if __name__ == "__main__":
    main()
