# -*- coding: utf-8 -*-
"""
i6 加创业板：多出来的弹性，值不值多承担的回撤？

创业板是放大器，不是减震器。要检验的不是「牛市里谁涨得多」——那不用算，
而是「把 10% 的红利换成创业板，全周期的风险调整后收益是变好还是变坏」。

口径与黄金实验保持一致：固定比例、不加闸门，基准 40/60，
替股 x 表示「红利 40−x / 创业板 x / 国债 60」，另加一档「替债 x=10」作对照，
用来证明它不是国债的替代品。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as bt
import experiment_gold as eg

CHINEXT = "data/159915.csv"
OUT = "output/exp/chinext.csv"
CHART = "charts/19_创业板_弹性与代价.png"
PERIODS = [("2015-01-05", "2015-06-12", "2015 上半年（牛）"),
           ("2015-06-12", "2016-01-28", "2015 下半年（崩）"),
           ("2019-01-01", "2021-02-10", "2019–2021 成长牛"),
           ("2021-02-10", "2024-09-13", "2021–2024 杀估值")]


def main():
    os.makedirs("output/exp", exist_ok=True)
    os.makedirs("charts", exist_ok=True)
    rows, navs, yearly = [], {}, []

    for ver in ["long", "trade"]:
        (op, cl), cfg = eg.prices_with_gold(ver, CHINEXT)
        alloc = pd.read_csv(f"output/alloc_{ver}_vol08.csv", parse_dates=["date"])
        dates = pd.DatetimeIndex(alloc["date"])
        start = dates[0]

        plans = [("基准 40/60", 40, 0, 60)]
        for x in [5, 10, 15, 20]:
            plans.append((f"替股 x={x}", 40 - x, x, 60))
        plans.append(("替债 x=10", 40, 10, 50))

        print("=" * 100)
        print(f"{cfg['label']}（{start.date()} 起，固定比例、不加闸门）")
        print("=" * 100)
        print(f"{'方案':<14}{'年化':>8}{'最大回撤':>10}{'Calmar':>8}{'波动':>8}"
              f"{'最差20日':>10}{'与红利相关':>11}")
        print("-" * 100)

        for name, dv, cx, bd in plans:
            nav, _ = eg.run(ver, dv, cx, bd, op, cl, cfg["exec_price"], dates)
            navs[(ver, name)] = nav
            m = bt.metrics(nav)
            corr = cl["dividend"].pct_change().corr(cl["broad"].pct_change())
            rows.append({"版本": ver, "方案": name, "红利": dv, "创业板": cx, "国债": bd,
                         "年化": m["年化收益率"], "回撤": m["最大回撤"],
                         "波动": m["年化波动"], "夏普": m["夏普比率"],
                         "Calmar": m["Calmar比率"], "最差20日": eg.worst_20d(nav),
                         "与红利相关": corr})
            print(f"{name:<14}{m['年化收益率']:>8.2%}{m['最大回撤']:>10.2%}"
                  f"{m['Calmar比率']:>8.2f}{m['年化波动']:>8.2%}"
                  f"{eg.worst_20d(nav):>10.2%}{corr:>11.2f}")

        # 分年度
        print()
        print("分年度：创业板把收益放大到了什么程度")
        print(f"{'年份':<8}{'创业板':>9}{'基准':>9}{'替股x=10':>11}{'替股x=20':>11}")
        print("-" * 50)
        for y in sorted({d.year for d in dates}):
            gs = cl["broad"][cl["broad"].index.year == y]
            gr = gs.iloc[-1] / gs.iloc[0] - 1 if len(gs) > 60 else np.nan
            vals = []
            for k in ["基准 40/60", "替股 x=10", "替股 x=20"]:
                nav = navs[(ver, k)]
                s = nav[nav.index.year == y]
                vals.append(s.iloc[-1] / s.iloc[0] - 1 if len(s) > 60 else np.nan)
            yearly.append({"版本": ver, "年份": int(y), "创业板": gr,
                           "基准": vals[0], "x10": vals[1], "x20": vals[2]})
            print(f"{y:<8}{gr:>9.1%}{vals[0]:>9.1%}{vals[1]:>11.1%}{vals[2]:>11.1%}")

        # 关键时段：放大器在涨和跌两边都放大
        print()
        print("关键时段的放大倍数（创业板 / 红利）")
        for a, b, label in PERIODS:
            seg = pd.DataFrame({"创业板": cl["broad"], "红利": cl["dividend"]})
            seg = seg[(seg.index >= pd.Timestamp(a)) & (seg.index <= pd.Timestamp(b))]
            if len(seg) < 20:
                continue
            cr = seg["创业板"].iloc[-1] / seg["创业板"].iloc[0] - 1
            dr = seg["红利"].iloc[-1] / seg["红利"].iloc[0] - 1
            if pd.isna(cr) or pd.isna(dr):
                continue          # 该时段早于本版红利数据起点
            ratio = cr / dr if abs(dr) > 1e-6 else np.nan
            print(f"  {label:<18}创业板 {cr:>9.1%}   红利 {dr:>9.1%}   倍数 {ratio:>5.1f}x")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8")
    pd.DataFrame(yearly).to_csv("output/exp/chinext_yearly.csv", index=False, encoding="utf-8")
    print(f"\n结果已保存：{OUT}")
    plot(navs, yearly)


def plot(navs, yearly):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), dpi=140,
                             gridspec_kw={"width_ratios": [1.75, 1]})

    ax = axes[0]
    base = navs[("long", "基准 40/60")]
    for name, color in [("替股 x=10", "#C77A1E"), ("替股 x=20", "#8B2A1D"),
                        ("替债 x=10", "#2E8B57")]:
        nav = navs[("long", name)]
        rel = nav / base.reindex(nav.index).ffill()
        ax.plot(rel.index, rel, color=color, linewidth=1.7, label=name)
    ax.axhline(1.0, color="#555", linewidth=0.9, linestyle=":")
    ax.set_title("加创业板后相对 40/60 的净值比（长周期版，含 2015）", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25, linestyle="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax = axes[1]
    y = [d for d in yearly if d["版本"] == "long"]
    years = [d["年份"] for d in y]
    w = 0.27
    x = np.arange(len(years))
    for i, (key, label, color) in enumerate([("基准", "基准 40/60", "#1F4E79"),
                                             ("x10", "替股 x=10", "#C77A1E"),
                                             ("x20", "替股 x=20", "#8B2A1D")]):
        vals = [d[key] for d in y]
        ax.bar(x + (i - 1) * w, vals, width=w, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in years], fontsize=8)
    ax.axhline(0, color="#333", linewidth=0.9)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("分年度：牛市放大、熊市也放大", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25, axis="y", linestyle="--")

    fig.tight_layout()
    fig.savefig(CHART, bbox_inches="tight")
    plt.close(fig)
    print(f"图已保存：{CHART}")


if __name__ == "__main__":
    main()
