# -*- coding: utf-8 -*-
"""
i2 加黄金：把黄金放进「红利 + 国债」，值不值？

两个口径必须分开，否则说不清黄金替代的是谁：
  口径一（替债）：红利 40 / 黄金 x / 国债 60−x   → 风险资产从 40% 升到 40+x
  口径二（替股）：红利 40−x / 黄金 x / 国债 60   → 风险资产仍是 40%，只是内部换人

基准是 40/60 固定比例（不加闸门），这样差异只来自资产，不来自仓位。
每档都报：年化、最大回撤、Calmar、最差 20 日、与基准的日收益相关。
另做两件事：
  1) 分年度，看黄金的优势是不是集中在 2019 之后的金价上行；
  2) 收益扣减压测（黄金年化扣 2 / 4 个百分点），看结论会不会翻转。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as bt

GOLD = "data/518880.csv"
OUT = "output/exp/gold.csv"
CHART = "charts/17_黄金_两个口径与年度.png"
XS = [5, 10, 15, 20]


def prices_with_gold(ver, gold_path=GOLD):
    cfg = bt.VERSIONS[ver]
    paths = dict(cfg["prices"])
    paths["broad"] = gold_path          # 借 broad 这个槽位放黄金
    return bt.load_prices(paths), cfg


def weights_table(dates, div, gold, bond):
    df = pd.DataFrame({"date": dates})
    df["w_dividend"] = div / 100.0
    df["w_broad"] = gold / 100.0
    df["w_bond"] = bond / 100.0
    df["w_cash"] = 0.0
    return df


def run(ver, div, gold, bond, op, cl, exec_price, start, gold_haircut=0.0):
    if gold_haircut:
        op, cl = op.copy(), cl.copy()
        c = cl["broad"].astype(float)
        first = c.first_valid_index()          # 黄金 2013-07 才上市，之前必须是 NaN
        r = c.pct_change().fillna(0.0) - gold_haircut / 252.0
        new_c = (1.0 + r).cumprod() * c.loc[first]
        new_c.loc[new_c.index < first] = np.nan
        f = new_c / c
        cl["broad"] = new_c
        op["broad"] = op["broad"].astype(float) * f
    dates = pd.DatetimeIndex(start) if not isinstance(start, pd.DatetimeIndex) else start
    w = weights_table(dates, div, gold, bond)
    nav, fee = bt.simulate(w, op, cl, exec_price, start=dates[0])
    return nav, fee


def worst_20d(nav):
    return float((nav / nav.shift(20) - 1).min())


def main():
    os.makedirs("output/exp", exist_ok=True)
    os.makedirs("charts", exist_ok=True)
    rows, navs, yearly = [], {}, []

    for ver in ["long", "trade"]:
        (op, cl), cfg = prices_with_gold(ver)
        alloc = pd.read_csv(f"output/alloc_{ver}_vol08.csv", parse_dates=["date"])
        dates = pd.DatetimeIndex(alloc["date"])
        start = dates[0]

        variants = [("基准 40/60", 40, 0, 60)]
        for x in XS:
            variants.append((f"替债 x={x}", 40, x, 60 - x))
        for x in XS:
            variants.append((f"替股 x={x}", 40 - x, x, 60))

        base_nav = None
        print("=" * 100)
        print(f"{cfg['label']}（{start.date()} 起，固定比例、不加闸门，只换资产）")
        print("=" * 100)
        print(f"{'方案':<14}{'年化':>8}{'最大回撤':>10}{'Calmar':>8}{'波动':>8}"
              f"{'最差20日':>10}{'与基准相关':>11}{'期末权益+黄金':>14}")
        print("-" * 100)

        for name, dv, gd, bd in variants:
            nav, fee = run(ver, dv, gd, bd, op, cl, cfg["exec_price"], dates)
            navs[(ver, name)] = nav
            if base_nav is None:
                base_nav = nav
            m = bt.metrics(nav)
            corr = nav.pct_change().corr(base_nav.pct_change())
            rows.append({"版本": ver, "方案": name, "红利": dv, "黄金": gd, "国债": bd,
                         "年化": m["年化收益率"], "回撤": m["最大回撤"],
                         "波动": m["年化波动"], "夏普": m["夏普比率"],
                         "Calmar": m["Calmar比率"], "最差20日": worst_20d(nav),
                         "与基准相关": corr})
            print(f"{name:<14}{m['年化收益率']:>8.2%}{m['最大回撤']:>10.2%}"
                  f"{m['Calmar比率']:>8.2f}{m['年化波动']:>8.2%}"
                  f"{worst_20d(nav):>10.2%}{corr:>11.2f}{(dv + gd):>13.0f}%")

        # 分年度：黄金自身 + 基准与最优两档的组合收益
        g = cl["broad"]
        g = g[g.index >= start]          # 与组合区间对齐，否则会画出组合还没开始的年份
        for y, s in g.groupby(g.index.year):
            if len(s) < 100:
                continue
            yearly.append({"版本": ver, "年份": int(y),
                           "黄金": s.iloc[-1] / s.iloc[0] - 1,
                           "红利": cl["dividend"][cl["dividend"].index.year == y].pipe(
                               lambda z: z.iloc[-1] / z.iloc[0] - 1) if len(s) > 0 else np.nan})

        print("分年度收益：黄金到底在哪几年救了组合")
        print(f"{'年份':<8}{'黄金':>9}{'基准40/60':>12}{'替股x=10':>11}{'替股x=20':>11}")
        print("-" * 52)
        for y in sorted({d.year for d in dates}):
            seg = {k: navs[(ver, k)] for k in ["基准 40/60", "替股 x=10", "替股 x=20"]}
            vals = {}
            for k, nav in seg.items():
                s = nav[nav.index.year == y]
                vals[k] = s.iloc[-1] / s.iloc[0] - 1 if len(s) > 60 else np.nan
            gs = g[g.index.year == y]
            gr = gs.iloc[-1] / gs.iloc[0] - 1 if len(gs) > 60 else np.nan
            print(f"{y:<8}{gr:>9.1%}{vals['基准 40/60']:>12.1%}"
                  f"{vals['替股 x=10']:>11.1%}{vals['替股 x=20']:>11.1%}")
        print()

        # 压测：替股 x=10 / x=20，黄金收益扣 2 / 4pct
        print()
        print("压测：黄金年化收益扣减后（替股口径）")
        print(f"{'方案':<20}{'扣0':>8}{'扣2pct':>9}{'扣4pct':>9}")
        print("-" * 50)
        for x in [10, 20]:
            line = f"替股 x={x:<14}"
            for hc in [0.0, 0.02, 0.04]:
                nav, _ = run(ver, 40 - x, x, 60, op, cl, cfg["exec_price"], dates, hc)
                m = bt.metrics(nav)
                line += f"{m['年化收益率']:>9.2%}" if hc else f"{m['年化收益率']:>8.2%}"
                rows.append({"版本": ver, "方案": f"压测 替股x={x} 扣{int(hc*100)}pct",
                             "红利": 40 - x, "黄金": x, "国债": 60,
                             "年化": m["年化收益率"], "回撤": m["最大回撤"],
                             "波动": m["年化波动"], "夏普": m["夏普比率"],
                             "Calmar": m["Calmar比率"], "最差20日": worst_20d(nav),
                             "与基准相关": np.nan})
            print(line)
        print()

    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")
    pd.DataFrame(yearly).to_csv("output/exp/gold_yearly.csv", index=False, encoding="utf-8")
    print(f"结果已保存：{OUT}")
    plot(navs)


def plot(navs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), dpi=140,
                             gridspec_kw={"width_ratios": [1.7, 1]})

    ax = axes[0]
    base = navs[("trade", "基准 40/60")]
    styles = {"替债 x=10": "#C77A1E", "替债 x=20": "#8B2A1D",
              "替股 x=10": "#2E8B57", "替股 x=20": "#1F4E79"}
    for name, color in styles.items():
        nav = navs[("trade", name)]
        rel = nav / base.reindex(nav.index).ffill()
        ax.plot(rel.index, rel, color=color, linewidth=1.8, label=name)
    ax.axhline(1.0, color="#555", linewidth=0.9, linestyle=":")
    ax.set_title("加黄金后相对 40/60 的净值比（可交易版）", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25, linestyle="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax = axes[1]
    y = pd.read_csv("output/exp/gold_yearly.csv")
    y = y[y["版本"] == "trade"]
    ax.bar(y["年份"], y["黄金"], color=["#B03A2E" if v >= 0 else "#2E8B57" for v in y["黄金"]],
           width=0.68)
    for yr, v in zip(y["年份"], y["黄金"]):
        ax.text(yr, v + (0.012 if v >= 0 else -0.028), f"{v:.0%}", ha="center", fontsize=7.2)
    ax.axhline(0, color="#333", linewidth=0.9)
    ax.set_ylim(float(y["黄金"].min()) * 1.22, float(y["黄金"].max()) * 1.18)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("黄金自身分年度收益（可交易版区间）", fontsize=11)
    ax.grid(alpha=0.25, axis="y", linestyle="--")

    fig.tight_layout()
    fig.savefig(CHART, bbox_inches="tight")
    plt.close(fig)
    print(f"图已保存：{CHART}")


if __name__ == "__main__":
    main()
