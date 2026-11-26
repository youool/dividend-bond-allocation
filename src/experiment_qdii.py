# -*- coding: utf-8 -*-
"""
i7 加纳斯达克 QDII：三类收益来源必须分开看

组合层面：与黄金/创业板同样的口径（固定比例、不加闸门、同起点）
  基准 40/60
  替股 x：红利 40−x / 纳指 x / 国债 60     （风险资产仍是 40%）
  替债 x：红利 40 / 纳指 x / 国债 60−x     （风险资产增加）

收益分解：513100 用人民币计价，涨跌 = 纳指(USD) × 汇率 × (1+溢价与跟踪误差)
  把三者拆开，才能回答「这部分收益可持续吗」。
  分解用月末对齐（美股收盘晚于 A 股，日用前一日收盘近似，误差很小）。

压测：纳指收益扣 4 / 8 个百分点；人民币每年升值 3 个百分点（汇率逆风）。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as bt
import experiment_gold as eg

QDII = "data/513100.csv"
OUT = "output/exp/qdii.csv"
OUT_D = "output/exp/qdii_decomp.csv"
CHART = "charts/20_纳指_组合效果与收益分解.png"


def load_series(path, col="close"):
    d = pd.read_csv(path, parse_dates=["date"]).set_index("date")[col].astype(float)
    return d[~d.index.duplicated()].sort_index()


def decompose():
    """月末对齐，把 ETF 的人民币收益拆成 纳指(USD) + 汇率 + 残差。"""
    etf = load_series(QDII)
    ndx = load_series("data/NDX.csv")
    fx = load_series("data/USDCNY.csv", col="mid")

    # 取每月最后一个可用值
    m_etf = etf.resample("ME").last()
    m_ndx = ndx.resample("ME").last()
    m_fx = fx.resample("ME").last()
    df = pd.DataFrame({"etf": m_etf, "ndx": m_ndx, "fx": m_fx}).dropna()
    df = df[df.index >= "2015-01-31"]

    rows = []
    for i in range(1, len(df)):
        p, c = df.iloc[i - 1], df.iloc[i]
        r_etf = c["etf"] / p["etf"] - 1
        r_ndx = c["ndx"] / p["ndx"] - 1
        r_fx = c["fx"] / p["fx"] - 1
        theo = (1 + r_ndx) * (1 + r_fx) - 1
        rows.append({"月份": df.index[i], "ETF": r_etf, "纳指USD": r_ndx,
                     "汇率": r_fx, "理论CNY": theo, "残差": (1 + r_etf) / (1 + theo) - 1})

    m = pd.DataFrame(rows).set_index("月份")
    yrows = []
    for y, g in m.groupby(m.index.year):
        r = {"年份": int(y)}
        for c in ["ETF", "纳指USD", "汇率", "理论CNY"]:
            r[c] = float((1 + g[c]).prod() - 1)
        r["残差"] = (1 + r["ETF"]) / (1 + r["理论CNY"]) - 1
        yrows.append(r)
    yearly = pd.DataFrame(yrows).set_index("年份")
    yearly.to_csv(OUT_D, encoding="utf-8")

    n = len(m)
    total = {c: (1 + m[c]).prod() - 1 for c in ["ETF", "纳指USD", "汇率", "理论CNY"]}
    ann = {k: (1 + v) ** (12 / n) - 1 for k, v in total.items()}
    print("=" * 78)
    print(f"513100 收益分解（{m.index[0].date()} → {m.index[-1].date()}，{n} 个月，月末对齐）")
    print("=" * 78)
    print(f"{'':<12}{'累计':>12}{'年化':>10}")
    for k in ["ETF", "纳指USD", "汇率", "理论CNY"]:
        print(f"{k:<12}{total[k]:>12.2%}{ann[k]:>10.2%}")
    print(f"{'残差(年化)':<12}{'':>12}{ann['ETF'] - ann['理论CNY']:>10.2%}")
    print()
    print("分年度贡献（百分点）：")
    print(f"{'年份':<8}{'纳指USD':>10}{'汇率':>9}{'理论':>9}{'ETF实际':>10}{'残差':>9}")
    for y, r in yearly.iterrows():
        print(f"{y:<8}{r['纳指USD']:>10.1%}{r['汇率']:>9.1%}{r['理论CNY']:>9.1%}"
              f"{r['ETF']:>10.1%}{r['残差']:>9.1%}")
    print()
    return yearly


def build_theoretical():
    """
    用「纳指 × 汇率」合成一条按净值成交的 QDII 曲线，用来剥离溢价。
    如果有溢价，ETF 价格会高于净值；买入即承担溢价，卖出时溢价可能消失。
    回测里若不剥离，2024/2026 那种溢价扩张会被记成「纳指涨得好」。
    """
    etf = load_series(QDII)
    ndx = load_series("data/NDX.csv").reindex(etf.index).ffill()
    fx = load_series("data/USDCNY.csv", col="mid").reindex(etf.index).ffill()
    df = pd.DataFrame({"ndx": ndx, "fx": fx}).dropna()
    r = (1 + df["ndx"].pct_change().fillna(0)) * (1 + df["fx"].pct_change().fillna(0)) - 1
    close = (1 + r).cumprod()
    out = pd.DataFrame({"date": close.index, "open": close.shift(1).fillna(close.iloc[0]),
                        "close": close.values})
    path = "data/synth_ndx_cny.csv"
    out.to_csv(path, index=False, encoding="utf-8")
    print(f"合成「纳指×汇率」净值曲线：{path}（{len(out)} 行，按净值成交）")
    return path


def build_gated(ver, qdii_share, base=0.40, floor=0.20, target=0.08, window=60, max_eq=0.50):
    """
    把纳指接进主线：风险资产仍是 40%，内部按 (1−share) : share 分给红利与纳指，
    波动率闸门按「真实的三资产组合」重算，而不是沿用原来的 40/0/60 权重表。
    qdii_share=0 时必须与 output/alloc_{ver}_vol08.csv 完全一致，用来验证实现。
    """
    (op, cl), cfg = eg.prices_with_gold(ver, QDII)
    alloc = pd.read_csv(f"output/alloc_{ver}_vol08.csv", parse_dates=["date"])
    dates = pd.DatetimeIndex(alloc["date"])
    rets = cl[["dividend", "broad", "bond"]].pct_change()

    rows, prev_eq = [], base
    for d in dates:
        hist = rets.loc[rets.index < d].tail(window)
        port = (prev_eq * ((1 - qdii_share) * hist["dividend"] + qdii_share * hist["broad"])
                + (1 - prev_eq) * hist["bond"])
        valid = int(port.notna().sum())
        realized = (float(port.std() * np.sqrt(252))
                    if valid >= 30 and valid == len(port) else np.nan)
        if np.isnan(realized) or realized <= 0:
            eq = base
        else:
            eq = float(np.clip(base * (target / realized), floor, max_eq))
        rows.append({"date": d, "w_dividend": eq * (1 - qdii_share),
                     "w_broad": eq * qdii_share, "w_bond": 1 - eq, "w_cash": 0.0})
        prev_eq = eq
    return pd.DataFrame(rows)


def main():
    os.makedirs("output/exp", exist_ok=True)
    decomp = decompose()
    theo_path = build_theoretical()

    rows, navs, yearly = [], {}, []
    for ver in ["long", "trade"]:
        (op, cl), cfg = eg.prices_with_gold(ver, QDII)
        alloc = pd.read_csv(f"output/alloc_{ver}_vol08.csv", parse_dates=["date"])
        dates = pd.DatetimeIndex(alloc["date"])
        start = dates[0]

        plans = [("基准 40/60", 40, 0, 60)]
        for x in [5, 10, 15, 20]:
            plans.append((f"替股 x={x}", 40 - x, x, 60))
        for x in [10, 20]:
            plans.append((f"替债 x={x}", 40, x, 60 - x))

        print("=" * 100)
        print(f"{cfg['label']}（{start.date()} 起，固定比例、不加闸门）")
        print("=" * 100)
        print(f"{'方案':<14}{'年化':>8}{'最大回撤':>10}{'Calmar':>8}{'波动':>8}"
              f"{'最差20日':>10}{'与红利相关':>11}")
        print("-" * 100)

        for name, dv, qd, bd in plans:
            nav, _ = eg.run(ver, dv, qd, bd, op, cl, cfg["exec_price"], dates)
            navs[(ver, name)] = nav
            m = bt.metrics(nav)
            corr = cl["dividend"].pct_change().corr(cl["broad"].pct_change())
            rows.append({"版本": ver, "方案": name, "红利": dv, "纳指": qd, "国债": bd,
                         "年化": m["年化收益率"], "回撤": m["最大回撤"],
                         "波动": m["年化波动"], "夏普": m["夏普比率"],
                         "Calmar": m["Calmar比率"], "最差20日": eg.worst_20d(nav),
                         "与红利相关": corr})
            print(f"{name:<14}{m['年化收益率']:>8.2%}{m['最大回撤']:>10.2%}"
                  f"{m['Calmar比率']:>8.2f}{m['年化波动']:>8.2%}"
                  f"{eg.worst_20d(nav):>10.2%}{corr:>11.2f}")

        print()
        print("分年度：纳指在哪几年救了组合，哪几年拖了后腿")
        print(f"{'年份':<8}{'纳指':>9}{'基准':>9}{'替股x=10':>11}{'替股x=20':>11}")
        print("-" * 52)
        for y in sorted({d.year for d in dates}):
            gs = cl["broad"][cl["broad"].index.year == y]
            gr = gs.iloc[-1] / gs.iloc[0] - 1 if len(gs) > 60 else np.nan
            vals = []
            for k in ["基准 40/60", "替股 x=10", "替股 x=20"]:
                nav = navs[(ver, k)]
                s = nav[nav.index.year == y]
                vals.append(s.iloc[-1] / s.iloc[0] - 1 if len(s) > 60 else np.nan)
            yearly.append({"版本": ver, "年份": int(y), "纳指": gr,
                           "基准": vals[0], "x10": vals[1], "x20": vals[2]})
            print(f"{y:<8}{gr:>9.1%}{vals[0]:>9.1%}{vals[1]:>11.1%}{vals[2]:>11.1%}")

        # 剥离溢价：把 ETF 价格换成「纳指×汇率」的合成净值，再跑同样两档
        (op2, cl2), _ = eg.prices_with_gold(ver, theo_path)
        print()
        print("剥离 QDII 溢价后（假设始终按净值成交）：")
        print(f"{'方案':<18}{'年化':>9}{'最大回撤':>10}{'Calmar':>9}")
        print("-" * 50)
        for x in [10, 20]:
            nav, _ = eg.run(ver, 40 - x, x, 60, op2, cl2, cfg["exec_price"], dates)
            m = bt.metrics(nav)
            rows.append({"版本": ver, "方案": f"扣溢价 替股 x={x}", "红利": 40 - x,
                         "纳指": x, "国债": 60, "年化": m["年化收益率"],
                         "回撤": m["最大回撤"], "波动": m["年化波动"],
                         "夏普": m["夏普比率"], "Calmar": m["Calmar比率"],
                         "最差20日": eg.worst_20d(nav), "与红利相关": np.nan})
            print(f"{'替股 x=' + str(x):<18}{m['年化收益率']:>9.2%}"
                  f"{m['最大回撤']:>10.2%}{m['Calmar比率']:>9.2f}")

        # 接入主线：闸门按三资产组合重算
        print()
        print("接入主线（风险资产仍 40%，闸门按真实三资产组合重算）：")
        print(f"{'方案':<20}{'年化':>9}{'最大回撤':>10}{'Calmar':>9}{'平均权益':>10}")
        print("-" * 60)
        for share, label in [(0.0, "主线 40/60"), (0.25, "红利30/纳指10/债60"),
                             (0.5, "红利20/纳指20/债60")]:
            w = build_gated(ver, share)
            if share == 0:
                diff = np.abs(w["w_dividend"].values - alloc["w_dividend"].values).max()
                print(f"  自检：share=0 与主线权重表最大差异 {diff:.2e}")
            nav, _ = bt.simulate(w, op, cl, cfg["exec_price"], start=dates[0])
            m = bt.metrics(nav)
            avg_eq = w["w_dividend"].mean() + w["w_broad"].mean()
            print(f"{label:<20}{m['年化收益率']:>9.2%}{m['最大回撤']:>10.2%}"
                  f"{m['Calmar比率']:>9.2f}{avg_eq:>10.1%}")
            rows.append({"版本": ver, "方案": f"接入主线 {label}", "红利": round((1 - share) * 40),
                         "纳指": round(share * 40), "国债": 60,
                         "年化": m["年化收益率"], "回撤": m["最大回撤"],
                         "波动": m["年化波动"], "夏普": m["夏普比率"],
                         "Calmar": m["Calmar比率"], "最差20日": eg.worst_20d(nav),
                         "与红利相关": np.nan})

        # 压测：美股收益打折 + 汇率逆风
        print()
        print("压测（替股 x=10 / x=20）：")
        print(f"{'情形':<22}{'x=10':>10}{'x=20':>10}")
        print("-" * 44)
        scenarios = [("基准情形", 0.0), ("纳指收益扣 4pct", 0.04),
                     ("纳指收益扣 8pct", 0.08), ("汇率逆风（每年 −3pct）", 0.03)]
        for label, hc in scenarios:
            line = f"{label:<22}"
            for x in [10, 20]:
                nav, _ = eg.run(ver, 40 - x, x, 60, op, cl, cfg["exec_price"], dates, hc)
                line += f"{bt.metrics(nav)['年化收益率']:>10.2%}"
            print(line)
            for x in [10, 20]:
                nav, _ = eg.run(ver, 40 - x, x, 60, op, cl, cfg["exec_price"], dates, hc)
                m = bt.metrics(nav)
                rows.append({"版本": ver, "方案": f"压测 {label} x={x}", "红利": 40 - x,
                             "纳指": x, "国债": 60, "年化": m["年化收益率"],
                             "回撤": m["最大回撤"], "波动": m["年化波动"],
                             "夏普": m["夏普比率"], "Calmar": m["Calmar比率"],
                             "最差20日": eg.worst_20d(nav), "与红利相关": np.nan})
        print()

    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")
    pd.DataFrame(yearly).to_csv("output/exp/qdii_yearly.csv", index=False, encoding="utf-8")
    print(f"结果已保存：{OUT} / {OUT_D}")
    plot(navs, decomp)


def plot(navs, decomp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9), dpi=140,
                             gridspec_kw={"width_ratios": [1.65, 1]})

    ax = axes[0]
    base = navs[("long", "基准 40/60")]
    for name, color in [("替股 x=10", "#C77A1E"), ("替股 x=20", "#8B2A1D"),
                        ("替债 x=20", "#2E8B57")]:
        nav = navs[("long", name)]
        ax.plot(nav.index, nav / base.reindex(nav.index).ffill(), color=color,
                linewidth=1.7, label=name)
    ax.axhline(1.0, color="#555", linewidth=0.9, linestyle=":")
    ax.set_title("加纳指后相对 40/60 的净值比（长周期版，含 2015）", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25, linestyle="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax = axes[1]
    years = decomp.index.astype(str)
    x = np.arange(len(years))
    w = 0.36
    ax.bar(x - w / 2, decomp["纳指USD"], width=w, color="#1F4E79", label="纳指（美元）")
    ax.bar(x + w / 2, decomp["汇率"], width=w, color="#C77A1E", label="汇率（人民币贬值）")
    ax.plot(x, decomp["ETF"], "o", color="#B03A2E", markersize=4.5, label="ETF 实际（人民币）")
    ax.axhline(0, color="#333", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=8, rotation=0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("513100 的收益拆解：纳指、汇率与残差", fontsize=11)
    ax.legend(fontsize=7.8, loc="upper left")
    ax.grid(alpha=0.25, axis="y", linestyle="--")

    fig.tight_layout()
    fig.savefig(CHART, bbox_inches="tight")
    plt.close(fig)
    print(f"图已保存：{CHART}")


if __name__ == "__main__":
    main()
