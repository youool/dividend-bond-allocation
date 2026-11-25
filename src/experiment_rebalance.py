# -*- coding: utf-8 -*-
"""
i1 再平衡的价值：频率、带宽、以及「完全不再平衡」。

要回答的问题：每月把比例拉回 40/60，这件事本身贡献了什么？
六种做法，同一张波动率闸门表、同一区间：

  A 月度 + 闸门          主线
  B 月度、不做闸门        固定 40/60，分离「再平衡」与「风控」
  C 季度 + 闸门
  D 半年 + 闸门
  E ±5% 带宽 + 闸门       偏离不足 5 个点就不动
  F ±3% 带宽 + 闸门
  G 完全不再平衡          买入 40/60 后任其漂移

为什么必须包含 G：再平衡溢价可能是负的。如果 G 反而更好，
那「每月拉回」就是成本而不是收益，这个结论比优化参数重要。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as bt

OUT = "output/exp/rebalance.csv"
CHART = "charts/16_再平衡_频率与带宽.png"


def simulate(weights, op, cl, band=0.0, rebalance=True, start=None,
             reset_months=None):
    """
    bt.simulate 的可选再平衡版本。
    band>0       : 偏离不足就不交易
    rebalance=F  : 只建仓一次，之后任其漂移
    reset_months : 只在这些月份做「拉回目标」，但闸门变化仍然当月执行
                   （用来把「再平衡频率」和「风控响应速度」拆开）
    """
    w = weights.sort_values("date").reset_index(drop=True)
    start_ts = pd.Timestamp(start) if start is not None else pd.Timestamp(w["date"].iloc[0])
    days = cl.index[cl.index >= start_ts]

    exec_map = {}
    for d in w["date"]:
        pos = days.searchsorted(d, side="right")
        if pos < len(days):
            exec_map[days[pos]] = d

    eq_idx = bt.ASSETS.index("dividend")
    shares = pd.Series(0.0, index=bt.ASSETS)
    nav_rows, eq_rows = [], []
    fee_total, n_trade = 0.0, 0
    invested = False
    last_target = None

    for t in days:
        if t in exec_map:
            s = exec_map[t]
            target = w.loc[w["date"] == s, list(bt.WEIGHT_COLS)].iloc[0].values.astype(float)
            px = op.loc[t, bt.ASSETS].values
            cur = shares.values * px

            if np.isnan(cur[shares.values != 0]).any():
                raise ValueError(f"{t.date()} 持仓价格缺失")

            total = float(np.nansum(cur)) if invested else bt.INIT_CAPITAL
            actual = (cur / total) if total > 0 else np.zeros(len(bt.ASSETS))

            if not rebalance:
                need = not invested
            elif not invested:
                need = True
            elif reset_months is not None:
                gate_moved = (last_target is None or
                              abs(target[eq_idx] - last_target[eq_idx]) > 1e-9)
                need = gate_moved or (t.month in reset_months)
            elif band > 0:
                need = abs(actual[eq_idx] - target[eq_idx]) >= band
            else:
                need = True

            if need:
                trade_amt = float(np.abs(total * target - np.nan_to_num(cur)).sum())
                fee = trade_amt * bt.FEE_RATE
                fee_total += fee
                n_trade += 1
                shares = pd.Series(
                    np.where(target > 0, (total - fee) * target / px, 0.0), index=bt.ASSETS)
                invested = True
                last_target = target.copy()

        if not invested:
            nav_rows.append((t, bt.INIT_CAPITAL))
            eq_rows.append((t, 0.0))
            continue

        vals = shares.values * cl.loc[t, bt.ASSETS].values
        if np.isnan(vals[shares.values != 0]).any():
            raise ValueError(f"{t.date()} 收盘价缺失")
        nav = float(np.nansum(vals))
        nav_rows.append((t, nav))
        eq_rows.append((t, float(shares["dividend"] * cl.loc[t, "dividend"] / nav)))

    return (pd.Series(dict(nav_rows)).sort_index(),
            fee_total, pd.Series(dict(eq_rows)).sort_index(), n_trade)


def main():
    os.makedirs("output/exp", exist_ok=True)
    os.makedirs("charts", exist_ok=True)
    rows, navs = [], {}

    for ver, label in [("long", "长周期版"), ("trade", "可交易版")]:
        cfg = bt.VERSIONS[ver]
        op, cl = bt.load_prices(cfg["prices"])
        base = pd.read_csv(f"output/alloc_{ver}_vol08.csv", parse_dates=["date"])

        # 无闸门版：同一批日期，比例恒为 40/60
        fixed = base.copy()
        fixed["w_dividend"], fixed["w_bond"] = 0.40, 0.60
        for c in ["w_broad", "w_cash"]:
            fixed[c] = 0.0

        # 季度 / 半年：只在这些月份的月末调仓
        q = base[base["date"].dt.month.isin([3, 6, 9, 12])]
        h = base[base["date"].dt.month.isin([6, 12])]

        plans = [
            ("A 月度+闸门（主线）", base, dict()),
            ("B 月度·无闸门", fixed, dict()),
            ("C 季度+闸门", q, dict()),
            ("C2 季度重置·闸门月度", base, dict(reset_months=[3, 6, 9, 12])),
            ("D 半年+闸门", h, dict()),
            ("D2 半年重置·闸门月度", base, dict(reset_months=[6, 12])),
            ("E ±5%带宽+闸门", base, dict(band=0.05)),
            ("F ±3%带宽+闸门", base, dict(band=0.03)),
            ("G 完全不再平衡", fixed, dict(rebalance=False)),
        ]

        print("=" * 96)
        print(f"{label}（{cfg['prices'] and ''}{base['date'].iloc[0].date()} 起，"
              f"至 {base['date'].iloc[-1].date()}）")
        print("=" * 96)
        print(f"{'方案':<20}{'年化':>8}{'最大回撤':>10}{'Calmar':>8}{'波动':>8}"
              f"{'累计换手':>10}{'调仓次数':>9}{'平均权益':>9}{'期末权益':>9}")
        print("-" * 96)

        for name, wdf, kw in plans:
            nav, fee, eq, n = simulate(wdf, op, cl, **kw)
            navs[(ver, name)] = nav
            m = bt.metrics(nav)
            turnover = fee / bt.FEE_RATE / bt.INIT_CAPITAL
            rows.append({
                "版本": ver, "方案": name, "年化": m["年化收益率"], "回撤": m["最大回撤"],
                "波动": m["年化波动"], "夏普": m["夏普比率"], "Calmar": m["Calmar比率"],
                "累计换手": turnover, "调仓次数": n,
                "平均权益": float(eq.mean()), "期末权益": float(eq.iloc[-1]),
            })
            print(f"{name:<20}{m['年化收益率']:>8.2%}{m['最大回撤']:>10.2%}"
                  f"{m['Calmar比率']:>8.2f}{m['年化波动']:>8.2%}"
                  f"{turnover:>10.2f}{n:>9d}{eq.mean():>9.1%}{eq.iloc[-1]:>9.1%}")

        # H：与「完全不再平衡」平均仓位相同的固定比例。
        # 不做这一步，G 的高收益会看起来像「不调仓更好」，其实只是权益漂到了 50% 以上。
        g_avg = float(rows[-1]["平均权益"])
        h_w = fixed.copy()
        h_w["w_dividend"], h_w["w_bond"] = g_avg, 1 - g_avg
        nav, fee, eq, n = simulate(h_w, op, cl)
        navs[(ver, "H 同暴露固定比例")] = nav
        m = bt.metrics(nav)
        turnover = fee / bt.FEE_RATE / bt.INIT_CAPITAL
        rows.append({"版本": ver, "方案": "H 同暴露固定比例", "年化": m["年化收益率"],
                     "回撤": m["最大回撤"], "波动": m["年化波动"], "夏普": m["夏普比率"],
                     "Calmar": m["Calmar比率"], "累计换手": turnover, "调仓次数": n,
                     "平均权益": float(eq.mean()), "期末权益": float(eq.iloc[-1])})
        print(f"{'H 同暴露固定(G的对照)':<20}{m['年化收益率']:>8.2%}{m['最大回撤']:>10.2%}"
              f"{m['Calmar比率']:>8.2f}{m['年化波动']:>8.2%}"
              f"{turnover:>10.2f}{n:>9d}{eq.mean():>9.1%}{eq.iloc[-1]:>9.1%}")
        print()

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"结果已保存：{OUT}")
    plot(navs)
    return df


def plot(navs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), dpi=140,
                             gridspec_kw={"width_ratios": [2.1, 1]})

    styles = {
        "A 月度+闸门（主线）": ("#1F4E79", 2.4, 1.0),
        "B 月度·无闸门": ("#8FA8BF", 1.3, 0.9),
        "C 季度+闸门": ("#C77A1E", 1.4, 0.95),
        "C2 季度重置·闸门月度": ("#E0A458", 1.4, 0.95),
        "D 半年+闸门": ("#7A5AA6", 1.4, 0.95),
        "D2 半年重置·闸门月度": ("#A58AC7", 1.4, 0.95),
        "E ±5%带宽+闸门": ("#2E8B57", 1.3, 0.9),
        "F ±3%带宽+闸门": ("#B03A2E", 1.3, 0.9),
        "G 完全不再平衡": ("#555555", 1.4, 1.0),
    }
    for ver in ["trade"]:
        base = navs[(ver, "A 月度+闸门（主线）")]
        ax = axes[0]
        for (v, name), nav in navs.items():
            if v != ver:
                continue
            color, lw, alpha = styles.get(name, ("#999", 1, 1))
            rel = nav / base.reindex(nav.index).ffill()
            ax.plot(rel.index, rel, color=color, linewidth=lw, alpha=alpha, label=name)
        ax.axhline(1.0, color="#1F4E79", linewidth=0.8, linestyle=":")
        ax.set_title("相对主线的净值比（可交易版，主线=1.0）", fontsize=11)
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(fontsize=7.6, loc="upper left", framealpha=0.92)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax = axes[1]
    d = pd.read_csv(OUT)
    d = d[d["版本"] == "trade"]
    y = np.arange(len(d))
    ax.barh(y, d["累计换手"], color="#B7C9D9", height=0.62)
    for i, (v, n) in enumerate(zip(d["累计换手"], d["调仓次数"])):
        ax.text(v + 0.15, i, f"{v:.1f}× / {n}次", va="center", fontsize=7.6, color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("+闸门", "").replace("（主线）", "") for s in d["方案"]], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("累计换手（倍，双边）")
    ax.set_xlim(0, float(d["累计换手"].max()) * 1.30)
    ax.set_title("换手成本：带宽能省多少", fontsize=11)
    ax.grid(alpha=0.25, axis="x", linestyle="--")

    fig.tight_layout()
    fig.savefig(CHART, bbox_inches="tight")
    plt.close(fig)
    print(f"图已保存：{CHART}")


if __name__ == "__main__":
    main()
