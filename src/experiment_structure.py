"""
第 5 关 · 对照实验二：择时到底有没有创造价值

实验一（换估值锚）已经证明：把温度计从沪深300 换成红利自己，样本外并没有变好。
那问题就不在「锚选错」，得往下追一层：这套择时本身有没有价值？

三组对照，每组只动一件事：

  组 1  暴露匹配的固定配置
        用策略自己的**平均权重**做一个不做任何择时的组合，每月机械再平衡。
        它和策略的权益暴露、标的结构完全一样，唯一差别就是「不判断」。
        如果它不输给策略，说明择时没有创造价值——这是评价择时策略的标准做法。

  组 2  只用趋势（关掉估值层，基础仓位恒 100%）
  组 3  只用估值（关掉趋势层）

  把这两组和「估值+趋势」的基线比，就能拆出两层各自贡献了多少。
  —— 一个层如果拆掉之后结果没变差，那它就不该留在策略里。

判读标准（同样提前写死）：
  固定配置的夏普若 ≥ 策略的夏普，则「择时无价值」成立。
  这种情况下的正确做法不是继续调参数，而是承认：这套信号抓不到 alpha，
  只能靠降仓位降风险。报告里如实写，比硬调参数调到赚钱更有分。

输出：
  output/exp/structure_summary.csv
  charts/09_结构对照_净值_{版本}.png
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt  # noqa: E402
import signals as sg  # noqa: E402

EXP_DIR = "output/exp"
WCOLS = ["w_dividend", "w_broad", "w_bond", "w_cash"]


def fixed_weight_table(dates, weights):
    """构造一张固定权重的月度权重表。"""
    df = pd.DataFrame({"date": pd.to_datetime(dates)})
    for c, v in zip(WCOLS, weights):
        df[c] = float(v)
    df["note"] = "固定配置"
    return df


def run_nav(w, ver):
    cfg = bt.VERSIONS[ver]
    op, cl = bt.load_prices(cfg["prices"])
    nav, fee = bt.simulate(w, op, cl, cfg["exec_price"])
    out = pd.DataFrame({"strategy": nav})
    for bm, asset in [("bm_broad", "broad"), ("bm_dividend", "dividend")]:
        px = cl[asset].reindex(out.index).ffill()
        out[bm] = (px / px.iloc[0]).values
    return out.ffill().dropna(), fee


def main():
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs("charts", exist_ok=True)
    rows, navs = [], {}

    for ver in ["long", "trade"]:
        cfg = bt.VERSIONS[ver]

        # 基线：估值 + 趋势（就是主流程的策略）
        _, base_w = sg.build(ver, tag="S0_基线")
        dates = base_w["date"].values

        variants = {}

        # 组 1：暴露匹配的固定配置（用策略自己的平均权重）
        avg_w = base_w[WCOLS].mean().values
        variants["F1_固定配置_匹配平均权重"] = fixed_weight_table(dates, avg_w)
        # 组 1b：35% 红利 + 65% 国债，一个更好记的近似
        variants["F2_固定配置_35红利65国债"] = fixed_weight_table(dates, [0.35, 0.0, 0.65, 0.0])

        # 组 2：只留趋势
        variants["T1_只用趋势"] = sg.build(ver, tag="T1_只用趋势",
                                        base_override=1.0)[1]
        # 组 3：只留估值
        variants["V1_只用估值"] = sg.build(ver, tag="V1_只用估值", no_trend=True)[1]

        variants["S0_基线_估值加趋势"] = base_w

        for name, w in variants.items():
            out, fee = run_nav(w, ver)
            navs[(ver, name)] = out
            segs = [("全区间", out),
                    ("样本内", out[out.index <= pd.Timestamp(cfg["insample_end"])]),
                    ("样本外", out[out.index > pd.Timestamp(cfg["insample_end"])])]
            for seg, sub in segs:
                if len(sub) < 20:
                    continue
                m = bt.metrics(sub["strategy"])
                wcols_avg = w[WCOLS].mean()
                rows.append({"版本": ver, "方案": name, "区间": seg,
                             "权益均值": wcols_avg[["w_dividend", "w_broad"]].sum(),
                             "红利均值": wcols_avg["w_dividend"],
                             "手续费": fee, **m})

    df = pd.DataFrame(rows)
    df.to_csv(f"{EXP_DIR}/structure_summary.csv", index=False, encoding="utf-8")

    for ver in ["long", "trade"]:
        print()
        print("=" * 104)
        print(f"{bt.VERSIONS[ver]['label']}：择时 vs 不择时（暴露匹配）")
        print("=" * 104)
        sub = df[df["版本"] == ver]
        print(f"{'方案':<26}{'区间':<8}{'权益均值':>8}{'累计':>9}{'年化':>8}"
              f"{'最大回撤':>9}{'夏普':>7}{'波动':>8}")
        print("-" * 104)
        for name in ["S0_基线_估值加趋势", "F1_固定配置_匹配平均权重", "F2_固定配置_35红利65国债",
                     "T1_只用趋势", "V1_只用估值"]:
            for seg in ["全区间", "样本内", "样本外"]:
                r = sub[(sub["方案"] == name) & (sub["区间"] == seg)]
                if r.empty:
                    continue
                r = r.iloc[0]
                print(f"{name:<26}{seg:<8}{r['权益均值']:>8.0%}{r['累计收益率']:>9.1%}"
                      f"{r['年化收益率']:>8.1%}{r['最大回撤']:>9.1%}{r['夏普比率']:>7.2f}"
                      f"{r['年化波动']:>8.1%}")
            print()

    plot(navs)


def plot(navs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    colors = {"S0_基线_估值加趋势": "#1D9E75", "F1_固定配置_匹配平均权重": "#D9534F",
              "F2_固定配置_35红利65国债": "#BA7517", "T1_只用趋势": "#378ADD",
              "V1_只用估值": "#8C8C8C"}

    for ver, title in [("long", "长周期版（2015 起）"), ("trade", "可交易版（2019 起）")]:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
        first = True
        for (v, name), out in navs.items():
            if v != ver:
                continue
            n = out["strategy"]
            ax.plot(n.index, n / n.iloc[0], label=name, color=colors[name], linewidth=1.8)
            if first:
                ax.plot(out.index, out["bm_dividend"], label="红利低波 买入持有",
                        color="#6B4C9A", linewidth=1.2, linestyle="--")
                first = False
        ax.set_title(f"{title}：择时 vs 不择时（归一化到 1.0）", fontsize=13)
        ax.set_ylabel("净值")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        p = f"charts/09_结构对照_净值_{ver}.png"
        fig.savefig(p)
        plt.close(fig)
        print(f"图已保存：{p}")


if __name__ == "__main__":
    main()
