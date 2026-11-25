"""
第 5 关（重构版）· 波动率目标化能不能补上 2015 股灾那个洞

固定 40/60 的账已经算清：可交易版年化 6.9%、回撤 −6.8%，几乎挑不出毛病。
唯一的弱点是长周期版 2015 年那次 −17.4%——股灾来得太快，固定比例来不及反应。

波动率目标化的思路是：**不预测方向，只盯着波动**。股灾的第一特征不是「跌」，
而是「波动突然放大」；波动一放大就按比例收缩权益，等波动回落再恢复。
这是唯一被保留下来的「动态」成分，因为它响应的是风险，不是收益预期。

要看的指标（按重要性）：
  1. 长周期版最大回撤能否从 −17.4% 压到 −12% 以内
  2. 为此付出的年化代价有多大（不超过 1 个百分点算划算）
  3. 样本外是否同步改善（不能只在样本内好看）

判读标准（提前写死）：
  回撤改善 ≥ 3 个百分点，且年化损失 ≤ 1 个百分点 → 采用。
  达不到 → 承认它没用，主策略就走纯固定比例。

输出：
  output/exp/vol_summary.csv
  charts/13_波动率目标化.png
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import allocation as al  # noqa: E402
import backtest as bt  # noqa: E402

EXP_DIR = "output/exp"


def main():
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs("charts", exist_ok=True)
    rows, navs = [], {}

    for ver in ["long", "trade"]:
        cfg = bt.VERSIONS[ver]
        ins_end = pd.Timestamp(cfg["insample_end"])
        op, cl = bt.load_prices(cfg["prices"])

        for mode in al.MODES:
            w, _ = al.build(ver, mode)
            nav, fee = bt.simulate(w, op, cl, cfg["exec_price"])
            navs[(ver, mode)] = nav

            ins, out = nav[nav.index <= ins_end], nav[nav.index > ins_end]
            mi, mo, mf = bt.metrics(ins), bt.metrics(out), bt.metrics(nav)
            rows.append({
                "版本": ver, "模式": al.MODES[mode]["label"],
                "平均红利": float(w["w_dividend"].mean()),
                "收缩月份": float((w["scale"] < 1).mean()),
                "内_年化": mi["年化收益率"], "内_回撤": mi["最大回撤"], "内_夏普": mi["夏普比率"],
                "外_年化": mo["年化收益率"], "外_回撤": mo["最大回撤"], "外_夏普": mo["夏普比率"],
                "全_年化": mf["年化收益率"], "全_回撤": mf["最大回撤"], "全_夏普": mf["夏普比率"],
                "手续费": fee,
            })

    df = pd.DataFrame(rows)
    df.to_csv(f"{EXP_DIR}/vol_summary.csv", index=False, encoding="utf-8")

    for ver in ["long", "trade"]:
        print()
        print("=" * 104)
        print(f"{bt.VERSIONS[ver]['label']}：固定比例 vs 波动率目标化")
        print("=" * 104)
        print(f"{'模式':<26}{'红利均值':>8}{'全回撤':>8}{'全年化':>8}{'全夏普':>7}"
              f"{'外年化':>8}{'外回撤':>8}{'外夏普':>7}")
        print("-" * 104)
        sub = df[df["版本"] == ver]
        for _, r in sub.iterrows():
            print(f"{r['模式']:<26}{r['平均红利']:>8.0%}{r['全_回撤']:>8.1%}{r['全_年化']:>8.1%}"
                  f"{r['全_夏普']:>7.2f}{r['外_年化']:>8.1%}{r['外_回撤']:>8.1%}{r['外_夏普']:>7.2f}")

        base = sub[sub["模式"].str.startswith("固定")].iloc[0]
        cand = sub[~sub["模式"].str.startswith("固定")].sort_values("全_回撤", ascending=True)
        top = cand.iloc[0]   # 回撤数值最小（跌得最少）的一档
        print()
        print(f"回撤压得最低的一档：{top['模式']}")
        print(f"  全区间回撤 {top['全_回撤']:.1%}（固定比例 {base['全_回撤']:.1%}，"
              f"改善 {base['全_回撤']-top['全_回撤']:.1%}）")
        print(f"  为此付出的年化代价 {base['全_年化']-top['全_年化']:+.1%}（负数表示年化反而更高）")

    plot(df, navs)


def plot(df, navs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    colors = {"fixed": "#8C8C8C", "vol12": "#1D9E75", "vol10": "#378ADD", "vol08": "#D9534F"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=140)
    for ax, ver, title in zip(axes, ["long", "trade"],
                              ["长周期版（2015 起）", "可交易版（2019 起）"]):
        for mode, color in colors.items():
            nav = navs[(ver, mode)]
            ax.plot(nav.index, nav / nav.iloc[0], label=al.MODES[mode]["label"],
                    color=color, linewidth=1.6)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("净值")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(fontsize=7.5, loc="upper left")
        # 标出 2015 股灾区间，方便看波动率目标化到底有没有起作用
        if ver == "long":
            ax.axvspan(pd.Timestamp("2015-06-01"), pd.Timestamp("2015-09-30"),
                       color="#D9534F", alpha=0.08)
            ax.annotate("2015 股灾", (pd.Timestamp("2015-07-15"), ax.get_ylim()[0]),
                        fontsize=7.5, color="#D9534F", ha="center", va="bottom")
    fig.suptitle("波动率目标化：能否在股灾里自动收缩（阴影为 2015 年 6–9 月）", fontsize=12)
    fig.tight_layout()
    p = "charts/13_波动率目标化.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"\n图已保存：{p}")


if __name__ == "__main__":
    main()
