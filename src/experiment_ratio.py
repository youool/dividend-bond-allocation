"""
第 5 关 · 对照实验四：如果改走「配置」路线，比例该定多少

前三个实验合起来说明了一件事：这套择时信号没有 alpha，把仓位翻来覆去地调，
只是在反复打断「长期持有红利」这件事。而一个不做任何判断的固定配置反而更好。

那就把问题换个问法：**不做择时，只决定比例，能做到什么水平？**

扫三样东西：
  红利比例   20% → 80%，步长 10%（剩下的给国债）
  再平衡频率 月度 / 季度 / 年度
  防御资产   国债 / 货币（国债的票息更高，但也更长端、会被利率上行伤到）

同样的纪律：比例只能按样本内挑，样本外只做检验。

输出：
  output/exp/ratio_summary.csv
  charts/11_配置比例_扫描.png
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt  # noqa: E402

EXP_DIR = "output/exp"
WCOLS = ["w_dividend", "w_broad", "w_bond", "w_cash"]

VERSIONS = {
    "long": dict(bt.VERSIONS["long"], insample_end="2020-12-31"),
    "trade": dict(bt.VERSIONS["trade"], insample_end="2022-12-31"),
}


def build_weight_table(dates, div_w, def_w, freq):
    """按指定频率生成调仓日，非调仓日的权重沿用上一期（simulate 只在这些日子动手）。"""
    d = pd.Series(pd.to_datetime(dates)).sort_values().reset_index(drop=True)
    if freq == "monthly":
        keep = d
    elif freq == "quarterly":
        keep = d[d.dt.month.isin([3, 6, 9, 12])].reset_index(drop=True)
    else:  # yearly
        keep = d[d.dt.month == 12].reset_index(drop=True)

    df = pd.DataFrame({"date": keep})
    df["w_dividend"] = div_w
    df["w_broad"] = 0.0
    df["w_bond"] = def_w
    df["w_cash"] = 1.0 - div_w - def_w
    return df


def run(dates, div_w, def_asset, freq, ver):
    cfg = VERSIONS[ver]
    def_w = 1.0 - div_w if def_asset == "bond" else 0.0
    cash_w = 0.0 if def_asset == "bond" else 1.0 - div_w
    w = build_weight_table(dates, div_w, def_w, freq)
    w["w_cash"] = cash_w
    op, cl = bt.load_prices(cfg["prices"])
    nav, fee = bt.simulate(w, op, cl, cfg["exec_price"])
    return nav, fee


def main():
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs("charts", exist_ok=True)
    rows = []

    for ver in ["long", "trade"]:
        cfg = VERSIONS[ver]
        ins_end = pd.Timestamp(cfg["insample_end"])

        # 用策略的月末日历作为调仓日历
        ref = pd.read_csv(f"output/weights_{ver}.csv", parse_dates=["date"])
        dates = ref["date"].values

        for def_asset, def_name in [("bond", "国债"), ("cash", "货币")]:
            for freq in ["monthly", "quarterly", "yearly"]:
                for div_w in np.arange(0.2, 0.81, 0.1):
                    nav, fee = run(dates, round(float(div_w), 2), def_asset, freq, ver)
                    ins, out = nav[nav.index <= ins_end], nav[nav.index > ins_end]
                    mi, mo = bt.metrics(ins), bt.metrics(out)
                    mf = bt.metrics(nav)
                    rows.append({
                        "版本": ver, "防御资产": def_name, "频率": freq, "红利比例": round(float(div_w), 2),
                        "内_年化": mi["年化收益率"], "内_回撤": mi["最大回撤"], "内_夏普": mi["夏普比率"],
                        "外_年化": mo["年化收益率"], "外_回撤": mo["最大回撤"], "外_夏普": mo["夏普比率"],
                        "全_年化": mf["年化收益率"], "全_回撤": mf["最大回撤"], "全_夏普": mf["夏普比率"],
                        "手续费": fee,
                    })

    df = pd.DataFrame(rows)
    df.to_csv(f"{EXP_DIR}/ratio_summary.csv", index=False, encoding="utf-8")

    for ver in ["long", "trade"]:
        print()
        print("=" * 100)
        print(f"{VERSIONS[ver]['label']}　固定比例配置（防御用国债，月度再平衡）")
        print("=" * 100)
        print(f"{'红利':>5}{'国债':>6}{'内年化':>9}{'内回撤':>9}{'内夏普':>8}"
              f"{'外年化':>9}{'外回撤':>9}{'外夏普':>8}{'全年化':>9}{'全夏普':>8}")
        print("-" * 100)
        sub = df[(df["版本"] == ver) & (df["防御资产"] == "国债") & (df["频率"] == "monthly")]
        for _, r in sub.iterrows():
            print(f"{r['红利比例']:>5.0%}{1-r['红利比例']:>6.0%}{r['内_年化']:>9.1%}{r['内_回撤']:>9.1%}"
                  f"{r['内_夏普']:>8.2f}{r['外_年化']:>9.1%}{r['外_回撤']:>9.1%}"
                  f"{r['外_夏普']:>8.2f}{r['全_年化']:>9.1%}{r['全_夏普']:>8.2f}")

        # 样本内最优比例 → 看它在样本外的表现
        s2 = sub.sort_values("内_夏普", ascending=False)
        best = s2.iloc[0]
        print()
        print(f"样本内夏普最优比例：红利 {best['红利比例']:.0%} / 国债 {1-best['红利比例']:.0%}　"
              f"内夏普 {best['内_夏普']:.2f} → 外年化 {best['外_年化']:.1%}，外夏普 {best['外_夏普']:.2f}")
        print(f"样本外夏普最优比例：红利 {s2.sort_values('外_夏普', ascending=False).iloc[0]['红利比例']:.0%}"
              "　（这个数只能看，不能用来选）")

        # 再平衡频率的敏感度
        print()
        print("再平衡频率敏感度（取样本内最优比例那一档）：")
        bw = best["红利比例"]
        for freq in ["monthly", "quarterly", "yearly"]:
            r = df[(df["版本"] == ver) & (df["防御资产"] == "国债") &
                   (df["频率"] == freq) & (df["红利比例"] == bw)].iloc[0]
            print(f"  {freq:<10}全年化 {r['全_年化']:>6.1%}　全夏普 {r['全_夏普']:.2f}　"
                  f"手续费 {r['手续费']:.2%}")

    plot(df)


def plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=140)
    for ax, ver, title in zip(axes, ["long", "trade"],
                              ["长周期版（2015 起）", "可交易版（2019 起）"]):
        sub = df[(df["版本"] == ver) & (df["防御资产"] == "国债") & (df["频率"] == "monthly")]
        x = sub["红利比例"] * 100
        ax.plot(x, sub["内_夏普"], marker="o", color="#378ADD", label="样本内夏普")
        ax.plot(x, sub["外_夏普"], marker="s", color="#D9534F", label="样本外夏普")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("红利低波占比（%），其余为国债")
        ax.set_ylabel("夏普比率")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(fontsize=8)
    fig.suptitle("固定比例配置：红利占比的敏感度（同一条线上两点差不多的区间，就是稳健区间）", fontsize=11.5)
    fig.tight_layout()
    p = "charts/11_配置比例_扫描.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"\n图已保存：{p}")


if __name__ == "__main__":
    main()
