"""
第 5 关 · 对照实验三：趋势层能不能修好

实验二已经给出一个难看的结论：把择时拆掉之后，什么都不判断的固定配置反而更好。
其中「只用趋势」单独跑最差（样本外 −11.9%），说明 120 日均线在 A 股震荡市里反复穿越，
把仓位砍在坑底、又追在半山腰。

本实验要回答的是：这个毛病是「均线这个工具不行」，还是「120 天这个参数不行」。
网格三组变量：

  均线周期   120 / 200 / 250     —— 越慢越少穿越，但也越迟钝
  仓位档位   80-60-30 / 100-80-60 —— 后者承认「不满仓本身就是收益损失」
  缓冲带     0 / 10%             —— 变动不足 10% 就不调仓，专治 whipsaw

一条纪律（比结果更重要）：
  **参数只能按样本内表现挑，样本外只用来检验，不能用来挑。**
  所以下表按样本内夏普排序，样本外单独列在右边。
  如果某个组合样本内好、样本外差，那是它过拟合，不是它被埋没。

  排序口径：样本内夏普。理由：夏普同时管收益和波动，比单看年化不容易选出
  「靠高仓位硬顶出来」的组合。

输出：
  output/exp/grid_summary.csv
  charts/10_参数网格_样本内vs样本外.png
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

GRID = []
for ma in [120, 200, 250]:
    for pos, pos_name in [((0.80, 0.60, 0.30), "80-60-30"),
                          ((1.00, 0.80, 0.60), "100-80-60")]:
        for buf in [0.0, 0.10]:
            GRID.append({"ma": ma, "pos": pos, "pos_name": pos_name, "buf": buf})


def run_nav(w, ver):
    cfg = bt.VERSIONS[ver]
    op, cl = bt.load_prices(cfg["prices"])
    nav, fee = bt.simulate(w, op, cl, cfg["exec_price"])
    return nav, fee


def fixed_35_65(dates, ver):
    """参照线：35% 红利 + 65% 国债，机械再平衡。"""
    df = pd.DataFrame({"date": pd.to_datetime(dates)})
    for c, v in zip(WCOLS, [0.35, 0.0, 0.65, 0.0]):
        df[c] = v
    return df


def main():
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs("charts", exist_ok=True)
    rows = []

    for ver in ["long", "trade"]:
        cfg = bt.VERSIONS[ver]
        ins_end = pd.Timestamp(cfg["insample_end"])

        # 参照：固定配置
        _, base_w = sg.build(ver, tag="REF")
        dates = base_w["date"].values
        fw = fixed_35_65(dates, ver)
        nav, fee = run_nav(fw, ver)
        ins, out = nav[nav.index <= ins_end], nav[nav.index > ins_end]
        mi, mo = bt.metrics(ins), bt.metrics(out)
        rows.append({"版本": ver, "组合": "固定35红利65国债", "均线": np.nan, "档位": "—",
                     "缓冲": np.nan, "权益均值": 0.35,
                     "内_年化": mi["年化收益率"], "内_回撤": mi["最大回撤"], "内_夏普": mi["夏普比率"],
                     "外_年化": mo["年化收益率"], "外_回撤": mo["最大回撤"], "外_夏普": mo["夏普比率"]})

        # 基线（现有规则）
        nav, fee = run_nav(base_w, ver)
        ins, out = nav[nav.index <= ins_end], nav[nav.index > ins_end]
        mi, mo = bt.metrics(ins), bt.metrics(out)
        rows.append({"版本": ver, "组合": "基线 120-80-60-30-无缓冲", "均线": 120, "档位": "80-60-30",
                     "缓冲": 0.0, "权益均值": float(base_w[WCOLS].mean()[["w_dividend", "w_broad"]].sum()),
                     "内_年化": mi["年化收益率"], "内_回撤": mi["最大回撤"], "内_夏普": mi["夏普比率"],
                     "外_年化": mo["年化收益率"], "外_回撤": mo["最大回撤"], "外_夏普": mo["夏普比率"]})

        for g in GRID:
            tag = f"G{ver[0]}{g['ma']}_{g['pos_name']}_{int(g['buf']*100)}"
            cfg2, w = sg.build(ver, tag=tag, ma_window=g["ma"], pos=g["pos"], buffer=g["buf"])
            nav, fee = run_nav(w, ver)
            ins, out = nav[nav.index <= ins_end], nav[nav.index > ins_end]
            mi, mo = bt.metrics(ins), bt.metrics(out)
            rows.append({"版本": ver, "组合": f"{g['ma']}-{g['pos_name']}-缓冲{int(g['buf']*100)}%",
                         "均线": g["ma"], "档位": g["pos_name"], "缓冲": g["buf"],
                         "权益均值": float(w[WCOLS].mean()[["w_dividend", "w_broad"]].sum()),
                         "内_年化": mi["年化收益率"], "内_回撤": mi["最大回撤"], "内_夏普": mi["夏普比率"],
                         "外_年化": mo["年化收益率"], "外_回撤": mo["最大回撤"], "外_夏普": mo["夏普比率"]})

    df = pd.DataFrame(rows)
    df.to_csv(f"{EXP_DIR}/grid_summary.csv", index=False, encoding="utf-8")

    for ver in ["long", "trade"]:
        sub = df[df["版本"] == ver].copy()
        sub = sub.sort_values("内_夏普", ascending=False)
        print()
        print("=" * 110)
        print(f"{bt.VERSIONS[ver]['label']}　按「样本内夏普」排序（样本外只做检验，不参与挑选）")
        print("=" * 110)
        print(f"{'组合':<28}{'权益':>6}{'内年化':>8}{'内回撤':>8}{'内夏普':>8}"
              f"{'外年化':>8}{'外回撤':>8}{'外夏普':>8}")
        print("-" * 110)
        for _, r in sub.iterrows():
            mark = "  ← 参照" if r["组合"].startswith("固定") else ""
            print(f"{r['组合']:<28}{r['权益均值']:>6.0%}{r['内_年化']:>8.1%}{r['内_回撤']:>8.1%}"
                  f"{r['内_夏普']:>8.2f}{r['外_年化']:>8.1%}{r['外_回撤']:>8.1%}{r['外_夏普']:>8.2f}{mark}")

        best_in = sub.iloc[1] if sub.iloc[0]["组合"].startswith("固定") else sub.iloc[0]
        print()
        print(f"样本内最优（不含参照）：{best_in['组合']}　"
              f"内夏普 {best_in['内_夏普']:.2f} → 外年化 {best_in['外_年化']:.1%}，外夏普 {best_in['外_夏普']:.2f}")
        ref = sub[sub["组合"].str.startswith("固定")].iloc[0]
        print(f"参照固定配置：　　　　　内夏普 {ref['内_夏普']:.2f} → 外年化 {ref['外_年化']:.1%}，外夏普 {ref['外_夏普']:.2f}")

    plot(df)


def plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), dpi=140)
    for ax, ver, title in zip(axes, ["long", "trade"],
                              ["长周期版（2015 起）", "可交易版（2019 起）"]):
        sub = df[(df["版本"] == ver) & (~df["组合"].str.startswith("固定"))]
        ref = df[(df["版本"] == ver) & (df["组合"].str.startswith("固定"))].iloc[0]

        ax.scatter(sub["内_夏普"], sub["外_夏普"], s=46, color="#378ADD", alpha=0.8,
                   label="参数组合", zorder=3)
        ax.axhline(ref["外_夏普"], color="#D9534F", linestyle="--", linewidth=1.4,
                   label=f"固定配置 外夏普 {ref['外_夏普']:.2f}")
        ax.axvline(ref["内_夏普"], color="#D9534F", linestyle=":", linewidth=1.0, alpha=0.7)

        for _, r in sub.iterrows():
            if r["外_夏普"] >= sub["外_夏普"].quantile(0.75) or r["内_夏普"] >= sub["内_夏普"].quantile(0.8):
                ax.annotate(r["组合"], (r["内_夏普"], r["外_夏普"]), fontsize=6.5,
                            xytext=(4, 3), textcoords="offset points", color="#444")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("样本内夏普（用来挑参数）")
        ax.set_ylabel("样本外夏普（只用来检验）")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(fontsize=7.5, loc="lower right")

    fig.suptitle("参数网格：样本内 vs 样本外（散点越靠右上越好；红虚线是不择时的参照）", fontsize=12)
    fig.tight_layout()
    p = "charts/10_参数网格_样本内vs样本外.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"\n图已保存：{p}")


if __name__ == "__main__":
    main()
