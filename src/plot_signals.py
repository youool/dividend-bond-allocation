"""
第 3 关 · 收尾：画仓位信号图，检查"该重的时候重、该轻的时候轻"

上面板：沪深300 归一化走势
下面板：权益仓位阶梯线
看两件事：
  1. 大跌段（2015 下半年、2018、2022）仓位是否降下来了
  2. 低位（2023–2024）仓位是否加上去了
如果反过来，说明规则写反了或数据接错了。
"""

import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot(weights_path, index_path, title, out_name):
    w = pd.read_csv(weights_path, parse_dates=["date"])
    idx = pd.read_csv(index_path, parse_dates=["date"]).set_index("date")["close"]
    idx = idx[idx.index >= w["date"].iloc[0]]
    norm = idx / idx.iloc[0]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), dpi=140, sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1]},
    )

    ax1.plot(norm.index, norm.values, color="#378ADD", linewidth=1.3)
    ax1.set_title(title, fontsize=13)
    ax1.set_ylabel("沪深300（归一化）")
    ax1.grid(alpha=0.25, linestyle="--")

    ax2.step(w["date"], w["equity"] * 100, where="post", color="#1D9E75", linewidth=1.6)
    ax2.fill_between(w["date"], 0, w["equity"] * 100, step="post", color="#1D9E75", alpha=0.12)
    ax2.set_ylabel("权益仓位（%）")
    ax2.set_ylim(-2, 90)
    ax2.grid(alpha=0.25, linestyle="--")

    for ax in (ax1, ax2):
        for d, c in [("2015-06-12", "#C0392B"), ("2018-01-24", "#BA7517"), ("2022-01-04", "#534AB7")]:
            ts = pd.Timestamp(d)
            if ts >= w["date"].iloc[0]:
                ax.axvline(ts, color=c, linestyle="--", linewidth=0.9, alpha=0.55)

    fig.tight_layout()
    path = os.path.join("charts", out_name)
    fig.savefig(path)
    print(f"已保存 {path}")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("charts", exist_ok=True)
    plot("output/weights_long.csv", "data/index/沪深300_全收益.csv",
         "长周期版：仓位信号 vs 沪深300（2015 起）", "03_仓位信号_长周期.png")
    plot("output/weights_trade.csv", "data/510300.csv",
         "可交易版：仓位信号 vs 沪深300ETF（2019 起）", "04_仓位信号_可交易.png")
