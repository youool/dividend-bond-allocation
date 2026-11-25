"""
第 2 关 · 收尾：数据总清点 + 长周期版曲线

产出两样东西：
  1. 一张数据清单（哪个文件、多少行、覆盖哪段），报告里可直接用
  2. 长周期版曲线图（2015 起，四条腿归一化），用来确认极端样本都在里面
"""

import glob
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CHART_DIR = "charts"

LONG_CYCLE = [
    ("data/index/红利低波_全收益.csv", "红利低波（全收益指数）"),
    ("data/index/沪深300_全收益.csv", "沪深300（全收益指数）"),
    ("data/511010.csv", "国债 ETF"),
    ("data/511880.csv", "货币 ETF"),
]


def inventory():
    print("=" * 76)
    print("数据清单")
    print("=" * 76)
    print(f"{'文件':<38}{'行数':>7}  {'起始':<12}{'结束':<12}")
    print("-" * 76)

    total = 0
    for path in sorted(glob.glob("data/**/*.csv", recursive=True)):
        df = pd.read_csv(path)
        date_col = "date" if "date" in df.columns else df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        total += len(df)
        print(f"{path:<38}{len(df):>7}  {str(df[date_col].iloc[0].date()):<12}{str(df[date_col].iloc[-1].date()):<12}")
    print("-" * 76)
    print(f"合计 {total} 行")
    return total


def plot_long_cycle():
    os.makedirs(CHART_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)

    for path, label in LONG_CYCLE:
        if not os.path.exists(path):
            print(f"缺少 {path}")
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        df = df[df["date"] >= "2015-01-01"]
        s = df.set_index("date")["close"]
        ax.plot(s.index, (s / s.iloc[0]).values, linewidth=1.4, label=label)

    ax.axvline(pd.Timestamp("2015-06-12"), color="#C0392B", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(pd.Timestamp("2018-01-24"), color="#BA7517", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(pd.Timestamp("2022-01-04"), color="#534AB7", linestyle="--", linewidth=1, alpha=0.7)

    ymax = ax.get_ylim()[1]
    ax.text(pd.Timestamp("2015-06-20"), ymax * 0.95, "2015 牛市顶", fontsize=9, color="#C0392B")
    ax.text(pd.Timestamp("2018-02-05"), ymax * 0.88, "2018 熊市起点", fontsize=9, color="#BA7517")
    ax.text(pd.Timestamp("2022-01-12"), ymax * 0.81, "2022 下跌起点", fontsize=9, color="#534AB7")

    ax.set_title("长周期版四条腿（2015 起，全收益，归一化到 1.0）", fontsize=13)
    ax.set_ylabel("归一化价格")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    out = os.path.join(CHART_DIR, "02_长周期版_四条腿.png")
    fig.savefig(out)
    print(f"\n长周期版曲线已保存：{out}")


if __name__ == "__main__":
    inventory()
    plot_long_cycle()
