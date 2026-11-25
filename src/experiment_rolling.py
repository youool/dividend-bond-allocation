# -*- coding: utf-8 -*-
"""
i4 滚动 3 年稳健性：任意时点买入，持有三年会怎样？

答辩时最容易被问的一句是「我 2021 年高点进场怎么办」。
全区间年化回答不了这个问题，只有滚动窗口能。

对每个交易日作为起点，取其后整三年的净值，算年化收益与窗口内最大回撤，
再统计分布：最差、5 分位、中位、最好，以及「三年还亏钱」的比例。
对照组：红利低波买入持有、沪深300 买入持有。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "output/exp/rolling3y.csv"
CHART = "charts/18_滚动三年_分布.png"
COLS = {"strategy": "本策略", "bm_dividend": "红利买入持有", "bm_broad": "沪深300"}


def rolling(nav, years=3):
    idx = nav.index
    rows = []
    for i, d in enumerate(idx):
        end_target = d + pd.DateOffset(years=years)
        j = idx.searchsorted(end_target)
        if j >= len(idx):
            break
        seg = nav.iloc[i:j + 1]
        days = (seg.index[-1] - seg.index[0]).days
        if days < 365 * years - 30:
            continue
        ann = (seg.iloc[-1] / seg.iloc[0]) ** (365.0 / days) - 1
        dd = float((seg / seg.cummax() - 1).min())
        rows.append({"起点": d, "终点": seg.index[-1], "年化": ann, "窗口回撤": dd})
    return pd.DataFrame(rows)


def main():
    os.makedirs("output/exp", exist_ok=True)
    os.makedirs("charts", exist_ok=True)
    all_rows = []

    for ver, label in [("long", "长周期版"), ("trade", "可交易版")]:
        nav = pd.read_csv(f"output/final_nav_{ver}.csv", index_col=0, parse_dates=True)
        print("=" * 92)
        print(f"{label}（{nav.index[0].date()} → {nav.index[-1].date()}）滚动 3 年")
        print("=" * 92)
        print(f"{'对象':<16}{'最差':>9}{'5分位':>9}{'中位':>9}{'最好':>9}"
              f"{'三年亏损占比':>13}{'最差窗口':>24}")
        print("-" * 92)

        for c, name in COLS.items():
            r = rolling(nav[c])
            all_rows.append(r.assign(版本=ver, 对象=name))
            q = r["年化"].quantile([0, 0.05, 0.5, 1.0]).values
            worst = r.loc[r["年化"].idxmin()]
            print(f"{name:<16}{q[0]:>9.2%}{q[1]:>9.2%}{q[2]:>9.2%}{q[3]:>9.2%}"
                  f"{(r['年化'] < 0).mean():>13.0%}"
                  f"{str(worst['起点'].date()) + '→' + str(worst['终点'].date()):>24}")
            # 顺带给出最差窗口的回撤，回答「亏多少」而不只是「年化多少」
            print(f"{'':<16}最差窗口内最大回撤 {worst['窗口回撤']:.2%}；"
                  f"中位窗口回撤 {r['窗口回撤'].median():.2%}")
        print()

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\n结果已保存：{OUT}")
    plot(df)


def plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 1, figsize=(12.6, 7.4), dpi=140, sharex=False)
    for ax, ver, label in [(axes[0], "long", "长周期版"), (axes[1], "trade", "可交易版")]:
        d = df[df["版本"] == ver]
        styles = {"本策略": ("#1F4E79", 2.2), "红利买入持有": ("#C77A1E", 1.5),
                  "沪深300": ("#8C8C8C", 1.3)}
        for name, (color, lw) in styles.items():
            s = d[d["对象"] == name]
            ax.plot(s["起点"], s["年化"], color=color, linewidth=lw, label=name)
        ax.axhline(0, color="#B03A2E", linewidth=1.0, linestyle="--")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(f"{label}：任意时点买入并持有 3 年的年化收益", fontsize=11)
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(fontsize=8.5, loc="lower left", ncol=3, framealpha=0.9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(CHART, bbox_inches="tight")
    plt.close(fig)
    print(f"图已保存：{CHART}")


if __name__ == "__main__":
    main()
