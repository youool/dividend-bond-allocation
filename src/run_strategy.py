"""
最终策略 · 主回测入口（重构后的唯一主流程）

一句话定义：
  长期持有 40% 红利低波 + 60% 国债，每月末把比例拉回目标；
  当组合近 60 日已实现波动超过年化 8% 时，按比例收缩权益（下限 20%），
  波动回落即恢复。不预测方向，只管理风险敞口。

为什么不是原来的版本：
  原 v2 规则（估值定仓位 + 120 日均线扣减 + 60 日动量倾斜）已被六组对照实验证伪，
  证据在 output/exp/ 与 charts/07–12。核心那条是：暴露匹配的固定配置，
  在样本外年化 5.0%，而择时版只有 1.5%。

这一步产出（报告里直接可用的三样东西）：
  output/final_nav_{版本}.csv       策略与两条基准的逐日净值
  output/final_metrics_{版本}.csv   四项必报指标 × 三个区间 × 三个对象
  charts/14_最终策略_{版本}.png     净值 / 回撤 / 波动与仓位 三面板图

四条课程硬要求在这里全部对上：
  四项指标（累计、年化、最大回撤、夏普）  → final_metrics
  归一化净值曲线 + 买卖点标注            → 面板 A（竖线为波动率触发的收缩月）
  样本内 / 样本外分别报告                → 三个区间并列
  基准选择理由                          → 见 README「基准」一节
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import allocation as al  # noqa: E402
import backtest as bt  # noqa: E402

FEE_RATE = bt.FEE_RATE


def build_drawdown(nav):
    return nav / nav.cummax() - 1.0


def main():
    os.makedirs("output", exist_ok=True)
    os.makedirs("charts", exist_ok=True)   # 换到干净目录时 charts/ 不存在，不建会直接崩在图那一步

    for ver in ["long", "trade"]:
        cfg = bt.VERSIONS[ver]
        ins_end = pd.Timestamp(cfg["insample_end"])

        alloc, alloc_path = al.build(ver, al.DEFAULT_MODE)
        op, cl = bt.load_prices(cfg["prices"])
        nav, fee = bt.simulate(alloc, op, cl, cfg["exec_price"])

        out = pd.DataFrame({"strategy": nav})
        for bm, asset in [("bm_broad", "broad"), ("bm_dividend", "dividend")]:
            px = cl[asset].reindex(out.index).ffill()
            out[bm] = (px / px.iloc[0]).values
        out = out.ffill().dropna()
        out.to_csv(f"output/final_nav_{ver}.csv", encoding="utf-8")

        rows = []
        for seg, sub in [("全区间", out),
                         ("样本内", out[out.index <= ins_end]),
                         ("样本外", out[out.index > ins_end])]:
            if len(sub) < 20:
                continue
            for col, cname in [("strategy", "策略"), ("bm_broad", "沪深300买入持有"),
                               ("bm_dividend", "红利低波买入持有")]:
                rows.append({"区间": seg, "对象": cname, **bt.metrics(sub[col])})
        mdf = pd.DataFrame(rows)
        mdf.to_csv(f"output/final_metrics_{ver}.csv", index=False, encoding="utf-8")

        print()
        print("=" * 88)
        print(f"{al.VERSIONS[ver]['label']}　{out.index[0].date()} → {out.index[-1].date()}　"
              f"{len(out)} 个交易日")
        print("=" * 88)
        print(f"{'区间':<8}{'对象':<18}{'累计':>9}{'年化':>8}{'最大回撤':>9}{'夏普':>7}{'Calmar':>8}{'波动':>8}")
        print("-" * 96)
        for _, r in mdf.iterrows():
            print(f"{r['区间']:<8}{r['对象']:<18}{r['累计收益率']:>9.1%}{r['年化收益率']:>8.1%}"
                  f"{r['最大回撤']:>9.1%}{r['夏普比率']:>7.2f}{r['Calmar比率']:>8.2f}"
                  f"{r['年化波动']:>8.1%}")
        print()
        print(f"累计手续费 {fee:.2%}　平均红利仓位 {alloc['w_dividend'].mean():.0%}　"
              f"波动触发收缩的月份 {int((alloc['scale'] < 1).sum())}/{len(alloc)}")

        plot(ver, out, alloc, mdf)


def plot(ver, out, alloc, mdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(13, 10.5), dpi=140)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.0, 1.0], hspace=0.32)

    idx = out.index
    ins_end = pd.Timestamp(bt.VERSIONS[ver]["insample_end"])
    vol_target = al.MODES[al.DEFAULT_MODE]["vol_target"]

    # ---- 面板 A：净值 ----
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(idx, out["strategy"], color="#1D9E75", linewidth=2.1, label="策略", zorder=4)
    ax1.plot(idx, out["bm_dividend"], color="#BA7517", linewidth=1.2,
             label="红利低波 买入持有", zorder=3)
    ax1.plot(idx, out["bm_broad"], color="#8C8C8C", linewidth=1.2,
             label="沪深300 买入持有", zorder=2)

    # 只标「强收缩」：缩放系数低于 0.85，即波动超预算 18% 以上才动手。
    # 全标会把图糊掉（月度信号本来就密），而且大部分收缩幅度很小、不值得讲。
    strong = pd.to_datetime(alloc.loc[alloc["scale"] < 0.85, "date"])
    bottom = ax1.get_ylim()[0]
    for d in strong:
        ax1.plot([d, d], [bottom, bottom + (ax1.get_ylim()[1] - bottom) * 0.055],
                 color="#D9534F", linewidth=1.6, alpha=0.75, zorder=1)
    if len(strong):
        ax1.plot([], [], color="#D9534F", linewidth=2,
                 label=f"强收缩月（{len(strong)} 次，仓位降幅 >15%）")

    ax1.axvline(ins_end, color="#333", linestyle="--", linewidth=1.1, alpha=0.65)
    # 样本内外标注放在虚线上方，避开左上的图例
    ax1.annotate("← 样本内　｜　样本外 →", xy=(ins_end, 0.93),
                 xycoords=("data", "axes fraction"), fontsize=9, color="#333",
                 ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none", alpha=0.75))

    ax1.set_title(f"{al.VERSIONS[ver]['label']}：策略与两条基准（归一化到 1.0）", fontsize=13)
    ax1.set_ylabel("净值")
    ax1.grid(alpha=0.25, linestyle="--")
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    # ---- 面板 B：回撤 ----
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    for col, name, color, lw in [("strategy", "策略", "#1D9E75", 1.8),
                                 ("bm_dividend", "红利低波", "#BA7517", 1.05),
                                 ("bm_broad", "沪深300", "#8C8C8C", 1.05)]:
        ax2.fill_between(idx, build_drawdown(out[col]), 0, color=color, alpha=0.10)
        ax2.plot(idx, build_drawdown(out[col]), color=color, linewidth=lw, label=name)
    ax2.set_ylabel("回撤")
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.grid(alpha=0.25, linestyle="--")
    ax2.legend(loc="lower left", fontsize=8, ncol=3)

    # ---- 面板 C：已实现波动 vs 风险预算 vs 红利仓位 ----
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ad = pd.to_datetime(alloc["date"])
    vol = alloc["realized_vol"].astype(float)
    ax3.fill_between(ad, vol, vol_target, where=(vol > vol_target),
                     color="#D9534F", alpha=0.22, label="超出风险预算")
    ax3.plot(ad, vol, color="#378ADD", linewidth=1.5, label="组合近 60 日已实现波动")
    ax3.axhline(vol_target, color="#D9534F", linestyle="--", linewidth=1.2,
                label=f"风险预算 {vol_target:.0%}")
    ax3.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax3.set_ylabel("年化波动")
    ax3.set_ylim(0, max(vol.max() * 1.45, vol_target * 2.0))
    ax3.grid(alpha=0.25, linestyle="--")

    ax4 = ax3.twinx()
    ax4.step(ad, alloc["w_dividend"], where="post", color="#1D9E75", linewidth=1.7,
             alpha=0.9, label="红利仓位（右轴）")
    ax4.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax4.set_ylabel("红利仓位")
    ax4.set_ylim(0, 0.5)

    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax4.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=7.2, ncol=4, framealpha=0.92,
               handlelength=1.6, columnspacing=1.2)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    path = f"charts/14_最终策略_{ver}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"图已保存：{path}")


if __name__ == "__main__":
    main()
