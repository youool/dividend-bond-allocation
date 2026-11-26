"""生成系统架构图（报告图 1）。

四层分离是课程要求的结构：数据接入 / 策略信号 / 回测引擎 / 结果输出。
这张图只画结构，不画数据流细节——细节在正文 2.1 节。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PRIMARY = "#1A3D6D"
ACCENT = "#8B2E2E"
BORDER = "#B9C2CF"
FILL = "#F4F7FB"
MUTED = "#4A4A4A"

LAYERS = [
    {
        "name": "数据接入层",
        "files": "src/fetch_etf.py · fetch_index.py · verify_data.py",
        "lines": [
            "输入：akshare 日线接口（ETF 与全收益指数）",
            "处理：自算后复权，识别分红与 512890 的 1:2 份额折算",
            "输出：data/*.csv　体检：行数、日期连续性、空值与零价格",
        ],
    },
    {
        "name": "策略 / 信号层",
        "files": "src/allocation.py",
        "lines": [
            "输入：价格序列 + 目标波动率 8%",
            "输出：月末目标权重表（40/60 基线，超预算时按比例收缩）",
            "说明：v2 择时信号已证伪，保留在 signals.py 供复现实验",
        ],
    },
    {
        "name": "回测引擎",
        "files": "src/backtest.py",
        "lines": [
            "份额法：调仓日换份额，期间随价格漂移，不假设每日免费再平衡",
            "执行约束：T 日信号、T+1 开盘成交；双边万 2，按成交额计",
            "自检：单资产权重须等于买入持有（实测偏离 < 0.05%）；缺价抛错",
        ],
    },
    {
        "name": "结果输出层",
        "files": "src/run_strategy.py",
        "lines": [
            "指标表：output/final_metrics_{long,trade}.csv（含样本内外分段）",
            "净值序列：output/final_nav_*.csv，逐日可核对",
            "图：charts/14_最终策略_*.png（净值 / 回撤 / 波动与仓位）",
        ],
    },
]

SIDE = {
    "name": "横切：审计与实验",
    "lines": [
        ("check_lookahead.py", "前视偏差检查"),
        ("experiment_*.py", "六组证伪实验"),
        ("audit_overfit.py", "对主线自身的五项审计"),
        (".github/workflows", "每次提交自动跑内核自检"),
    ],
}


def main() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.9), dpi=220)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    left, right = 4, 71
    top, height, gap = 89, 18.0, 3.4

    for i, layer in enumerate(LAYERS):
        y = top - i * (height + gap) - height
        ax.add_patch(
            FancyBboxPatch(
                (left, y), right - left, height,
                boxstyle="round,pad=0.5,rounding_size=1.0",
                linewidth=1.0, edgecolor=BORDER, facecolor=FILL, zorder=2,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (left + 0.9, y + 0.9), 0.8, height - 1.8,
                boxstyle="square,pad=0", linewidth=0, facecolor=PRIMARY, zorder=3,
            )
        )
        ax.text(left + 2.6, y + height - 2.4, layer["name"], fontsize=10.5,
                fontweight="bold", color=PRIMARY, va="top", ha="left", zorder=4)
        ax.text(left + 2.6, y + height - 6.0, layer["files"], fontsize=7.8,
                color=MUTED, va="top", ha="left", style="italic", zorder=4)
        for j, line in enumerate(layer["lines"]):
            ax.text(left + 2.6, y + height - 9.2 - j * 2.95, line, fontsize=8.1,
                    color="#1A1A1A", va="top", ha="left", zorder=4)

        if i < len(LAYERS) - 1:
            ax.annotate(
                "", xy=(left - 0.2, y - gap + 0.7), xytext=(left - 0.2, y - 0.5),
                arrowprops=dict(arrowstyle="-|>", color=PRIMARY, lw=1.3,
                                shrinkA=0, shrinkB=0, mutation_scale=11),
                zorder=5,
            )

    # 右侧横切关注点：覆盖上两层侧边
    sx, sw = 74.0, 23.5
    sh = 2 * height + gap
    sy = top - sh
    ax.add_patch(
        FancyBboxPatch((sx, sy), sw, sh, boxstyle="round,pad=0.4,rounding_size=1.0",
                       linewidth=1.0, edgecolor=ACCENT, facecolor="#FBF4F4",
                       linestyle=(0, (4, 2)), zorder=2)
    )
    ax.text(sx + 1.3, top - 2.6, SIDE["name"], fontsize=9.4, fontweight="bold",
            color=ACCENT, va="top", ha="left", zorder=4)
    for j, (fname, desc) in enumerate(SIDE["lines"]):
        yy = top - 7.4 - j * 8.2
        ax.text(sx + 1.3, yy, fname, fontsize=7.4, color="#1A1A1A",
                va="top", ha="left", zorder=4, family="Consolas")
        ax.text(sx + 1.3, yy - 2.9, "　" + desc, fontsize=7.3, color=MUTED,
                va="top", ha="left", zorder=4)

    ax.text(50, 96.6, "四层分离：每层可单独运行、单独验证，换数据源不影响回测逻辑",
            fontsize=10.2, fontweight="bold", color=PRIMARY, ha="center", va="top")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig("charts/00_系统架构.png", dpi=220, facecolor="white")
    print("saved charts/00_系统架构.png")


if __name__ == "__main__":
    main()
