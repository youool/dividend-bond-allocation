"""
第 2 关 · 步骤 4-5：画曲线验真 + 核对数据完整性

做两件事：
  1. 把 4 只 ETF 的收盘价画在一张图上（归一化到 1.0），肉眼判断数据是不是真历史
  2. 逐只检查：行数、起止日期、重复日期、空值、零价格、最大单日跳空

为什么要肉眼验：
  接口出错时不会报错，只会给你一段截断或错位的数据。
  图上没有 2015 股灾 / 2018 熊市 / 2022 下跌，就说明数据没拉全。
"""

import glob
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = "data"
CHART_DIR = "charts"
NAMES = {
    "512890": "红利低波ETF",
    "510300": "沪深300ETF",
    "511010": "国债ETF",
    "511880": "货币ETF",
}


def load_all():
    frames = {}
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))):
        code = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path, parse_dates=["date"])
        frames[code] = df
    return frames


def check_one(code, df):
    """返回这一只的体检结果。"""
    issues = []
    if df["date"].duplicated().any():
        issues.append(f"重复日期 {df['date'].duplicated().sum()} 个")
    if df[["open", "high", "low", "close"]].isna().any().any():
        issues.append("存在空值")
    if (df["close"] <= 0).any():
        issues.append("存在 0 或负价格")
    gaps = df["date"].diff().dt.days
    max_gap = gaps.max()
    if pd.notna(max_gap) and max_gap > 30:
        issues.append(f"最长空档 {int(max_gap)} 天（可能是长假或数据缺失）")
    # 单日涨跌幅超过 15% 的，ETF 里多半是数据问题（除权没处理或错位）
    ret = df["close"].pct_change()
    big = (ret.abs() > 0.15).sum()
    if big > 0:
        issues.append(f"单日涨跌超 15% 的有 {big} 天，需核对是否复权问题")
    return issues


def main():
    os.makedirs(CHART_DIR, exist_ok=True)
    frames = load_all()
    if not frames:
        print("data/ 下没有 csv，先跑 fetch_etf.py")
        return

    print("=" * 78)
    print("数据体检")
    print("=" * 78)
    print(f"{'代码':<8}{'名称':<12}{'行数':>7}  {'起始':<12}{'结束':<12}{'结论'}")
    print("-" * 78)

    all_ok = True
    for code, df in frames.items():
        name = NAMES.get(code, code)
        issues = check_one(code, df)
        verdict = "通过" if not issues else "；".join(issues)
        if issues:
            all_ok = False
        print(f"{code:<8}{name:<12}{len(df):>7}  {df['date'].iloc[0].date()}   "
              f"{df['date'].iloc[-1].date()}   {verdict}")

    print("-" * 78)
    starts = {c: df["date"].iloc[0] for c, df in frames.items()}
    if starts:
        latest_code = max(starts, key=starts.get)
        print(f"共同可用起点：{starts[latest_code].date()}（由 {latest_code} 决定）")

    # 画图
    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    for code, df in frames.items():
        s = df.set_index("date")["close"]
        norm = s / s.iloc[0]
        ax.plot(norm.index, norm.values, linewidth=1.3, label=f"{code} {NAMES.get(code, '')}")

    ax.axvline(pd.Timestamp("2015-06-12"), color="#C0392B", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(pd.Timestamp("2018-01-24"), color="#BA7517", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(pd.Timestamp("2022-01-04"), color="#534AB7", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(pd.Timestamp("2015-06-20"), ax.get_ylim()[1] * 0.95, "2015 牛市顶", fontsize=9, color="#C0392B")
    ax.text(pd.Timestamp("2018-02-01"), ax.get_ylim()[1] * 0.88, "2018 熊市起点", fontsize=9, color="#BA7517")
    ax.text(pd.Timestamp("2022-01-12"), ax.get_ylim()[1] * 0.81, "2022 下跌起点", fontsize=9, color="#534AB7")

    ax.set_title("四只标的归一化收盘价（后复权，起点=1.0）", fontsize=13)
    ax.set_ylabel("归一化价格")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    out = os.path.join(CHART_DIR, "01_价格曲线_验真.png")
    fig.savefig(out)
    print(f"\n图已保存：{out}")

    print()
    if all_ok:
        print("全部通过：数据可以进入第 3 关。")
    else:
        print("有项目未通过，先修数据再往下走——数据不对，后面全错。")


if __name__ == "__main__":
    main()
