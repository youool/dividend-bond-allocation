"""
第 5 关 · 对照实验：换估值锚

要检验的假设（来自第 4 关的诊断）：
  回测失效的首要原因是「估值锚错配」——用沪深300 的 PE 判断整个组合的贵贱，
  但底仓是红利低波。2023 年后 300 的 PE 分位常年偏高，把仓位压在 30%，
  而红利低波自己还在低位、一路新高。

所以本实验只改一个变量：估值锚。其余规则（120 日均线、60 日动量、70/30 防御）完全不动。

四个锚：
  A0  沪深300 PE-TTM 分位      —— 现状，作为基线
  A1  上证红利 PE-TTM 分位      —— 换成「红利类资产自己的估值」（乐咕，2005 起）
  A2  红利低波自身价格分位       —— 纯价格代理，零外部依赖，最坏情况下的降级方案
  A3  无估值层（基础仓位恒 60%）—— 对照，用来回答「估值层到底有没有贡献」

判读标准（提前写死，不许跑完再挑）：
  1. 样本外年化收益是否高于 A0
  2. 最大回撤是否没有明显恶化（放宽到 5 个百分点以内）
  3. 换手率是否没有大幅上升
  三条都满足才算「修好了」。

输出：
  output/exp/weights_{锚}_{版本}.csv
  output/exp/summary.csv          全部指标汇总
  output/exp/anchor_compare.csv   只看全区间与样本外的关键列
  charts/07_估值锚对比_净值.png
  charts/08_估值锚对比_仓位.png
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt  # noqa: E402
import signals as sg  # noqa: E402

EXP_DIR = "output/exp"
PE_RED_PATH = "data/pe_red.csv"
DIVIDEND_PATHS = {
    "long": "data/index/红利低波_全收益.csv",
    "trade": "data/512890.csv",
}


# ---------------- 估值锚 ----------------

def _pct_series(pe_series):
    """每个点在过去 120 个月里的分位（与 valuation.py 口径一致）。"""
    out = []
    vals = pe_series.values
    for i in range(len(vals)):
        window = vals[max(0, i - 119): i + 1]
        out.append(float((window < vals[i]).mean()))
    return pd.Series(out, index=pe_series.index)


def fetch_pe_red():
    """上证红利指数 PE-TTM（乐咕月度，2005 起）。缓存到 data/pe_red.csv。"""
    if os.path.exists(PE_RED_PATH):
        df = pd.read_csv(PE_RED_PATH, parse_dates=["date"])
        return df

    import akshare as ak
    raw = ak.stock_index_pe_lg(symbol="上证红利")
    df = raw[["日期", "滚动市盈率"]].copy()
    df.columns = ["date", "pe_ttm"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["pe_ttm"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    df["pe_pct"] = _pct_series(df["pe_ttm"]).round(4).values
    df.to_csv(PE_RED_PATH, index=False, encoding="utf-8")
    return df


def anchor_broad_pe(ver):
    """A0：沪深300 PE-TTM 分位（现状）。"""
    df = pd.read_csv("data/pe_300.csv", parse_dates=["date"])
    return df[["date", "pe_ttm", "pe_pct"]]


def anchor_red_pe(ver):
    """A1：上证红利 PE-TTM 分位。"""
    return fetch_pe_red()[["date", "pe_ttm", "pe_pct"]]


def anchor_price_pct(ver):
    """
    A2：红利低波自身的价格分位。
    口径：过去 5 年（1260 个交易日）里，有多少比例的交易日收盘价低于今天。
    不足 5 年时窗口自然扩张，最少要求 1 年（252 日）才开始给信号。
    这不是真估值，但它锚对了标的——用底仓自己的价格位置判断贵贱。
    """
    s = sg.load_close(DIVIDEND_PATHS[ver])
    win = 1260
    pct = s.rolling(win, min_periods=252).apply(lambda w: float((w[:-1] < w[-1]).mean()), raw=True)
    df = pd.DataFrame({"date": s.index, "pe_ttm": s.values, "pe_pct": pct.values})
    return df.dropna(subset=["pe_pct"])


def anchor_none(ver):
    """
    A3：无估值层。pe_pct 恒为 0.5，落进 30%–70% 区间，基础仓位恒为 60%。
    用来回答「估值层到底贡献了什么」。
    """
    dates = anchor_broad_pe(ver)["date"]
    return pd.DataFrame({"date": dates, "pe_ttm": np.nan, "pe_pct": 0.5})


ANCHORS = {
    "A0_沪深300PE": (anchor_broad_pe, "沪深300 PE-TTM 分位（现状）"),
    "A1_上证红利PE": (anchor_red_pe, "上证红利 PE-TTM 分位"),
    "A2_红利自身价格分位": (anchor_price_pct, "红利低波自身价格分位"),
    "A3_无估值层": (anchor_none, "无估值层，基础仓位恒 60%"),
}


# ---------------- 回测 ----------------

def run_one(anchor_key, ver):
    fn = ANCHORS[anchor_key][0]
    anchor = fn(ver)
    cfg, w = sg.build(ver, anchor_df=anchor, tag=anchor_key)

    bcfg = bt.VERSIONS[ver]
    op, cl = bt.load_prices(bcfg["prices"])
    nav, fee = bt.simulate(w, op, cl, bcfg["exec_price"])

    out = pd.DataFrame({"strategy": nav})
    for bm, asset in [("bm_broad", "broad"), ("bm_dividend", "dividend")]:
        px = cl[asset].reindex(out.index).ffill()
        out[bm] = (px / px.iloc[0]).values
    out = out.ffill().dropna()

    # 换手：相邻两期权重变动的绝对值之和 / 2（单边换手）
    wcols = ["w_dividend", "w_broad", "w_bond", "w_cash"]
    turnover = float(w[wcols].diff().abs().sum(axis=1).sum() / 2)
    years = (pd.Timestamp(w["date"].iloc[-1]) - pd.Timestamp(w["date"].iloc[0])).days / 365.25
    return {
        "anchor": anchor_key, "version": ver, "nav": out, "weights": w,
        "fee": fee, "turnover": turnover, "years": years,
        "ann_turnover": turnover / years if years > 0 else np.nan,
        "avg_equity": float(w["equity"].mean()),
    }


def collect_metrics(res):
    cfg = bt.VERSIONS[res["version"]]
    out = res["nav"]
    rows = []
    for seg, sub in [
        ("全区间", out),
        ("样本内", out[out.index <= pd.Timestamp(cfg["insample_end"])]),
        ("样本外", out[out.index > pd.Timestamp(cfg["insample_end"])]),
    ]:
        if len(sub) < 20:
            continue
        for col, cname in [("strategy", "策略"), ("bm_broad", "沪深300"), ("bm_dividend", "红利低波")]:
            m = bt.metrics(sub[col])
            rows.append({"锚": res["anchor"], "版本": res["version"], "区间": seg,
                         "对象": cname, **m})
    return rows


def main():
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs("charts", exist_ok=True)
    all_metrics, results = [], []
    for ver in ["long", "trade"]:
        for key in ANCHORS:
            res = run_one(key, ver)
            results.append(res)
            all_metrics.extend(collect_metrics(res))
            print(f"  {ver:<6}{key:<20} 权益均值 {res['avg_equity']:>5.0%}　"
                  f"换手/年 {res['ann_turnover']:>4.1f}　手续费 {res['fee']:.2%}")

    mdf = pd.DataFrame(all_metrics)
    mdf.to_csv(f"{EXP_DIR}/summary.csv", index=False, encoding="utf-8")

    # 精简对比表：只看策略，全区间 + 样本内 + 样本外
    piv = mdf[mdf["对象"] == "策略"].pivot_table(
        index=["版本", "锚"], columns="区间",
        values=["累计收益率", "年化收益率", "最大回撤", "夏普比率"])
    piv.to_csv(f"{EXP_DIR}/anchor_compare.csv", encoding="utf-8")

    for ver in ["long", "trade"]:
        print()
        print("=" * 96)
        print(f"{bt.VERSIONS[ver]['label']}：四种估值锚的策略表现（其余规则完全相同）")
        print("=" * 96)
        sub = mdf[(mdf["对象"] == "策略") & (mdf["版本"] == ver)]
        print(f"{'锚':<22}{'区间':<8}{'累计':>9}{'年化':>8}{'最大回撤':>9}{'夏普':>7}{'波动':>8}")
        print("-" * 96)
        for key in ANCHORS:
            for seg in ["全区间", "样本内", "样本外"]:
                r = sub[(sub["锚"] == key) & (sub["区间"] == seg)]
                if r.empty:
                    continue
                r = r.iloc[0]
                print(f"{key:<22}{seg:<8}{r['累计收益率']:>9.1%}{r['年化收益率']:>8.1%}"
                      f"{r['最大回撤']:>9.1%}{r['夏普比率']:>7.2f}{r['年化波动']:>8.1%}")
            print()

    plot(results)


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    colors = {"A0_沪深300PE": "#8C8C8C", "A1_上证红利PE": "#1D9E75",
              "A2_红利自身价格分位": "#378ADD", "A3_无估值层": "#D9534F"}

    for ver, tag, title in [("long", "长周期版", "长周期版（2015 起）"),
                            ("trade", "可交易版", "可交易版（2019 起）")]:
        rs = [r for r in results if r["version"] == ver]
        if not rs:
            continue

        # 图 1：净值
        fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
        for r in rs:
            n = r["nav"]["strategy"]
            ax.plot(n.index, n / n.iloc[0], label=r["anchor"], color=colors[r["anchor"]], linewidth=1.7)
        bm = rs[0]["nav"]
        ax.plot(bm.index, bm["bm_dividend"], label="红利低波 买入持有", color="#BA7517",
                linewidth=1.1, linestyle="--")
        ax.plot(bm.index, bm["bm_broad"], label="沪深300 买入持有", color="#5B7DB1",
                linewidth=1.1, linestyle=":")
        ax.set_title(f"{title}：换成不同估值锚后的策略净值（归一化到 1.0）", fontsize=13)
        ax.set_ylabel("净值")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        p1 = f"charts/07_估值锚对比_净值_{ver}.png"
        fig.savefig(p1)
        plt.close(fig)

        # 图 2：权益仓位
        fig, ax = plt.subplots(figsize=(12, 4.2), dpi=140)
        for r in rs:
            w = r["weights"]
            ax.step(pd.to_datetime(w["date"]), w["equity"], where="post",
                    label=r["anchor"], color=colors[r["anchor"]], linewidth=1.5)
        ax.set_title(f"{title}：不同估值锚下的权益仓位", fontsize=13)
        ax.set_ylabel("权益仓位")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        fig.tight_layout()
        p2 = f"charts/08_估值锚对比_仓位_{ver}.png"
        fig.savefig(p2)
        plt.close(fig)
        print(f"图已保存：{p1}　{p2}")


if __name__ == "__main__":
    main()
