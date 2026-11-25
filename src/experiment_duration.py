"""
自检（v4）· 国债端的久期选择：影响最大的一块，之前从没检验过

为什么这是最该先做的一步：
  前面所有实验都在调权益那 40%，怎么调都是四成的事。
  国债占 60%，久期一变，六成仓位的收益与风险都变。

做法（刻意保持干净）：
  权重表完全不动，只把 bond 这一列的价格换成另一只国债 ETF。
  这样测出来的差异 100% 来自「持有了什么」，不掺任何策略改动。
  口径与现有一致：后复权（分红再投资）、月末调仓、次日成交、双边万 2。

三个必须看的横截面：
  1. 债券单资产：五年 vs 十年 vs 三十年，年化 / 回撤 / 波动
  2. 组合层面：同一张权重表下，换成十年国债后年化与回撤各变多少
  3. 关键时段：2020 年 5 月债市转熊、2022 年 11 月理财赎回潮、2024 年 9 月股债跷跷板
     ——久期风险不是理论上的，这三次是真实发生过的

必须先说清的局限（结论要带着这句话走）：
  2019–2026 是中国利率的单边下行期，十年期国债收益率从 3.3% 一路降到 1.8% 附近。
  久期越长，在这段里受益越大。这是**环境**给的，不是久期本身的alpha。
  所以本实验只能回答「过去七年哪个更好」，不能回答「未来哪个更好」。
  真正的结论要落在一句话上：长久期是用更大的利率风险，换更高的票息与资本利得。

输出：
  output/exp/v4_duration.csv
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt  # noqa: E402

EXP_DIR = "output/exp"

BONDS = {
    "五年 511010": "data/511010.csv",
    "十年 511260": "data/511260.csv",
    "三十年 511090": "data/511090.csv",
}
BASE_PRICES = {
    "dividend": "data/512890.csv",
    "broad": "data/510300.csv",
    "bond": "data/511010.csv",
    "cash": "data/511880.csv",
}
PERIODS = [("2019-01-18", "2020-04-30"), ("2020-05-01", "2022-10-31"),
           ("2022-11-01", "2024-08-31"), ("2024-09-01", "2026-11-23")]


def seg_metrics(nav):
    return bt.metrics(nav)


def main():
    os.makedirs(EXP_DIR, exist_ok=True)
    rows = []

    # ---------- 1. 债券单资产 ----------
    for name, path in BONDS.items():
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        px = df["close"].dropna()
        for start in ["2017-08-24", "2019-01-18", "2023-06-13"]:
            sub = px[px.index >= pd.Timestamp(start)]
            if len(sub) < 250:
                continue
            m = bt.metrics(sub / sub.iloc[0])
            rows.append({"对象": f"债券|{name}", "区间": f"{start}起",
                         "年化": m["年化收益率"], "回撤": m["最大回撤"],
                         "波动": m["年化波动"], "Calmar": m["Calmar比率"]})

    # ---------- 2. 组合层面：同一张权重表，只换 bond ----------
    w = pd.read_csv("output/alloc_trade_vol08.csv", parse_dates=["date"])

    def build_nav(path, tag):
        """从 tag 与「该债券数据首日」中较晚的那个开始，保证三只从同一时点建仓。"""
        op, cl = bt.load_prices(dict(BASE_PRICES, bond=path))
        b0 = cl["bond"].dropna().index[0]
        s = max(pd.Timestamp(tag), b0)
        ww = w[w["date"] >= s]
        if len(ww) < 6:
            return None
        s = ww["date"].iloc[0]
        nav, _ = bt.simulate(ww, op.loc[op.index >= s], cl.loc[cl.index >= s], "open", start=s)
        if nav.nunique() < 10:
            print(f"  跳过：净值几乎不动，数据不完整")
            return None
        return nav

    navs, navs_23 = {}, {}
    for tag, store in [("2019-01-18", navs), ("2023-06-13", navs_23)]:
        for name, path in BONDS.items():
            nav = build_nav(path, tag)
            if nav is None:
                continue
            store[name] = nav
            m = bt.metrics(nav)
            rows.append({"对象": f"组合|{name}", "区间": f"{nav.index[0].date()}起",
                         "年化": m["年化收益率"], "回撤": m["最大回撤"],
                         "波动": m["年化波动"], "Calmar": m["Calmar比率"]})

    pd.DataFrame(rows).to_csv(f"{EXP_DIR}/v4_duration.csv", index=False, encoding="utf-8")

    print("=" * 92)
    print("一、债券单资产：久期越长，收益与回撤同步放大")
    print("=" * 92)
    print(f"{'对象':<18}{'区间':<14}{'年化':>9}{'最大回撤':>10}{'年化波动':>10}{'Calmar':>9}")
    print("-" * 92)
    for _, r in pd.DataFrame(rows).iterrows():
        print(f"{r['对象']:<18}{r['区间']:<14}{r['年化']:>9.1%}{r['回撤']:>10.1%}"
              f"{r['波动']:>10.1%}{r['Calmar']:>9.2f}")

    print()
    print("=" * 92)
    print("二、组合层面：权重表完全不动，只把国债换掉")
    print("=" * 92)
    for title, store in [("2019-01 起（五年 vs 十年，两只都有数据）", navs),
                         ("2023-06 起（三只共同区间，三十年只能看这一段）", navs_23)]:
        print(f"\n{title}")
        print(f"{'组合':<18}{'年化':>9}{'最大回撤':>10}{'年化波动':>10}{'Calmar':>9}{'夏普':>8}")
        print("-" * 92)
        for name, nav in store.items():
            m = bt.metrics(nav)
            print(f"{name:<18}{m['年化收益率']:>9.1%}{m['最大回撤']:>10.1%}"
                  f"{m['年化波动']:>10.1%}{m['Calmar比率']:>9.2f}{m['夏普比率']:>8.2f}")

    print()
    print("=" * 92)
    print("三、关键时段：久期风险不是理论上的，这三次真实发生过")
    print("=" * 92)
    print(f"{'时段':<24}" + "".join(f"{n:>22}" for n in navs))
    print("-" * 92)
    for a, b in PERIODS:
        line = f"{a[:7]}→{b[:7]:<12}"
        for name, nav in navs.items():
            sub = nav[(nav.index >= pd.Timestamp(a)) & (nav.index <= pd.Timestamp(b))]
            if len(sub) < 20:
                line += f"{'—':>22}"
                continue
            ret = sub.iloc[-1] / sub.iloc[0] - 1
            dd = float((sub / sub.cummax() - 1).min())
            line += f"{ret:>11.1%} / {dd:>8.1%}"
        print(line)

    # 组合之间的相关性
    rdf = pd.DataFrame({k: v.pct_change() for k, v in navs.items()}).dropna()
    print()
    print("组合净值日收益相关系数：")
    print(rdf.corr().round(3).to_string())


if __name__ == "__main__":
    main()
