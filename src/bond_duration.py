"""
支线 · 债端久期：把 60% 国债从「纯五年」换成混合久期

背景（第七轮自检的结论，有数据支撑）：
  同一张权重表、只替换 60% 的国债标的，2019–2026 十年 511260 比五年 511010
  组合年化高 0.6 个百分点、回撤完全不变（7.5%/−6.3% vs 6.9%/−6.3%，
  Calmar 1.19 vs 1.09）。
  但分年度拆开看：2024 一年（+2.70%）就占了总差异的 29%，且 2025 起连续吃亏。
  利率从 3.3% 降到 1.8% 附近，久期的资本利得部分不可持续。
  → 全押十年等于赌利率继续下行；混合久期才是「取中间」的做法。

这个模块只做一件事：
  把两只国债 ETF 按给定比例合成成一条标准价格曲线（date/open/close），
  之后所有回测代码把它当普通标的读，不需要改任何回测逻辑。
  这样"换债"与"改规则"彻底分离，测出来的差异 100% 来自持有什么。

合成口径（与回测内核的份额法一致，不是简单的收益加权）：
  - 首日按收盘价建仓，份额按目标比例分配
  - 每月最后一个交易日把比例拉回目标（与策略自身的调仓节奏同步，贴近实盘）
  - 开盘价按「昨日收盘的份额 × 今日开盘价」推算，保证 open 与 close 自洽，
    否则可交易版用开盘价成交会失真
  - 不计交易成本：国债 ETF 双边佣金约万 0.2，月度再平衡的年化拖累在 0.01 个百分点
    量级，比久期差异小两个数量级

输出：
  data/synth_bond_mix{30,50,70}.csv
"""

import os
import sys

import numpy as np
import pandas as pd

P5 = "data/511010.csv"    # 五年国债 ETF（主线在用）
P10 = "data/511260.csv"   # 十年国债 ETF（2017-08 起才有数据）

# 十年国债在 60% 里的占比。0 和 1 直接用原始文件，不合成，
# 避免在两端引入任何合成误差，也让"主线 = mix00"这件事在文件层面成立。
PLANS = {
    "mix00": {"label": "十年 0%（纯五年，主线）", "tenor": 0.0, "file": P5},
    "mix30": {"label": "十年 30%", "tenor": 0.3, "file": "data/synth_bond_mix30.csv"},
    "mix50": {"label": "十年 50%（支线）", "tenor": 0.5, "file": "data/synth_bond_mix50.csv"},
    "mix70": {"label": "十年 70%", "tenor": 0.7, "file": "data/synth_bond_mix70.csv"},
    "mix100": {"label": "十年 100%（上限对照）", "tenor": 1.0, "file": P10},
}

ORDER = ["mix00", "mix30", "mix50", "mix70", "mix100"]


def _read(path):
    df = pd.read_csv(path, parse_dates=["date"]).drop_duplicates("date").sort_values("date")
    df = df.set_index("date")
    if "open" not in df.columns:
        df["open"] = df["close"]
    return df[["open", "close"]].astype(float)


def synthesize(tenor):
    """
    五年 + 十年 按 (1-tenor, tenor) 合成一条曲线。

    为什么不用「日收益加权平均」：那等价于每天再平衡，比实盘更频繁，
    而且会让合成曲线隐含一个策略里并不存在的日频交易假设。
    这里用月末再平衡 + 份额法，与策略自身的调仓节奏完全对齐。
    """
    if tenor <= 0 or tenor >= 1:
        raise ValueError("单腿情形直接用原始文件，不要走合成")

    legs = [(P5, 1 - tenor), (P10, tenor)]
    frames = {p: _read(p) for p, _ in legs}

    idx = None
    for f in frames.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    idx = idx.sort_values()
    if len(idx) < 250:
        raise RuntimeError(f"两只债的共同交易日只有 {len(idx)} 天，无法合成")

    C = pd.DataFrame({p: frames[p].loc[idx, "close"] for p, _ in legs})
    O = pd.DataFrame({p: frames[p].loc[idx, "open"] for p, _ in legs})
    w0 = np.array([w for _, w in legs], dtype=float)
    w0 = w0 / w0.sum()

    # 每个自然月的最后一个交易日 = 再平衡日
    me = pd.DatetimeIndex(
        pd.Series(idx, index=idx).groupby([idx.year, idx.month]).max().values)

    shares = w0 / C.iloc[0].values
    nav_o, nav_c = [], []
    for t in idx:
        val_o = float((shares * O.loc[t].values).sum())
        val_c = float((shares * C.loc[t].values).sum())
        nav_o.append(val_o)
        nav_c.append(val_c)
        if t in me:
            # 月末收盘拉回目标比例，次日起生效
            shares = val_c * w0 / C.loc[t].values

    out = pd.DataFrame({"date": idx, "open": nav_o, "close": nav_c})
    return out, len(me)


def ensure(plan_key, verbose=True):
    """确保该方案的数据文件存在，返回路径。已存在则不重算。"""
    plan = PLANS[plan_key]
    if plan["tenor"] <= 0 or plan["tenor"] >= 1:
        return plan["file"]            # 单腿用原始文件
    path = plan["file"]
    if os.path.exists(path):
        return path
    df, n_reb = synthesize(plan["tenor"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    if verbose:
        print(f"合成 {path}：{len(df)} 个交易日，{n_reb} 次月度再平衡，"
              f"{df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")
    return path


def main():
    for k in ORDER:
        path = ensure(k)
        d = _read(path)["close"]
        yrs = len(d) / 252.0
        ann = (d.iloc[-1] / d.iloc[0]) ** (1 / yrs) - 1
        vol = d.pct_change().std() * np.sqrt(252)
        print(f"{PLANS[k]['label']:<24}{str(d.index[0].date()):<12}"
              f"年化 {ann:>7.2%}　波动 {vol:>6.2%}　{path}")

    print()
    print("说明：合成曲线的起点由十年国债决定（2017-08），")
    print("      所以长周期版做久期对比只能从 2017 年末开始，2015–2017 那两年只有五年国债。")
    print("      这是数据边界，不是选择，结果里必须标注。")


if __name__ == "__main__":
    main()
