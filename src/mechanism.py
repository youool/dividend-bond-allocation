"""
自检（v4）· 机制库：把「配置型策略」拆成四个可独立开关的旋钮

为什么要有这个文件：
  v3 的写法把三件事焊死在一起——基准比例 40%、上限 40%、线性缩放。
  结果是 max_equity == base_equity，scale 被 min(1.0, ...) 压住，
  所以「动态」只有一个方向：减仓。78% 的月份仓位锁在 40%，
  剩下 22% 的月份里又有大半只是从 40% 微调到 38%、39%。
  这不是一个动态策略，是「固定 40/60 + 偶发小幅减仓」。

本文件把 v3 拆成四个可以单独开关、单独检验的旋钮：

  旋钮 1｜波动闸门
    - 线性模式：eq = base × target / realized      （v3 的做法）
    - 分档模式：按 realized 落在哪个区间直接给档位   （更稳，不会顶格）
    - 关键改动：允许 mx > base，即低波动时能加仓

  旋钮 2｜回撤线（用户要的「盈亏线」）
    组合从历史高点回撤超过阈值 → 把权益压到地板；
    回撤修复到阈值的 recover 倍以内 → 恢复。
    注意：这是**响应已实现亏损**，不是预测方向。和已被证伪的均线择时不是一回事。

  旋钮 3｜再平衡带
    月末目标与当前漂移权重的偏离小于 band 就不动，省手续费与摩擦。

  旋钮 4｜波动度量
    std（双侧，v3 用）/ downside（只算负收益，更贴合「怕亏」这件事）。

两个必须讲清楚的口径：

  a) 回撤信号用「恒定 base 组合」的净值算，不用策略自身净值。
     原因：策略净值依赖权重、权重又依赖回撤，会形成循环依赖，无法无未来函数地实现。
     恒定组合的含义是「如果我躺着不动，账户从高点跌了多少」——这个量每个月末都真实可观测。

  b) 漂移权重按两资产相对涨幅推算，用来驱动再平衡带。
     v3 直接用上期目标比例算历史波动，忽略了月内漂移，是个已知近似。

输出与 al.build 同构，可直接喂 bt.simulate。
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt  # noqa: E402

VERSIONS = {
    "long": dict(bt.VERSIONS["long"], start="2015-01-01"),
    "trade": dict(bt.VERSIONS["trade"], start="2019-01-18"),
}


def month_ends(series, start):
    idx = series.index.to_series()
    d = pd.DatetimeIndex(idx.groupby([idx.dt.year, idx.dt.month]).max().sort_values().values)
    return d[d >= pd.Timestamp(start)]


def realized_vol(port_ret, measure="std"):
    x = port_ret.dropna()
    if len(x) < 30:
        return np.nan
    if measure == "std":
        return float(x.std() * np.sqrt(252))
    # 下行波动：只把负收益计入，正收益不当作风险
    neg = np.minimum(x.values, 0.0)
    return float(np.sqrt((neg ** 2).mean()) * np.sqrt(252))


def build(ver, base=0.40, mn=0.20, mx=0.40, vol_target=None, vol_window=60,
          measure="std", bands=None, dd_stop=None, dd_recover=0.5,
          tp_trigger=None, tp_to=None, band=0.0, smooth=1.0):
    """
    bands: 分档模式，[(上限波动, 权益比例), ...] 按波动从小到大。
           给了 bands 就忽略 vol_target（两者互斥，避免语义混乱）。
    dd_stop: 组合回撤阈值（正数，如 0.05 表示回撤 5% 触发）
    tp_trigger: 止盈触发（组合从 252 日低点上涨的幅度）
    """
    cfg = VERSIONS[ver]
    op, cl = bt.load_prices(cfg["prices"])
    div = cl["dividend"].dropna()
    dates = month_ends(div, cfg["start"])

    rets = cl[["dividend", "bond"]].pct_change()
    px = cl[["dividend", "bond"]]

    # 恒定 base 组合：既用来当回撤/止盈的信号源，也是「躺着不动」的参照
    const_ret = (base * rets["dividend"] + (1 - base) * rets["bond"]).fillna(0.0)
    const_nav = (1.0 + const_ret).cumprod()
    const_dd = const_nav / const_nav.cummax() - 1.0

    rows = []
    prev_w = base
    prev_d = None
    halted = False
    took_profit = False

    for d in dates:
        hist = rets.loc[rets.index < d].tail(vol_window)
        port = prev_w * hist["dividend"] + (1 - prev_w) * hist["bond"]
        rv = realized_vol(port, measure)

        # 月内漂移后的真实权重
        if prev_d is None:
            drift = base
        else:
            gd = prev_w * (px.loc[d, "dividend"] / px.loc[prev_d, "dividend"])
            gb = (1 - prev_w) * (px.loc[d, "bond"] / px.loc[prev_d, "bond"])
            drift = float(gd / (gd + gb)) if (gd + gb) > 0 else prev_w

        # ---- 旋钮 1：波动闸门 ----
        note = "固定比例"
        if np.isnan(rv) or rv <= 0:
            eq = base
        elif bands is not None:
            eq = bands[-1][1]
            for ub, w in bands:
                if rv <= ub:
                    eq = w
                    break
            note = "分档定仓"
        elif vol_target is not None:
            eq = float(np.clip(base * vol_target / rv, mn, mx))
            note = "波动超预算→收缩" if eq < base else ("波动低于预算→加仓" if eq > base else "波动在预算内")
        else:
            eq = base

        # ---- 平滑：只走一部分路，别每个月都把仓位翻一遍 ----
        # 打开上行后波动信号会频繁穿越 8%，不平滑就会出现「每月调 6 个点、方向切换上百次」，
        # 那种策略在纸上很漂亮，实盘执行不了。smooth=0.5 表示每次只走到目标的一半。
        if smooth < 1.0:
            raw = eq
            eq = smooth * eq + (1.0 - smooth) * prev_w
            eq = float(np.clip(eq, mn, mx))
            if abs(raw - eq) > 1e-9:
                note = "平滑过渡"

        # ---- 旋钮 2：回撤线 ----
        # 回撤线不平滑：灾难时要立刻降，不能慢慢来
        dd = float(const_dd.loc[d])
        if dd_stop is not None:
            if dd <= -dd_stop:
                if not halted:
                    note = f"回撤{dd:.1%}→压到地板"
                halted = True
            elif halted and dd >= -dd_stop * dd_recover:
                halted = False
                note = "回撤修复→恢复"
            if halted:
                eq = mn

        # ---- 止盈线 ----
        if tp_trigger is not None and dd_stop is None:
            lo = const_nav.loc[:d].tail(252).min()
            up = float(const_nav.loc[d] / lo - 1.0)
            if up >= tp_trigger:
                took_profit = True
            elif up <= tp_trigger * 0.3:
                took_profit = False
            if took_profit:
                eq = min(eq, tp_to)
                note = f"涨{up:.1%}→止盈减仓"

        # ---- 旋钮 3：再平衡带 ----
        # 用「漂移后的真实权重」判断，而不是上期目标权重：
        # 上期目标 40%、这个月红利涨得多漂到 44%，目标仍是 40%，
        # 偏离 4% 已经该调了，若拿 40% 去比就会误判成「没偏离」。
        if band > 0 and abs(drift - eq) < band:
            eq = drift
            note = "偏离不足→不动"

        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "w_dividend": round(float(eq), 4),
            "w_broad": 0.0,
            "w_bond": round(1.0 - float(eq), 4),
            "w_cash": 0.0,
            "realized_vol": round(rv, 4) if not np.isnan(rv) else np.nan,
            "scale": round(float(eq) / base, 4) if base else 1.0,
            "note": note,
        })
        prev_w = float(eq)
        prev_d = d

    return pd.DataFrame(rows)
