"""实验：双向波动率目标化——仓位随风险浮动，而不是锁死在 40%。

当前主线的闸门是单向的：
    eq = 40% × min(1, 8% / 实际波动)，且上限 40%
实际波动低于 8% 时（风险预算没用完），仓位不会上升，永远停在 40%。
用户的问题是：能不能让仓位双向浮动，把风险预算真正用满？

    eq = 40% × (8% / 实际波动)，上下限 [20%, cap]

cap 取 40%（= 现状）/ 50% / 60% / 80%，逐一比较。
关键纪律仍然是暴露匹配：动态仓位如果只是把平均仓位抬高，收益变高不算本事。
每个动态方案都要和一个「平均仓位相同、但完全不判断」的固定配置比。

输出：output/exp/dynamic_vol.csv
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import allocation as al
import backtest as bt

VOL_TARGET = 0.08
VOL_WINDOW = 60
BASE = 0.40
FLOOR = 0.20


def build_dyn(ver, op, cl, cap: float, two_way: bool = True, base: float = BASE):
    cfg = al.VERSIONS[ver]
    div = cl["dividend"].dropna()
    dates = pd.DatetimeIndex(al.month_end_dates(div))
    dates = dates[dates >= pd.Timestamp(cfg["start"])]

    rets = cl[["dividend", "bond"]].pct_change()
    rows = []
    prev_eq = base
    for d in dates:
        hist = rets.loc[rets.index < d].tail(VOL_WINDOW)
        port = prev_eq * hist["dividend"] + (1 - prev_eq) * hist["bond"]
        valid = int(port.notna().sum())
        realized = (float(port.std() * np.sqrt(252))
                    if valid >= 30 and valid == len(port) else np.nan)

        if np.isnan(realized) or realized <= 0:
            eq = base
        else:
            scale = VOL_TARGET / realized
            if not two_way:
                scale = min(1.0, scale)
            eq = float(np.clip(base * scale, FLOOR, cap))

        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "w_dividend": round(eq, 4),
            "w_broad": 0.0,
            "w_bond": round(1.0 - eq, 4),
            "w_cash": 0.0,
            "realized_vol": round(realized, 4) if not np.isnan(realized) else np.nan,
        })
        prev_eq = eq
    return pd.DataFrame(rows)


def build_fixed(ver, op, cl, eq: float):
    cfg = al.VERSIONS[ver]
    div = cl["dividend"].dropna()
    dates = pd.DatetimeIndex(al.month_end_dates(div))
    dates = dates[dates >= pd.Timestamp(cfg["start"])]
    return pd.DataFrame([{
        "date": d.strftime("%Y-%m-%d"), "w_dividend": eq, "w_broad": 0.0,
        "w_bond": round(1 - eq, 4), "w_cash": 0.0,
    } for d in dates])


def run(ver, op, cl, w):
    cfg = al.VERSIONS[ver]
    ww = w.copy()
    ww["date"] = pd.to_datetime(ww["date"])
    nav, _ = bt.simulate(ww, op, cl, exec_price=cfg["exec_price"], start=cfg["start"])
    m = bt.metrics(nav)
    eq = w["w_dividend"].values
    d = np.abs(np.diff(eq))
    sign = np.sign(np.diff(eq)); sign = sign[sign != 0]
    m["平均权益pct"] = float(eq.mean() * 100)
    m["月均调整pct"] = float(d.mean() * 100)
    m["最大单月pct"] = float(d.max() * 100)
    m["方向切换"] = int((np.diff(sign) != 0).sum()) if len(sign) > 1 else 0
    m["加仓月数"] = int((np.diff(eq) > 0).sum())
    return m


def match_fixed(ver, op, cl, target_avg):
    lo, hi = FLOOR, 0.95
    best = None
    for _ in range(12):
        mid = (lo + hi) / 2
        m = run(ver, op, cl, build_fixed(ver, op, cl, mid))
        best = (mid, m)
        if m["平均权益pct"] / 100 < target_avg:
            lo = mid
        else:
            hi = mid
    return best


def main() -> None:
    os.makedirs("output/exp", exist_ok=True)
    rows = []
    for ver, label in [("long", "长周期 2015 起"), ("trade", "可交易 2019 起")]:
        cfg = al.VERSIONS[ver]
        op, cl = bt.load_prices(cfg["prices"])
        print(f"\n{'='*104}\n{label}　单向 vs 双向波动率目标化（目标 8%，下限 20%）\n{'='*104}")
        print(f"{'方案':<34s}{'年化':>8}{'回撤':>9}{'Calmar':>8}{'夏普':>7}"
              f"{'均仓':>7}{'月换手':>8}{'切换':>6}{'加仓月':>7}")

        for cap, two_way, name in [
            (0.40, False, "单向（现状）：上限 40%"),
            (0.50, True, "双向：上限 50%"),
            (0.60, True, "双向：上限 60%"),
            (0.80, True, "双向：上限 80%"),
            (1.00, True, "双向：无上限（满仓封顶）"),
        ]:
            w = build_dyn(ver, op, cl, cap=cap, two_way=two_way)
            m = run(ver, op, cl, w)
            rows.append({"版本": label, "方案": name, **m})
            print(f"{name:<34s}{m['年化收益率']*100:>7.2f}%{m['最大回撤']*100:>8.2f}%"
                  f"{m['Calmar比率']:>8.3f}{m['夏普比率']:>7.2f}{m['平均权益pct']:>6.1f}%"
                  f"{m['月均调整pct']:>7.1f}%{m['方向切换']:>6d}{m['加仓月数']:>6d}")

            # 暴露匹配
            fx, mf = match_fixed(ver, op, cl, m["平均权益pct"] / 100)
            rows.append({"版本": label, "方案": f"　└ 同仓位固定 {fx*100:.1f}%（对照）", **mf})
            print(f"{'　└ 同仓位固定 ' + f'{fx*100:.1f}%':<34s}"
                  f"{mf['年化收益率']*100:>7.2f}%{mf['最大回撤']*100:>8.2f}%"
                  f"{mf['Calmar比率']:>8.3f}{mf['夏普比率']:>7.2f}{mf['平均权益pct']:>6.1f}%"
                  f"{mf['月均调整pct']:>7.1f}%{'-':>6}{'-':>7}"
                  f"　超额 {m['Calmar比率']-mf['Calmar比率']:+.3f}")

    pd.DataFrame(rows).to_csv("output/exp/dynamic_vol.csv", index=False, encoding="utf-8")
    print("\n已写入 output/exp/dynamic_vol.csv")


if __name__ == "__main__":
    main()
