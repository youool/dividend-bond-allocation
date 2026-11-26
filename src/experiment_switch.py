"""实验：牛市把红利换成沪深300，到底能不能做。

用户的想法：牛市时减少债券与红利，增加沪深300；并质疑「必须守住 7% 回撤」这个约束。

本脚本回答两件事：
  A. 用「沪深300 在 120 日均线上方」判定牛市，牛市持有沪深300、非牛市持有红利，
     与三种不判断的固定配置比。关键对照是「红利 20% + 沪深300 20% + 债 60%」——
     切换策略长期看有一半时间在持有沪深300，这个混合对照才是它真正的替代品。
  B. 在相同闸门下扫权益比例 20%–80%，给出「约束—收益」前沿，
     看 Calmar 是稳住了还是随仓位下降。这决定 7% 预算是不是可以放宽。

两个版本、同起点、同成本（双边万 2）、同 8% 波动闸门。
输出：output/exp/switch_variants.csv、output/exp/frontier.csv
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
MIN_EQUITY = 0.20


def signal_series(cl: pd.DataFrame, kind: str) -> pd.Series:
    broad = cl["broad"].dropna()
    if kind == "ma120":
        return (broad > broad.rolling(120).mean()).rename(kind)
    if kind == "ma250":
        return (broad > broad.rolling(250).mean()).rename(kind)
    raise ValueError(kind)


def build(ver: str, op, cl, mix_fn, cap: float = 1.0, kind: str | None = "ma120"):
    """mix_fn(is_bull) -> (w_div, w_broad)；债券为剩余。闸门只降不升，保持两腿比例。"""
    cfg = al.VERSIONS[ver]
    div = cl["dividend"].dropna()
    dates = pd.DatetimeIndex(al.month_end_dates(div))
    dates = dates[dates >= pd.Timestamp(cfg["start"])]
    sig = signal_series(cl, kind) if kind else None

    rets = cl[["dividend", "broad", "bond"]].pct_change()
    rows = []
    prev = mix_fn(False)  # 首期按非牛市配置起算
    prev = (prev[0], prev[1], max(0.0, 1 - prev[0] - prev[1]))

    switches = 0
    last_state = None
    for d in dates:
        hist = rets.loc[rets.index < d].tail(VOL_WINDOW)
        port = (prev[0] * hist["dividend"] + prev[1] * hist["broad"]
                + prev[2] * hist["bond"])
        valid = int(port.notna().sum())
        realized = (float(port.std() * np.sqrt(252))
                    if valid >= 30 and valid == len(port) else np.nan)

        is_bull = bool(sig.asof(d)) if sig is not None else False
        if last_state is not None and is_bull != last_state:
            switches += 1
        last_state = is_bull

        wd, wb = mix_fn(is_bull)
        eq_raw = wd + wb
        if np.isnan(realized) or realized <= 0 or eq_raw <= 0:
            eq = eq_raw
        else:
            scale = min(1.0, VOL_TARGET / realized)
            eq = float(np.clip(eq_raw * scale, MIN_EQUITY, min(eq_raw, cap)))
        k = eq / eq_raw if eq_raw > 0 else 0.0
        wd, wb = wd * k, wb * k

        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "w_dividend": round(wd, 4),
            "w_broad": round(wb, 4),
            "w_bond": round(1.0 - wd - wb, 4),
            "w_cash": 0.0,
            "bull": int(is_bull),
            "realized_vol": round(realized, 4) if not np.isnan(realized) else np.nan,
        })
        prev = (wd, wb, 1.0 - wd - wb)

    df = pd.DataFrame(rows)
    df.attrs["switches"] = switches
    return df


def run(ver, op, cl, w):
    cfg = al.VERSIONS[ver]
    ww = w.copy()
    ww["date"] = pd.to_datetime(ww["date"])
    nav, _ = bt.simulate(ww, op, cl, exec_price=cfg["exec_price"], start=cfg["start"])
    m = bt.metrics(nav)
    eq = (w["w_dividend"] + w["w_broad"]).values
    d = np.abs(np.diff(eq))
    m["平均权益pct"] = float(eq.mean() * 100)
    m["月均调整pct"] = float(d.mean() * 100)
    m["信号切换次数"] = int(w.attrs.get("switches", 0))
    m["牛市月份pct"] = float(w["bull"].mean() * 100)
    return m


def main() -> None:
    os.makedirs("output/exp", exist_ok=True)

    # ---------- A. 牛市换资产 ----------
    variants = [
        ("固定 红利40／债60（主线，无信号）", lambda b: (0.40, 0.00)),
        ("固定 沪深300 40／债60", lambda b: (0.00, 0.40)),
        ("固定 红利20／沪深300 20／债60（混合对照）", lambda b: (0.20, 0.20)),
        ("切换 牛市沪深300 40／非牛市红利40", lambda b: (0.00, 0.40) if b else (0.40, 0.00)),
        ("切换 牛市沪深300 60／非牛市红利60", lambda b: (0.00, 0.60) if b else (0.60, 0.00)),
        ("切换 牛市沪深300 60／非牛市红利40", lambda b: (0.00, 0.60) if b else (0.40, 0.00)),
    ]

    rows = []
    for ver, label in [("long", "长周期 2015 起"), ("trade", "可交易 2019 起")]:
        cfg = al.VERSIONS[ver]
        op, cl = bt.load_prices(cfg["prices"])
        print(f"\n{'='*100}\n{label}　牛市换资产（沪深300 > 120 日均线）\n{'='*100}")
        print(f"{'方案':<44s}{'年化':>8}{'回撤':>9}{'Calmar':>8}{'夏普':>7}{'均仓':>7}{'月换手':>8}{'切换':>6}")
        for name, fn in variants:
            w = build(ver, op, cl, fn)
            m = run(ver, op, cl, w)
            rows.append({"版本": label, "方案": name, **m})
            print(f"{name:<44s}{m['年化收益率']*100:>7.2f}%{m['最大回撤']*100:>8.2f}%"
                  f"{m['Calmar比率']:>8.3f}{m['夏普比率']:>7.2f}{m['平均权益pct']:>6.1f}%"
                  f"{m['月均调整pct']:>7.1f}%{m['信号切换次数']:>6d}")

    pd.DataFrame(rows).to_csv("output/exp/switch_variants.csv", index=False, encoding="utf-8")

    # ---------- B. 约束—收益前沿（同闸门） ----------
    frows = []
    for ver, label in [("long", "长周期 2015 起"), ("trade", "可交易 2019 起")]:
        cfg = al.VERSIONS[ver]
        op, cl = bt.load_prices(cfg["prices"])
        print(f"\n{'='*100}\n{label}　权益比例扫描（全部带 8% 波动闸门，只降不升）\n{'='*100}")
        print(f"{'权益比例':>8}{'年化':>9}{'最大回撤':>10}{'Calmar':>9}{'夏普':>8}{'平均仓位':>9}")
        for eq in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
            w = build(ver, op, cl, lambda b, e=eq: (e, 0.0), cap=eq, kind=None)
            m = run(ver, op, cl, w)
            frows.append({"版本": label, "权益比例": eq, **m})
            print(f"{eq:>8.0%}{m['年化收益率']*100:>8.2f}%{m['最大回撤']*100:>9.2f}%"
                  f"{m['Calmar比率']:>9.2f}{m['夏普比率']:>8.2f}{m['平均权益pct']:>8.1f}%")

    pd.DataFrame(frows).to_csv("output/exp/frontier.csv", index=False, encoding="utf-8")
    print("\n已写入 output/exp/switch_variants.csv、frontier.csv")


if __name__ == "__main__":
    main()
