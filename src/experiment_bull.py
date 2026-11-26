"""实验：牛市加仓红利，到底有没有用。

用户提出的直觉：策略在牛市里跟不上红利低波，能不能在牛市时提高红利仓位。

必须拆成两层，否则会把「加仓」记成「判断」的功劳：
  A. 事实层——牛市里红利低波是不是真的更强？如果牛市里红利并不强，
     「牛市加仓红利」从前提上就不成立。
  B. 归因层——就算加仓后收益变高，那是「判断对了」还是「仓位变高了」？
     做法是构造暴露完全相同（平均权益仓位一致）但不判断的固定配置，两者相减。

信号只做四组，覆盖两类直觉：
  牛市信号（沪深300 趋势）：MA120 / MA250
  红利自己的动量（红利在涨就多买）：MA120 / 60 日涨幅 > 0
  全部只用当日及之前的数据，月末判断、次日成交，与主线口径一致。

结论（见输出表）：
  1. 牛市里红利低波年化落后沪深300 10–16pct；非牛市反而领先 18–28pct。
     红利是「熊市资产」，不是「牛市资产」——「牛市加红利」加在了它最弱的时候。
  2. 四种信号全部输给同仓位的固定配置：暴露匹配后 Calmar 超额为负。
  3. 换手是主线的 5–10 倍，远超「月均 6.5pct」的一票否决线。

输出：output/exp/bull_signal_stats.csv、bull_variants.csv
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
BULL_EQ = 0.60
MAX_EQ = 0.60

SIGNALS = {
    "hs300_ma120": "沪深300 > 120 日均线（牛市）",
    "hs300_ma250": "沪深300 > 250 日均线（长期牛市）",
    "div_ma120": "红利低波 > 120 日均线（红利趋势）",
    "div_mom60": "红利低波 60 日涨幅 > 0（红利动量）",
}


def make_signal(kind: str, cl: pd.DataFrame) -> pd.Series:
    """返回月末可用的布尔序列（只用当日及之前数据）。"""
    div = cl["dividend"].dropna()
    broad = cl["broad"].dropna()
    if kind == "hs300_ma120":
        return (broad > broad.rolling(120).mean()).rename(kind)
    if kind == "hs300_ma250":
        return (broad > broad.rolling(250).mean()).rename(kind)
    if kind == "div_ma120":
        return (div > div.rolling(120).mean()).rename(kind)
    if kind == "div_mom60":
        return (div.pct_change(60) > 0).rename(kind)
    raise ValueError(kind)


def build_weights(ver: str, op, cl, kind: str | None, base_eq: float,
                  bull_eq: float = BULL_EQ, max_eq: float = MAX_EQ):
    """生成月末权重表。闸门逻辑与 src/allocation.py 完全一致，只把基准仓位换成信号函数。"""
    cfg = al.VERSIONS[ver]
    div = cl["dividend"].dropna()
    dates = pd.DatetimeIndex(al.month_end_dates(div))
    dates = dates[dates >= pd.Timestamp(cfg["start"])]

    sig = make_signal(kind, cl) if kind else None

    rets = cl[["dividend", "bond"]].pct_change()
    rows = []
    prev_eq = base_eq
    for d in dates:
        hist = rets.loc[rets.index < d].tail(VOL_WINDOW)
        port_ret = prev_eq * hist["dividend"] + (1 - prev_eq) * hist["bond"]
        valid = int(port_ret.notna().sum())
        realized = (float(port_ret.std() * np.sqrt(252))
                    if valid >= 30 and valid == len(port_ret) else np.nan)

        on = bool(sig.asof(d)) if sig is not None else False
        b = bull_eq if on else base_eq

        if np.isnan(realized) or realized <= 0:
            eq, scale = b, 1.0
        else:
            scale = min(1.0, VOL_TARGET / realized)
            eq = float(np.clip(b * scale, MIN_EQUITY, max_eq))

        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "w_dividend": round(eq, 4),
            "w_broad": 0.0,
            "w_bond": round(1.0 - eq, 4),
            "w_cash": 0.0,
            "signal_on": int(on) if sig is not None else 0,
            "realized_vol": round(realized, 4) if not np.isnan(realized) else np.nan,
            "scale": round(scale, 4),
        })
        prev_eq = eq
    return pd.DataFrame(rows)


def turnover(w: pd.DataFrame) -> dict:
    eq = w["w_dividend"].values
    d = np.abs(np.diff(eq))
    sign = np.sign(np.diff(eq))
    sign = sign[sign != 0]
    flips = int((np.diff(sign) != 0).sum()) if len(sign) > 1 else 0
    return {"平均权益pct": float(eq.mean() * 100),
            "月均调整pct": float(d.mean() * 100),
            "最大单月pct": float(d.max() * 100),
            "方向切换": flips}


def run_w(ver, op, cl, w):
    cfg = al.VERSIONS[ver]
    ww = w.copy()
    ww["date"] = pd.to_datetime(ww["date"])
    nav, _ = bt.simulate(ww, op, cl, exec_price=cfg["exec_price"], start=cfg["start"])
    m = bt.metrics(nav)
    m.update(turnover(w))
    return m


def match_exposure(ver, op, cl, target_avg: float, iters: int = 12):
    """找固定比例，使其在同样闸门下的平均权益仓位等于 target_avg。"""
    lo, hi = MIN_EQUITY, 0.90
    w, avg = None, None
    for _ in range(iters):
        mid = (lo + hi) / 2
        w = build_weights(ver, op, cl, None, base_eq=mid)
        avg = float(w["w_dividend"].mean())
        if avg < target_avg:
            lo = mid
        else:
            hi = mid
    return w, avg


def signal_stats(ver: str, op, cl) -> pd.DataFrame:
    """牛市 / 非牛市状态下红利与沪深300 的表现对比（用沪深300 MA120 定义牛市）。"""
    cfg = al.VERSIONS[ver]
    sig = make_signal("hs300_ma120", cl)
    r = cl[["dividend", "broad"]].pct_change()
    df = pd.DataFrame({"on": sig.reindex(r.index).ffill(),
                       "div": r["dividend"], "broad": r["broad"]}).dropna()
    df = df[df.index >= pd.Timestamp(cfg["start"])]

    out = []
    for label, sub in [("牛市（沪深300 > 120 日均线）", df[df["on"]]),
                       ("非牛市（≤ 120 日均线）", df[~df["on"]])]:
        n = len(sub)
        ann_div = (1 + sub["div"]).prod() ** (252 / n) - 1
        ann_broad = (1 + sub["broad"]).prod() ** (252 / n) - 1
        out.append({
            "状态": label, "交易日数": n,
            "占比pct": round(n / len(df) * 100, 1),
            "红利年化pct": round(ann_div * 100, 2),
            "沪深300年化pct": round(ann_broad * 100, 2),
            "红利−沪深300 pct": round((ann_div - ann_broad) * 100, 2),
            "红利日胜率pct": round((sub["div"] > sub["broad"]).mean() * 100, 1),
        })
    return pd.DataFrame(out)


def main() -> None:
    os.makedirs("output/exp", exist_ok=True)
    stats_all, var_all = [], []

    for ver, label in [("long", "长周期 2015 起"), ("trade", "可交易 2019 起")]:
        cfg = al.VERSIONS[ver]
        op, cl = bt.load_prices(cfg["prices"])

        st = signal_stats(ver, op, cl)
        st.insert(0, "版本", label)
        stats_all.append(st)
        print(f"\n{'='*94}\n{label}　牛市与非牛市：红利 vs 沪深300\n{'='*94}")
        print(st.to_string(index=False))

        w0 = build_weights(ver, op, cl, None, base_eq=0.40, max_eq=0.40)
        m0 = run_w(ver, op, cl, w0)

        print(f"\n{label}　主线 V0：年化 {m0['年化收益率']*100:.2f}%　"
              f"回撤 {m0['最大回撤']*100:.2f}%　Calmar {m0['Calmar比率']:.3f}　"
              f"平均仓位 {m0['平均权益pct']:.1f}%")

        rows = [{"版本": label, "方案": "V0 主线（基准 40%，上限 40%）", **m0}]
        for kind, desc in SIGNALS.items():
            w1 = build_weights(ver, op, cl, kind, base_eq=0.40)
            m1 = run_w(ver, op, cl, w1)
            w2, avg2 = match_exposure(ver, op, cl, m1["平均权益pct"] / 100)
            m2 = run_w(ver, op, cl, w2)
            rows.append({"版本": label, "方案": f"V1 {desc}", **m1})
            rows.append({"版本": label,
                         "方案": f"V2 暴露匹配固定 {avg2*100:.1f}%（不判断）", **m2})
            print(f"　{desc:<28s} 年化 {m1['年化收益率']*100:5.2f}%  "
                  f"回撤 {m1['最大回撤']*100:6.2f}%  Calmar {m1['Calmar比率']:.3f}  "
                  f"| 同仓位 {avg2*100:.1f}% → {m2['年化收益率']*100:5.2f}% / {m2['Calmar比率']:.3f}  "
                  f"| 超额 {m1['Calmar比率']-m2['Calmar比率']:+.3f}  "
                  f"| 月均调仓 {m1['月均调整pct']:.1f}pct")

        var_all.append(pd.DataFrame(rows))

    pd.concat(stats_all).to_csv("output/exp/bull_signal_stats.csv", index=False, encoding="utf-8")
    out = pd.concat(var_all)
    out.to_csv("output/exp/bull_variants.csv", index=False, encoding="utf-8")

    print(f"\n{'='*94}\n汇总：信号加仓 vs 同仓位固定配置（暴露匹配后的 Calmar 超额）\n{'='*94}")
    for ver in ["长周期 2015 起", "可交易 2019 起"]:
        sub = out[out["版本"] == ver].reset_index(drop=True)
        for i in range(1, len(sub) - 1, 2):
            a, b = sub.loc[i], sub.loc[i + 1]
            print(f"{ver}　{a['方案']:<32s} 超额 {a['Calmar比率']-b['Calmar比率']:+.3f}")
    print("\n已写入 output/exp/bull_signal_stats.csv、bull_variants.csv")


if __name__ == "__main__":
    main()
