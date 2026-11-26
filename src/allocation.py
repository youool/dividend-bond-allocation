"""
第 3 关（重构版）· 目标比例表：配置型策略

为什么整个换掉：
  重构前的 v2 规则（估值定上限 + 120 日均线扣减 + 60 日动量倾斜）已经用六组对照实验
  判死（见 src/experiment_*.py 与 output/exp/）：
    - 换估值锚无效（实验一）
    - 择时整体是负贡献，暴露匹配的固定配置全面更好（实验二）
    - 24 组参数全输给不择时（实验三）
    - 窄幅择时的超额夏普经 100 次随机重排检验后不显著（实验六）
  结论：这套策略没有 alpha，它的收益全部来自「持有了什么」，不是「什么时候持有」。
  所以重构的方向不是再找信号，而是把「持有什么」这件事做对。

新策略一句话：
  长期固定持有 40% 红利低波 + 60% 国债，每月末把比例拉回目标；
  当组合已实现波动明显超过风险预算时，按比例收缩权益，波动回落后再恢复。
  —— 不预测方向，只管理风险敞口。

三个设计决定，都要能当场解释：

  1. 为什么是 40%？
     先定风险预算（可交易口径最大回撤不超过 7%），再反推比例。
     实验四的扫描结果：40% → 回撤 −6.8%；50% → −8.5%；60% → −10.2%。
     40% 是「不突破 7% 回撤」约束下红利占比的上限，不是拍出来的。

  2. 为什么国债占 60%？
     它是这个组合里唯一与权益长期负相关的资产，且波动只有 2% 上下。
     60% 国债把组合波动压到 6% 附近，这是「稳健」在数字上的含义。

  3. 为什么要波动率目标化？
     固定 40% 的唯一弱点是 2015 股灾：长周期版最大回撤 −17.4%。
     股灾的特征是「波动先飙升」——用已实现波动做闸门，不需要预测方向，
     就能在最危险的时候自动收缩。这是唯一被保留的「动态」成分，
     因为它响应的是风险，不是收益预期。

不做的事（都试过、都无效）：
  - 不做方向择时（估值分位、均线、动量排名）
  - 不在标的之间轮动
  - 不因为「感觉便宜」加仓

输出：
  output/alloc_{版本}_{模式}.csv    每行含调仓日、目标比例、当期波动、缩放系数
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt  # noqa: E402

# ---------- 参数 ----------
BASE_EQUITY = 0.40      # 红利低波的基准比例
MIN_EQUITY = 0.20       # 波动率目标化后的下限（不低于 20%，否则组合失去意义）
MAX_EQUITY = 0.50       # 上限：风险预算没用完时允许适度加仓，最多 50%
# 为什么是 50% 而不是 40%：闸门改为双向之后，上限扫描 0.42–0.60 是一条平台，
# 长周期最优 0.48（Calmar 0.583）、可交易最优 0.50（1.280）；0.55 以上开始变差。
# 50% 比基准高 10 个百分点，是「风险预算没用完时的适度加仓」，不是激进加杠杆。
VOL_WINDOW = 60         # 算已实现波动的窗口（交易日）
VOL_TARGET = None       # None 表示不启用波动率目标化

MODES = {
    "fixed": {"label": "固定 40/60（月度再平衡）", "vol_target": None},
    "vol12": {"label": "波动率目标 12%（月度）", "vol_target": 0.12},
    "vol10": {"label": "波动率目标 10%（月度）", "vol_target": 0.10},
    "vol08": {"label": "波动率目标 8%（月度）", "vol_target": 0.08},
}

# 最终采用的一档。依据见 experiment_vol.py：
#   vol08 把长周期版最大回撤从 −17.4% 压到 −12.1%，年化反而从 6.2% 抬到 6.3%
#   （收缩的月份正好是坏月份，所以「错过上涨」的担忧没有发生）
DEFAULT_MODE = "vol08"

VERSIONS = {
    "long": dict(bt.VERSIONS["long"], label="长周期版（2015 起）", start="2015-01-01"),
    "trade": dict(bt.VERSIONS["trade"], label="可交易版（2019 起）", start="2019-01-18"),
}


def month_end_dates(series):
    idx = series.index.to_series()
    return idx.groupby([idx.dt.year, idx.dt.month]).max().sort_values().values


def build(version_key, mode="fixed", base_equity=BASE_EQUITY,
          vol_window=VOL_WINDOW, out_tag=None, bond_path=None, write=True):
    """
    返回一张调仓表：每行是「哪天、目标比例是多少、为什么」。
    波动率用**上一期持有的比例**算组合历史波动，避免用当期还没发生的价格。

    bond_path：把国债腿换成另一条价格曲线（支线做久期对比时用）。
    write：设为 False 则只算不落盘，用于诊断，不会覆盖主线产物。
    """
    cfg = VERSIONS[version_key]
    prices = dict(cfg["prices"])
    if bond_path:
        prices["bond"] = bond_path
    op, cl = bt.load_prices(prices)

    div = cl["dividend"].dropna()
    dates = pd.DatetimeIndex(month_end_dates(div))
    dates = dates[dates >= pd.Timestamp(cfg["start"])]

    vol_target = MODES[mode]["vol_target"]

    rets = cl[["dividend", "bond"]].pct_change()
    rows = []
    prev_eq = base_equity   # 首期按基准比例起算

    for d in dates:
        # 用上一期比例算组合在 d 之前的已实现波动
        hist = rets.loc[rets.index < d].tail(vol_window)
        port_ret = prev_eq * hist["dividend"] + (1 - prev_eq) * hist["bond"]
        # 窗口内一旦有缺失（某条腿那时还没数据），std() 会跳过缺失用剩下的算，
        # 长度检查却照样通过 —— 结果是一个样本不足、被低估的波动率。
        # 这种情况直接判为不可用，退回固定比例，不要拿半截数据去开闸门。
        valid = int(port_ret.notna().sum())
        realized = (float(port_ret.std() * np.sqrt(252))
                    if valid >= 30 and valid == len(port_ret) else np.nan)

        if vol_target is None or np.isnan(realized) or realized <= 0:
            eq = base_equity
            scale = 1.0
            note = "固定比例"
        else:
            # 双向：实际波动低于目标时按缺口加仓（上限 MAX_EQUITY），
            # 高于目标时按比例收缩（下限 MIN_EQUITY）。
            # 单向版本（min(1,·)）会把风险预算永远用不满：低波动时仓位停在 40%，
            # 而低波动恰恰是权益资产最便宜、最该多拿的时候。
            scale = vol_target / realized
            eq = float(np.clip(base_equity * scale, MIN_EQUITY, MAX_EQUITY))
            if scale < 1:
                note = "波动超预算→收缩"
            elif eq > base_equity + 1e-9:
                note = "波动低于预算→加仓"
            else:
                note = "波动在预算内"

        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "w_dividend": round(eq, 4),
            "w_broad": 0.0,
            "w_bond": round(1.0 - eq, 4),
            "w_cash": 0.0,
            "realized_vol": round(realized, 4) if not np.isnan(realized) else np.nan,
            "scale": round(scale, 4),
            "note": note,
        })
        prev_eq = eq

    df = pd.DataFrame(rows)
    if not write:
        return df, None
    os.makedirs("output", exist_ok=True)
    path = f"output/alloc_{version_key}_{out_tag or mode}.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    return df, path


def main():
    for ver in VERSIONS:
        for mode in MODES:
            df, path = build(ver, mode)
            avg = df["w_dividend"].mean()
            shrink = (df["scale"] < 1).mean()
            print(f"{VERSIONS[ver]['label']:<16}{MODES[mode]['label']:<24}"
                  f"平均红利 {avg:>5.0%}　收缩月份占比 {shrink:>4.0%}　→ {path}")

    print()
    print("说明：波动率目标化只降不升（上限锁在基准 40%），所以启用后平均仓位一定低于 40%。")
    print("      这不是缺陷，是「只在风险升高时收缩」这一设计的必然结果。")


if __name__ == "__main__":
    main()
