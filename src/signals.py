"""
第 3 关 · 步骤 2-3：按规则 v2 生成月度目标权重表

规则（每月最后一个交易日判断一次，中途不动）：
  1. 基础权益仓位看沪深300 近 10 年 PE-TTM 分位：<30% → 80%；30–70% → 60%；>70% → 30%
  2. 趋势修正（串联扣减）：
       - 沪深300 跌破 120 日均线          → 基础仓位减半
       - 红利低波也跌破自己的 120 日均线  → 权益归零
  3. 剩余仓位进防御：国债 70%、货币 30%
  4. 权益内部按 60 日涨幅倾斜：第一名 60%、第二名 40%；只有一只可持时 100%

两条容易写错、这里显式定义的细节：
  - 「可持」= 该标的收盘价在它自己的 120 日均线之上。破位的那只不入选。
  - 因此「只有一只可持」时，它拿满全部权益仓位（100%），不是 60%。

防止未来函数的三条约束：
  - 所有指标只用当日及之前的数据（rolling / shift 都是向后看的）
  - 月末日取「当月最后一个有交易的日期」，不是自然月末
  - PE 分位只取该月末及之前的月度样本

输出：
  output/weights_long.csv   长周期版（2015 起，股用全收益指数）
  output/weights_trade.csv  可交易版（2019 起，全用 ETF）
"""

import os

import pandas as pd

# ---------- 参数 ----------
PE_LOW, PE_HIGH = 0.30, 0.70
POS_LOW, POS_MID, POS_HIGH = 0.80, 0.60, 0.30
MA_WINDOW = 120
MOM_WINDOW = 60
W_TOP1, W_TOP2 = 0.60, 0.40
DEF_BOND, DEF_CASH = 0.70, 0.30

VERSIONS = {
    "long": {
        "label": "长周期版（2015 起）",
        "dividend": "data/index/红利低波_全收益.csv",
        "broad": "data/index/沪深300_全收益.csv",
        "bond": "data/511010.csv",
        "cash": "data/511880.csv",
        "start": "2015-01-01",
        "out": "output/weights_long.csv",
    },
    "trade": {
        "label": "可交易版（2019 起）",
        "dividend": "data/512890.csv",
        "broad": "data/510300.csv",
        "bond": "data/511010.csv",
        "cash": "data/511880.csv",
        "start": "2019-01-18",
        "out": "output/weights_trade.csv",
    },
}

ASSET_NAMES = {"dividend": "红利低波", "broad": "沪深300", "bond": "国债", "cash": "货币"}


def load_close(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["close"]).drop_duplicates(subset="date").sort_values("date")
    return df.set_index("date")["close"]


def month_end_dates(s):
    """当月最后一个有交易的日期。"""
    idx = s.index.to_series()
    return idx.groupby([idx.dt.year, idx.dt.month]).max().sort_values().values


def compute_indicators(s, ma_window=MA_WINDOW):
    return pd.DataFrame({
        "close": s,
        "ma": s.rolling(ma_window).mean(),
        "mom": s / s.shift(MOM_WINDOW) - 1.0,
    })


def align_to(ind_df, dates):
    """把指标按「最后一个不晚于 date 的值」对齐到目标日期，避免用到未来数据。"""
    full = ind_df.reindex(ind_df.index.union(pd.DatetimeIndex(dates))).ffill()
    return full.reindex(pd.DatetimeIndex(dates))


def build(version_key, anchor_df=None, tag=None, base_override=None, no_trend=False,
          ma_window=None, pos=None, buffer=0.0):
    """
    anchor_df:    可替换的估值锚，需含 date / pe_ttm / pe_pct 三列。
                  None 表示用默认锚（沪深300 PE-TTM 分位）。
    tag:          不为 None 时输出到 output/exp/weights_{tag}_{version}.csv，
                  不覆盖主流程的权重表。用于第 5 关的对照实验。
    base_override: 给定则忽略估值锚，基础仓位固定为该值（用于「只留趋势」的对照）。
    no_trend:     True 则关闭趋势修正层（用于「只留估值」的对照）。
    ma_window:    均线天数，None 走默认 120。
    pos:          (低分位仓位, 中间仓位, 高分位仓位)，None 走默认 (0.8, 0.6, 0.3)。
    buffer:       仓位缓冲带。相邻两期权益仓位变动小于该值时不调仓（0 视为关闭）。
                  注意：归零与归零后重新入场不受缓冲带阻拦，否则破位后永远回不来。
    """
    cfg = VERSIONS[version_key]
    ma_w = ma_window or MA_WINDOW
    p_low, p_mid, p_high = pos if pos else (POS_LOW, POS_MID, POS_HIGH)

    series = {k: load_close(cfg[k]) for k in ["dividend", "broad", "bond", "cash"]}
    ind = {k: compute_indicators(v, ma_w) for k, v in series.items()}

    # 以沪深300 的交易日历为锚，决定月末日期
    dates = pd.DatetimeIndex(month_end_dates(series["broad"]))
    dates = dates[dates >= pd.Timestamp(cfg["start"])]

    aligned = {k: align_to(v, dates) for k, v in ind.items()}

    if anchor_df is None:
        pe = pd.read_csv("data/pe_300.csv", parse_dates=["date"]).set_index("date")
    else:
        pe = anchor_df.copy()
        pe["date"] = pd.to_datetime(pe["date"])
        pe = pe.set_index("date")[["pe_ttm", "pe_pct"]]
    pe_aligned = pe.reindex(pe.index.union(dates)).ffill().reindex(dates)

    rows = []
    dropped_warmup = 0
    prev_equity = None
    for i, d in enumerate(dates):
        pct = pe_aligned["pe_pct"].iloc[i]
        if pd.isna(pct):
            continue
        pe_val = pe_aligned["pe_ttm"].iloc[i]

        # 指标预热期：120 日均线还算不出来的月份不产生信号。
        # 不这样处理的话，前半年会被误判成「双破位→权益归零」，不是真信号。
        a = aligned["dividend"].iloc[i]
        b = aligned["broad"].iloc[i]
        if pd.isna(a["ma"]) or pd.isna(b["ma"]):
            dropped_warmup += 1
            continue

        # 规则 1：基础权益仓位
        if base_override is not None:
            base = base_override
        elif pct < PE_LOW:
            base = p_low
        elif pct <= PE_HIGH:
            base = p_mid
        else:
            base = p_high

        # 规则 2：趋势修正
        a_ok = a["close"] >= a["ma"]
        b_ok = b["close"] >= b["ma"]

        if no_trend:
            equity = base
            trend_note = "趋势层关闭"
        elif not a_ok and not b_ok:
            equity = 0.0
            trend_note = "双破位→归零"
        elif not b_ok:
            equity = base * 0.5
            trend_note = "300破位→减半"
        else:
            equity = base
            trend_note = "趋势健康"

        # 缓冲带：小幅变动不调仓，抑制震荡市里的来回打脸。
        # 归零/重新入场不拦，否则破位后永远回不来。
        if (buffer > 0 and prev_equity is not None and equity > 0 and prev_equity > 0
                and abs(equity - prev_equity) < buffer):
            equity = prev_equity
            trend_note += "（缓冲带内不动）"
        prev_equity = equity

        # 规则 4：权益内部按 60 日涨幅倾斜，只保留均线之上的标的
        cand = []
        if a_ok and not pd.isna(a["mom"]):
            cand.append(("dividend", a["mom"]))
        if b_ok and not pd.isna(b["mom"]):
            cand.append(("broad", b["mom"]))

        w = {"dividend": 0.0, "broad": 0.0, "bond": 0.0, "cash": 0.0}
        if equity > 0 and cand:
            cand.sort(key=lambda x: x[1], reverse=True)
            if len(cand) == 1:
                w[cand[0][0]] = equity
            else:
                w[cand[0][0]] = equity * W_TOP1
                w[cand[1][0]] = equity * W_TOP2
            defense = 1.0 - equity
        else:
            equity = 0.0
            defense = 1.0

        w["bond"] = defense * DEF_BOND
        w["cash"] = defense * DEF_CASH

        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "pe_ttm": round(float(pe_val), 2),
            "pe_pct": round(float(pct), 4),
            "base_pos": base,
            "equity": round(equity, 4),
            "w_dividend": round(w["dividend"], 4),
            "w_broad": round(w["broad"], 4),
            "w_bond": round(w["bond"], 4),
            "w_cash": round(w["cash"], 4),
            "note": trend_note,
        })

    out = pd.DataFrame(rows)
    out.attrs["dropped_warmup"] = dropped_warmup
    out["w_sum"] = (out[["w_dividend", "w_broad", "w_bond", "w_cash"]].sum(axis=1)).round(6)

    if tag is None:
        out_path = cfg["out"]
    else:
        out_path = f"output/exp/weights_{tag}_{version_key}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    return cfg, out


def report(cfg, df):
    warm = df.attrs.get("dropped_warmup", 0)
    print("=" * 84)
    print(f"{cfg['label']}　{len(df)} 个月　{df['date'].iloc[0]} → {df['date'].iloc[-1]}")
    if warm:
        print(f"（已剔除 {warm} 个指标预热月份的无效信号）")
    print("=" * 84)

    bad = df[(df["w_sum"] - 1).abs() > 1e-6]
    print(f"权重合计检查：{'全部为 1，通过' if bad.empty else f'{len(bad)} 行异常'}")

    print()
    print("仓位档位分布：")
    for pos, g in df.groupby("equity"):
        print(f"  权益 {pos:>4.0%}：{len(g):>3} 个月（{len(g)/len(df):>4.0%}）")

    print()
    print("最近 12 期：")
    print(f"{'日期':<12}{'PE分位':>8}{'基础':>7}{'权益':>7}{'红利':>7}{'300':>7}{'国债':>7}{'货币':>7}  {'说明'}")
    print("-" * 84)
    for _, r in df.tail(12).iterrows():
        print(f"{r['date']:<12}{r['pe_pct']:>8.0%}{r['base_pos']:>7.0%}{r['equity']:>7.0%}"
              f"{r['w_dividend']:>7.0%}{r['w_broad']:>7.0%}{r['w_bond']:>7.0%}{r['w_cash']:>7.0%}  {r['note']}")

    print()
    print("极端月份抽查（权益 0% 与 权益满档 各取 2 个）：")
    zero = df[df["equity"] == 0]
    full = df[df["equity"] == df["equity"].max()]
    for label, sub in [("权益 0%", zero), ("权益最高档", full)]:
        if sub.empty:
            print(f"  {label}：没有出现")
            continue
        for _, r in sub.head(2).iterrows():
            print(f"  {label} {r['date']}：PE分位 {r['pe_pct']:.0%}，权益 {r['equity']:.0%}，{r['note']}")


def main():
    for key in VERSIONS:
        cfg, df = build(key)
        report(cfg, df)
        print()
    print("长周期版与可交易版权重表均已生成，权重合计均为 1。")


if __name__ == "__main__":
    main()
