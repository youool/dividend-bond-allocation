"""
自检审计：主线策略的结果有多少是「构造出来的」

写这个脚本的原因
  前几轮所有严格检验（暴露匹配、100 次随机重排、24 组参数网格）都打在**被否决的方案**上，
  主线（40/60 + 波动率闸门）只接受了软检验：相邻参数不敏感、样本内外一致、滚动三年为正。
  这些检验有个共同点——它们无法失败。所以必须换一组**能证伪主线**的检验。

五项审计，每项都能把主线判死：

  A 回撤预算是不是循环论证
     如果「最大回撤 ≈ 常数 × 权益比例」，那么「先定 7% 预算、再反推 40%」
     等于「40% = 7% ÷ 常数」，参数没有信息量，纪律 40% 的兑现是恒真句。
     检验：扫 20%–60%，看回撤对权益比例的线性程度（R² 与一阶差分）。

  B 波动率闸门有没有信息量
     闸门的作用是「在坏月份收缩」。如果随便挑同样多的月份收缩也能得到同样结果，
     那闸门就只是「平均仓位略低」的另一种写法。
     检验：固定权重序列的多重集，随机重排时间对应关系 N 次，看真实 Calmar 落在第几分位。
     （这是当初判死估值信号用的同一把刀，只是这次对准主线。）

  C 收益来自「资产」还是来自「结构」
     40/60 + 月度再平衡 + 闸门 是一套结构。把权益腿换成沪深300 / 创业板，
     结构一字不改。如果换掉之后 Calmar 大幅塌陷，说明结果完全押在「选了红利低波」这一个
     事前不可知的决策上，结构本身不产生超额；如果 Calmar 依然高，说明所谓优势只是
     「债券占比高」的算术结果。两种结局都对主线的叙事不利。

  D 稳健是谁贡献的
     把收益拆成权益腿与债券腿的加总贡献。如果「低回撤」主要由 60% 国债提供，
     那么报告里「风险控制有效」的说法就站不住——那是资产配置的常识，不是策略的功劳。

  E 参数是用哪段数据挑出来的
     用样本内数据（long ≤ 2020-12、trade ≤ 2022-12）单独重跑「7% 预算反推比例」，
     看它会不会挑出同样的 40%。若挑出的比例与 40% 不同，则 40% 含后视信息。

输出：output/exp/audit_overfit.csv（分项指标）与若干 charts
不覆盖主线任何产物：本脚本只读 output/final_*，写 output/exp/audit_*。
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt  # noqa: E402

EXP_DIR = "output/exp"
N_SHUFFLE = 300
SEED = 20261125

BOND = "data/511010.csv"
CASH = "data/511880.csv"
EQUITY_POOL = {
    "红利低波": {"long": "data/index/红利低波_全收益.csv", "trade": "data/512890.csv"},
    "沪深300": {"long": "data/index/沪深300_全收益.csv", "trade": "data/510300.csv"},
    "创业板": {"long": "data/159915.csv", "trade": "data/159915.csv"},
}
START = {"long": "2015-01-01", "trade": "2019-01-18"}
EXEC = {"long": "close", "trade": "open"}


def month_ends(index, start):
    s = index.to_series()
    d = s.groupby([s.dt.year, s.dt.month]).max().sort_values().values
    d = pd.DatetimeIndex(d)
    return d[d >= pd.Timestamp(start)]


def make_weights(equity_path, dates, base_equity, vol_target=None,
                 vol_window=60, floor=0.20):
    """与 allocation.build 同构，但资产可替换。用于 C 项。"""
    cl = pd.read_csv(equity_path, parse_dates=["date"]).drop_duplicates("date").set_index("date")["close"]
    bond = pd.read_csv(BOND, parse_dates=["date"]).drop_duplicates("date").set_index("date")["close"]
    df = pd.concat([cl.rename("eq"), bond.rename("bd")], axis=1).sort_index().ffill()
    rets = df.pct_change()

    rows, prev = [], base_equity
    for d in dates:
        if d not in df.index:
            continue
        hist = rets.loc[rets.index < d].tail(vol_window)
        pr = prev * hist["eq"] + (1 - prev) * hist["bd"]
        valid = int(pr.notna().sum())
        vol = float(pr.std() * np.sqrt(252)) if valid >= 30 and valid == len(pr) else np.nan
        if vol_target is None or not np.isfinite(vol) or vol <= 0:
            eq, scale = base_equity, 1.0
        else:
            scale = min(1.0, vol_target / vol)
            eq = float(np.clip(base_equity * scale, floor, base_equity))
        rows.append({"date": d, "w_dividend": eq, "w_broad": 0.0,
                     "w_bond": 1.0 - eq, "w_cash": 0.0,
                     "realized_vol": vol, "scale": scale})
        prev = eq
    return pd.DataFrame(rows)


def run_nav(weights, equity_path, version):
    """用 equities 当权益腿、511010 当债券腿跑一遍。w_broad 表示权益（可替换）。"""
    op, cl = bt.load_prices({"dividend": equity_path, "broad": equity_path,
                             "bond": BOND, "cash": CASH})
    w = weights.copy()
    # 统一走 broad 列，避免两列同一资产被重复计算
    w["w_broad"] = w["w_dividend"] + w.get("w_broad", 0.0)
    w["w_dividend"] = 0.0
    nav, _ = bt.simulate(w, op, cl, EXEC[version], start=START[version])
    return nav


def summarize(nav):
    m = bt.metrics(nav)
    return {k: m[k] for k in ("年化收益率", "最大回撤", "夏普比率", "Calmar比率", "年化波动")}


def audit_a(rows):
    """A：回撤预算的循环论证。固定比例口径，扫 20%–60%。"""
    print("\n=== A 回撤对权益比例的线性程度（固定比例、无闸门）===")
    for ver in ("long", "trade"):
        eqp = EQUITY_POOL["红利低波"][ver]
        cl = pd.read_csv(eqp, parse_dates=["date"]).drop_duplicates("date").set_index("date")["close"]
        dates = month_ends(cl.index, START[ver])
        pts = []
        for beq in np.arange(0.20, 0.61, 0.05):
            w = make_weights(eqp, dates, beq, vol_target=None)
            m = summarize(run_nav(w, eqp, ver))
            pts.append((beq, m["年化收益率"], m["最大回撤"], m["Calmar比率"]))
            rows.append({"审计": "A_回撤线性", "版本": ver, "权益比例": round(beq, 2),
                         "年化": round(m["年化收益率"], 4), "最大回撤": round(m["最大回撤"], 4),
                         "Calmar": round(m["Calmar比率"], 4), "备注": "固定比例"})
        arr = np.array(pts)
        w_, dd = arr[:, 0], -arr[:, 2]
        k, b = np.polyfit(w_, dd, 1)
        r2 = 1 - ((dd - (k * w_ + b)) ** 2).sum() / ((dd - dd.mean()) ** 2).sum()
        print(f"  {ver}: 回撤 ≈ {k:.3f}×w + {b:.4f}  R²={r2:.3f}")
        for w0, a, d, c in pts:
            print(f"    w={w0:.0%}  年化 {a:>6.2%}  回撤 {d:>7.2%}  Calmar {c:>5.2f}")
        # 「7% 预算反推比例」的显式解
        w_implied = (0.07 - b) / k
        print(f"    → 按「回撤≤7%」反解出的权益比例 = {w_implied:.1%}（与 40% 比较）")


def audit_b(rows):
    """B：波动率闸门的安慰剂检验。重排同一组权重的时间顺序。"""
    print("\n=== B 闸门安慰剂：重排权重序列，问『随便收缩能不能也行』===")
    rng = np.random.default_rng(SEED)
    for ver in ("long", "trade"):
        eqp = EQUITY_POOL["红利低波"][ver]
        alloc = pd.read_csv(f"output/alloc_{ver}_vol08.csv", parse_dates=["date"])
        real = summarize(run_nav(alloc[["date", "w_dividend", "w_broad", "w_bond", "w_cash"]].assign(w_broad=0.0), eqp, ver))
        pool = alloc["w_dividend"].values.copy()
        sims = []
        for _ in range(N_SHUFFLE):
            sh = pool.copy()
            rng.shuffle(sh)
            w = pd.DataFrame({"date": alloc["date"], "w_dividend": sh,
                              "w_broad": 0.0, "w_bond": 1 - sh, "w_cash": 0.0})
            sims.append(summarize(run_nav(w, eqp, ver)))
        cal = np.array([s["Calmar比率"] for s in sims])
        ann = np.array([s["年化收益率"] for s in sims])
        dd = np.array([s["最大回撤"] for s in sims])
        p_cal = float((cal < real["Calmar比率"]).mean())
        p_ann = float((ann < real["年化收益率"]).mean())
        print(f"  {ver}: 真实 Calmar {real['Calmar比率']:.2f}｜重排 中位 {np.median(cal):.2f} "
              f"95分位 {np.quantile(cal, .95):.2f}｜真实落在第 {p_cal:.0%} 分位")
        print(f"        真实年化 {real['年化收益率']:.2%}｜重排中位 {np.median(ann):.2%}，分位 {p_ann:.0%}")
        print(f"        真实回撤 {real['最大回撤']:.2%}｜重排中位 {np.median(dd):.2%}")
        rows.append({"审计": "B_闸门安慰剂", "版本": ver, "指标": "Calmar",
                     "真实值": round(real["Calmar比率"], 4), "重排中位": round(float(np.median(cal)), 4),
                     "重排95分位": round(float(np.quantile(cal, .95)), 4),
                     "分位": round(p_cal, 3),
                     "备注": f"N={N_SHUFFLE}"})
        rows.append({"审计": "B_闸门安慰剂", "版本": ver, "指标": "年化",
                     "真实值": round(real["年化收益率"], 4), "重排中位": round(float(np.median(ann)), 4),
                     "重排95分位": round(float(np.quantile(ann, .95)), 4),
                     "分位": round(p_ann, 3), "备注": f"N={N_SHUFFLE}"})
        np.save(f"{EXP_DIR}/audit_b_{ver}_calmar.npy", cal)


def audit_c(rows):
    """C：同结构、不同权益资产。"""
    print("\n=== C 同结构（40/60+闸门）换权益腿 ===")
    for ver in ("long", "trade"):
        base = EQUITY_POOL["红利低波"][ver]
        cl0 = pd.read_csv(base, parse_dates=["date"]).drop_duplicates("date").set_index("date")["close"]
        dates = month_ends(cl0.index, START[ver])
        print(f"  -- {ver} --")
        for name, paths in EQUITY_POOL.items():
            eqp = paths[ver]
            for tag, vt in (("固定", None), ("闸门8%", 0.08)):
                try:
                    w = make_weights(eqp, dates, 0.40, vol_target=vt)
                    if len(w) < 12:
                        print(f"    {name:<6}{tag:<7} 数据不足（{len(w)} 期）")
                        continue
                    nav = run_nav(w, eqp, ver)
                    m = summarize(nav)
                    print(f"    {name:<6}{tag:<7} 年化 {m['年化收益率']:>6.2%}  回撤 {m['最大回撤']:>7.2%}  "
                          f"夏普 {m['夏普比率']:>5.2f}  Calmar {m['Calmar比率']:>5.2f}")
                    rows.append({"审计": "C_换资产", "版本": ver, "权益": name, "模式": tag,
                                 "年化": round(m["年化收益率"], 4), "最大回撤": round(m["最大回撤"], 4),
                                 "夏普": round(m["夏普比率"], 4), "Calmar": round(m["Calmar比率"], 4)})
                except Exception as e:  # 数据不足等
                    print(f"    {name:<6}{tag:<7} 跳过：{type(e).__name__} {e}")


def audit_d(rows):
    """D：收益归因——权益腿 vs 债券腿。"""
    print("\n=== D 收益归因（月度权重 × 月度收益，算术分解）===")
    for ver in ("long", "trade"):
        eqp = EQUITY_POOL["红利低波"][ver]
        alloc = pd.read_csv(f"output/alloc_{ver}_vol08.csv", parse_dates=["date"]).set_index("date")
        e = pd.read_csv(eqp, parse_dates=["date"]).drop_duplicates("date").set_index("date")["close"]
        b = pd.read_csv(BOND, parse_dates=["date"]).drop_duplicates("date").set_index("date")["close"]
        px = pd.concat([e.rename("eq"), b.rename("bd")], axis=1).ffill()
        mth = px.resample("ME").last()
        r = mth.pct_change().dropna()
        w = alloc.reindex(r.index, method="ffill")
        w = w[w.index >= r.index[0]]
        r = r.loc[w.index]
        c_eq = float((w["w_dividend"] * r["eq"]).sum())
        c_bd = float((w["w_bond"] * r["bd"]).sum())
        tot = c_eq + c_bd
        print(f"  {ver}: 权益腿贡献 {c_eq:>6.1%}（占 {c_eq/tot:>5.0%}）｜债券腿 {c_bd:>6.1%}"
              f"（占 {c_bd/tot:>5.0%}）｜合计 {tot:.1%}，期数 {len(r)}")
        rows.append({"审计": "D_归因", "版本": ver, "权益腿贡献": round(c_eq, 4),
                     "债券腿贡献": round(c_bd, 4), "债券腿占比": round(c_bd / tot, 3),
                     "备注": "月度算术分解"})


def audit_e(rows):
    """E：只用样本内数据反推比例，看是不是同一个数。"""
    print("\n=== E 用样本内数据重做「7% 预算→反推比例」 ===")
    in_end = {k: bt.VERSIONS[k]["insample_end"] for k in ("long", "trade")}
    for ver in ("long", "trade"):
        eqp = EQUITY_POOL["红利低波"][ver]
        cl = pd.read_csv(eqp, parse_dates=["date"]).drop_duplicates("date").set_index("date")["close"]
        dates = month_ends(cl.index, START[ver])
        cut = pd.Timestamp(in_end[ver])
        best = None
        for beq in np.arange(0.20, 0.66, 0.05):
            w = make_weights(eqp, dates, beq, vol_target=None)
            nav = run_nav(w, eqp, ver)
            m_in = bt.metrics(nav[nav.index <= cut])
            dd_in = -m_in["最大回撤"]
            if dd_in <= 0.07:
                best = (beq, dd_in, m_in["年化收益率"])
        for beq, dd_in, a_in in ([best] if best else []):
            print(f"  {ver}: 样本内（截至 {in_end[ver]}）满足回撤≤7% 的最大比例 = {beq:.0%}"
                  f"（样本内回撤 {dd_in:.2%}／年化 {a_in:.2%}）")
            w = make_weights(eqp, dates, beq, vol_target=None)
            nav = run_nav(w, eqp, ver)
            m_all = bt.metrics(nav)
            m_out = bt.metrics(nav[nav.index > cut])
            print(f"        该比例用到全期：回撤 {m_all['最大回撤']:.2%}｜Calmar {m_all['Calmar比率']:.2f}"
                  f"；样本外年化 {m_out['年化收益率']:.2%} 回撤 {m_out['最大回撤']:.2%}")
            rows.append({"审计": "E_样本内反推", "版本": ver, "样本内选出的比例": round(beq, 2),
                         "样本内回撤": round(dd_in, 4), "全期回撤": round(m_all["最大回撤"], 4),
                         "全期Calmar": round(m_all["Calmar比率"], 4),
                         "样本外年化": round(m_out["年化收益率"], 4),
                         "备注": f"对照：主线用 40%"})
        if best is None:
            print(f"  {ver}: 样本内没有比例能满足回撤≤7%")


def main():
    os.makedirs(EXP_DIR, exist_ok=True)
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else {"A", "B", "C", "D", "E"}
    rows = []
    if "A" in only:
        audit_a(rows)
    if "B" in only:
        audit_b(rows)
    if "C" in only:
        audit_c(rows)
    if "D" in only:
        audit_d(rows)
    if "E" in only:
        audit_e(rows)
    df = pd.DataFrame(rows)
    p = f"{EXP_DIR}/audit_overfit_{''.join(sorted(only))}.csv"
    df.to_csv(p, index=False, encoding="utf-8")
    print(f"\n→ {p}（{len(df)} 行）")


if __name__ == "__main__":
    main()
