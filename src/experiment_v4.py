"""
自检（v4）· 实验：把「灵活化」拆成可独立检验的开关，逐个量它的账

背景（自检发现的问题）：
  v3 写死了 MAX_EQUITY = BASE_EQUITY = 0.40，scale 又被 min(1.0, ...) 压住，
  所以动态成分只有一个方向——减仓。实测 143 个月里 112 个月（78%）就停在 40%，
  剩下 22% 里大半只是 40%→38% 的微调。
  用户说的「太死板」不是主观感受，是仓位分布本身长这样。

本次检验的六组开关（每组都跑长周期版与可交易版）：

  G1 打开上限       固定 40% 基准，允许低波动时加到 50%/60%/70%
  G2 暴露匹配对照   固定 45%/50%/55%/60%，用来判断 G1 的超额是不是只来自「仓位更大」
  G3 分档式定仓     按已实现波动落在哪一档直接给仓位，避免线性缩放常年顶格
  G4 回撤线         组合从高点回撤 3%/5%/8% → 压到 20%，回撤修复一半再恢复
  G5 止盈线         组合从一年低点上涨 10%/20% → 减到 30%
  G6 再平衡带       偏离不足 3%/5% 不动，省摩擦

判读标准（跑之前先写死，避免事后挑参数）：
  采用要求同时满足——
    a) 全区间 Calmar 比现状高 ≥ 0.10
    b) 样本外 Calmar 不下降
    c) 最大回撤不恶化超过 1 个百分点
  若某组平均仓位明显高于 40%，必须和 G2 里同仓位的固定版比：
  只有比固定版更好，才说明「动态」本身有价值，否则只是变相加仓。

输出：
  output/exp/v4_switches.csv
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import allocation as al  # noqa: E402
import backtest as bt  # noqa: E402
import mechanism as mc  # noqa: E402

EXP_DIR = "output/exp"


def evaluate(w, cfg, op, cl):
    nav, fee = bt.simulate(w, op, cl, cfg["exec_price"])
    ins_end = pd.Timestamp(cfg["insample_end"])
    mi = bt.metrics(nav[nav.index <= ins_end])
    mo = bt.metrics(nav[nav.index > ins_end])
    mf = bt.metrics(nav)
    return {
        "平均权益": float(w["w_dividend"].mean()),
        "权益最低": float(w["w_dividend"].min()),
        "权益最高": float(w["w_dividend"].max()),
        "全年化": mf["年化收益率"], "全回撤": mf["最大回撤"],
        "全夏普": mf["夏普比率"], "全Calmar": mf["Calmar比率"],
        "内年化": mi["年化收益率"], "内回撤": mi["最大回撤"], "内Calmar": mi["Calmar比率"],
        "外年化": mo["年化收益率"], "外回撤": mo["最大回撤"], "外Calmar": mo["Calmar比率"],
        "手续费": fee, "累计成交额": fee / bt.FEE_RATE,
    }, nav


# 分档模式的档位表：(波动上限, 权益比例)
BANDS_MID = [(0.05, 0.50), (0.07, 0.40), (0.09, 0.30), (0.12, 0.25), (9.9, 0.20)]
BANDS_HI = [(0.05, 0.60), (0.07, 0.45), (0.09, 0.35), (0.12, 0.28), (9.9, 0.20)]
BANDS_SOFT = [(0.05, 0.45), (0.07, 0.40), (0.09, 0.36), (0.12, 0.30), (9.9, 0.24)]


def variants():
    """返回 [(组, 代号, 说明, build 参数)]"""
    v = []
    # G1 打开上限：唯一改动是 mx
    for mx in [0.50, 0.60, 0.70]:
        v.append(("G1 打开上限", f"UP{int(mx*100)}",
                  f"基准 40% + 波动目标 8% + 上限 {mx:.0%}",
                  dict(base=0.40, mn=0.20, mx=mx, vol_target=0.08)))
    # G2 暴露匹配对照：纯固定比例，没有动态
    for w0 in [0.40, 0.45, 0.50, 0.55, 0.60]:
        v.append(("G2 固定对照", f"FIX{int(w0*100)}",
                  f"固定 {w0:.0%}/{(1-w0):.0%}，无动态",
                  dict(base=w0, mn=w0, mx=w0)))
    # G3 分档
    for tag, bands, desc in [("MID", BANDS_MID, "<5%→50, 5-7→40, 7-9→30, 9-12→25, >12→20"),
                             ("HI", BANDS_HI, "<5%→60, 5-7→45, 7-9→35, 9-12→28, >12→20"),
                             ("SOFT", BANDS_SOFT, "<5%→45, 5-7→40, 7-9→36, 9-12→30, >12→24")]:
        v.append(("G3 分档定仓", f"BAND{tag}", desc,
                  dict(base=0.40, mn=0.20, mx=0.60, bands=bands)))
    # G4 回撤线
    for dd in [0.03, 0.05, 0.08]:
        v.append(("G4 回撤线", f"DD{int(dd*100)}",
                  f"组合回撤 {dd:.0%} → 权益压到 20%，修复一半恢复",
                  dict(base=0.40, mn=0.20, mx=0.40, vol_target=0.08, dd_stop=dd)))
    # G4b 回撤线叠加「打开上限」
    for dd in [0.05, 0.08]:
        v.append(("G4 回撤线", f"UP60+DD{int(dd*100)}",
                  f"上限 60% + 回撤 {dd:.0%} 压到 20%",
                  dict(base=0.40, mn=0.20, mx=0.60, vol_target=0.08, dd_stop=dd)))
    # G5 止盈
    for tp in [0.10, 0.20]:
        v.append(("G5 止盈线", f"TP{int(tp*100)}",
                  f"组合自一年低点涨 {tp:.0%} → 减到 30%",
                  dict(base=0.40, mn=0.20, mx=0.40, tp_trigger=tp, tp_to=0.30)))
    # G6 再平衡带
    for bd in [0.03, 0.05]:
        v.append(("G6 再平衡带", f"RB{int(bd*100)}",
                  f"波动目标 8% + 偏离不足 {bd:.0%} 不动",
                  dict(base=0.40, mn=0.20, mx=0.40, vol_target=0.08, band=bd)))
    return v


def main():
    os.makedirs(EXP_DIR, exist_ok=True)
    rows = []

    for ver in ["long", "trade"]:
        cfg = bt.VERSIONS[ver]
        op, cl = bt.load_prices(cfg["prices"])

        # 基线：直接复用 v3 的 build，确保对照是「现在这份策略」而不是我的复刻
        w_ref, _ = al.build(ver, "vol08")
        m, _ = evaluate(w_ref, cfg, op, cl)
        rows.append({"版本": ver, "组": "G0 现状", "代号": "V3",
                     "说明": "40/60 基准 + 波动目标 8% + 上限锁 40%", **m})

        for grp, tag, desc, kw in variants():
            w = mc.build(ver, **kw)
            m, _ = evaluate(w, cfg, op, cl)
            rows.append({"版本": ver, "组": grp, "代号": tag, "说明": desc, **m})

    df = pd.DataFrame(rows)
    df.to_csv(f"{EXP_DIR}/v4_switches.csv", index=False, encoding="utf-8")

    cols = ["代号", "平均权益", "全年化", "全回撤", "全Calmar", "外年化", "外Calmar", "手续费"]
    for ver in ["long", "trade"]:
        print()
        print("=" * 108)
        print(f"{bt.VERSIONS[ver]['label']}：六组开关逐项检验")
        print("=" * 108)
        sub = df[df["版本"] == ver]
        print(f"{'代号':<14}{'平均权益':>9}{'全年化':>8}{'全回撤':>8}{'全Calmar':>10}"
              f"{'外年化':>8}{'外Calmar':>10}{'手续费':>8}")
        print("-" * 108)
        for grp in sub["组"].unique():
            print(f"  ── {grp}")
            for _, r in sub[sub["组"] == grp].iterrows():
                print(f"{r['代号']:<14}{r['平均权益']:>9.0%}{r['全年化']:>8.1%}{r['全回撤']:>8.1%}"
                      f"{r['全Calmar']:>10.2f}{r['外年化']:>8.1%}{r['外Calmar']:>10.2f}{r['手续费']:>8.2%}")


if __name__ == "__main__":
    main()
