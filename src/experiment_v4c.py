"""
自检（v4c）· 两件事：参数曲面 + 「机制还是标的」的压力测试

问题一：UP50 的超额是不是靠红利这十年涨得好？
  2015–2026 红利低波年化 9.7%，是它历史上最好的一段。在最好的标的上加仓，
  事后看一定对——但这不能证明机制有效，只能证明「过去该多买」。
  做法：把红利低波的日收益整体扣掉 d（0/1/2/3 个百分点/年），重跑同一批权重。
  如果 UP50 相对固定 45% 的优势在 d=3 时消失，说明它只是杠杆化了红利的高收益；
  如果优势还在，说明「低波动加仓、高波动减仓」本身在创造价值。

问题二：UP50 是不是刚好挑中了 8% 和 50% 这两个数？
  做法：target ∈ {7,8,9,10}% × 上限 ∈ {45,50,55,60}% 的二维曲面。
  判据不是「最优点在哪」，而是「最优点周围是不是一片平台」。
  孤峰 = 过拟合；平台 = 机制真的有效。

输出：
  output/exp/v4_surface.csv
  output/exp/v4_stress.csv
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
INS = {"long": "2020-12-31", "trade": "2022-12-31"}
BANDS_MID = [(0.05, 0.50), (0.07, 0.40), (0.09, 0.30), (0.12, 0.25), (9.9, 0.20)]


def shifted(cfg, drag):
    """把红利价格按 drag（年化）整体衰减，用来模拟「红利没那么好」的世界。"""
    if drag == 0:
        return bt.load_prices(cfg["prices"])
    op, cl = bt.load_prices(cfg["prices"])
    n = len(cl)
    adj = pd.Series(np.exp(-drag / 252.0 * np.arange(n)), index=cl.index)
    cl2, op2 = cl.copy(), op.copy()
    cl2["dividend"] = cl["dividend"] * adj
    op2["dividend"] = op["dividend"] * adj
    return op2, cl2


def summarize(nav, ver):
    m = bt.metrics(nav)
    mi = bt.metrics(nav[nav.index <= pd.Timestamp(INS[ver])])
    mo = bt.metrics(nav[nav.index > pd.Timestamp(INS[ver])])
    return (m["年化收益率"], m["最大回撤"], m["Calmar比率"],
            mi["Calmar比率"], mo["Calmar比率"])


def main():
    os.makedirs(EXP_DIR, exist_ok=True)
    surf, stress = [], []

    for ver in ["long", "trade"]:
        cfg = bt.VERSIONS[ver]

        # ---------- 1. 二维曲面 ----------
        for tv in [0.07, 0.08, 0.09, 0.10]:
            for mx in [0.45, 0.50, 0.55, 0.60]:
                op, cl = bt.load_prices(cfg["prices"])
                w = mc.build(ver, base=0.40, mn=0.20, mx=mx, vol_target=tv)
                nav, fee = bt.simulate(w, op, cl, cfg["exec_price"])
                a, dd, cal, ci, co = summarize(nav, ver)
                surf.append({"版本": ver, "目标波动": tv, "上限": mx, "平均权益": float(w["w_dividend"].mean()),
                             "年化": a, "回撤": dd, "Calmar": cal, "内Calmar": ci, "外Calmar": co,
                             "手续费": fee})

        # ---------- 2. 压力测试 ----------
        # 权重只算一次，四套方案各自在不同 drag 下重跑，保证只有「标的收益」在变
        plans = {
            "V3 现状": al.build(ver, "vol08")[0],
            "UP50 打开上限": mc.build(ver, base=0.40, mn=0.20, mx=0.50, vol_target=0.08),
            "BANDMID 分档": mc.build(ver, base=0.40, mn=0.20, mx=0.60, bands=BANDS_MID),
            "FIX40 固定40": mc.build(ver, base=0.40, mn=0.40, mx=0.40),
            "FIX45 固定45": mc.build(ver, base=0.45, mn=0.45, mx=0.45),
        }
        for drag in [0.00, 0.01, 0.02, 0.03]:
            op, cl = shifted(cfg, drag)
            for name, w in plans.items():
                nav, _ = bt.simulate(w, op, cl, cfg["exec_price"])
                a, dd, cal, ci, co = summarize(nav, ver)
                stress.append({"版本": ver, "红利收益扣减": drag, "方案": name,
                               "年化": a, "回撤": dd, "Calmar": cal, "内Calmar": ci, "外Calmar": co})

    sdf = pd.DataFrame(surf)
    sdf.to_csv(f"{EXP_DIR}/v4_surface.csv", index=False, encoding="utf-8")
    stdf = pd.DataFrame(stress)
    stdf.to_csv(f"{EXP_DIR}/v4_stress.csv", index=False, encoding="utf-8")

    for ver in ["long", "trade"]:
        print()
        print("=" * 96)
        print(f"{bt.VERSIONS[ver]['label']}：全区间 Calmar 曲面（行=目标波动，列=权益上限）")
        print("=" * 96)
        p = sdf[sdf["版本"] == ver].pivot(index="目标波动", columns="上限", values="Calmar")
        print((p * 100).round(0).astype(int).to_string())
        print("\n样本外 Calmar 曲面：")
        p2 = sdf[sdf["版本"] == ver].pivot(index="目标波动", columns="上限", values="外Calmar")
        print((p2 * 100).round(0).astype(int).to_string())

    print()
    print("=" * 96)
    print("压力测试：红利年化被扣减后，各方案还剩多少")
    print("=" * 96)
    for ver in ["long", "trade"]:
        print(f"\n{bt.VERSIONS[ver]['label']}")
        print(f"{'扣减':<8}{'方案':<16}{'年化':>8}{'回撤':>8}{'Calmar':>8}{'外Calmar':>10}")
        print("-" * 96)
        for drag in [0.00, 0.01, 0.02, 0.03]:
            sub = stdf[(stdf["版本"] == ver) & (np.isclose(stdf["红利收益扣减"], drag))]
            for _, r in sub.iterrows():
                print(f"{drag:<8.0%}{r['方案']:<16}{r['年化']:>8.1%}{r['回撤']:>8.1%}"
                      f"{r['Calmar']:>8.2f}{r['外Calmar']:>10.2f}")
            print("-" * 96)


if __name__ == "__main__":
    main()
