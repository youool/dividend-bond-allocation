"""对「双向波动率目标化（上限 50%）」的三项检验。

任何要进主线的改动，都必须先过这三关，否则就是把「挑参数」当成「发现规律」：

  1. 样本内外分段：样本外只能检验、不能挑选。
  2. 安慰剂：打乱波动率与月份的时间对应，保持仓位分布完全不变。
     如果真实信号排不进随机分布的前 5%，说明高 Calmar 来自「仓位结构」而不是「时机」。
  3. 成本敏感：换手从 1.1pct/月升到 6.3pct/月，手续费与冲击成本必须重新算。
     分别按万 2 / 万 5 / 千 1 重跑，看优势是否消失。

输出：output/exp/dynamic_checks.csv、dynamic_placebo.csv
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import allocation as al
import backtest as bt
from experiment_dynamic import build_dyn, run, BASE, FLOOR, VOL_TARGET

CAP_NEW = 0.50
CAP_OLD = 0.40
NP_ITER = 100


def split_metrics(nav: pd.Series, split: str):
    s = pd.Timestamp(split)
    a = nav[nav.index <= s]
    b = nav[nav.index > s]
    return bt.metrics(a), bt.metrics(b)


def main() -> None:
    os.makedirs("output/exp", exist_ok=True)
    rows = []

    for ver, label in [("long", "长周期 2015 起"), ("trade", "可交易 2019 起")]:
        cfg = al.VERSIONS[ver]
        op, cl = bt.load_prices(cfg["prices"])
        split = cfg["insample_end"]

        w_old = build_dyn(ver, op, cl, cap=CAP_OLD, two_way=False)
        w_new = build_dyn(ver, op, cl, cap=CAP_NEW, two_way=True)

        def nav_of(w):
            ww = w.copy(); ww["date"] = pd.to_datetime(ww["date"])
            nav, _ = bt.simulate(ww, op, cl, exec_price=cfg["exec_price"], start=cfg["start"])
            return nav

        nav_o, nav_n = nav_of(w_old), nav_of(w_new)

        print(f"\n{'='*86}\n{label}　样本内外分段（分界 {split}）\n{'='*86}")
        print(f"{'区间':<10}{'方案':<18}{'年化':>9}{'回撤':>9}{'Calmar':>9}{'夏普':>8}")

        io_o, io_n = split_metrics(nav_o, split)
        oo_o, oo_n = split_metrics(nav_n, split)
        for tag, mo, mn in [("全区间", bt.metrics(nav_o), bt.metrics(nav_n)),
                            ("样本内", io_o, oo_o), ("样本外", io_n, oo_n)]:
            for nm, m in [("单向 40（现状）", mo), ("双向 50（候选）", mn)]:
                print(f"{tag:<10}{nm:<18}{m['年化收益率']*100:>8.2f}%{m['最大回撤']*100:>8.2f}%"
                      f"{m['Calmar比率']:>9.3f}{m['夏普比率']:>8.2f}")
                rows.append({"版本": label, "区间": tag, "方案": nm, **m})

        # ---- 安慰剂：打乱波动率的时间对应 ----
        vols = w_new["realized_vol"].values.astype(float)
        valid = ~np.isnan(vols)
        real_cal = bt.metrics(nav_n)["Calmar比率"]
        rng = np.random.default_rng(20261126)
        sims = []
        for _ in range(NP_ITER):
            perm = vols.copy()
            perm[valid] = rng.permutation(vols[valid])
            w2 = w_new.copy()
            eq = np.where(np.isnan(perm), BASE,
                          np.clip(BASE * VOL_TARGET / np.where(np.isnan(perm), 1.0, perm),
                                  FLOOR, CAP_NEW))
            w2["w_dividend"] = np.round(eq, 4)
            w2["w_bond"] = np.round(1 - eq, 4)
            ww = w2.copy(); ww["date"] = pd.to_datetime(ww["date"])
            nav2, _ = bt.simulate(ww, op, cl, exec_price=cfg["exec_price"], start=cfg["start"])
            sims.append(bt.metrics(nav2)["Calmar比率"])
        sims = np.array(sims)
        pct = float((sims < real_cal).mean() * 100)
        print(f"\n　安慰剂（{NP_ITER} 次随机重排波动率时序）：真实 Calmar {real_cal:.3f}，"
              f"随机 95 分位 {np.percentile(sims, 95):.3f}，真实值排第 {pct:.0f} 分位")
        rows.append({"版本": label, "区间": "安慰剂", "方案": "双向 50 真实",
                     "Calmar比率": real_cal, "安慰剂95分位": float(np.percentile(sims, 95)),
                     "分位": pct})

        # ---- 成本敏感 ----
        print(f"\n　成本敏感（手续费率 ×2.5 / ×5）")
        for fee, tag in [(0.0002, "万2（现状）"), (0.0005, "万5"), (0.0010, "千1")]:
            old = bt.FEE_RATE
            bt.FEE_RATE = fee
            mo = bt.metrics(nav_of(w_old))
            mn = bt.metrics(nav_of(w_new))
            bt.FEE_RATE = old
            print(f"　　{tag:<10} 单向 {mo['年化收益率']*100:5.2f}% / {mo['Calmar比率']:.3f}"
                  f"　双向 {mn['年化收益率']*100:5.2f}% / {mn['Calmar比率']:.3f}"
                  f"　差额 {mn['Calmar比率']-mo['Calmar比率']:+.3f}")
            rows.append({"版本": label, "区间": f"成本{tag}", "方案": "双向50−单向40",
                         "Calmar差": mn["Calmar比率"] - mo["Calmar比率"],
                         "年化差": mn["年化收益率"] - mo["年化收益率"]})

    pd.DataFrame(rows).to_csv("output/exp/dynamic_checks.csv", index=False, encoding="utf-8")
    print("\n已写入 output/exp/dynamic_checks.csv")


if __name__ == "__main__":
    main()
