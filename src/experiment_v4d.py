"""
自检（v4d）· 把「可执行性」当作一票否决项

上一轮发现：UP50 年化确实高（长周期 7.1% vs 6.3%），但它是靠
  每月平均调 6.5 个点的权益仓位、方向反复切换 103 次换来的。
这种策略写进报告很漂亮，实盘执行不了：滑点没算、心理扛不住、而且
一百多次小决策里任何一次微调都可能只是噪声。

所以这一轮加约束：
  smooth = 每次只走到目标仓位的一部分（0.5 / 0.3）
  vol_window = 拉长到 120 日，让信号本身更迟钝
  dd_stop = 8% 灾难保险（只在 2015/2018 那种级别触发，平时不干扰）

判据（同时也是给用户的取舍依据）：
  1. 月度仓位变动均值降到 3% 以内、方向切换降到 40 次以内 → 才算可执行
  2. 在此前提下，全区间 Calmar 不低于现状 0.52（长）/ 1.14（可交易）
  3. 样本外不明显恶化

输出：
  output/exp/v4_exec.csv
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


def turnover(w):
    ch = w["w_dividend"].diff().abs().dropna()
    flips = int((np.sign(w["w_dividend"].diff()).diff().abs() > 0).sum())
    return float(ch.mean()), float(ch.max()), flips


def main():
    os.makedirs(EXP_DIR, exist_ok=True)
    rows = []

    plans = {
        "V3 现状": dict(base=0.40, mn=0.20, mx=0.40, vol_target=0.08),
        "UP50 无平滑": dict(base=0.40, mn=0.20, mx=0.50, vol_target=0.08),
        "UP50+平滑0.5": dict(base=0.40, mn=0.20, mx=0.50, vol_target=0.08, smooth=0.5),
        "UP50+平滑0.3": dict(base=0.40, mn=0.20, mx=0.50, vol_target=0.08, smooth=0.3),
        "UP50+窗口120": dict(base=0.40, mn=0.20, mx=0.50, vol_target=0.08, vol_window=120),
        "UP50+窗120+平0.5": dict(base=0.40, mn=0.20, mx=0.50, vol_target=0.08,
                              vol_window=120, smooth=0.5),
        "UP50+平滑0.5+险8%": dict(base=0.40, mn=0.20, mx=0.50, vol_target=0.08,
                              smooth=0.5, dd_stop=0.08),
        "UP50+窗120+险8%": dict(base=0.40, mn=0.20, mx=0.50, vol_target=0.08,
                             vol_window=120, dd_stop=0.08),
        "UP55+平滑0.5": dict(base=0.40, mn=0.20, mx=0.55, vol_target=0.08, smooth=0.5),
    }

    for ver in ["long", "trade"]:
        cfg = bt.VERSIONS[ver]
        op, cl = bt.load_prices(cfg["prices"])
        for name, kw in plans.items():
            w = al.build(ver, "vol08")[0] if name == "V3 现状" else mc.build(ver, **kw)
            nav, fee = bt.simulate(w, op, cl, cfg["exec_price"])
            m = bt.metrics(nav)
            mi = bt.metrics(nav[nav.index <= pd.Timestamp(INS[ver])])
            mo = bt.metrics(nav[nav.index > pd.Timestamp(INS[ver])])
            avg_ch, max_ch, flips = turnover(w)
            rows.append({"版本": ver, "方案": name, "平均权益": float(w["w_dividend"].mean()),
                         "月均变动": avg_ch, "最大变动": max_ch, "方向切换": flips,
                         "年化": m["年化收益率"], "回撤": m["最大回撤"],
                         "Calmar": m["Calmar比率"], "内Calmar": mi["Calmar比率"],
                         "外Calmar": mo["Calmar比率"], "外回撤": mo["最大回撤"],
                         "手续费": fee})

    df = pd.DataFrame(rows)
    df.to_csv(f"{EXP_DIR}/v4_exec.csv", index=False, encoding="utf-8")

    for ver in ["long", "trade"]:
        print()
        print("=" * 112)
        print(f"{bt.VERSIONS[ver]['label']}：可执行性约束下的取舍")
        print("=" * 112)
        print(f"{'方案':<20}{'平均权益':>9}{'月均变动':>9}{'最大变动':>9}{'切换':>6}"
              f"{'年化':>8}{'回撤':>8}{'Calmar':>8}{'外Calmar':>10}{'外回撤':>8}")
        print("-" * 112)
        for _, r in df[df["版本"] == ver].iterrows():
            flag = "  " if (r["月均变动"] <= 0.03 and r["方向切换"] <= 40) else " *"
            print(f"{r['方案']:<20}{r['平均权益']:>9.0%}{r['月均变动']:>9.1%}{r['最大变动']:>9.1%}"
                  f"{r['方向切换']:>6}{r['年化']:>8.1%}{r['回撤']:>8.1%}{r['Calmar']:>8.2f}"
                  f"{r['外Calmar']:>10.2f}{r['外回撤']:>8.1%}{flag}")
        print("  * = 换手未达「月均 3%、切换 40 次」的可执行线")


if __name__ == "__main__":
    main()
