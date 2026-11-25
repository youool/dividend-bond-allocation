"""
自检（v4b）· 暴露匹配：判断一个机制是「真有本事」还是「只是变相加仓」

为什么必须做这一步：
  上一轮里 TP10 平均仓位只有 32%，Calmar 却高达 1.276；DD5 平均 34%、0.611。
  看着都比现状好，但这里有个陷阱——**仓位越低，Calmar 天然越高**。
  因为国债占大头时组合夏普偏高、回撤偏小，减仓本身就是「改善指标」的捷径。
  所以「Calmar 变好」不能证明机制有效，只能证明它减了仓。

  正确做法：先画出「固定比例 30%→60%」的 Calmar 曲线，
  再用候选策略的平均仓位去这条曲线上插值，得到「同等仓位但不做任何动态」的基准。
  候选策略的 Calmar 减去这个基准，才是机制本身的贡献（下称「暴露匹配超额」）。

  这个口径对「加仓型」机制同样成立：UP50 平均 45%，就必须和固定 45% 比，
  不能和固定 40% 比，否则会把「多承担的暴露」记成「择时的功劳」。

同时补三件事：
  1. 固定网格加密到 2 个百分点，让插值更准
  2. 分组年度表：看机制在 2015 股灾、2018 熊市、2023–2026 红利牛里各自做了什么
  3. 分档阈值的稳健性：把档位边界整体平移 ±20%，看结论会不会翻

输出：
  output/exp/v4_exposure_curve.csv    固定比例网格
  output/exp/v4_excess.csv            每个候选的暴露匹配超额
  output/exp/v4_yearly.csv            分年度表现
  output/exp/v4_robust.csv            阈值扰动
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


def perf(nav, ins_end):
    m = bt.metrics(nav)
    mi = bt.metrics(nav[nav.index <= pd.Timestamp(ins_end)])
    mo = bt.metrics(nav[nav.index > pd.Timestamp(ins_end)])
    return m, mi, mo


def run_one(ver, w, cfg, op, cl):
    nav, fee = bt.simulate(w, op, cl, cfg["exec_price"])
    m, mi, mo = perf(nav, INS[ver])
    return {
        "平均权益": float(w["w_dividend"].mean()),
        "全年化": m["年化收益率"], "全回撤": m["最大回撤"],
        "全夏普": m["夏普比率"], "全Calmar": m["Calmar比率"], "全波动": m["年化波动"],
        "内Calmar": mi["Calmar比率"], "内年化": mi["年化收益率"],
        "外Calmar": mo["Calmar比率"], "外年化": mo["年化收益率"], "外回撤": mo["最大回撤"],
        "手续费": fee,
    }, nav


def main():
    os.makedirs(EXP_DIR, exist_ok=True)
    curve_rows, cand_rows, yearly_rows, robust_rows = [], [], [], []

    for ver in ["long", "trade"]:
        cfg = bt.VERSIONS[ver]
        op, cl = bt.load_prices(cfg["prices"])

        # ---------- 1. 固定比例网格 ----------
        grid = {}
        for w0 in np.arange(0.30, 0.61, 0.02):
            w = mc.build(ver, base=float(w0), mn=float(w0), mx=float(w0))
            m, nav = run_one(ver, w, cfg, op, cl)
            grid[round(float(w0), 3)] = m
            curve_rows.append({"版本": ver, "固定比例": round(float(w0), 3), **m})

        gx = np.array(sorted(grid.keys()))
        gy = np.array([grid[k]["全Calmar"] for k in gx])
        gy_out = np.array([grid[k]["外Calmar"] for k in gx])

        def baseline(avg_w, ys):
            return float(np.interp(avg_w, gx, ys))

        # ---------- 2. 候选 ----------
        w_ref, _ = al.build(ver, "vol08")
        cands = [
            ("V3 现状", w_ref),
            ("BANDMID 分档", mc.build(ver, base=0.40, mn=0.20, mx=0.60, bands=BANDS_MID)),
            ("UP50 打开上限", mc.build(ver, base=0.40, mn=0.20, mx=0.50, vol_target=0.08)),
            ("DD5 回撤线", mc.build(ver, base=0.40, mn=0.20, mx=0.40, vol_target=0.08, dd_stop=0.05)),
            ("TP10 止盈", mc.build(ver, base=0.40, mn=0.20, mx=0.40, tp_trigger=0.10, tp_to=0.30)),
            ("BAND+DD5", mc.build(ver, base=0.40, mn=0.20, mx=0.60, bands=BANDS_MID, dd_stop=0.05)),
            ("UP50+DD5", mc.build(ver, base=0.40, mn=0.20, mx=0.50, vol_target=0.08, dd_stop=0.05)),
            ("UP50+BAND", mc.build(ver, base=0.40, mn=0.20, mx=0.55, bands=BANDS_MID)),
        ]
        navs = {}
        for name, w in cands:
            m, nav = run_one(ver, w, cfg, op, cl)
            navs[name] = nav
            bw, bwo = baseline(m["平均权益"], gy), baseline(m["平均权益"], gy_out)
            cand_rows.append({
                "版本": ver, "候选": name, **m,
                "同仓位固定Calmar": bw,
                "全超额": m["全Calmar"] - bw,
                "外超额": m["外Calmar"] - bwo,
            })

        # ---------- 3. 分年度 ----------
        for name, nav in navs.items():
            for y, sub in nav.groupby(nav.index.year):
                if len(sub) < 20:
                    continue
                yr = sub.iloc[-1] / sub.iloc[0] - 1
                dd = float((sub / sub.cummax() - 1).min())
                yearly_rows.append({"版本": ver, "候选": name, "年": y,
                                    "收益": yr, "年内回撤": dd})

        # ---------- 4. 阈值稳健性（只对 BANDMID） ----------
        for tag, mult in [("边界×0.8", 0.8), ("边界×1.2", 1.2), ("档位+5pct", None), ("档位-5pct", None)]:
            if mult is not None:
                bands = [(ub * mult, w) for ub, w in BANDS_MID[:-1]] + [(9.9, 0.20)]
            else:
                shift = 0.05 if tag == "档位+5pct" else -0.05
                bands = [(ub, max(0.15, min(0.65, w + shift))) for ub, w in BANDS_MID[:-1]] + [(9.9, 0.20)]
            w = mc.build(ver, base=0.40, mn=0.20, mx=0.60, bands=bands)
            m, _ = run_one(ver, w, cfg, op, cl)
            robust_rows.append({"版本": ver, "扰动": tag,
                                "平均权益": m["平均权益"], "全年化": m["全年化"],
                                "全回撤": m["全回撤"], "全Calmar": m["全Calmar"],
                                "外Calmar": m["外Calmar"]})

    pd.DataFrame(curve_rows).to_csv(f"{EXP_DIR}/v4_exposure_curve.csv", index=False, encoding="utf-8")
    cdf = pd.DataFrame(cand_rows)
    cdf.to_csv(f"{EXP_DIR}/v4_excess.csv", index=False, encoding="utf-8")
    pd.DataFrame(yearly_rows).to_csv(f"{EXP_DIR}/v4_yearly.csv", index=False, encoding="utf-8")
    pd.DataFrame(robust_rows).to_csv(f"{EXP_DIR}/v4_robust.csv", index=False, encoding="utf-8")

    for ver in ["long", "trade"]:
        print()
        print("=" * 106)
        print(f"{bt.VERSIONS[ver]['label']}：暴露匹配后的真实超额")
        print("=" * 106)
        print(f"{'候选':<16}{'平均权益':>9}{'全年化':>8}{'全回撤':>8}{'全Calmar':>10}"
              f"{'同仓位固定':>11}{'全超额':>9}{'外超额':>9}")
        print("-" * 106)
        sub = cdf[cdf["版本"] == ver]
        for _, r in sub.iterrows():
            print(f"{r['候选']:<16}{r['平均权益']:>9.0%}{r['全年化']:>8.1%}{r['全回撤']:>8.1%}"
                  f"{r['全Calmar']:>10.2f}{r['同仓位固定Calmar']:>11.2f}"
                  f"{r['全超额']:>+9.2f}{r['外超额']:>+9.2f}")

    print()
    print("=" * 106)
    print("分档阈值稳健性（BANDMID 的档位边界/比例各扰动一档）")
    print("=" * 106)
    rdf = pd.DataFrame(robust_rows)
    print(f"{'版本':<8}{'扰动':<14}{'平均权益':>9}{'全年化':>9}{'全回撤':>9}{'全Calmar':>10}{'外Calmar':>10}")
    print("-" * 106)
    for _, r in rdf.iterrows():
        print(f"{r['版本']:<8}{r['扰动']:<14}{r['平均权益']:>9.0%}{r['全年化']:>9.1%}"
              f"{r['全回撤']:>9.1%}{r['全Calmar']:>10.2f}{r['外Calmar']:>10.2f}")


if __name__ == "__main__":
    main()
