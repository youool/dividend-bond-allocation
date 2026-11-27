"""
归因：本策略的收益是 alpha 还是 beta？

拆法（从下往上，每一层都能量化）：
  1. 纯 beta 底座：静态 40/60 月度再平衡（不做任何判断）
  2. 其中红利腿贡献 = 平均权重 × 红利年化；国债腿同理
  3. 再平衡溢价 = 静态再平衡 − 静态买入持有（两者资产完全相同，只差「拉不拉回比例」）
  4. 闸门增量 = 主线策略 − 静态再平衡（只差「波动率触发时缩不缩仓」）
  5. 回归检验：策略日收益对「静态 40/60 组合」回归 → alpha 是否显著异于 0、R² 多少
  6. 对沪深300 回归 → 市场 beta 多少（说明跑赢它靠不靠选股能力）

输出：output/exp/attribution_{ver}.csv / attribution_summary.csv
不覆盖主线任何产物。
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import allocation as al  # noqa: E402
import backtest as bt  # noqa: E402

BASE_EQ = 0.40
VOL_TARGET = 0.08


def ann(nav, col=None):
    """与 bt.metrics 完全一致的年化口径（交易日数/252），保证和报告数字对得上。"""
    s = (nav if col is None else nav[col]).dropna()
    return bt.metrics(s)["年化收益率"]


def mdd(s):
    s = s.dropna()
    return float((s / s.cummax() - 1.0).min())


def ols_full(y, x):
    """单变量 OLS，返回 dict：alpha 日均、beta、R²、alpha 年化、alpha t 值、残差波动。"""
    d = pd.concat([y, x], axis=1, sort=True).dropna()
    Y = d.iloc[:, 0].values
    X = d.iloc[:, 1].values
    X1 = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X1, Y, rcond=None)
    pred = X1 @ coef
    ss_res = float(((Y - pred) ** 2).sum())
    ss_tot = float(((Y - Y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    alpha_d, beta = float(coef[0]), float(coef[1])
    n, k = len(Y), 2
    sigma2 = ss_res / (n - k) if n > k else np.nan
    try:
        cov = sigma2 * np.linalg.inv(X1.T @ X1)
        se_alpha = float(np.sqrt(cov[0, 0]))
    except Exception:
        se_alpha = np.nan
    t_alpha = alpha_d / se_alpha if se_alpha and se_alpha > 0 else np.nan
    resid_vol = float(np.std(Y - pred, ddof=k) * np.sqrt(252))
    return {"alpha_d": alpha_d, "beta": beta, "r2": r2, "alpha_ann": alpha_d * 252,
            "t_alpha": float(t_alpha), "resid_vol": resid_vol, "n": n}


def static_weights(dates, eq):
    df = pd.DataFrame({"date": pd.to_datetime(dates)})
    df["w_dividend"] = eq
    df["w_broad"] = 0.0
    df["w_bond"] = 1.0 - eq
    df["w_cash"] = 0.0
    return df


def main():
    os.makedirs("output/exp", exist_ok=True)
    summary_rows = []

    for ver in ["trade", "long"]:
        cfg = bt.VERSIONS[ver]
        op, cl = bt.load_prices(cfg["prices"])
        price = op if cfg["exec_price"] == "open" else cl

        # 1) 主线（含闸门），与 final_nav 同口径
        alloc, _ = al.build(ver, al.DEFAULT_MODE, base_equity=BASE_EQ,
                            vol_target=VOL_TARGET)
        nav_s, fee_s = bt.simulate(alloc, op, cl, cfg["exec_price"])
        alloc = alloc.copy()
        alloc["date"] = pd.to_datetime(alloc["date"])
        dates = alloc["date"]

        # 2) 静态 40/60 月度再平衡（同执行日、同费率）
        w_static = static_weights(dates, BASE_EQ)
        nav_r, fee_r = bt.simulate(w_static, op, cl, cfg["exec_price"])

        # 3) 静态 40/60 买入持有（只在首日建一次仓，之后权重自由漂移）
        w_bh = static_weights([dates.iloc[0]], BASE_EQ)
        nav_b, fee_b = bt.simulate(w_bh, op, cl, cfg["exec_price"])

        # 对齐到共同区间
        idx = nav_s.index.intersection(nav_r.index).intersection(nav_b.index)
        s, r, b = nav_s[idx], nav_r[idx], nav_b[idx]

        # ---- 期内收益分解：按执行日切段，目标权重 × 资产收益 ----
        # 执行日 = 信号日的下一个交易日
        tdays = cl.index
        exec_dates = []
        for d in dates:
            pos = tdays.searchsorted(d, side="right")
            if pos < len(tdays):
                exec_dates.append(tdays[pos])
        exec_dates = [d for d in exec_dates if d in idx][1:]  # 首期无上期，跳过
        # 首期起点补第一个执行日
        first_exec = [tdays[tdays.searchsorted(d, side="right")]
                      for d in dates if tdays.searchsorted(d, side="right") < len(tdays)][0]
        bounds = [first_exec] + exec_dates

        contrib = pd.DataFrame(0.0, index=bounds[:-1], columns=["dividend", "bond", "resid"])
        prev_eq = 0.0
        for k in range(len(bounds) - 1):
            d0, d1 = bounds[k], bounds[k + 1]
            if d0 not in idx or d1 not in idx:
                continue
            # 该期期初的目标权益比例（静态 40/60 恒定；主线用于对照）
            row = alloc[alloc["date"] < d0]
            eq = float(row["w_dividend"].iloc[-1]) if len(row) else BASE_EQ
            rd = float(price.loc[d1, "dividend"] / price.loc[d0, "dividend"] - 1)
            rb = float(price.loc[d1, "bond"] / price.loc[d0, "bond"] - 1)
            contrib.loc[d0, "dividend"] = eq * rd
            contrib.loc[d0, "bond"] = (1 - eq) * rb
            contrib.loc[d0, "resid"] = 0.0

        # 静态口径的贡献：权重恒为 40/60
        contrib_static = pd.DataFrame(0.0, index=bounds[:-1], columns=["dividend", "bond"])
        for k in range(len(bounds) - 1):
            d0, d1 = bounds[k], bounds[k + 1]
            if d0 not in idx or d1 not in idx:
                continue
            rd = float(price.loc[d1, "dividend"] / price.loc[d0, "dividend"] - 1)
            rb = float(price.loc[d1, "bond"] / price.loc[d0, "bond"] - 1)
            contrib_static.loc[d0, "dividend"] = BASE_EQ * rd
            contrib_static.loc[d0, "bond"] = (1 - BASE_EQ) * rb

        # 期内收益转成累计：几何链接
        def link(col):
            return float(np.prod(1.0 + col.values) - 1.0)

        years = (idx[-1] - idx[0]).days / 365.25
        n_per = len(contrib_static)
        per_year = n_per / years if years > 0 else np.nan

        # 年化贡献（几何链接后按年化折算）
        def annualize_contrib(col):
            cum = link(col)
            return (1.0 + cum) ** (1.0 / years) - 1.0 if years > 0 else np.nan

        c_div_static = annualize_contrib(contrib_static["dividend"])
        c_bond_static = annualize_contrib(contrib_static["bond"])

        # ---- 回归 ----
        rs = s.pct_change().dropna()
        rr = r.pct_change().dropna()
        rb_ = b.pct_change().dropna()
        if "broad" in cl.columns:
            rmkt = cl["broad"].pct_change().dropna()
        else:
            rmkt = None

        R1 = ols_full(rs, rr)                           # 对静态 40/60
        R2_ = ols_full(rs, rmkt) if rmkt is not None else {k: np.nan for k in
                                                          ("beta", "r2", "alpha_ann", "t_alpha", "resid_vol")}
        # 相对静态 40/60 的信息比率
        te = float((rs - rr).std() * np.sqrt(252))
        ir = (bt.metrics(s)["年化收益率"] - bt.metrics(r)["年化收益率"]) / te if te > 0 else np.nan

        # 三个组合的平均权益权重（暴露是否可比）
        avg_eq_main = float(alloc["w_dividend"].mean())
        avg_eq_static = BASE_EQ
        nav_bh_v = nav_b["dividend"] if "dividend" in nav_b else None

        # 双因子：红利 + 国债
        dd = pd.concat([rs, cl["dividend"].pct_change(), cl["bond"].pct_change()], axis=1, sort=True).dropna()
        dd.columns = ["y", "x1", "x2"]
        X1 = np.column_stack([np.ones(len(dd)), dd["x1"].values, dd["x2"].values])
        coef, *_ = np.linalg.lstsq(X1, dd["y"].values, rcond=None)
        pred = X1 @ coef
        r2m = 1 - ((dd["y"].values - pred) ** 2).sum() / ((dd["y"].values - dd["y"].mean()) ** 2).sum()
        ann_a_m = float(coef[0]) * 252

        # ---- 汇总 ----
        rows = [
            ("全期年化_主线含闸门", ann(s)),
            ("全期年化_静态40/60再平衡", ann(r)),
            ("全期年化_静态40/60买入持有", ann(b)),
            ("红利买入持有年化", ann(cl["dividend"])),
            ("国债买入持有年化", ann(cl["bond"])),
            ("静态口径_红利腿年化贡献", c_div_static),
            ("静态口径_国债腿年化贡献", c_bond_static),
            ("静态贡献合计", c_div_static + c_bond_static),
            ("再平衡溢价(静态再平衡-买入持有)", ann(r) - ann(b)),
            ("闸门增量(主线-静态再平衡)", ann(s) - ann(r)),
            ("", np.nan),
            ("回归_对静态40/60_beta", R1["beta"]),
            ("回归_对静态40/60_R2", R1["r2"]),
            ("回归_对静态40/60_alpha年化", R1["alpha_ann"]),
            ("回归_对静态40/60_alpha_t值", R1["t_alpha"]),
            ("相对静态40/60_跟踪误差", te),
            ("相对静态40/60_信息比率", ir),
            ("回归_对沪深300_beta", R2_["beta"]),
            ("回归_对沪深300_R2", R2_["r2"]),
            ("回归_对沪深300_alpha年化", R2_["alpha_ann"]),
            ("回归_对沪深300_alpha_t值", R2_["t_alpha"]),
            ("平均权益权重_主线", avg_eq_main),
            ("回归_双因子(红利+国债)_alpha年化", ann_a_m),
            ("回归_双因子_R2", r2m),
            ("", np.nan),
            ("最大回撤_主线", mdd(s)),
            ("最大回撤_静态再平衡", mdd(r)),
            ("最大回撤_买入持有40/60", mdd(b)),
            ("累计手续费_主线", fee_s),
            ("累计手续费_静态再平衡", fee_r),
            ("调仓次数(年化)", per_year),
            ("区间", f"{idx[0].date()} → {idx[-1].date()}"),
        ]
        for name, val in rows:
            if name == "":
                summary_rows.append({"版本": ver, "项目": "", "数值": np.nan})
            else:
                summary_rows.append({"版本": ver, "项目": name,
                                     "数值": val if isinstance(val, str) else round(float(val), 6)})

        detail = pd.DataFrame({
            "nav_strategy": s, "nav_static": r, "nav_bh": b,
        })
        detail.to_csv(f"output/exp/attribution_nav_{ver}.csv", encoding="utf-8")
        contrib_static.to_csv(f"output/exp/attribution_contrib_{ver}.csv", encoding="utf-8")

    out = pd.DataFrame(summary_rows)
    out.to_csv("output/exp/attribution_summary.csv", index=False, encoding="utf-8")
    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    for ver in ["trade", "long"]:
        sub = out[out["版本"] == ver][["项目", "数值"]]
        print(f"\n{'='*70}\n{ver}\n{'='*70}")
        print(sub.to_string(index=False))
    print("\n已写出 output/exp/attribution_summary.csv")


if __name__ == "__main__":
    main()
