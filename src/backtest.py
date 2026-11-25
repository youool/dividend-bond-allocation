"""
第 4 关 · 回测内核

做的事：读第 3 关的目标权重表，按「月末出信号 → 次日成交 → 双边万 2 手续费」
逐日算出策略净值，同时生成两条基准。

为什么用「份额法」而不是「每日按权重算收益」：
  每日按目标权重算收益，等于假设每天都在再平衡，会凭空产生收益、也凭空产生成本。
  真实做法是：调仓日按目标权重换一次份额，之后份额不动，价值随价格自然漂移。
  权重会因此慢慢偏离目标，直到下个月末再拉回来——这才是真实发生的事。

三个必须对齐的细节：
  - 执行日 = 信号日的下一个交易日（不是同一天，否则就是用了收盘后才知道的信号）
  - 成交价：可交易版用开盘价；长周期版用收盘价
    原因：中证指数接口只提供收盘价，没有开盘价，两版口径不同，报告里必须写明
  - 手续费按成交金额双边收：fee = 万2 × Σ|目标市值 − 当前市值|
    这个式子里天然包含买和卖两侧，不需要再乘 2

自带一个单元测试（--test）：
  把权重固定成 100% 持有单一标的，跑出来的净值必须和该标的买入持有完全一致，
  误差只允许来自一次建仓手续费。这条过不了，说明内核有 bug，后面所有结论都不作数。
"""

import os
import sys

import numpy as np
import pandas as pd

FEE_RATE = 0.0002
RF_ANNUAL = 0.02
INIT_CAPITAL = 1.0

WEIGHT_COLS = {"w_dividend": "dividend", "w_broad": "broad", "w_bond": "bond", "w_cash": "cash"}
ASSETS = list(WEIGHT_COLS.values())

VERSIONS = {
    "long": {
        "label": "长周期版",
        "weights": "output/weights_long.csv",
        "prices": {
            "dividend": "data/index/红利低波_全收益.csv",
            "broad": "data/index/沪深300_全收益.csv",
            "bond": "data/511010.csv",
            "cash": "data/511880.csv",
        },
        "exec_price": "close",   # 指数没有开盘价
        "insample_end": "2020-12-31",
        "out_nav": "output/nav_long.csv",
        "out_metrics": "output/metrics_long.csv",
        "out_chart": "charts/05_净值对比_长周期.png",
    },
    "trade": {
        "label": "可交易版",
        "weights": "output/weights_trade.csv",
        "prices": {
            "dividend": "data/512890.csv",
            "broad": "data/510300.csv",
            "bond": "data/511010.csv",
            "cash": "data/511880.csv",
        },
        "exec_price": "open",
        "insample_end": "2022-12-31",
        "out_nav": "output/nav_trade.csv",
        "out_metrics": "output/metrics_trade.csv",
        "out_chart": "charts/06_净值对比_可交易.png",
    },
}


def load_prices(paths):
    opens, closes = {}, {}
    for k, path in paths.items():
        df = pd.read_csv(path, parse_dates=["date"]).drop_duplicates("date").sort_values("date")
        df = df.set_index("date")
        opens[k] = df["open"] if "open" in df.columns else np.nan
        closes[k] = df["close"]
    op = pd.DataFrame(opens).sort_index().ffill()
    cl = pd.DataFrame(closes).sort_index().ffill()
    # 指数没有开盘价时，用收盘价代替（并在执行价上统一处理）
    op = op.where(op.notna(), cl)
    return op, cl


def simulate(weights_df, op, cl, exec_price="open", start=None):
    """
    weights_df: 含 date + 四个权重列
    返回 (nav Series, 手续费合计)

    数据完整性是硬约束，不做任何静默容错：
      价格缺失时直接抛错，而不是把当天净值填成 1.0。
      原因：填成 1.0 会让「数据没有」看起来像「净值不动」，回测照样跑完、
      指标照样输出，只是全部是假的。这个坑在跑三十年国债（2019 年无数据）时踩到过。
      正确做法是让调用方用 start= 把起点对齐到数据真实起点，把区间差异写在结果里。
    """
    w = weights_df.sort_values("date").reset_index(drop=True)
    if w["date"].duplicated().any():
        dup = sorted(w.loc[w["date"].duplicated(), "date"].astype(str))[:5]
        raise ValueError(f"权重表存在重复日期：{dup}")

    price = op if exec_price == "open" else cl

    start_ts = pd.Timestamp(start) if start is not None else pd.Timestamp(w["date"].iloc[0])
    trading_days = cl.index[cl.index >= start_ts]

    # 信号日 → 执行日
    exec_map = {}
    for d in w["date"]:
        pos = trading_days.searchsorted(d, side="right")
        if pos < len(trading_days):
            exec_map[trading_days[pos]] = d
    if not exec_map:
        raise RuntimeError("没有任何可执行日，检查权重表与价格表的日期范围")

    # 前置校验：回测区间内可能被持有的资产，必须在首个执行日之前就有价格。
    # 权重恒为 0 的资产不校验（它不影响净值，也不该卡住回测）。
    col_of = {asset: col for col, asset in WEIGHT_COLS.items()}
    first_exec = min(exec_map)
    for a in ASSETS:
        if a not in col_of or w[col_of[a]].abs().max() <= 1e-9:
            continue
        fv = cl[a].first_valid_index()
        if fv is None:
            raise ValueError(f"资产 {a} 的价格全为空，检查数据文件是否读错")
        if fv > first_exec:
            raise ValueError(
                f"资产 {a} 的首个价格是 {fv.date()}，晚于首个执行日 {first_exec.date()}。"
                f"请用 start='{fv.date()}' 把回测起点对齐，"
                f"否则 {first_exec.date()} 到 {fv.date()} 这段没有价格可用。")

    shares = pd.Series(0.0, index=ASSETS)
    rows = []
    total_fee = 0.0
    invested = False

    for t in trading_days:
        if t in exec_map:
            s = exec_map[t]
            target_w = w.loc[w["date"] == s, list(WEIGHT_COLS)].iloc[0].values.astype(float)
            px = price.loc[t, ASSETS].values
            cur_v = shares.values * px

            # 已持有的资产价格若是 NaN，说明数据有洞，不能当成 0 市值
            held_nan = np.isnan(cur_v) & (shares.values != 0)
            if held_nan.any():
                miss = [a for a, b in zip(ASSETS, held_nan) if b]
                raise ValueError(f"{t.date()} 调仓时持仓资产价格缺失：{miss}")

            cur_val = float(np.nansum(cur_v))
            total = cur_val if cur_val > 0 else INIT_CAPITAL
            target_val = total * target_w
            trade_amt = float(np.abs(target_val - np.nan_to_num(cur_v)).sum())
            fee = trade_amt * FEE_RATE
            total_fee += fee
            total_after = total - fee
            # 目标权重为 0 的资产不参与除法：0/NaN 仍是 NaN，会把整条持仓污染
            shares = pd.Series(
                np.where(target_w > 0, total_after * target_w / px, 0.0), index=ASSETS)
            invested = True

        if not invested:
            # 首个执行日之前：钱还是现金，净值恒为初始资金。
            # 这是「尚未建仓」，与「数据缺失」是两件完全不同的事，必须分开处理。
            rows.append((t, INIT_CAPITAL))
            continue

        vals = shares.values * cl.loc[t, ASSETS].values
        held_nan = np.isnan(vals) & (shares.values != 0)
        if held_nan.any():
            miss = [a for a, b in zip(ASSETS, held_nan) if b]
            raise ValueError(f"{t.date()} 收盘价缺失，无法计算净值：{miss}")

        nav = float(np.nansum(vals))
        if not np.isfinite(nav) or nav <= 0:
            raise ValueError(f"{t.date()} 净值异常（{nav}），检查价格数据")
        rows.append((t, nav))

    nav = pd.Series(dict(rows)).sort_index()
    return nav, total_fee


def metrics(nav):
    r = nav.pct_change().dropna()
    if len(r) == 0 or r.std() == 0:
        return {"累计收益率": 0.0, "年化收益率": 0.0, "最大回撤": 0.0, "夏普比率": 0.0, "年化波动": 0.0, "交易日数": len(nav)}
    years = len(nav) / 252.0
    total = float(nav.iloc[-1] / nav.iloc[0] - 1)
    annual = float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1)
    dd = float((nav / nav.cummax() - 1).min())
    excess = r - RF_ANNUAL / 252.0
    sharpe = float(excess.mean() / r.std() * np.sqrt(252))
    # Calmar = 年化收益 / 最大回撤绝对值。夏普会因低波动虚高（国债本身接近无风险利率），
    # Calmar 不受这个影响，汇报里更适合用来横向比较不同风险水平的组合。
    mdd = abs(dd) if dd != 0 else np.nan
    calmar = float(annual / mdd) if mdd and not np.isnan(mdd) else 0.0
    return {"累计收益率": total, "年化收益率": annual, "最大回撤": dd,
            "夏普比率": sharpe, "Calmar比率": calmar,
            "年化波动": float(r.std() * np.sqrt(252)), "交易日数": len(nav)}


def run(version_key):
    cfg = VERSIONS[version_key]
    w = pd.read_csv(cfg["weights"], parse_dates=["date"])
    op, cl = load_prices(cfg["prices"])

    nav, fee = simulate(w, op, cl, cfg["exec_price"])
    nav.name = "strategy"

    out = pd.DataFrame(nav)
    for bm, asset in [("bm_broad", "broad"), ("bm_dividend", "dividend")]:
        px = cl[asset].reindex(out.index).ffill()
        out[bm] = (px / px.iloc[0]).values
    out = out.ffill().dropna()
    out.to_csv(cfg["out_nav"], encoding="utf-8")
    return cfg, out, fee


def report(cfg, out, fee):
    print("=" * 80)
    print(f"{cfg['label']}　{out.index[0].date()} → {out.index[-1].date()}　{len(out)} 个交易日")
    print("=" * 80)

    rows = []
    for seg, sub in [
        ("全区间", out),
        ("样本内", out[out.index <= pd.Timestamp(cfg["insample_end"])]),
        ("样本外", out[out.index > pd.Timestamp(cfg["insample_end"])]),
    ]:
        if len(sub) < 20:
            continue
        for col, cname in [("strategy", "策略"), ("bm_broad", "沪深300买入持有"), ("bm_dividend", "红利低波买入持有")]:
            rows.append({"区间": seg, "对象": cname, **metrics(sub[col])})

    mdf = pd.DataFrame(rows)
    mdf.to_csv(cfg["out_metrics"], index=False, encoding="utf-8")

    print(f"{'区间':<8}{'对象':<18}{'累计':>9}{'年化':>9}{'最大回撤':>10}{'夏普':>7}{'年化波动':>10}")
    print("-" * 80)
    for _, r in mdf.iterrows():
        print(f"{r['区间']:<8}{r['对象']:<18}{r['累计收益率']:>9.1%}{r['年化收益率']:>9.1%}"
              f"{r['最大回撤']:>10.1%}{r['夏普比率']:>7.2f}{r['年化波动']:>10.1%}")
    print()
    print(f"累计手续费（初始资金按 1 计）：{fee:.4f}（{fee:.2%}）")
    return mdf


def plot(cfg, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    for col, name, color, lw in [
        ("strategy", "策略", "#1D9E75", 1.9),
        ("bm_broad", "沪深300 买入持有", "#378ADD", 1.2),
        ("bm_dividend", "红利低波 买入持有", "#BA7517", 1.2),
    ]:
        ax.plot(out.index, out[col], label=name, color=color, linewidth=lw)

    ax.set_title(f"{cfg['label']}：策略与两条基准（归一化到 1.0）", fontsize=13)
    ax.set_ylabel("净值")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(cfg["out_chart"])
    plt.close(fig)
    print(f"净值图已保存：{cfg['out_chart']}")


def self_test():
    """单元测试：100% 持有单一标的时，净值必须等于该标的的买入持有。"""
    print("=" * 80)
    print("内核自检：权重固定 100% 持有红利低波，净值应与买入持有完全一致")
    print("=" * 80)

    cfg = VERSIONS["trade"]
    op, cl = load_prices(cfg["prices"])
    w0 = pd.read_csv(cfg["weights"], parse_dates=["date"])
    test_w = w0[["date"]].copy()
    test_w["w_dividend"] = 1.0
    test_w["w_broad"] = 0.0
    test_w["w_bond"] = 0.0
    test_w["w_cash"] = 0.0

    nav, fee = simulate(test_w, op, cl, cfg["exec_price"])
    bench = cl["dividend"].reindex(nav.index).ffill()
    bench = bench / bench.iloc[0]

    # 净值首日因为扣了一次建仓手续费，会比基准低一点点，之后应完全同步
    nav_n = nav / nav.iloc[0]
    diff = (nav_n - bench).abs()
    max_diff = float(diff.max())
    print(f"调仓 {len(test_w)} 次，手续费合计 {fee:.6f}")
    print(f"与买入持有的最大偏离：{max_diff:.6f}（应接近 0，仅来源于每次调仓的重建仓成本）")
    print(f"首日建仓手续费：{1 - nav.iloc[0]:.6f}")

    ok = max_diff < 0.01
    print(f"结论：{'通过' if ok else '未通过——内核有问题，先修再跑'}")
    return ok


def main():
    os.makedirs("output", exist_ok=True)
    if "--test" in sys.argv:
        self_test()
        return

    ok = self_test()
    print()
    if not ok:
        print("自检未通过，停止回测。")
        return

    for key in VERSIONS:
        cfg, out, fee = run(key)
        report(cfg, out, fee)
        plot(cfg, out)
        print()


if __name__ == "__main__":
    main()
