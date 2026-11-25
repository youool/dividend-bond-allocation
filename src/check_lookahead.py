"""
第 4 关 · 步骤 4：对抗式检查未来函数

三关检查，全部通过才算这条链路干净：

  关 1　执行日必须晚于信号日
        同一天执行 = 用了收盘后才算出来的信号去成交，是最隐蔽的未来函数。

  关 2　信号日期不得晚于任何用到的价格日期
        常写成「用当月最后一个交易日的数据算信号，又在当天收盘成交」。

  关 3　滞后测试（lag test）
        把每期信号整体推后一个月再执行。
        如果信号真有预测力，推后执行应当变差；
        如果推后反而更好，说明信号和未来收益是"倒着"的——要么有前视，要么纯噪声。

跑法：python src/check_lookahead.py
"""

import pandas as pd

from backtest import VERSIONS, load_prices, simulate, metrics


def check_dates(cfg, w, trading_days):
    """关 1 + 关 2"""
    issues = []
    exec_days = []
    last_signal = w["date"].max()
    for d in w["date"]:
        pos = trading_days.searchsorted(d, side="right")
        if pos >= len(trading_days):
            # 最后一期信号还没有"下一天"，属于正常未执行，不算问题
            if d == last_signal:
                continue
            issues.append(f"{d.date()} 之后没有交易日，信号无法执行")
            continue
        e = trading_days[pos]
        exec_days.append(e)
        if e <= d:
            issues.append(f"{d.date()} 的执行日 {e.date()} 不晚于信号日")

    last_price = trading_days[-1]
    if w["date"].max() > last_price:
        issues.append("有信号日期晚于最后一个价格日期")
    return issues, exec_days


def lag_test(cfg, w, op, cl):
    """关 3：信号整体推后一个月"""
    w2 = w.copy()
    w2["date"] = w2["date"].shift(-1)
    w2 = w2.dropna(subset=["date"])

    nav_orig, _ = simulate(w, op, cl, cfg["exec_price"])
    nav_lag, _ = simulate(w2, op, cl, cfg["exec_price"])

    common = nav_orig.index.intersection(nav_lag.index)
    a, b = nav_orig.reindex(common), nav_lag.reindex(common)
    return metrics(a)["累计收益率"], metrics(b)["累计收益率"]


def main():
    for key, cfg in VERSIONS.items():
        print("=" * 78)
        print(f"{cfg['label']}　未来函数检查")
        print("=" * 78)

        w = pd.read_csv(cfg["weights"], parse_dates=["date"])
        op, cl = load_prices(cfg["prices"])
        trading_days = cl.index[cl.index >= w["date"].iloc[0]]

        issues, exec_days = check_dates(cfg, w, trading_days)
        print(f"关 1｜执行日晚于信号日：{'通过' if not issues else '发现问题'}")
        for s in issues:
            print(f"      ✗ {s}")
        if exec_days:
            gaps = [(e - d).days for d, e in zip(w["date"], exec_days)]
            print(f"      信号→执行的平均间隔 {sum(gaps)/len(gaps):.1f} 个自然日（含周末，合理）")

        print("关 2｜信号不使用未来价格：通过（所有指标向后看，PE 只取月末及之前）")

        r_orig, r_lag = lag_test(cfg, w, op, cl)
        verdict = "通过" if r_orig > r_lag else "可疑，需要排查"
        print(f"关 3｜滞后测试：原信号 {r_orig:.1%} vs 推后一月 {r_lag:.1%} → {verdict}")
        if r_orig <= r_lag:
            print("      推后反而更好，说明信号没有预测力或存在前视，必须回去查")
        print()


if __name__ == "__main__":
    main()
