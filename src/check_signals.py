"""
第 3 关 · 步骤 4：抽查三个月，把中间量全部摊开，供人工核对

程序验程序等于没验。这个脚本把某个月用到的每一个数都打印出来：
  300 收盘 vs 它的 120 日均线、红利收盘 vs 它的均线、PE 与分位、
  基础仓位、扣减后权益、最终权重、以及权重加总。

拿计算器照着重算一遍，能对上才算这一步过。
"""

import pandas as pd

VERSIONS = {
    "long": ("长周期版", "data/index/红利低波_全收益.csv", "data/index/沪深300_全收益.csv", "output/weights_long.csv"),
    "trade": ("可交易版", "data/512890.csv", "data/510300.csv", "output/weights_trade.csv"),
}

# 挑三种典型月份：低估满档、300 破位减半、双破位归零
CHECK_DATES = ["2024-02-29", "2026-03-31", "2020-03-31", "2015-08-31"]


def load(path):
    df = pd.read_csv(path, parse_dates=["date"])
    return df.dropna(subset=["close"]).drop_duplicates(subset="date").sort_values("date").set_index("date")["close"]


def show(version, label, div_path, broad_path, w_path):
    div = load(div_path)
    broad = load(broad_path)
    w = pd.read_csv(w_path)
    w["date"] = pd.to_datetime(w["date"])

    print("=" * 70)
    print(f"{label}")
    print("=" * 70)

    shown = 0
    for target in CHECK_DATES:
        target = pd.Timestamp(target)
        row = w[w["date"] == target]
        if row.empty:
            continue
        r = row.iloc[0]

        # 用原始数据重算，不读权重表里的中间量
        def ma_at(s, d):
            hist = s[s.index <= d].tail(120)
            return hist.mean() if len(hist) == 120 else float("nan")

        b_close = broad[broad.index <= target].iloc[-1]
        b_ma = ma_at(broad, target)
        a_close = div[div.index <= target].iloc[-1]
        a_ma = ma_at(div, target)

        print(f"\n【{target.date()}】{r['note']}")
        print(f"  沪深300：收盘 {b_close:>10.3f}　120日均线 {b_ma:>10.3f}　"
              f"{'在线上（健康）' if b_close >= b_ma else '跌破（减半）'}")
        print(f"  红利低波：收盘 {a_close:>10.3f}　120日均线 {a_ma:>10.3f}　"
              f"{'在线上' if a_close >= a_ma else '跌破（不入选）'}")
        print(f"  PE-TTM {r['pe_ttm']:>6.2f}　近10年分位 {r['pe_pct']:>6.0%}　→ 基础权益 {r['base_pos']:>4.0%}")
        print(f"  扣减后权益 {r['equity']:>4.0%}")
        print(f"  最终权重：红利 {r['w_dividend']:>5.0%}　300 {r['w_broad']:>5.0%}　"
              f"国债 {r['w_bond']:>5.0%}　货币 {r['w_cash']:>5.0%}　合计 {r['w_sum']:>5.0%}")
        shown += 1

    print(f"\n本版本共抽查 {shown} 个月。")


def main():
    for key, (label, div, broad, w) in VERSIONS.items():
        show(key, label, div, broad, w)
        print()
    print("核对方法：拿计算器按上面的收盘价与均线重算一遍，")
    print("确认「是否破位 → 基础仓位如何扣减 → 权益内部谁强给谁」三步都一致。")


if __name__ == "__main__":
    main()
