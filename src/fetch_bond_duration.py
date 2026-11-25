"""
自检（v4）· 补数据：国债端的久期选择

为什么要补这一步：
  组合 60% 在国债上，这是最大的一块。前面六组实验全在调「权益那一侧」的
  40%，怎么调都只影响四成。国债端一动，影响的是六成。

  当前用的是 511010（五年国债）。如果换成 511260（十年国债），
  在利率下行期久期越长涨得越多；但利率反弹时也跌得更多。
  这是个真问题，必须用数据回答，不能靠「过去几年利率一直下行」这种后视印象。

可得区间（决定了能做多长的检验）：
  511010 五年国债   2013-04 起
  511260 十年国债   2017-08 起   → 可交易版（2019 起）可完整对比
  511090 三十年国债 2023-06 起   → 太短，只作参考，不进结论

沿用 fetch_etf 的后复权口径（分红再投资），保证与现有数据同源可比。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fetch_etf as fe  # noqa: E402

BONDS = [("sh511260", "511260", "十年国债ETF"), ("sh511090", "511090", "三十年国债ETF")]


def main():
    os.makedirs("data", exist_ok=True)
    for sym, code, name in BONDS:
        path = f"data/{code}.csv"
        prices = fe.get_prices(sym)
        events = fe.get_dividends(sym)
        adj, used = fe.adjust_hfq(prices, events, [])
        adj.to_csv(path, index=False, encoding="utf-8")
        print(f"{code} {name}: {len(adj)} 行  {adj.date.iloc[0]} → {adj.date.iloc[-1]}  "
              f"分红 {sum(1 for u in used if u[1]=='分红')} 次  末段复权因子 {adj.adj_factor.iloc[-1]:.4f}")


if __name__ == "__main__":
    main()
