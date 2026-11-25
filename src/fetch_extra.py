# -*- coding: utf-8 -*-
"""
i2 / i6 的数据准备：黄金 ETF（518880）与创业板 ETF（159915）。

复用 fetch_etf.py 的抓取与后复权逻辑，不另写一套。
这两只经检查都没有分红与份额折算事件，adj_factor 恒为 1；
仍然走同一个函数，是为了以后再补别的标的时口径一致。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_etf as fe

EXTRA = {
    "sh518880": ("518880", "黄金ETF", []),
    "sz159915": ("159915", "创业板ETF", []),
    "sh513100": ("513100", "纳指ETF", [("2022-01-14", 5.0)]),   # 1:5 份额折算
}


def main():
    force = "--force" in sys.argv
    print(f"{'代码':<8}{'名称':<12}{'行数':>6}  {'起始':<12}{'结束':<12}{'复权事件':>8}")
    print("-" * 62)
    for symbol, (code, name, splits) in EXTRA.items():
        path = os.path.join("data", f"{code}.csv")
        if os.path.exists(path) and not force:
            old = __import__("pandas").read_csv(path)
            print(f"{code:<8}{name:<12}{len(old):>6}  {old['date'].iloc[0]:<12}"
                  f"{old['date'].iloc[-1]:<12}{'已存在':>8}")
            continue
        prices = fe.get_prices(symbol)
        events = fe.get_dividends(symbol)
        adj, used = fe.adjust_hfq(prices, events, splits)
        adj.to_csv(path, index=False, encoding="utf-8")
        print(f"{code:<8}{name:<12}{len(adj):>6}  {adj['date'].iloc[0]:<12}"
              f"{adj['date'].iloc[-1]:<12}{len(used):>8}")


if __name__ == "__main__":
    main()
