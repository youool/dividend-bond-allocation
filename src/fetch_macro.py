# -*- coding: utf-8 -*-
"""
i7 的辅助数据：纳斯达克100 指数（美元）与美元兑人民币中间价。

为什么要这两个：
  513100 用人民币计价，它的涨跌 = 纳指涨幅 × 汇率变化 × (1 + 溢价/跟踪误差)。
  只看 ETF 价格，会把「人民币贬值」和「QDII 溢价扩大」都误记成「纳指涨得好」。
  拆开之后才能回答：这部分收益可持续吗？
"""
import os
import sys

import pandas as pd

import akshare as ak

OUT_NDX = "data/NDX.csv"
OUT_FX = "data/USDCNY.csv"


def main():
    os.makedirs("data", exist_ok=True)
    force = "--force" in sys.argv

    if force or not os.path.exists(OUT_NDX):
        ndx = ak.index_us_daily_sina(symbol=".NDX")
        ndx = ndx[["date", "close"]].copy()
        ndx["date"] = pd.to_datetime(ndx["date"])
        ndx = ndx.drop_duplicates("date").sort_values("date")
        ndx.to_csv(OUT_NDX, index=False, encoding="utf-8")
        print(f"纳指100：{len(ndx)} 行  {ndx.date.iloc[0].date()} → {ndx.date.iloc[-1].date()}")
    else:
        print("纳指100：已存在")

    if force or not os.path.exists(OUT_FX):
        fx = ak.fx_boc_rate_sina(symbol="美元", start_date="20150101",
                                 end_date=pd.Timestamp.today().strftime("%Y%m%d"))
        fx = fx[["日期", "央行中间价"]].copy()
        fx.columns = ["date", "mid"]
        fx["date"] = pd.to_datetime(fx["date"])
        fx["mid"] = fx["mid"].astype(float) / 100.0     # 每 100 美元报价 → 每 1 美元
        fx = fx.drop_duplicates("date").sort_values("date")
        fx.to_csv(OUT_FX, index=False, encoding="utf-8")
        print(f"USD/CNY：{len(fx)} 行  {fx.date.iloc[0].date()} → {fx.date.iloc[-1].date()}")
    else:
        print("USD/CNY：已存在")


if __name__ == "__main__":
    main()
