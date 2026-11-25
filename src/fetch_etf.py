"""
第 2 关 · 步骤 2：拉 4 只 ETF 的日线，并自算后复权，落成本地 csv

为什么不用东财的 hfq 直接下载：
  东财 push2 对高频请求会临时封 IP（本次实测触发了 40 分钟级别的限流）。
  新浪接口稳定且不限流，但只给不复权价格，所以分红要自己加回去。

后复权的做法（这是本脚本的核心）：
  持有一份份额，除息日拿到每股分红 D，用 D 再买入同一只 ETF，
  份额变成 (1 + D / 除息前收盘价) 倍。
  把每个除息日的这个倍数连乘，得到累计复权因子 F(t)，
  后复权价 = 原价 × F(t)。这样早期价格不动，后期价格被放大，
  整条序列就等价于"分红全部再投资"的总回报。

  不做这一步会怎样：510300 每年分红 2%–3%、货币 ETF 累计分红 13 元以上，
  不复权回测会把这块收益整段漏掉，红利类策略直接被判死。
"""

import os
import sys
import time

import akshare as ak
import pandas as pd

ETFS = {
    "sh512890": ("512890", "红利低波ETF"),
    "sh510300": ("510300", "沪深300ETF"),
    "sh511010": ("511010", "国债ETF"),
    "sh511880": ("511880", "货币ETF"),
}

OUT_DIR = "data"
COLS = ["date", "open", "high", "low", "close", "volume"]

# 已知的份额折算（拆分）：不是分红，但会让价格腰斩，必须按倍数补回去。
# 512890 在 2021-10-25 做过 1:2 份额折算（价格 1.639 → 0.801），
# akshare 的分红接口把它记成了一条 0 元分红，所以这里显式处理。
# 1:2 折算在数学上等价于「100% 分红再投资」，复权倍数就是 2。
SPLITS = {
    "512890": [("2021-10-25", 2.0)],
    # 513100 在 2022-01-14 做 1:5 份额折算（5.192 → 1.015，中间 1-13 停牌），
    # 新浪不把它记为分红，不补因子会凭空出现一次 −80% 的假暴跌。
    "513100": [("2022-01-14", 5.0)],
}


def get_prices(symbol):
    df = ak.fund_etf_hist_sina(symbol=symbol)
    if df is None or df.empty:
        raise RuntimeError("价格返回空")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return df


def get_dividends(symbol):
    """返回 [(日期, 每股分红), ...]；新浪给的是累计分红，需要做差。"""
    try:
        raw = ak.fund_etf_dividend_sina(symbol=symbol)
    except Exception:
        return []
    if raw is None or raw.empty:
        return []

    raw = raw.copy()
    raw["日期"] = pd.to_datetime(raw["日期"]).dt.strftime("%Y-%m-%d")
    raw = raw.sort_values("日期").reset_index(drop=True)

    events = []
    prev = 0.0
    for _, row in raw.iterrows():
        cum = float(row["累计分红"])
        diff = cum - prev
        prev = cum
        if diff > 1e-9:  # 只保留真正派了钱的
            events.append((row["日期"], diff))
    return events


def adjust_hfq(df, events, splits):
    """
    把不复权价格序列转成后复权序列。
    events: [(除息日, 每股分红), ...]，按日期升序
    splits: [(折算日, 倍数), ...]，份额折算
    返回新的 df 和用到的复权事件明细
    """
    df = df.copy()
    factor = 1.0
    factors = []
    used = []

    # 分红与折算合并成一张事件表：分红存金额，折算存倍数
    ev = [(d, "分红", amt) for d, amt in events] + \
         [(d, "份额折算", mult) for d, mult in splits]
    ev.sort(key=lambda x: x[0])

    idx = 0
    closes = df["close"].tolist()
    dates = df["date"].tolist()

    for i in range(len(df)):
        while idx < len(ev) and ev[idx][0] <= dates[i]:
            d_date, kind, val = ev[idx]
            if kind == "分红":
                prev_close = closes[i - 1] if i > 0 else closes[0]
                mult = (1.0 + val / prev_close) if prev_close > 0 else 1.0
                used.append((d_date, kind, val, prev_close, round(mult, 6)))
            else:
                mult = val
                used.append((d_date, kind, val, 0.0, round(mult, 6)))
            factor *= mult
            idx += 1
        factors.append(factor)

    df["adj_factor"] = factors
    for c in ["open", "high", "low", "close"]:
        df[c] = (df[c] * df["adj_factor"]).round(6)

    return df[COLS + ["adj_factor"]], used


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    force = "--force" in sys.argv

    print(f"{'代码':<8}{'名称':<12}{'行数':>6}  {'起始':<12}{'结束':<12}{'分红次数':>8}  {'末段复权因子':>10}")
    print("-" * 82)

    div_rows = []
    for symbol, (code, name) in ETFS.items():
        path = os.path.join(OUT_DIR, f"{code}.csv")
        if os.path.exists(path) and not force:
            old = pd.read_csv(path)
            print(f"{code:<8}{name:<12}{len(old):>6}  {old['date'].iloc[0]:<12}{old['date'].iloc[-1]:<12}{'已存在':>8}")
            continue

        try:
            prices = get_prices(symbol)
            events = get_dividends(symbol)
            adj, used = adjust_hfq(prices, events, SPLITS.get(code, []))
            adj.to_csv(path, index=False, encoding="utf-8")

            last_factor = adj["adj_factor"].iloc[-1]
            n_div = sum(1 for u in used if u[1] == "分红")
            n_split = len(used) - n_div
            print(f"{code:<8}{name:<12}{len(adj):>6}  {adj['date'].iloc[0]:<12}{adj['date'].iloc[-1]:<12}{n_div:>6}次  折算{n_split}  {last_factor:>8.4f}")

            for d_date, kind, val, prev_close, mult in used:
                div_rows.append({
                    "code": code, "name": name, "日期": d_date, "类型": kind,
                    "数值": val, "除息前收盘": prev_close, "复权倍数": mult,
                })
        except Exception as e:
            print(f"{code:<8}{name:<12}{'失败':>6}  {str(e)[:50]}")
        time.sleep(1.5)

    print("-" * 82)

    if div_rows:
        ddf = pd.DataFrame(div_rows)
        dpath = os.path.join("output", "dividends.csv")
        os.makedirs("output", exist_ok=True)
        ddf.to_csv(dpath, index=False, encoding="utf-8")
        print(f"分红事件明细已存：{dpath}（共 {len(ddf)} 次）")
        print()
        print("按标的汇总：")
        for code, g in ddf.groupby("code"):
            div = g[g["类型"] == "分红"]
            split = g[g["类型"] == "份额折算"]
            parts = [f"{len(div)} 次分红，累计每股 {div['数值'].sum():.3f} 元"]
            if len(split):
                parts.append("份额折算 " + "、".join(f"{r['日期']}×{r['数值']:g}" for _, r in split.iterrows()))
            print(f"  {code}: " + "；".join(parts))

    files = [f for f in os.listdir(OUT_DIR) if f.endswith(".csv")]
    print()
    print(f"data/ 下已有 {len(files)} 个 csv，数据已本地化。")


if __name__ == "__main__":
    main()
