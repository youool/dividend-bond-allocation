"""
第 2 关 · 步骤 1：验证每个标的的可用起点

目的：在拉数据之前，先搞清楚每只标的的日线最早能追溯到哪一天。
      红利低波 ETF（512890）上市最晚，它会直接决定可交易版回测的起点。

输出：一张表，每只标的的起始日、结束日、行数。
"""

import sys
import time

import akshare as ak

ETFS = {
    "512890": "红利低波ETF",
    "510300": "沪深300ETF",
    "511010": "国债ETF",
    "511880": "货币ETF",
}

# 长周期版需要指数数据，这里一并探测哪些接口能拿到
INDEX_CANDIDATES = [
    ("sh000300", "沪深300 价格指数"),
    ("sh000922", "中证红利 价格指数"),
    ("H30269", "中证红利低波动 价格指数"),
]


def probe_etf(code, name):
    """尝试拿一只 ETF 的全部历史日线，返回 (起始日, 结束日, 行数) 或错误信息。"""
    try:
        df = ak.fund_etf_hist_em(
            symbol=code,
            period="daily",
            start_date="19900101",
            end_date="20261231",
            adjust="hfq",  # 后复权：分红按再投资处理，红利类策略必须用这个
        )
        if df is None or df.empty:
            return name, code, "空数据", "", "", 0
        first = str(df.iloc[0]["日期"])
        last = str(df.iloc[-1]["日期"])
        return name, code, "OK", first, last, len(df)
    except Exception as e:
        return name, code, "ERROR: " + str(e)[:60], "", "", 0


def probe_index(symbol, name):
    try:
        df = ak.stock_zh_index_daily_em(symbol=symbol)
        if df is None or df.empty:
            return name, symbol, "空数据", "", "", 0
        first = str(df.iloc[0]["date"])
        last = str(df.iloc[-1]["date"])
        return name, symbol, "OK", first, last, len(df)
    except Exception as e:
        return name, symbol, "ERROR: " + str(e)[:60], "", "", 0


def main():
    lines = []
    lines.append("=" * 72)
    lines.append("ETF 可用区间探测（后复权）")
    lines.append("=" * 72)
    lines.append(f"{'名称':<12}{'代码':<9}{'状态':<26}{'起始日':<12}{'结束日':<12}{'行数':>6}")

    starts = {}
    for code, name in ETFS.items():
        name, code, status, first, last, rows = probe_etf(code, name)
        lines.append(f"{name:<12}{code:<9}{status:<26}{first:<12}{last:<12}{rows:>6}")
        if status == "OK":
            starts[code] = first
        time.sleep(0.4)  # 轻微限速，避免接口被判异常

    lines.append("")
    lines.append("=" * 72)
    lines.append("指数接口探测（长周期版用）")
    lines.append("=" * 72)
    for symbol, name in INDEX_CANDIDATES:
        name, symbol, status, first, last, rows = probe_index(symbol, name)
        lines.append(f"{name:<18}{symbol:<10}{status:<26}{first:<12}{last:<12}{rows:>6}")
        time.sleep(0.4)

    lines.append("")
    lines.append("=" * 72)
    if starts:
        latest = max(starts.values())
        earliest = min(starts.values())
        lines.append(f"共同可用起点（取最晚）：{latest}")
        lines.append(f"最早的一只可回溯到：{earliest}")
        lines.append("")
        lines.append("结论：可交易版回测的起点由最晚上市的那只决定。")
    else:
        lines.append("没有任何一只探测成功，检查网络或数据源接口。")
    lines.append("=" * 72)

    report = "\n".join(lines)
    print(report)
    with open("output/probe_report.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        print("selftest ok")
    else:
        main()
