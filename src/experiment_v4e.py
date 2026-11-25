"""
自检（v4e）· 资产端的最后两组对比：久期要不要混、权益要不要分散

上一轮结论：十年国债（511260）在 2019–2026 全面优于五年（511010）——
  组合年化 7.8% vs 7.2%，最大回撤同为 −6.3%，Calmar 1.24 vs 1.14。
但这是利率单边下行期给的。真把钱全押十年，等于赌利率继续下行。

这一轮回答两个「要不要」：
  1. 久期要不要混？  60% 里一半五年一半十年
  2. 权益要不要分散？ 40% 权益里拿 30% 给沪深300

反面教材一起放：三十年国债（511090）单资产年化 7.8% 很诱人，
  但把组合回撤从 −5.0% 拉到 −8.3%，Calmar 反而更低。

口径（这次特意做干净）：
  同一版本内所有方案从「最晚可得起点」起跑。十年国债 2017-08 才有数据，
  所以长周期版的久期对比只能从 2017-09 开始；2015–2017 只有五年国债，
  拿它比十年不公平，直接不比。
  替换债券时把整行数据（含 open）一起写进临时文件，避免执行价被填成空。
  长周期版的权益腿必须用全收益指数，可交易版才用 512890——两版口径本来就不同。

输出：
  output/exp/v4_asset.csv
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt  # noqa: E402

EXP_DIR = "output/exp"
TMP = "data/_tmp_bond.csv"

BASE = {
    "long": {"dividend": "data/index/红利低波_全收益.csv", "broad": "data/index/沪深300_全收益.csv",
             "bond": "data/511010.csv", "cash": "data/511880.csv"},
    "trade": {"dividend": "data/512890.csv", "broad": "data/510300.csv",
              "bond": "data/511010.csv", "cash": "data/511880.csv"},
}
P5, P10, P30 = "data/511010.csv", "data/511260.csv", "data/511090.csv"


def load(path):
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")["close"].sort_index()


def synth(paths_weights):
    """合成虚拟债券（日度再平衡），返回 date/open/close 表，open 用 close 代替。"""
    rets, wsum = None, 0.0
    for path, wt in paths_weights:
        r = load(path).pct_change()
        rets = r * wt if rets is None else rets.add(r * wt, fill_value=0.0)
        wsum += wt
    px = (1 + (rets / wsum).fillna(0)).cumprod()
    return pd.DataFrame({"date": px.index, "open": px.values, "close": px.values})


def prep_bond(spec, s):
    """把债券数据截到 s 之后写成临时文件，保留 open/close 两列。"""
    if isinstance(spec, pd.DataFrame):
        df = spec.copy()
    else:
        df = pd.read_csv(spec, parse_dates=["date"])
    df = df[df["date"] >= pd.Timestamp(s)].sort_values("date")
    if len(df) < 30:
        raise RuntimeError("债券数据不足")
    df[["date", "open", "close"]].to_csv(TMP, index=False, encoding="utf-8")
    return df["date"].iloc[0]


def cleanup():
    """临时文件必须在 finally 里删，否则中途抛异常会把 data/ 污染成
    「下次运行读到上一次的半截数据」，而且不会有任何提示。"""
    if os.path.exists(TMP):
        os.remove(TMP)


def run(ver, spec, start, wmod=None):
    s0 = pd.Timestamp(start)
    # 先按 start 截债券，再按债券实际首日对齐权重，保证三者同起点
    tmp_start = max(s0, pd.Timestamp("1990-01-01"))
    first = prep_bond(spec, tmp_start)
    s = max(s0, first)

    op, cl = bt.load_prices(dict(BASE[ver], bond=TMP))
    cl, op = cl[cl.index >= s], op[op.index >= s]

    w = pd.read_csv(f"output/alloc_{ver}_vol08.csv", parse_dates=["date"])
    w = w[w["date"] >= s]
    if wmod:
        w = wmod(w)
    if len(w) < 6:
        raise RuntimeError("权重期数不足")
    nav, _ = bt.simulate(w, op, cl, bt.VERSIONS[ver]["exec_price"], start=s)
    return nav


def main():
    os.makedirs(EXP_DIR, exist_ok=True)
    rows = []

    def add(ver, name, nav):
        if nav.nunique() < 10:
            print(f"跳过 {ver}/{name}: 净值几乎不动")
            return
        m = bt.metrics(nav)
        rows.append({"版本": ver, "方案": name, "起点": nav.index[0].date(),
                     "年化": m["年化收益率"], "回撤": m["最大回撤"],
                     "波动": m["年化波动"], "Calmar": m["Calmar比率"],
                     "夏普": m["夏普比率"]})

    def mix_eq(w):
        w = w.copy()
        eq = w["w_dividend"].copy()
        w["w_dividend"] = (eq * 0.7).round(4)
        w["w_broad"] = (eq * 0.3).round(4)
        w["w_bond"] = (1 - eq).round(4)
        return w

    for ver in ["long", "trade"]:
        start = "2017-09-01" if ver == "long" else "2019-02-01"
        plans = [
            ("债 五年100%", P5, None),
            ("债 十年100%", P10, None),
            ("债 五年50+十年50", synth([(P5, 0.5), (P10, 0.5)]), None),
            ("债 十年 + 权益红利70/300_30", P10, mix_eq),
        ]
        for name, spec, mod in plans:
            try:
                add(ver, name, run(ver, spec, start, mod))
            except Exception as e:
                print(f"跳过 {ver}/{name}: {type(e).__name__} {e}")

        for name, spec in [("债 五年100%", P5), ("债 十年100%", P10), ("债 三十年100%", P30)]:
            try:
                add(ver, f"{name}［2023-07起］", run(ver, spec, "2023-07-01"))
            except Exception as e:
                print(f"跳过 {ver}/{name} 2023: {type(e).__name__} {e}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{EXP_DIR}/v4_asset.csv", index=False, encoding="utf-8")

    for ver in ["long", "trade"]:
        sub = df[(df["版本"] == ver) & (~df["方案"].str.contains("［"))]
        sub23 = df[(df["版本"] == ver) & (df["方案"].str.contains("［"))]
        print()
        print("=" * 104)
        print(f"{bt.VERSIONS[ver]['label']}：资产端对比（策略规则一个字没改，只换持有什么）")
        print("=" * 104)
        print(f"{'方案':<30}{'起点':<12}{'年化':>8}{'回撤':>9}{'波动':>8}{'Calmar':>8}{'夏普':>7}")
        print("-" * 104)
        for _, r in sub.iterrows():
            print(f"{r['方案']:<30}{str(r['起点']):<12}{r['年化']:>8.1%}{r['回撤']:>9.1%}"
                  f"{r['波动']:>8.1%}{r['Calmar']:>8.2f}{r['夏普']:>7.2f}")
        if not sub23.empty:
            print("\n共同区间 2023-07 起（三只都在，含三十年）：")
            for _, r in sub23.iterrows():
                print(f"{r['方案'].replace('［2023-07起］',''):<30}{str(r['起点']):<12}"
                      f"{r['年化']:>8.1%}{r['回撤']:>9.1%}{r['波动']:>8.1%}"
                      f"{r['Calmar']:>8.2f}{r['夏普']:>7.2f}")



if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()
