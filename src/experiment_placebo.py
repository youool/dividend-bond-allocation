"""
第 5 关 · 安慰剂检验：估值方向到底有没有信息量

上一轮的「反向微调」对照被证明是有缺陷的：
  C1（估值微调）平均持有红利 32%，C2（反向）平均持有 49%。
  两者的夏普差异主要来自承担了不同的波动，而不是来自「方向对不对」。
  这样比出来的结论站不住。

正确的对照是「暴露完全匹配、只破坏时间对应关系」：
  把 C1 的红利比例序列**随机打乱**，重新贴回同一串日期。
  打乱后：比例分布一模一样（同样的 30/40/50 各占多少个月）、平均暴露一模一样、
  手续费量级一模一样，唯一被破坏的是「低估值时多配、高估值时少配」这个对应关系。

  跑 N 次随机重排，得到夏普的分布。
  如果 C1 的实际夏普落在分布的 95 分位以上，才可以说「估值方向有信息量」；
  落在分布中间，就是随机波动，应该直接承认信号无效。

这是判断一个信号真假的底线检验——它不看你赚了多少，只问：
「随便乱配，能不能也配出这个结果？」

输出：
  output/exp/placebo.csv
  charts/12_安慰剂检验.png
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt  # noqa: E402
import signals as sg  # noqa: E402

EXP_DIR = "output/exp"
DIVIDEND_PATH = {"long": "data/index/红利低波_全收益.csv", "trade": "data/512890.csv"}
N_SHUFFLE = 100
SEED = 20261124


def price_pct_series(ver):
    s = sg.load_close(DIVIDEND_PATH[ver])
    pct = s.rolling(1260, min_periods=252).apply(lambda w: float((w[:-1] < w[-1]).mean()), raw=True)
    return pct.dropna()


def tilt_ratios(dates, pct):
    aligned = pct.reindex(pct.index.union(pd.DatetimeIndex(dates))).ffill().reindex(pd.DatetimeIndex(dates))
    out = []
    for p in aligned.values:
        if np.isnan(p):
            out.append(0.40)
        elif p < 0.30:
            out.append(0.50)
        elif p <= 0.70:
            out.append(0.40)
        else:
            out.append(0.30)
    return np.array(out)


def table_from_ratios(dates, ratios):
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "w_dividend": ratios, "w_broad": 0.0,
        "w_bond": 1.0 - ratios, "w_cash": 0.0,
    })


def main():
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs("charts", exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows, dist = [], {}

    for ver in ["long", "trade"]:
        cfg = bt.VERSIONS[ver]
        ins_end = pd.Timestamp(cfg["insample_end"])
        dates = pd.read_csv(f"output/weights_{ver}.csv", parse_dates=["date"])["date"].values
        ratios = tilt_ratios(dates, price_pct_series(ver))

        op, cl = bt.load_prices(cfg["prices"])

        # 真实信号
        nav, _ = bt.simulate(table_from_ratios(dates, ratios), op, cl, cfg["exec_price"])
        real_all = bt.metrics(nav)["夏普比率"]
        real_out = bt.metrics(nav[nav.index > ins_end])["夏普比率"]
        real_ann = bt.metrics(nav[nav.index > ins_end])["年化收益率"]

        # 随机重排
        shuffles = []
        for i in range(N_SHUFFLE):
            perm = rng.permutation(ratios)
            nv, _ = bt.simulate(table_from_ratios(dates, perm), op, cl, cfg["exec_price"])
            shuffles.append({
                "全夏普": bt.metrics(nv)["夏普比率"],
                "外夏普": bt.metrics(nv[nv.index > ins_end])["夏普比率"],
                "外年化": bt.metrics(nv[nv.index > ins_end])["年化收益率"],
                "红利均值": float(perm.mean()),
            })
            if (i + 1) % 25 == 0:
                print(f"  {ver} 已完成 {i+1}/{N_SHUFFLE} 次重排")

        sdf = pd.DataFrame(shuffles)
        dist[ver] = (sdf, real_all, real_out, real_ann)

        p_all = float((sdf["全夏普"] >= real_all).mean())
        p_out = float((sdf["外夏普"] >= real_out).mean())

        rows.append({
            "版本": ver,
            "真实_全年化": bt.metrics(nav)["年化收益率"], "真实_全夏普": real_all,
            "真实_外年化": real_ann, "真实_外夏普": real_out,
            "随机_全夏普均值": sdf["全夏普"].mean(), "随机_全夏普_95分位": sdf["全夏普"].quantile(0.95),
            "随机_外夏普均值": sdf["外夏普"].mean(), "随机_外夏普_95分位": sdf["外夏普"].quantile(0.95),
            "真实_全夏普_超过随机的比例": p_all, "真实_外夏普_超过随机的比例": p_out,
            "随机_红利均值": sdf["红利均值"].mean(),
        })

    out = pd.DataFrame(rows)
    out.to_csv(f"{EXP_DIR}/placebo.csv", index=False, encoding="utf-8")

    for ver in ["long", "trade"]:
        sdf, real_all, real_out, real_ann = dist[ver]
        r = out[out["版本"] == ver].iloc[0]
        print()
        print("=" * 92)
        print(f"{bt.VERSIONS[ver]['label']}　{N_SHUFFLE} 次随机重排（暴露分布完全相同，只打乱时间）")
        print("=" * 92)
        print(f"真实信号：    全年化 {r['真实_全年化']:>6.1%}　全夏普 {real_all:>5.2f}　"
              f"外年化 {real_ann:>6.1%}　外夏普 {real_out:>5.2f}")
        print(f"随机重排均值：              全夏普 {r['随机_全夏普均值']:>5.2f}　              "
              f"　　　　外夏普 {r['随机_外夏普均值']:>5.2f}")
        print(f"随机重排95分位：            全夏普 {r['随机_全夏普_95分位']:>5.2f}　              "
              f"　　　　外夏普 {r['随机_外夏普_95分位']:>5.2f}")
        print()
        p1, p2 = r["真实_全夏普_超过随机的比例"], r["真实_外夏普_超过随机的比例"]
        print(f"真实夏普在随机分布中的位置：全区间 {p1:.0%}　样本外 {p2:.0%}")
        if p1 >= 0.95 and p2 >= 0.95:
            print("→ 两个区间都超过 95% 分位，估值方向有信息量，可以保留。")
        elif p1 >= 0.95 or p2 >= 0.95:
            print("→ 只有一个区间通过，证据不充分。报告里只能写「方向可能有效，但未达稳健标准」。")
        else:
            print("→ 落在随机分布中间，**没有证据支持估值方向有效**。")
            print("  这时正确的做法是：不把它当 alpha 讲，只把「30/40/50 的窄幅区间」当成风险预算。")

    plot(dist)


def plot(dist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=140)
    for ax, ver, title in zip(axes, ["long", "trade"],
                              ["长周期版（2015 起）", "可交易版（2019 起）"]):
        sdf, real_all, real_out, _ = dist[ver]
        ax.hist(sdf["外夏普"], bins=18, color="#B8C9E0", edgecolor="white",
                label="随机重排 100 次")
        ax.axvline(real_out, color="#D9534F", linewidth=2.0, label=f"真实信号 {real_out:.2f}")
        ax.axvline(sdf["外夏普"].quantile(0.95), color="#1D9E75", linestyle="--",
                   linewidth=1.4, label=f"95% 分位 {sdf['外夏普'].quantile(0.95):.2f}")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("样本外夏普")
        ax.set_ylabel("出现次数")
        ax.grid(alpha=0.2, linestyle="--")
        ax.legend(fontsize=7.5)
    fig.suptitle("安慰剂检验：真实信号 vs 把比例随机打乱（暴露完全相同）", fontsize=12)
    fig.tight_layout()
    p = "charts/12_安慰剂检验.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"\n图已保存：{p}")


if __name__ == "__main__":
    main()
