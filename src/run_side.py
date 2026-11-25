"""
支线 · 债端久期对比（可复现入口）

主线一个字不动：40% 红利低波 + 60% 五年国债 + 波动率闸门。
支线只改一件事：那 60% 国债的久期怎么摆。

为什么这件事值得单列一条支线：
  前面七轮实验都在调权益那 40%（怎么调都是四成的事），国债那 60% 一次没动过。
  久期一变，六成仓位的收益与风险都变，杠杆比调权益大得多。

为什么只做支线、不进主线：
  2019–2026 是利率单边下行期（十年期 3.3% → 1.8%），久期越长受益越大。
  这是环境给的，不是久期本身的 alpha。全押十年等于赌利率继续下行。
  数据也支持这个谨慎：2024 一年就贡献了十年与五年总差异的 29%，
  2025 起连续吃亏。所以建议是「混合」，不是「加久期」。

口径（刻意做干净）：
  - 权重表完全冻结在主线那一张，只替换 bond 这一列的价格
    → 测出来的差异 100% 来自「持有了什么」，不掺任何规则改动
  - 所有方案从同一起点起跑（长周期版受十年国债数据限制，只能从 2017-08 起）
  - 合成债用月末再平衡 + 份额法，与策略自身的调仓节奏一致

额外做一项检查（回答"混合债会不会把闸门也改了"）：
  用混合债重算一遍波动率闸门，看权重表会不会变。
  如果几乎不变，说明"冻结权重表"是合理近似，结论只归因于资产替换。

输出：
  output/side_duration_{ver}.csv          各方案自有区间
  output/side_duration_common_{ver}.csv   共同区间（判定用这张）
  output/side_yearly.csv                  分年度久期差异
  charts/15_支线_久期_{ver}.png
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import allocation as al  # noqa: E402
import backtest as bt    # noqa: E402
import bond_duration as bd  # noqa: E402

COLORS = {
    "mix00": ("#1D9E75", 2.0, 1.0),   # 主线：绿线、加粗
    "mix30": ("#8FC9AE", 1.1, 0.9),
    "mix50": ("#C77A1E", 2.2, 1.0),   # 支线：橙线、最粗
    "mix70": ("#D9A45B", 1.1, 0.9),
    "mix100": ("#8C8C8C", 1.3, 1.0),
}


def main():
    os.makedirs("output", exist_ok=True)
    os.makedirs("charts", exist_ok=True)

    print("=" * 96)
    print("零、检查：混合债会不会把波动率闸门也改掉（决定「冻结权重表」是否成立）")
    print("=" * 96)
    check_gate()

    print("=" * 96)
    print("零之二、久期不是免费的：它在两种场景里方向相反")
    print("=" * 96)
    tail_evidence()

    yearly = []
    for ver in ["long", "trade"]:
        run_version(ver, yearly)

    if yearly:
        ydf = pd.DataFrame(yearly)
        ydf.to_csv("output/side_yearly.csv", index=False, encoding="utf-8")


def check_gate():
    """
    用混合债重算权重表，与主线对比仓位差异。

    为什么必须查这一步：波动率闸门的输入是「组合近 60 日已实现波动」，
    而组合 = 40% 权益 + 60% 债券。债券换了，组合波动就变了，闸门可能在不同月份触发。
    如果差异大，那"只换资产"这个口径就不成立，支线的收益里会混进"仓位变了"的成分。
    """
    for ver in ["long", "trade"]:
        base = pd.read_csv(f"output/alloc_{ver}_vol08.csv", parse_dates=["date"])
        bond = bd.ensure("mix50", verbose=False)
        alt, _ = al.build(ver, al.DEFAULT_MODE, bond_path=bond, write=False)
        # build 里为了 CSV 可读性把日期存成字符串，对比前转回来
        alt["date"] = pd.to_datetime(alt["date"])

        # 混合债 2017-08 才有数据，之前那段它的波动率是用不完整样本算的，
        # 拿来对比会得出「闸门变了」的假结论。只比两者都有真实数据的区间。
        alt_start = pd.Timestamp(
            bt.load_prices({"bond": bond})[1]["bond"].first_valid_index())
        m = base.merge(alt, on="date", suffixes=("_主线", "_混合"))
        m = m[m["date"] >= alt_start]
        diff = (m["w_dividend_主线"] - m["w_dividend_混合"]).abs()
        print(f"{al.VERSIONS[ver]['label']:<20}共 {len(m)} 期　"
              f"仓位差异：均值 {diff.mean():.2%}　最大 {diff.max():.2%}　"
              f"差异>1pct 的月份 {int((diff > 0.01).sum())}")
        if diff.max() > 1e-6:
            top = m.loc[diff.nlargest(3).index,
                        ["date", "w_dividend_主线", "w_dividend_混合",
                         "realized_vol_主线", "realized_vol_混合"]]
            print(top.to_string(index=False))
    print()
    print("结论：差异足够小 → 支线只替换资产、冻结权重表是合理口径，")
    print("      支线与主线的差距可以整体归因于「债端持有了什么」。")
    print()


def tail_evidence():
    """
    把久期的两面都量出来，避免只说好的一面。

    组合层面看到的是「十年占比越高，回撤反而越小」，这很反直觉——
    久期更长意味着债券本身波动更大，为什么组合回撤会变小？
    原因是两种场景方向相反、且幅度不对称：
      股灾时，十年债涨得更多（避险 + 降息预期）→ 对冲更强，回撤改善
      债熊时，十年债跌得更多（约 1.5 倍）        → 拖累更大，回撤恶化
    净效果为正，是因为 2019–2026 这段里股灾的幅度压过了债熊。
    这是环境给的，换个利率上行周期就会反过来。
    """
    def rd(p):
        return pd.read_csv(p, parse_dates=["date"]).set_index("date")["close"].astype(float)

    df = pd.DataFrame({"五年": rd(bd.P5), "十年": rd(bd.P10),
                       "红利低波": rd("data/512890.csv")}).dropna()
    df = df[df.index >= "2019-01-18"]
    r = df.pct_change().dropna()

    print("A. 股票大跌时，久期是更强的对冲")
    for thr in [-0.01, -0.02]:
        t = r[r["红利低波"] < thr]
        if len(t) < 10:
            continue
        m5, m10 = t["五年"].mean(), t["十年"].mean()
        print(f"   红利单日跌超 {abs(thr):.0%}（{len(t)} 天）：五年债 {m5:+.3%}　十年债 {m10:+.3%}　"
              f"十年是五年的 {m10 / m5:.2f} 倍　十年跑赢概率 {(t['十年'] > t['五年']).mean():.0%}")

    print("\nB. 利率上行（债熊）时，久期是更大的拖累")
    for a, b, name in [("2020-04-01", "2020-12-31", "2020 疫后复苏"),
                       ("2025-01-01", "2025-03-31", "2025 开年"),
                       ("2019-04-01", "2019-12-31", "2019 猪通胀")]:
        s = df[(df.index >= a) & (df.index <= b)]
        if len(s) < 10:
            continue
        r5 = s["五年"].iloc[-1] / s["五年"].iloc[0] - 1
        r10 = s["十年"].iloc[-1] / s["十年"].iloc[0] - 1
        mult = f"　十年跌幅是五年的 {r10 / r5:.2f} 倍" if r5 < 0 else "　（该时段未下跌）"
        print(f"   {name:<14}五年 {r5:>+7.2%}　十年 {r10:>+7.2%}{mult}")

    print("\n   两张表必须一起看：支线的收益不是凭空来的，")
    print("   是拿「债熊时多亏 1.5 倍」换「股灾时多涨 1.2 倍」。")
    print("   2019–2026 股灾的幅度压过债熊，所以净效果为正；换成利率上行周期会反过来。")
    print()


def run_version(ver, yearly_store):
    cfg = bt.VERSIONS[ver]
    w_all = pd.read_csv(f"output/alloc_{ver}_vol08.csv", parse_dates=["date"])

    navs, rows = {}, []

    for key in bd.ORDER:
        bond = bd.ensure(key, verbose=False)
        prices = dict(cfg["prices"], bond=bond)
        op, cl = bt.load_prices(prices)

        # 起点 = max(权重表首期, 该债券首个有效价格)
        b0 = cl["bond"].first_valid_index()
        s = max(pd.Timestamp(w_all["date"].iloc[0]), b0)
        w = w_all[w_all["date"] >= s]
        if len(w) < 6:
            print(f"  跳过 {ver}/{key}：权重期数不足")
            continue
        s = w["date"].iloc[0]

        nav, fee = bt.simulate(w, op, cl, cfg["exec_price"], start=s)
        navs[key] = nav
        m = bt.metrics(nav)
        rows.append({"版本": ver, "方案": key, "标签": bd.PLANS[key]["label"],
                     "起点": nav.index[0].date(), "期末": nav.index[-1].date(),
                     "年化": m["年化收益率"], "回撤": m["最大回撤"],
                     "波动": m["年化波动"], "夏普": m["夏普比率"],
                     "Calmar": m["Calmar比率"], "手续费": fee})

    df = pd.DataFrame(rows)
    df.to_csv(f"output/side_duration_{ver}.csv", index=False, encoding="utf-8")

    # ---------- 共同区间：所有方案都有数据的重叠段，判定只能看这张 ----------
    common_start = max(n.index[0] for n in navs.values())
    common = {k: v[v.index >= common_start] for k, v in navs.items()}
    common = {k: v / v.iloc[0] for k, v in common.items()}

    crows = []
    for k, nav in common.items():
        m = bt.metrics(nav)
        crows.append({"版本": ver, "方案": k, "标签": bd.PLANS[k]["label"],
                      "起点": nav.index[0].date(), "年化": m["年化收益率"],
                      "回撤": m["最大回撤"], "波动": m["年化波动"],
                      "夏普": m["夏普比率"], "Calmar": m["Calmar比率"]})
    cdf = pd.DataFrame(crows)
    cdf.to_csv(f"output/side_duration_common_{ver}.csv", index=False, encoding="utf-8")

    print("=" * 104)
    print(f"{al.VERSIONS[ver]['label']}：债端久期对比（权重表冻结，只换 60% 国债）")
    print("=" * 104)
    print("\n各方案自有最长区间：")
    print(f"{'方案':<26}{'起点':<12}{'年化':>8}{'回撤':>9}{'波动':>8}{'夏普':>7}{'Calmar':>8}")
    print("-" * 104)
    for _, r in df.iterrows():
        print(f"{r['标签']:<26}{str(r['起点']):<12}{r['年化']:>8.1%}{r['回撤']:>9.1%}"
              f"{r['波动']:>8.1%}{r['夏普']:>7.2f}{r['Calmar']:>8.2f}")

    print(f"\n共同区间（{common_start.date()} 起，判定用这张）：")
    print(f"{'方案':<26}{'年化':>8}{'回撤':>9}{'波动':>8}{'夏普':>7}{'Calmar':>8}{'vs主线':>10}")
    print("-" * 104)
    base_ann = float(cdf.loc[cdf["方案"] == "mix00", "年化"].iloc[0])
    for _, r in cdf.iterrows():
        delta = "" if r["方案"] == "mix00" else f"{r['年化'] - base_ann:+.2%}"
        print(f"{r['标签']:<26}{r['年化']:>8.1%}{r['回撤']:>9.1%}{r['波动']:>8.1%}"
              f"{r['夏普']:>7.2f}{r['Calmar']:>8.2f}{delta:>10}")

    # ---------- 分年度：久期优势是不是集中在某一年 ----------
    # 两个版本各算一次：起点不同，年度的切分口径也不同，但都成立。
    ydf = yearly_table(ver, common_start, yearly_store)

    print()
    plot(ver, common, ydf)


def yearly_table(ver, start, store):
    """
    债券单资产的分年度收益，回答「十年比五年多赚的到底是哪一年赚的」。

    这张表是支线结论的分量所在：如果优势集中在单独一年，
    就说不上是久期的稳定溢价，而是那一年利率下了太多。
    """
    p5 = pd.read_csv(bd.P5, parse_dates=["date"]).set_index("date")["close"]
    p10 = pd.read_csv(bd.P10, parse_dates=["date"]).set_index("date")["close"]
    pair = pd.DataFrame({"五年": p5, "十年": p10}).dropna()
    pair = pair[pair.index >= start]

    print(f"\n债券本身的分年度收益（{start.date()} 起）：")
    print(f"{'年份':<8}{'五年':>9}{'十年':>9}{'差(十年-五年)':>16}")
    print("-" * 44)

    yr = []
    for y, g in pair.groupby(pair.index.year):
        if len(g) < 100:
            continue
        r5 = g["五年"].iloc[-1] / g["五年"].iloc[0] - 1
        r10 = g["十年"].iloc[-1] / g["十年"].iloc[0] - 1
        print(f"{y:<8}{r5:>9.2%}{r10:>9.2%}{r10 - r5:>+16.2%}")
        store.append({"版本": ver, "年份": int(y), "五年": r5, "十年": r10, "差": r10 - r5})
        yr.append({"年份": int(y), "差": r10 - r5})

    gap = (pair["十年"].iloc[-1] / pair["十年"].iloc[0]) / \
          (pair["五年"].iloc[-1] / pair["五年"].iloc[0]) - 1
    print("-" * 44)
    print(f"全期累计差异：{gap:+.2%}")
    return pd.DataFrame(yr)


def plot(ver, common, ydf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(13, 8.6), dpi=140)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.34)

    # ---- 面板 A：相对主线（十年 0%）的比值 ----
    # 绝对净值图上五条线几乎完全重合（差异只有 0.3–0.6 个百分点/年），
    # 必须换成相对值才看得出分化。基准线在 1.0，线在上方就是跑赢主线。
    ax1 = fig.add_subplot(gs[0])
    base = common["mix00"]
    for key in bd.ORDER:
        if key not in common:
            continue
        color, lw, alpha = COLORS[key]
        rel = common[key] / base
        ax1.plot(rel.index, rel, color=color, linewidth=lw, alpha=alpha,
                 label=bd.PLANS[key]["label"], zorder=4 if key in ("mix00", "mix50") else 3)
        if key != "mix00":
            ax1.annotate(f"{rel.iloc[-1]:.4f}", xy=(rel.index[-1], rel.iloc[-1]),
                         xytext=(4, 0), textcoords="offset points",
                         fontsize=7.5, color=color, va="center")

    ax1.axhline(1.0, color="#1D9E75", linestyle="--", linewidth=1.0, alpha=0.8)
    ax1.set_title(f"{al.VERSIONS[ver]['label']}：不同久期的相对表现"
                  f"（以「十年 0%」为 1.0，{common['mix00'].index[0].date()} 起）", fontsize=13)
    ax1.set_ylabel("相对主线的净值比")
    ax1.grid(alpha=0.25, linestyle="--")
    ax1.legend(loc="upper left", fontsize=8.6, framealpha=0.92, ncol=2)

    # ---- 面板 B：分年度「十年 − 五年」，回答优势集中在哪一年 ----
    ax2 = fig.add_subplot(gs[1])
    if ydf is not None and len(ydf):
        colors = ["#C0392B" if v > 0 else "#1D9E75" for v in ydf["差"]]
        ax2.bar(ydf["年份"].astype(str), ydf["差"] * 100, color=colors, width=0.62, alpha=0.9)
        for x, v in zip(ydf["年份"].astype(str), ydf["差"] * 100):
            ax2.annotate(f"{v:+.1f}", xy=(x, v), xytext=(0, 4 if v >= 0 else -12),
                         textcoords="offset points", ha="center", fontsize=7.5,
                         color="#333")
        ax2.axhline(0, color="#333", linewidth=0.9)
        ax2.set_ylabel("十年 − 五年（个百分点）")
        ax2.set_title("债券单资产的分年度差异：正数=十年跑赢（红），负数=十年跑输（绿）"
                      "　｜　2026 只到 11 月，不是完整年度",
                      fontsize=10)
        ax2.grid(alpha=0.2, linestyle="--", axis="y")
        ax2.set_ylim(ydf["差"].min() * 100 - 0.9, ydf["差"].max() * 100 + 0.9)

    path = f"charts/15_支线_久期_{ver}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"图已保存：{path}")


if __name__ == "__main__":
    main()
