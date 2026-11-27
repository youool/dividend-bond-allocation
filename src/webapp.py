"""
本地 Web 看板：在浏览器里改参数，由 Python 端的同一套回测内核重新计算。

为什么不用 Flask：项目依赖清单里没有 Web 框架，为了课堂演示再装一个包，
换机器就要重新配环境。这里只用标准库 http.server，零依赖、断网可用，
但它确实是一个带后端的应用 —— 页面上的每一条曲线都来自 /api/run 的实时计算，
不是预先画好的图片。

启动：
    python src/webapp.py                # 默认 127.0.0.1:8777
    python src/webapp.py --port 9000
然后浏览器打开提示的地址。

接口：
    GET /                          页面
    GET /api/run?ver=trade&vol=8&cap=50&base=40&window=60
        返回归一化净值、回撤、仓位、指标、本次计算耗时
"""

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import allocation as al  # noqa: E402
import backtest as bt  # noqa: E402

CACHE = {}
CACHE_LOCK = threading.Lock()


def _round_list(series, nd=4):
    return [None if pd.isna(v) else round(float(v), nd) for v in series]


def run_backtest(ver, vol_target, max_eq, min_eq, base_eq, window):
    """跑一次完整回测，返回给前端的一切。不落盘，不覆盖主线产物。"""
    if ver not in al.VERSIONS:
        raise ValueError(f"未知版本 {ver}")

    cfg = al.VERSIONS[ver]
    alloc, _ = al.build(ver, al.DEFAULT_MODE,
                        base_equity=base_eq,
                        vol_window=window,
                        write=False,
                        vol_target=vol_target,
                        max_equity=max_eq,
                        min_equity=min_eq)

    op, cl = bt.load_prices(cfg["prices"])
    nav, fee = bt.simulate(alloc, op, cl, cfg["exec_price"])

    frame = pd.DataFrame({"strategy": nav})
    for col, asset in [("bm_broad", "broad"), ("bm_dividend", "dividend")]:
        px = cl[asset].reindex(frame.index).ffill()
        frame[col] = (px / px.iloc[0]).values
    frame = frame.ffill().dropna()

    ins_end = pd.Timestamp(bt.VERSIONS[ver]["insample_end"])
    segs = {}
    for name, sub in [("全区间", frame),
                      ("样本内", frame[frame.index <= ins_end]),
                      ("样本外", frame[frame.index > ins_end])]:
        if len(sub) < 20:
            continue
        segs[name] = {
            "策略": bt.metrics(sub["strategy"]),
            "红利低波买入持有": bt.metrics(sub["bm_dividend"]),
            "沪深300买入持有": bt.metrics(sub["bm_broad"]),
        }

    dd = frame["strategy"] / frame["strategy"].cummax() - 1.0
    wdates = pd.to_datetime(alloc["date"])

    return {
        "meta": {
            "ver": ver,
            "label": al.VERSIONS[ver]["label"],
            "vol_target": vol_target,
            "max_equity": max_eq,
            "min_equity": min_eq,
            "base_equity": base_eq,
            "window": window,
            "fee_total": round(float(fee), 6),
            "months": int(len(alloc)),
            "avg_weight": round(float(alloc["w_dividend"].mean()), 4),
            "shrink_months": int((alloc["scale"] < 1).sum()),
            "start": str(frame.index[0].date()),
            "end": str(frame.index[-1].date()),
            "days": int(len(frame)),
        },
        "series": {
            "dates": [d.strftime("%Y-%m-%d") for d in frame.index],
            "strategy": _round_list(frame["strategy"]),
            "bm_dividend": _round_list(frame["bm_dividend"]),
            "bm_broad": _round_list(frame["bm_broad"]),
            "drawdown": _round_list(dd),
            "weight_dates": [d.strftime("%Y-%m-%d") for d in wdates],
            "weights": _round_list(alloc["w_dividend"]),
            "vols": _round_list(alloc["realized_vol"]),
        },
        "metrics": segs,
    }


def cached_run(params):
    key = (params["ver"], round(params["vol"], 4), round(params["cap"], 4),
           round(params["min"], 4), round(params["base"], 4), int(params["window"]))
    with CACHE_LOCK:
        if key in CACHE:
            return CACHE[key], True
    t0 = time.perf_counter()
    data = run_backtest(params["ver"], params["vol"] / 100.0, params["cap"] / 100.0,
                        params["min"] / 100.0, params["base"] / 100.0, params["window"])
    data["meta"]["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    with CACHE_LOCK:
        CACHE[key] = data
    return data, False


PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>红利低波 + 国债 · 参数看板</title>
<style>
  :root{
    --bg:#f6f7f9; --panel:#fff; --ink:#1a1d21; --muted:#6b7280; --line:#e3e6ea;
    --green:#1d9e75; --amber:#ba7517; --gray:#9aa1a9; --red:#d9534f; --blue:#3b82f6;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font-family:-apple-system,"Segoe UI","Microsoft YaHei",system-ui,sans-serif;}
  .wrap{max-width:1120px;margin:0 auto;padding:26px 22px 60px}
  h1{font-size:21px;margin:0 0 4px;letter-spacing:.2px}
  .sub{color:var(--muted);font-size:13px;margin-bottom:20px}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-bottom:14px}
  .ctrl{display:grid;grid-template-columns:repeat(4,1fr);gap:18px}
  .ctrl label{display:block;font-size:12px;color:var(--muted);margin-bottom:6px}
  .ctrl .val{font-size:15px;font-weight:600;color:var(--ink);margin-left:6px}
  input[type=range]{width:100%;accent-color:var(--green)}
  .seg{display:inline-flex;border:1px solid var(--line);border-radius:7px;overflow:hidden;margin-bottom:14px}
  .seg button{border:0;background:#fff;padding:7px 16px;font-size:13px;cursor:pointer;color:var(--muted)}
  .seg button.on{background:var(--green);color:#fff;font-weight:600}
  .kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:12px}
  .kpi{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:11px 13px}
  .kpi .k{font-size:11px;color:var(--muted);margin-bottom:5px}
  .kpi .v{font-size:19px;font-weight:650;letter-spacing:-.3px}
  .kpi .n{font-size:11px;color:var(--muted);margin-top:3px}
  .chartbox{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 10px 6px;margin-bottom:14px}
  .chartbox h3{margin:2px 0 6px;padding-left:8px;font-size:13px;font-weight:600;color:#333}
  svg{display:block;width:100%;height:auto}
  .legend{display:flex;gap:16px;padding:0 10px 6px;font-size:12px;color:var(--muted);flex-wrap:wrap}
  .legend i{display:inline-block;width:16px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
  table{width:100%;border-collapse:collapse;font-size:12.5px}
  th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right}
  th:first-child,td:first-child{text-align:left}
  th{color:var(--muted);font-weight:500;font-size:11.5px}
  .stamp{font-size:12px;color:var(--muted);margin-top:10px}
  .stamp b{color:var(--green)}
  .err{color:var(--red);font-size:13px}
</style>
</head>
<body>
<div class="wrap">
  <h1>红利低波 + 国债 · 参数看板</h1>
  <div class="sub">拖动参数，Python 后端用同一套回测内核重新计算，曲线与指标实时更新。不是预渲染的图片。</div>

  <div class="panel">
    <div class="seg" id="seg">
      <button data-ver="trade" class="on">可交易版 2019 起</button>
      <button data-ver="long">长周期版 2015 起</button>
    </div>
    <div class="ctrl">
      <div><label>目标波动 <span class="val" id="v-vol">8%</span></label>
        <input type="range" id="vol" min="4" max="16" step="1" value="8"></div>
      <div><label>权益上限 <span class="val" id="v-cap">50%</span></label>
        <input type="range" id="cap" min="30" max="60" step="1" value="50"></div>
      <div><label>基准仓位 <span class="val" id="v-base">40%</span></label>
        <input type="range" id="base" min="20" max="60" step="1" value="40"></div>
      <div><label>波动窗口 <span class="val" id="v-win">60 日</span></label>
        <input type="range" id="win" min="20" max="120" step="10" value="60"></div>
    </div>
  </div>

  <div class="kpis" id="kpis"></div>

  <div class="chartbox">
    <h3>归一化净值（起点 = 1.0）</h3>
    <div class="legend">
      <span><i style="background:#1d9e75"></i>策略</span>
      <span><i style="background:#ba7517"></i>红利低波 买入持有</span>
      <span><i style="background:#9aa1a9"></i>沪深300 买入持有</span>
    </div>
    <svg id="nav" viewBox="0 0 1000 360" preserveAspectRatio="none"></svg>
  </div>

  <div class="chartbox">
    <h3>红利仓位（每月调仓一次，阶梯线）</h3>
    <div class="legend"><span><i style="background:#1d9e75"></i>实际仓位</span>
      <span><i style="background:#d9534f"></i>上限</span></div>
    <svg id="wchart" viewBox="0 0 1000 190" preserveAspectRatio="none"></svg>
  </div>

  <div class="panel">
    <table id="tab">
      <thead><tr><th>区间</th><th>累计收益</th><th>年化</th><th>最大回撤</th><th>夏普</th><th>Calmar</th></tr></thead>
      <tbody></tbody>
    </table>
    <div class="stamp" id="stamp"></div>
  </div>
</div>

<script>
const NS = "http://www.w3.org/2000/svg";
let VER = "trade";
let timer = null, calls = 0;

function el(tag, attrs, parent){
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function fmtPct(x, nd){ return (x*100).toFixed(nd===undefined?1:nd) + "%"; }

function drawNav(s){
  const svg = document.getElementById("nav");
  svg.innerHTML = "";
  const W=1000,H=360,P={l:62,r:18,t:14,b:26};
  const S = s.series;
  const n = S.dates.length;
  const all = S.strategy.concat(S.bm_dividend, S.bm_broad);
  let lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
  const pad = (hi-lo)*0.06; lo-=pad; hi+=pad;
  const X = i => P.l + (W-P.l-P.r)*(i/(n-1));
  const Y = v => H-P.b - (H-P.t-P.b)*((v-lo)/(hi-lo));

  for (let g=0; g<=4; g++){
    const v = lo + (hi-lo)*g/4, y = Y(v);
    el("line",{x1:P.l,x2:W-P.r,y1:y,y2:y,stroke:"#eceef1","stroke-width":1}, svg);
    const t = el("text",{x:P.l-9,y:y+4,"text-anchor":"end","font-size":11,fill:"#8b929a"}, svg);
    t.textContent = v.toFixed(2);
  }
  const years = {};
  S.dates.forEach((d,i)=>{ const y=d.slice(0,4); if(!(y in years)) years[y]=i; });
  Object.keys(years).forEach(y=>{
    const i = years[y], x = X(i);
    el("line",{x1:x,x2:x,y1:P.t,y2:H-P.b,stroke:"#f2f3f5","stroke-width":1}, svg);
    const t = el("text",{x:x,y:H-8,"text-anchor":"middle","font-size":11,fill:"#8b929a"}, svg);
    t.textContent = y;
  });

  function line(vals, color, w){
    let d = "";
    for (let i=0;i<n;i++){ d += (i?"L":"M") + X(i).toFixed(1) + " " + Y(vals[i]).toFixed(1) + " "; }
    el("path",{d:d,fill:"none",stroke:color,"stroke-width":w,"stroke-linejoin":"round"}, svg);
  }
  line(S.bm_broad, "#9aa1a9", 1.3);
  line(S.bm_dividend, "#ba7517", 1.3);
  line(S.strategy, "#1d9e75", 2.3);
}

function drawWeight(s){
  const svg = document.getElementById("wchart");
  svg.innerHTML = "";
  const W=1000,H=190,P={l:62,r:18,t:12,b:24};
  const S = s.series;
  const ws = S.weights, ds = S.weight_dates, n = ws.length;
  const hi = 0.65, lo = 0;
  const X = i => P.l + (W-P.l-P.r)*(i/(Math.max(n-1,1)));
  const Y = v => H-P.b - (H-P.t-P.b)*((v-lo)/(hi-lo));

  [0,0.2,0.4,0.6].forEach(v=>{
    const y = Y(v);
    el("line",{x1:P.l,x2:W-P.r,y1:y,y2:y,stroke:"#eceef1","stroke-width":1}, svg);
    const t = el("text",{x:P.l-9,y:y+4,"text-anchor":"end","font-size":11,fill:"#8b929a"}, svg);
    t.textContent = Math.round(v*100)+"%";
  });
  const years = {};
  ds.forEach((d,i)=>{ const y=d.slice(0,4); if(!(y in years)) years[y]=i; });
  Object.keys(years).forEach(y=>{
    const x = X(years[y]);
    const t = el("text",{x:x,y:H-6,"text-anchor":"middle","font-size":11,fill:"#8b929a"}, svg);
    t.textContent = y;
  });

  el("line",{x1:P.l,x2:W-P.r,y1:Y(s.meta.max_equity),y2:Y(s.meta.max_equity),
             stroke:"#d9534f","stroke-width":1.2,"stroke-dasharray":"5 4"}, svg);

  let d = "M" + X(0) + " " + Y(ws[0]);
  for (let i=1;i<n;i++){ d += " L" + X(i) + " " + Y(ws[i-1]) + " L" + X(i) + " " + Y(ws[i]); }
  el("path",{d:d,fill:"none",stroke:"#1d9e75","stroke-width":1.8}, svg);
}

function kpis(s){
  const m = s.metrics["全区间"]["策略"];
  const items = [
    ["累计收益", fmtPct(m["累计收益率"]), ""],
    ["年化", fmtPct(m["年化收益率"],2), ""],
    ["最大回撤", fmtPct(m["最大回撤"],2), "预算 −7%"],
    ["夏普", m["夏普比率"].toFixed(2), ""],
    ["Calmar", m["Calmar比率"].toFixed(2), ""],
    ["平均仓位", fmtPct(s.meta.avg_weight,0), "收缩 " + s.meta.shrink_months + "/" + s.meta.months + " 月"],
  ];
  document.getElementById("kpis").innerHTML = items.map(it =>
    `<div class="kpi"><div class="k">${it[0]}</div><div class="v">${it[1]}</div><div class="n">${it[2]}</div></div>`
  ).join("");
}

function table(s){
  const rows = [];
  ["全区间","样本内","样本外"].forEach(seg=>{
    if (!s.metrics[seg]) return;
    const m = s.metrics[seg]["策略"];
    rows.push(`<tr><td>${seg}　策略</td><td>${fmtPct(m["累计收益率"])}</td><td>${fmtPct(m["年化收益率"],2)}</td>
      <td>${fmtPct(m["最大回撤"],2)}</td><td>${m["夏普比率"].toFixed(2)}</td><td>${m["Calmar比率"].toFixed(2)}</td></tr>`);
  });
  const bm = s.metrics["全区间"]["红利低波买入持有"];
  rows.push(`<tr><td>全区间　红利低波买入持有</td><td>${fmtPct(bm["累计收益率"])}</td><td>${fmtPct(bm["年化收益率"],2)}</td>
    <td>${fmtPct(bm["最大回撤"],2)}</td><td>${bm["夏普比率"].toFixed(2)}</td><td>${bm["Calmar比率"].toFixed(2)}</td></tr>`);
  document.querySelector("#tab tbody").innerHTML = rows.join("");
}

async function run(){
  calls++;
  const q = new URLSearchParams({
    ver: VER,
    vol: document.getElementById("vol").value,
    cap: document.getElementById("cap").value,
    min: 20,
    base: document.getElementById("base").value,
    window: document.getElementById("win").value,
    _: calls
  });
  document.getElementById("stamp").innerHTML = "正在由 Python 后端重算…";
  try {
    const r = await fetch("/api/run?" + q.toString());
    if (!r.ok) throw new Error("HTTP " + r.status);
    const s = await r.json();
    kpis(s); drawNav(s); drawWeight(s); table(s);
    document.getElementById("stamp").innerHTML =
      `本次由后端重算完成，用时 <b>${s.meta.elapsed_ms} ms</b>　·　` +
      `${s.meta.start} → ${s.meta.end}，${s.meta.days} 个交易日　·　` +
      `参数：波动 ${(s.meta.vol_target*100).toFixed(0)}%、基准 ${(s.meta.base_equity*100).toFixed(0)}%、` +
      `区间 [${(s.meta.min_equity*100).toFixed(0)}%, ${(s.meta.max_equity*100).toFixed(0)}%]、窗口 ${s.meta.window} 日`;
  } catch (e) {
    document.getElementById("stamp").innerHTML = '<span class="err">计算失败：' + e.message + '</span>';
  }
}

function bind(){
  const pairs = [["vol","v-vol","%"],["cap","v-cap","%"],["base","v-base","%"],["win","v-win"," 日"]];
  pairs.forEach(([id,lab,suf])=>{
    const inp = document.getElementById(id);
    const out = document.getElementById(lab);
    out.textContent = inp.value + suf;
    inp.addEventListener("input", ()=>{
      out.textContent = inp.value + suf;
      clearTimeout(timer);
      timer = setTimeout(run, 180);
    });
  });
  document.querySelectorAll("#seg button").forEach(b=>{
    b.addEventListener("click", ()=>{
      document.querySelectorAll("#seg button").forEach(x=>x.classList.remove("on"));
      b.classList.add("on");
      VER = b.dataset.ver;
      run();
    });
  });
}

bind();
run();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "StrategyBoard/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def _send(self, code, body, ctype):
        raw = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if parsed.path == "/api/run":
            return self._api(parse_qs(parsed.query))
        return self._send(404, "not found", "text/plain; charset=utf-8")

    def _api(self, q):
        try:
            params = {
                "ver": (q.get("ver", ["trade"])[0]),
                "vol": float(q.get("vol", ["8"])[0]),
                "cap": float(q.get("cap", ["50"])[0]),
                "min": float(q.get("min", ["20"])[0]),
                "base": float(q.get("base", ["40"])[0]),
                "window": int(q.get("window", ["60"])[0]),
            }
            if params["min"] >= params["cap"]:
                raise ValueError("下限必须小于上限")
            if not (0 < params["base"] <= params["cap"]):
                raise ValueError("基准仓位必须在 (0, 上限] 之间")
            data, hit = cached_run(params)
            payload = dict(data)
            payload["meta"] = dict(data["meta"], cached=hit)
            return self._send(200, json.dumps(payload, ensure_ascii=False),
                              "application/json; charset=utf-8")
        except Exception as exc:  # noqa: BLE001
            return self._send(400, json.dumps({"error": str(exc)}, ensure_ascii=False),
                              "application/json; charset=utf-8")


def main():
    ap = argparse.ArgumentParser(description="策略参数看板（本地 Web）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    url = f"http://{args.host}:{args.port}/"
    print(f"看板已启动：{url}")
    print("按 Ctrl+C 停止。")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
