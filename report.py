"""Morning report, 09:15 IST. Two completely separate blocks, same strategy in each:
  BLOCK 1 - Nifty 50      : heat map, sector heat, top gainers / losers, trade ideas, Elliott Wave
  BLOCK 2 - Nifty Next 50 : the same five things for the Next 50 stocks and their index
Works with the existing bot.py / ew_scan.py without editing them.   python report.py [--dry] [--force]
"""
import io, json, os, sys
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import bot
import ew_scan
from ew_scan import BG, CFG, GREEN, MUTED, RED, best_count, caption_for, chart_png, fp, tile_color
from universe import NEXT50, NIFTY50, SECTORS, SECTOR_OF, UNIVERSE

try:
    import yfinance as yf
except ImportError:
    yf = None

IST = bot.IST
TOP_N = 5          # top gainers / losers shown in each list
MAX_IDEAS = 8      # trade ideas shown per block
MAX_CHARTS = 5     # Elliott charts per block (plus the index chart)

GROUPS = [
    {"title": "Nifty 50", "emoji": "🔵", "syms": NIFTY50, "idx_name": "NIFTY 50", "idx_tick": "^NSEI"},
    {"title": "Nifty Next 50", "emoji": "🟠", "syms": NEXT50, "idx_name": "NIFTY NEXT 50", "idx_tick": "^NSMIDCP"},
]

# ---- make bot.py use the full Nifty 100 list, whatever its own import line says
bot.NIFTY50, bot.SECTORS, bot.SECTOR_OF = UNIVERSE, SECTORS, SECTOR_OF


def _no_nse():
    raise RuntimeError("NSE skipped (only has Nifty 50)")


bot.fetch_nse = _no_nse            # NSE would give only 50 stocks; Angel -> Yahoo give all 100


# ------------------------------------------------------------ text blocks
def movers(rows, n=TOP_N):
    s = sorted(rows, key=lambda r: r["chg"], reverse=True)
    fmt = lambda lst: "\n".join(f"{r['sym']:<11}{r['chg']:+6.2f}%  {r['ltp']:,.2f}" for r in lst)
    return fmt(s[:n]), fmt(list(reversed(s[-n:])))


def index_line(g, idx):
    if not idx:
        return f"{g['title']}: डेटा उपलब्ध नहीं"
    if idx.get("approx"):
        return f"{g['title']} (शेयरों का औसत): {idx['chg']:+.2f}%"
    return f"{g['title']}: {idx['ltp']:,.2f} ({idx['chg']:+.2f}%)"


def build_direction(g, rows, idx, source, now):
    ts = now.strftime("%a %d %b %Y, %H:%M IST")
    adv = sum(r["chg"] > 0 for r in rows)
    heat = "\n".join(f"{'▲' if c >= 0 else '▼'} {s:<9}{c:+.2f}%" for s, c in bot.sector_heat(rows))
    gain, lose = movers(rows)
    t = g["title"]
    return (f"{g['emoji']} *{t} — बाज़ार की दिशा* — {ts}\n{index_line(g, idx)}\n"
            f"बढ़त {adv} | गिरावट {len(rows) - adv}\n\n"
            f"🏭 *सेक्टर हीट* ({t} शेयरों की औसत चाल)\n```\n{heat}\n```\n"
            f"🟢 *{t} टॉप {TOP_N} गेनर*\n```\n{gain}\n```\n"
            f"🔴 *{t} टॉप {TOP_N} लूज़र*\n```\n{lose}\n```\n"
            f"_स्रोत: {source}। केवल जानकारी, निवेश सलाह नहीं।_")


def build_trading(g, rows, vr, cfg, source, now):
    ts = now.strftime("%a %d %b %Y, %H:%M IST")
    movers_n = sum(abs(r["chg"]) >= cfg["min_move_pct"] for r in rows)
    ideas = bot.build_ideas(rows, vr, dict(cfg, max_ideas=MAX_IDEAS))
    body = ("\n".join(f"{i['side']:<4} {i['sym']}  {i['chg']:+.2f}%  Vol {i['vol']:.1f}x\n"
                      f" Entry {i['entry']:.2f}  SL {i['sl']:.2f}\n Target {i['target']:.2f}" for i in ideas)
            if ideas else "कोई शेयर शर्तें पूरी नहीं करता")
    vol_known = sum(vr.get(r["sym"]) is not None for r in rows) if vr else 0
    vol_note = "" if vol_known else "\n⚠️ Volume डेटा नहीं मिला, इसलिए कोई सिफ़ारिश नहीं।"
    return (f"{g['emoji']} *{g['title']} — ट्रेड आइडिया* — {ts}\n"
            f"±{cfg['min_move_pct']:g}% से ऊपर चलने वाले शेयर: {movers_n}\n\n"
            f"✅ चाल ≥ ±{cfg['min_move_pct']:g}% और Volume ≥ {cfg['min_volume_ratio']:g}x\n"
            f"```\n{body}\n```{vol_note}\n"
            f"_Entry = मौजूदा भाव, Target {cfg['target_pct']:g}%, SL {cfg['stop_pct']:g}%। "
            f"Vol = अब तक का volume ÷ पिछले दिनों का उसी समय तक का औसत। "
            f"शुरुआती मिनटों में चाल अस्थिर होती है। स्रोत: {source}। केवल जानकारी, निवेश सलाह नहीं।_")


def heatmap_png(g, rows, idx, source, now):
    """Sector heat map image for one block."""
    per = 8
    by = {}
    for r in rows:
        by.setdefault(SECTOR_OF.get(r["sym"], "Other"), []).append(r)
    secs = sorted(by.items(), key=lambda kv: -sum(x["chg"] for x in kv[1]) / len(kv[1]))
    lines = sum((len(v) + per - 1) // per for _, v in secs)
    W, rh, head = 12.0, 0.62, 1.4
    H = head + rh * lines
    fig = plt.figure(figsize=(W, H), dpi=100, facecolor=BG)
    ax = fig.add_axes([0.01, 0.01, 0.98, 1 - (head - 0.1) / H], facecolor=BG)
    ax.set_xlim(0, 12); ax.set_ylim(lines, 0); ax.axis("off")
    tw = (12 - 2.2) / per
    y = 0
    for sec, lst in secs:
        avg = sum(x["chg"] for x in lst) / len(lst)
        ax.text(0.05, y + 0.32, sec, color="white", fontsize=12, fontweight="bold", va="center")
        ax.text(0.05, y + 0.68, f"{avg:+.2f}% avg", color=GREEN if avg >= 0 else RED, fontsize=10, va="center")
        for j, r in enumerate(sorted(lst, key=lambda x: -x["chg"])):
            x0, y0 = 2.2 + (j % per) * tw, y + j // per
            ax.add_patch(Rectangle((x0, y0 + 0.05), tw - 0.07, 0.90, fc=tile_color(r["chg"]), ec="none"))
            ax.text(x0 + (tw - 0.07) / 2, y0 + 0.35, r["sym"], color="white", fontsize=8,
                    fontweight="bold", ha="center", va="center")
            ax.text(x0 + (tw - 0.07) / 2, y0 + 0.68, f"{r['chg']:+.2f}%", color="white", fontsize=8,
                    ha="center", va="center")
        y += (len(lst) + per - 1) // per
    sub = index_line(g, idx)
    up = (idx or {}).get("chg", 0) >= 0
    fig.text(0.02, 1 - 0.35 / H, f"{g['title']} · sector heat map", color="white", fontsize=15, fontweight="bold")
    fig.text(0.02, 1 - 0.75 / H, sub, color=GREEN if up else RED, fontsize=11, fontweight="bold")
    fig.text(0.98, 1 - 0.35 / H, now.strftime("%d %b %H:%M IST"), color=MUTED, fontsize=8.5, ha="right")
    fig.text(0.98, 1 - 0.75 / H, f"source: {source} · colour = % change vs prev close (±3% scale)",
             color=MUTED, fontsize=8, ha="right")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    plt.close(fig)
    return buf.getvalue()


# --------------------------------------------------------------- Elliott
def load_ew(now):
    """1h candles: both indices + all 100 stocks -> {name: DataFrame}."""
    tick = {g["idx_name"]: g["idx_tick"] for g in GROUPS}
    tick.update({s: s + ".NS" for s in UNIVERSE})
    import candles                      # Angel One first, Yahoo Finance as backup
    return candles.load(tick, 60, 60, now)


def group_ctx(rows):
    sec = {}
    for r in rows:
        sec.setdefault(SECTOR_OF.get(r["sym"], "Other"), []).append(r["chg"])
    return {"sector": {k: sum(v) / len(v) for k, v in sec.items()}, "chg": {r["sym"]: r["chg"] for r in rows}}


def ew_summary(g, results, idx, now, ctx):
    ts = now.strftime("%a %d %b %Y, %H:%M IST")
    parts = [f"{g['emoji']} *{g['title']} — Elliott Wave (1h)* — {ts}", index_line(g, idx)]
    nc = results.get(g["idx_name"])
    if nc and nc["valid"]:
        parts.append(f"इंडेक्स: {nc['side']} setup ({nc['status']}) — Entry {fp(nc['entry'])}, "
                     f"SL {fp(nc['stop'])}, Target {fp(nc['target'])}")
    else:
        parts.append("इंडेक्स: कोई वैध entry गिनती नहीं")
    stocks = {k: v for k, v in results.items() if k != g["idx_name"] and v and v["valid"]}
    for status, title in (("TRADE", "✅ *TRADE* (गिनती ठीक + H1 breakout)"),
                          ("WATCH", "👀 *WATCH* (गिनती ठीक, H1 breakout अभी नहीं)")):
        lst = [(k, v) for k, v in stocks.items() if v["status"] == status]
        if lst:
            more = f"\n… और {len(lst) - CFG['max_list']}" if len(lst) > CFG["max_list"] else ""
            parts.append(title + f" — {len(lst)}\n```\n"
                         + "\n".join(ew_scan.row(k, v, ctx) for k, v in lst[: CFG["max_list"]]) + more + "\n```")
    if not stocks:
        parts.append("शेयरों में कोई वैध Elliott setup नहीं मिला।")
    parts.append(ew_scan.FOOTER)
    return "\n\n".join(parts), stocks


def chart_names_for(g, results, stocks):
    order = sorted(stocks, key=lambda k: (stocks[k]["status"] != "TRADE", stocks[k]["stop_pct"]))
    return ([g["idx_name"]] if results.get(g["idx_name"]) else []) + order[:MAX_CHARTS]


# ------------------------------------------------------------------ main
def next50_index(rows):
    """Live Nifty Next 50 level from Yahoo; falls back to the average of its stocks."""
    try:
        fi = yf.Ticker("^NSMIDCP").fast_info
        ltp, prev = float(fi.last_price), float(fi.previous_close)
        if ltp > 0 and prev > 0:
            return {"ltp": ltp, "chg": (ltp / prev - 1) * 100}
    except Exception as e:
        print(f"Next 50 index quote unavailable: {e}", file=sys.stderr)
    if rows:
        return {"ltp": 0.0, "chg": sum(r["chg"] for r in rows) / len(rows), "approx": True}
    return None


def out_text(text, dry):
    print(text, "\n") if dry else ew_scan.send_text(text)


def out_photo(png, caption, dry, name):
    if dry:
        os.makedirs("out", exist_ok=True)
        open(f"out/{name}.png", "wb").write(png)
        print("image:", name, "-", caption.splitlines()[0])
    else:
        ew_scan.send_photo(png, caption)


def main():
    now = datetime.now(IST)
    dry = "--dry" in sys.argv
    if now.weekday() >= 5 and "--force" not in sys.argv:
        print("Weekend - market closed."); return

    cfg = dict(bot.DEFAULT_CFG)
    try:
        cfg.update(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "criteria.json"))))
    except Exception as e:
        print(f"criteria.json not read ({e}); using defaults", file=sys.stderr)

    rows = vr = index = source = None
    try:
        rows, index, source, vr = bot.fetch_all(now)
        if len(rows) < 40:
            raise RuntimeError(f"only {len(rows)} quotes fetched")
        if vr is None or sum(v is not None for v in vr.values()) < 25:
            vr = bot.fetch_volume_ratios(now.date())
        print(f"quotes: {len(rows)}")
    except BaseException as e:
        print(f"market data failed: {e!r}", file=sys.stderr)
        rows = None

    data = results = None
    try:
        if yf is None:
            raise RuntimeError("yfinance not installed")
        data = load_ew(now)
        if len(data) < 40:
            raise RuntimeError(f"only {len(data)} symbols loaded")
        results = {n: best_count(df) for n, df in data.items()}
    except BaseException as e:
        print(f"Elliott data failed: {e!r}", file=sys.stderr)

    for g in GROUPS:
        gs = set(g["syms"])
        grows = [r for r in rows if r["sym"] in gs] if rows else []
        idx = index if g["title"] == "Nifty 50" else (next50_index(grows) if rows else None)
        if grows:
            try:
                out_photo(heatmap_png(g, grows, idx, source, now),
                          f"{g['emoji']} {g['title']} सेक्टर हीट मैप — स्रोत: {source}", dry,
                          "heatmap_" + g["title"].replace(" ", "_"))
                out_text(build_direction(g, grows, idx, source, now), dry)
                out_text(build_trading(g, grows, vr, cfg, source, now), dry)
            except BaseException as e:
                print(f"{g['title']} market block failed: {e!r}", file=sys.stderr)
        if results:
            try:
                gres = {k: v for k, v in results.items() if k in gs or k == g["idx_name"]}
                summary, stocks = ew_summary(g, gres, idx, now, group_ctx(grows) if grows else None)
                out_text(summary, dry)
                for nm in chart_names_for(g, gres, stocks):
                    c = gres[nm]
                    cap = (caption_for(nm, c, group_ctx(grows) if grows else None) if c["valid"]
                           else f"📊 {nm} — {c['side']} गिनती: अभी entry नहीं।\nकारण: {c['fails'][0] if c['fails'] else '-'}")
                    out_photo(chart_png(nm, data[nm], c, now), cap, dry, "ew_" + nm.replace(" ", "_").replace("&", "n"))
            except BaseException as e:
                print(f"{g['title']} Elliott block failed: {e!r}", file=sys.stderr)


if __name__ == "__main__":
    main()
