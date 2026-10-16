"""Morning report, run at 09:15 IST (one command runs everything):
  1) Market direction  - Nifty 50: heat map image + sector heat + top gainers / losers
  2) Trading           - Nifty 100: Next-50 top gainers / losers + trade ideas (move >= 1% and volume >= 1.5x)
  3) Elliott Wave scan - ew_scan.py (index + Nifty 100 stocks)
Works with the existing bot.py / ew_scan.py without editing them.   python report.py [--dry] [--force]
"""
import io, json, os, sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import bot
import ew_scan
from ew_scan import BG, GREEN, MUTED, RED, tile_color
from universe import NEXT50, NIFTY50, SECTORS, SECTOR_OF, UNIVERSE

IST = bot.IST
TOP_N = 5          # top gainers / losers shown in each list
MAX_IDEAS = 10     # trade ideas shown (Nifty 100)

# ---- make bot.py use the full Nifty 100 list, whatever its own import line says
bot.NIFTY50, bot.SECTORS, bot.SECTOR_OF = UNIVERSE, SECTORS, SECTOR_OF


def _no_nse():
    raise RuntimeError("NSE skipped (only has Nifty 50)")


bot.fetch_nse = _no_nse            # NSE would give only 50 stocks; Angel -> Yahoo give all 100
_orig_fetch, _cache = bot.fetch_all, {}


def _fetch_once(now):              # fetch once, share with ew_scan's sector context
    if "v" not in _cache:
        _cache["v"] = _orig_fetch(now)
    return _cache["v"]


bot.fetch_all = _fetch_once


def movers(rows, n=TOP_N):
    s = sorted(rows, key=lambda r: r["chg"], reverse=True)
    top, bottom = s[:n], [r for r in reversed(s[-n:])]
    fmt = lambda lst: "\n".join(f"{r['sym']:<11}{r['chg']:+6.2f}%  {r['ltp']:,.2f}" for r in lst)
    return fmt(top), fmt(bottom)


def build_direction(rows50, index, source, now):
    ts = now.strftime("%a %d %b %Y, %H:%M IST")
    adv = sum(r["chg"] > 0 for r in rows50)
    nifty = f"Nifty 50: {index['ltp']:,.2f} ({index['chg']:+.2f}%)" if index else "Nifty 50: डेटा उपलब्ध नहीं"
    heat = "\n".join(f"{'▲' if c >= 0 else '▼'} {s:<9}{c:+.2f}%" for s, c in bot.sector_heat(rows50))
    gain, lose = movers(rows50)
    return (f"📊 *बाज़ार की दिशा — Nifty 50* — {ts}\n{nifty}\n"
            f"बढ़त {adv} | गिरावट {len(rows50) - adv}\n\n"
            f"🏭 *सेक्टर हीट* (Nifty 50 शेयरों की औसत चाल)\n```\n{heat}\n```\n"
            f"🟢 *Nifty 50 टॉप {TOP_N} गेनर*\n```\n{gain}\n```\n"
            f"🔴 *Nifty 50 टॉप {TOP_N} लूज़र*\n```\n{lose}\n```\n"
            f"_स्रोत: {source}। केवल जानकारी, निवेश सलाह नहीं।_")


def build_trading(rows, vr, cfg, source, now):
    ts = now.strftime("%a %d %b %Y, %H:%M IST")
    nxt = [r for r in rows if r["sym"] in set(NEXT50)]
    adv = sum(r["chg"] > 0 for r in rows)
    movers_n = sum(abs(r["chg"]) >= cfg["min_move_pct"] for r in rows)
    gain, lose = movers(nxt)
    cfg = dict(cfg, max_ideas=MAX_IDEAS)
    ideas = bot.build_ideas(rows, vr, cfg)
    body = ("\n".join(f"{i['side']:<4} {i['sym']}  {i['chg']:+.2f}%  Vol {i['vol']:.1f}x\n"
                      f" Entry {i['entry']:.2f}  SL {i['sl']:.2f}\n Target {i['target']:.2f}" for i in ideas)
            if ideas else "कोई शेयर शर्तें पूरी नहीं करता")
    vol_known = sum(v is not None for v in (vr or {}).values())
    vol_note = "" if vol_known else "\n⚠️ Volume डेटा नहीं मिला, इसलिए कोई सिफ़ारिश नहीं।"
    return (f"🎯 *ट्रेडिंग — Nifty 100* — {ts}\n"
            f"बढ़त {adv} | गिरावट {len(rows) - adv} | ±{cfg['min_move_pct']:g}% से ऊपर: {movers_n}\n\n"
            f"🟢 *बाकी 50 (Next 50) टॉप {TOP_N} गेनर*\n```\n{gain}\n```\n"
            f"🔴 *बाकी 50 (Next 50) टॉप {TOP_N} लूज़र*\n```\n{lose}\n```\n"
            f"✅ *ट्रेड आइडिया — सभी 100 में* (चाल ≥ ±{cfg['min_move_pct']:g}% और Volume ≥ {cfg['min_volume_ratio']:g}x)\n"
            f"```\n{body}\n```{vol_note}\n"
            f"_Entry = मौजूदा भाव, Target {cfg['target_pct']:g}%, SL {cfg['stop_pct']:g}%। "
            f"Vol = अब तक का volume ÷ पिछले दिनों का उसी समय तक का औसत। "
            f"शुरुआती मिनटों में चाल अस्थिर होती है। स्रोत: {source}। केवल जानकारी, निवेश सलाह नहीं।_")


def heatmap50(rows, index, source, now):
    """Nifty 50 sector heat map image (same look as the Nifty 100 one)."""
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
    nifty = f"Nifty 50 {index['ltp']:,.2f} ({index['chg']:+.2f}%)" if index else "Nifty 50"
    fig.text(0.02, 1 - 0.35 / H, "Nifty 50 · sector heat map", color="white", fontsize=15, fontweight="bold")
    fig.text(0.02, 1 - 0.75 / H, nifty, color=GREEN if (index and index["chg"] >= 0) else RED,
             fontsize=11, fontweight="bold")
    fig.text(0.98, 1 - 0.35 / H, now.strftime("%d %b %H:%M IST"), color=MUTED, fontsize=8.5, ha="right")
    fig.text(0.98, 1 - 0.75 / H, f"source: {source} · colour = % change vs prev close (±3% scale)",
             color=MUTED, fontsize=8, ha="right")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    plt.close(fig)
    return buf.getvalue()


def market_part(now, dry):
    cfg = dict(bot.DEFAULT_CFG)
    try:
        cfg.update(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "criteria.json"))))
    except Exception as e:
        print(f"criteria.json not read ({e}); using defaults", file=sys.stderr)
    rows, index, source, vr = bot.fetch_all(now)
    if len(rows) < 40:
        sys.exit(f"Only {len(rows)} quotes fetched; aborting.")
    if vr is None or sum(v is not None for v in vr.values()) < 25:
        vr = bot.fetch_volume_ratios(now.date())
    n50 = set(NIFTY50)
    rows50 = [r for r in rows if r["sym"] in n50]
    print(f"quotes: {len(rows)} (Nifty 50 part: {len(rows50)})")
    m1 = build_direction(rows50, index, source, now)
    m2 = build_trading(rows, vr, cfg, source, now)
    png = heatmap50(rows50, index, source, now)
    if dry:
        os.makedirs("out", exist_ok=True)
        open("out/heatmap50.png", "wb").write(png)
        print(m1, "\n\n", m2)
        return
    ew_scan.send_photo(png, f"🏭 Nifty 50 सेक्टर हीट मैप — स्रोत: {source}")
    ew_scan.send_text(m1)
    ew_scan.send_text(m2)


def main():
    now = datetime.now(IST)
    if now.weekday() >= 5 and "--force" not in sys.argv:
        print("Weekend - market closed."); return
    try:
        market_part(now, "--dry" in sys.argv)
    except BaseException as e:           # a data problem must not stop the Elliott scan
        print(f"market part failed: {e!r}", file=sys.stderr)
    # Elliott scan: its own Nifty-100 heat map is replaced by the Nifty-50 one sent above
    ew_scan.heatmap_png = lambda *a, **k: None
    _send_photo = ew_scan.send_photo
    ew_scan.send_photo = lambda png, cap: None if png is None else _send_photo(png, cap)
    ew_scan.main()


if __name__ == "__main__":
    main()
