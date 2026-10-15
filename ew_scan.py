"""Elliott Wave scanner: Nifty 50 index + its 50 stocks on 1h candles -> Telegram charts.

Looks for an impulse count 0-1-2-3-4 (long = uptrend, short = downtrend) and, when wave 4
sits in the 23.6-50% zone of wave 3, gives entry / stop / target (wave 5, 2R).
Run after 09:15 IST on trading days:   python ew_scan.py [--dry] [--force]
"""
import io, os, sys, textwrap
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    import yfinance as yf
except ImportError:
    yf = None

IST = ZoneInfo("Asia/Kolkata")

NIFTY50 = """ADANIENT ADANIPORTS APOLLOHOSP ASIANPAINT AXISBANK BAJAJ-AUTO BAJFINANCE BAJAJFINSV
BEL BHARTIARTL CIPLA COALINDIA DRREDDY EICHERMOT ETERNAL GRASIM HCLTECH HDFCBANK HDFCLIFE
HINDALCO HINDUNILVR ICICIBANK INDIGO INFY ITC JIOFIN JSWSTEEL KOTAKBANK LT M&M MARUTI
MAXHEALTH NESTLEIND NTPC ONGC POWERGRID RELIANCE SBILIFE SBIN SHRIRAMFIN SUNPHARMA
TATACONSUM TMPV TATASTEEL TCS TECHM TITAN TRENT ULTRACEMCO WIPRO""".split()

SECTORS = {
    "IT": "TCS INFY HCLTECH WIPRO TECHM",
    "Banks": "HDFCBANK ICICIBANK SBIN KOTAKBANK AXISBANK",
    "Finance": "BAJFINANCE BAJAJFINSV SHRIRAMFIN JIOFIN HDFCLIFE SBILIFE",
    "Auto": "MARUTI M&M BAJAJ-AUTO EICHERMOT TMPV",
    "Metals": "TATASTEEL JSWSTEEL HINDALCO COALINDIA",
    "Energy": "RELIANCE ONGC",
    "Power": "NTPC POWERGRID",
    "Pharma": "SUNPHARMA CIPLA DRREDDY APOLLOHOSP MAXHEALTH",
    "FMCG": "HINDUNILVR ITC NESTLEIND TATACONSUM",
    "Infra": "LT BEL ADANIENT ADANIPORTS BHARTIARTL",
    "Cement": "ULTRACEMCO GRASIM",
    "Consumer": "TITAN TRENT ASIANPAINT ETERNAL INDIGO",
}
SECTOR_OF = {s: sec for sec, names in SECTORS.items() for s in names.split()}

CFG = {
    "zone_lo": 0.236,         # wave 4 must retrace at least this much of wave 3 ...
    "zone_hi": 0.50,          # ... and at most this much
    "rr": 2.0,                # target = entry -/+ rr x risk
    "max_stop_pct": 4.5,      # stop wider than this is rejected
    "h1_lookback": 10,        # H1 breakout = close beyond the previous N candles' high/low
    "h1_window": 4,           # ... within the last N hours
    "atr_mult": (7, 5, 3.5),  # zig-zag swing sizes tried (x ATR); big swings first
    "max_charts": 8,
}

BG, GRID, TXT, MUTED = "#1c1c1f", "#2c2c30", "#e8e8ea", "#8d8d93"
GREEN, RED, BLUE, GOLD = "#0ecb81", "#f6465d", "#4da3ff", "#f5b800"


# ------------------------------------------------------------------- data
def load_all(now):
    """-> {name: DataFrame(Open, High, Low, Close) with IST index}; unfinished candle dropped."""
    tick = {"NIFTY 50": "^NSEI"}
    tick.update({s: s + ".NS" for s in NIFTY50})
    data = yf.download(list(tick.values()), period="60d", interval="1h", group_by="ticker",
                       progress=False, threads=True, auto_adjust=False)
    out = {}
    for name, t in tick.items():
        try:
            df = data[t][["Open", "High", "Low", "Close"]].dropna().copy()
            df.index = df.index.tz_convert(IST) if df.index.tz is not None else df.index.tz_localize(IST)
            if len(df) and df.index[-1] + timedelta(hours=1) > now:
                df = df.iloc[:-1]  # candle still forming
            if len(df) >= 60:
                out[name] = df
        except Exception as e:
            print(f"skip {name}: {e}", file=sys.stderr)
    return out


# ------------------------------------------------------------------ waves
def zigzag(H, L, dev):
    """Alternating swing pivots [(index, price), ...]; the last one is the running extreme."""
    n = len(H)
    piv, d, ext = [], 0, 0
    hi = lo = 0
    for i in range(1, n):
        if d == 0:
            if H[i] > H[hi]:
                hi = i
            if L[i] < L[lo]:
                lo = i
            if H[hi] - L[i] >= dev and hi < i:
                piv.append((hi, H[hi])); d, ext = -1, i
            elif H[i] - L[lo] >= dev and lo < i:
                piv.append((lo, L[lo])); d, ext = 1, i
        elif d == 1:
            if H[i] > H[ext]:
                ext = i
            elif H[ext] - L[i] >= dev:
                piv.append((ext, H[ext])); d, ext = -1, i
        else:
            if L[i] < L[ext]:
                ext = i
            elif H[i] - L[ext] >= dev:
                piv.append((ext, L[ext])); d, ext = 1, i
    if d == 1:
        piv.append((ext, H[ext]))
    elif d == -1:
        piv.append((ext, L[ext]))
    return piv


def analyse(df, pts):
    """Test five pivots as impulse 0-1-2-3-4. Everything is done in 'long space'
    (prices x -1 for a downtrend) so one set of rules serves both directions."""
    s = 1 if pts[0][1] < pts[1][1] else -1
    side = "LONG" if s == 1 else "SHORT"
    q = [s * p for _, p in pts]
    O, H, L, C = (df[c].values for c in ("Open", "High", "Low", "Close"))
    qH, qC = (H, C) if s == 1 else (-L, -C)
    close_q, n = qC[-1], len(C)
    w1, w3 = q[1] - q[0], q[3] - q[2]
    zl, zh = CFG["zone_lo"], CFG["zone_hi"]
    r4 = (q[3] - q[4]) / w3 if w3 > 0 else 9.0
    r_now = (q[3] - close_q) / w3 if w3 > 0 else 9.0
    fails = []
    if q[2] <= q[0]:
        fails.append("Wave 2 retraced past the start of wave 1")
    if q[3] <= q[1]:
        fails.append("Wave 3 did not pass the end of wave 1")
    elif w3 < w1:
        fails.append("Wave 3 is shorter than wave 1")
    if q[4] <= q[1]:
        fails.append("Wave 4 overlaps wave 1")
    if r4 > zh:
        fails.append(f"Wave 4 ({side.lower()}) has pulled back {r4 * 100:.0f}% of wave 3, past the "
                     f"{zh * 100:.0f}% limit: this count cannot give an entry (a new wave 1-2-3 is needed).")
    elif r4 < zl:
        fails.append(f"Wave 4 ({side.lower()}) pulled back only {r4 * 100:.0f}% of wave 3, under {zl * 100:.1f}%.")
    elif not (zl <= r_now <= zh):
        fails.append(f"Price sits at {r_now * 100:.0f}% of wave 3, outside the "
                     f"{zl * 100:.1f}-{zh * 100:.0f}% entry zone.")
    risk = close_q - q[4]
    stop_pct = risk / abs(close_q) * 100 if close_q else 99
    target_q = close_q + CFG["rr"] * risk
    if risk <= 0:
        fails.append("Price is already beyond the wave-4 extreme (stop).")
    else:
        if stop_pct > CFG["max_stop_pct"]:
            fails.append(f"Stop width {stop_pct:.1f}% is above the {CFG['max_stop_pct']:g}% maximum.")
        if target_q <= q[3]:
            fails.append("Target does not clear the end of wave 3.")
    # H1 breakout the same way, within the last few hours
    h1_ago = None
    lb = CFG["h1_lookback"]
    for k in range(n - 1, max(n - 1 - CFG["h1_window"], lb) - 1, -1):
        if qC[k] > qH[k - lb:k].max():
            h1_ago = n - 1 - k
            break
    valid = not fails
    return {
        "side": side, "s": s, "pts": pts, "valid": valid, "fails": fails,
        "r4": r4, "r_now": r_now, "stop_pct": stop_pct, "h1_ago": h1_ago,
        "entry": s * close_q, "stop": s * q[4], "target": s * target_q,
        "zone": (s * (q[3] - zl * w3), s * (q[3] - zh * w3)),
        "status": ("TRADE" if h1_ago is not None else "WATCH") if valid else None,
    }


def best_count(df):
    tr = np.maximum.reduce([df.High - df.Low, (df.High - df.Close.shift()).abs(),
                            (df.Low - df.Close.shift()).abs()])
    atr = float(pd.Series(tr).tail(60).mean())
    H, L = df.High.values, df.Low.values
    cands = []
    for k in CFG["atr_mult"]:
        piv = zigzag(H, L, k * atr)
        for off in (0, 1):  # off=1: wave 5 has just started after the wave-4 pivot
            if len(piv) >= 5 + off:
                c = analyse(df, piv[len(piv) - 5 - off: len(piv) - off])
                c["k"] = k
                cands.append(c)
    if not cands:
        return None
    cands.sort(key=lambda c: (not c["valid"], len(c["fails"]), c["h1_ago"] is None, -c["pts"][4][0]))
    return cands[0]


# ------------------------------------------------------------------ chart
def fp(x):
    return f"{x:,.2f}" if abs(x) >= 1 else f"{x:.5f}"


def chart_png(name, df, c, now):
    s, pts = c["s"], c["pts"]
    i0 = max(0, pts[0][0] - 12)
    d = df.iloc[i0:]
    n = len(d)
    xs = [i - i0 for i, _ in pts]
    right = n + 3
    ok = c["valid"]
    fig = plt.figure(figsize=(10, 6), dpi=100, facecolor=BG)
    ax = fig.add_axes([0.02, 0.16, 0.89, 0.70], facecolor=BG)

    for j, (o, h, l, cl) in enumerate(zip(d.Open, d.High, d.Low, d.Close)):
        col = GREEN if cl >= o else RED
        ax.plot([j, j], [l, h], color=col, lw=1, zorder=2)
        ax.add_patch(Rectangle((j - 0.32, min(o, cl)), 0.64, max(abs(cl - o), 1e-9),
                               fc=col, ec=col, zorder=3))

    levels = [d.High.max(), d.Low.min()]
    if ok:
        levels += [c["stop"], c["target"]]
        lo_z, hi_z = sorted(c["zone"])
        ax.add_patch(Rectangle((xs[3], lo_z), right - xs[3], hi_z - lo_z, fc=GOLD, alpha=0.16, ec="none", zorder=1))
        ax.text(right, lo_z, "entry zone 23.6-50%", color=GOLD, fontsize=8.5, ha="right", va="top")
        ax.hlines(c["stop"], xs[4], right, colors=RED, linestyles="--", lw=1.3, zorder=4)
        ax.text(right, c["stop"], f"stop {fp(c['stop'])}", color=RED, fontsize=8.5, ha="right",
                va="bottom" if s == -1 else "top")
        ax.hlines(c["target"], xs[3], right, colors=GREEN, linestyles="--", lw=1.3, zorder=4)
        ax.text(right, c["target"], f"target (wave 5) {fp(c['target'])}", color=GREEN, fontsize=8.5,
                ha="right", va="bottom")
    ymin, ymax = min(levels), max(levels)
    pad = (ymax - ymin) * 0.10
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlim(-1, right + 1)

    ax.plot(xs, [p for _, p in pts], color=BLUE, lw=1.8, zorder=5)
    for num, (x, (_, p)) in enumerate(zip(xs, pts)):
        is_high = (num % 2 == 1) if s == 1 else (num % 2 == 0)
        ax.annotate(str(num), (x, p), xytext=(0, 15 if is_high else -15), textcoords="offset points",
                    ha="center", va="center", color="#111" if num == 4 else "white", fontsize=9,
                    fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="circle,pad=0.32", fc=GOLD if num == 4 else BLUE, ec="none"))

    last = float(d.Close.iloc[-1])
    ax.axhline(last, color="#9a9aa0", lw=0.8, zorder=1)
    ax.text(1.004, last, fp(last), transform=ax.get_yaxis_transform(), ha="left", va="center",
            color="#111", fontsize=9, bbox=dict(boxstyle="round,pad=0.25", fc="#e8e8ea", ec="none"))
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", colors=MUTED, labelsize=8.5, length=0)
    ax.set_xticks([])
    ax.grid(axis="y", color=GRID, lw=0.8)
    for sp in ax.spines.values():
        sp.set_visible(False)

    fig.text(0.025, 0.945, f"{name} 1h · Elliott read", color="white", fontsize=15, fontweight="bold")
    fig.text(0.975, 0.955, now.astimezone(timezone.utc).strftime("%d %b %H:%M UTC"), color=MUTED,
             fontsize=8.5, ha="right")
    sub = f"wave 4 under way · {c['side']} " + ("setup" if ok else "count (no entry)")
    fig.text(0.025, 0.895, sub, color=(GREEN if c["status"] == "TRADE" else GOLD) if ok else MUTED,
             fontsize=11, fontweight="bold")
    right_txt = (f"entry {fp(c['entry'])} · stop {fp(c['stop'])} · target {fp(c['target'])} ({CFG['rr']:.1f}R)"
                 if ok else "entry now: none")
    fig.text(0.975, 0.897, right_txt, color=TXT, fontsize=10, ha="right")

    if ok:
        l1 = (f"Wave 4 ({c['side'].lower()}) has pulled back {c['r4'] * 100:.0f}% of wave 3, inside the "
              f"23.6-50% zone. Wave 5 should run beyond the end of wave 3.")
    else:
        l1 = c["fails"][0] if c["fails"] else ""
    if c["h1_ago"] is not None:
        l2 = f"H1 breakout the same way {c['h1_ago']} candle(s) ago: confirmed."
    else:
        l2 = (f"No H1 breakout the same way in the last {CFG['h1_window']}h. "
              f"H1 comes first (at the wave-3 push), so this count is unlikely to trade.")
    l3 = f"Stop width now {c['stop_pct']:.1f}% (max {CFG['max_stop_pct']:g}%)" if c["stop_pct"] > 0 else ""
    for y, txt, col in ((0.105, l1, TXT), (0.072, l2, MUTED), (0.039, l3, MUTED)):
        fig.text(0.025, y, textwrap.shorten(txt, 150, placeholder="..."), color=col, fontsize=9)
    fig.text(0.975, 0.012, "projection, not a call · EW + H1 rules", color=MUTED, fontsize=8, ha="right")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    plt.close(fig)
    return buf.getvalue()


# --------------------------------------------------------------- telegram
def _creds():
    return ("".join(os.environ["TELEGRAM_BOT_TOKEN"].split()),
            "".join(os.environ["TELEGRAM_CHAT_ID"].split()))


def send_text(text):
    token, chat = _creds()
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": text, "parse_mode": "Markdown"}, timeout=20)
    if not r.ok:
        print(f"Telegram text error {r.status_code}: {r.text}", file=sys.stderr)


def send_photo(png, caption):
    token, chat = _creds()
    r = requests.post(f"https://api.telegram.org/bot{token}/sendPhoto",
                      data={"chat_id": chat, "caption": caption[:1000]},
                      files={"photo": ("ew.png", png, "image/png")}, timeout=60)
    if not r.ok:
        print(f"Telegram photo error {r.status_code}: {r.text}", file=sys.stderr)


# ---------------------------------------------------------------- message
def ctx_note(name, c, ctx):
    """Live-market confluence: is the stock's sector moving the same way as the setup?"""
    if not ctx:
        return ""
    out, sec = [], SECTOR_OF.get(name)
    if sec in ctx["sector"]:
        a = ctx["sector"][sec]
        out.append(f"{sec} {a:+.1f}% {'✓' if (a > 0) == (c['side'] == 'LONG') else '✗'}")
    if name in ctx["chg"]:
        out.append(f"आज {ctx['chg'][name]:+.1f}%")
    return " | ".join(out)


def caption_for(name, c, ctx=None):
    tag = "TRADE (H1 पक्का)" if c["status"] == "TRADE" else "WATCH (H1 breakout अभी नहीं)"
    return (f"{'📈' if c['side'] == 'LONG' else '📉'} {name} — {c['side']} — {tag}\n"
            f"Entry {fp(c['entry'])} | SL {fp(c['stop'])} | Target {fp(c['target'])} ({CFG['rr']:.1f}R)\n"
            f"Stop चौड़ाई {c['stop_pct']:.1f}% | Wave 4 pullback {c['r4'] * 100:.0f}%\n"
            + (ctx_note(name, c, ctx) + "\n" if ctx_note(name, c, ctx) else "")
            + "केवल जानकारी, निवेश सलाह नहीं।")


def row(name, c, ctx=None):
    note = ctx_note(name, c, ctx)
    return (f"{c['side']:<6}{name}  (w4 {c['r4'] * 100:.0f}%)\n"
            f" E {fp(c['entry'])}  SL {fp(c['stop'])}\n T {fp(c['target'])}  stop {c['stop_pct']:.1f}%"
            + (f"\n {note}" if note else ""))


def build_summary(results, index_df, now, ctx=None):
    ts = now.strftime("%a %d %b %Y, %H:%M IST")
    parts = [f"📐 *Elliott Wave स्कैन (1h)* — {ts}"]
    if ctx and ctx.get("index"):
        parts.append(f"Nifty 50 (अभी): {ctx['index']['ltp']:,.2f} ({ctx['index']['chg']:+.2f}%)")
    elif index_df is not None:
        close = index_df.Close
        prev = close[close.index.date < close.index[-1].date()]
        chg = (close.iloc[-1] / prev.iloc[-1] - 1) * 100 if len(prev) else 0.0
        parts.append(f"Nifty 50: {close.iloc[-1]:,.2f} ({chg:+.2f}%)")
    nc = results.get("NIFTY 50")
    if nc and nc["valid"]:
        parts.append(f"Nifty index: {nc['side']} setup ({nc['status']}) — Entry {fp(nc['entry'])}, "
                     f"SL {fp(nc['stop'])}, Target {fp(nc['target'])}")
    else:
        parts.append("Nifty index: कोई वैध entry गिनती नहीं")
    stocks = {k: v for k, v in results.items() if k != "NIFTY 50" and v and v["valid"]}
    for status, title in (("TRADE", "✅ *TRADE* (गिनती ठीक + H1 breakout)"),
                          ("WATCH", "👀 *WATCH* (गिनती ठीक, H1 breakout अभी नहीं)")):
        lst = [(k, v) for k, v in stocks.items() if v["status"] == status]
        if lst:
            parts.append(title + f" — {len(lst)}\n```\n" + "\n".join(row(k, v, ctx) for k, v in lst) + "\n```")
    if not stocks:
        parts.append("शेयरों में कोई वैध Elliott setup नहीं मिला।")
    parts.append("_Wave 4, wave 3 का 23.6–50% लौटा हो; Stop = wave-4 का सिरा (अधिकतम "
                 f"{CFG['max_stop_pct']:g}%); Target = {CFG['rr']:g}R। 1h गिनती अनुमान है, पक्की नहीं। "
                 "केवल जानकारी, निवेश सलाह नहीं।_")
    return "\n\n".join(parts)


def tile_color(chg):
    t = max(-3.0, min(3.0, chg)) / 3.0
    neutral, tgt = (0x3a, 0x3a, 0x40), ((0x0f, 0x9d, 0x62) if t >= 0 else (0xc0, 0x2a, 0x3a))
    return tuple((n + (g - n) * abs(t)) / 255 for n, g in zip(neutral, tgt))


def heatmap_png(rows, index, source, now):
    """Sector-wise heat map: one row per sector, one tile per stock, colour = % change."""
    by = {}
    for r in rows:
        by.setdefault(SECTOR_OF.get(r["sym"], "Other"), []).append(r)
    secs = sorted(by.items(), key=lambda kv: -sum(x["chg"] for x in kv[1]) / len(kv[1]))
    n = len(secs)
    fig = plt.figure(figsize=(10, 1.5 + 0.8 * n), dpi=100, facecolor=BG)
    ax = fig.add_axes([0.01, 0.02, 0.98, 1 - 1.3 / (1.5 + 0.8 * n)], facecolor=BG)
    ax.set_xlim(0, 10); ax.set_ylim(n, 0); ax.axis("off")
    tw = 8.0 / 6
    for i, (sec, lst) in enumerate(secs):
        avg = sum(x["chg"] for x in lst) / len(lst)
        ax.text(0.05, i + 0.38, sec, color="white", fontsize=12, fontweight="bold", va="center")
        ax.text(0.05, i + 0.68, f"{avg:+.2f}% avg", color=GREEN if avg >= 0 else RED, fontsize=10, va="center")
        for j, r in enumerate(sorted(lst, key=lambda x: -x["chg"])):
            x0 = 1.95 + j * tw
            ax.add_patch(Rectangle((x0, i + 0.06), tw - 0.07, 0.88, fc=tile_color(r["chg"]), ec="none"))
            ax.text(x0 + (tw - 0.07) / 2, i + 0.36, r["sym"], color="white", fontsize=8.5,
                    fontweight="bold", ha="center", va="center")
            ax.text(x0 + (tw - 0.07) / 2, i + 0.64, f"{r['chg']:+.2f}%", color="white", fontsize=8.5,
                    ha="center", va="center")
    h = 1.5 + 0.8 * n
    nifty = f"Nifty 50 {index['ltp']:,.2f} ({index['chg']:+.2f}%)" if index else "Nifty 50"
    fig.text(0.02, 1 - 0.35 / h, "Nifty 50 · sector heat map", color="white", fontsize=15, fontweight="bold")
    fig.text(0.02, 1 - 0.75 / h, nifty, color=GREEN if (index and index["chg"] >= 0) else RED, fontsize=11,
             fontweight="bold")
    fig.text(0.98, 1 - 0.35 / h, now.strftime("%d %b %H:%M IST"), color=MUTED, fontsize=8.5, ha="right")
    fig.text(0.98, 1 - 0.75 / h, f"source: {source} · colour = % change vs prev close (±3% scale)",
             color=MUTED, fontsize=8, ha="right")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    plt.close(fig)
    return buf.getvalue()


def live_context(now):
    """Live quotes from bot.py (Angel -> NSE -> Yahoo). None if unavailable."""
    try:
        import bot
        rows, index, source, _ = bot.fetch_all(now)
        if len(rows) < 40:
            return None
        sec = {}
        for r in rows:
            sec.setdefault(SECTOR_OF.get(r["sym"], "Other"), []).append(r["chg"])
        return {"rows": rows, "index": index, "source": source,
                "sector": {k: sum(v) / len(v) for k, v in sec.items()},
                "chg": {r["sym"]: r["chg"] for r in rows}}
    except Exception as e:
        print(f"live context unavailable: {e}", file=sys.stderr)
        return None


def main():
    now = datetime.now(IST)
    if now.weekday() >= 5 and "--force" not in sys.argv:
        print("Weekend - market closed."); return
    if yf is None:
        sys.exit("yfinance not installed")
    data = load_all(now)
    if len(data) < 40:
        sys.exit(f"Only {len(data)} symbols loaded; aborting.")
    results = {name: best_count(df) for name, df in data.items()}
    ctx = live_context(now)
    summary = build_summary(results, data.get("NIFTY 50"), now, ctx)

    order = [k for k, v in results.items()
