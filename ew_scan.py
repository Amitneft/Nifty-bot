"""Elliott Wave engine (rules, charts, captions, Telegram senders). Used by report.py and ew_alert.py.

Looks for an impulse count 0-1-2-3-4 (long = uptrend, short = downtrend) and, when wave 4
sits in the 14.6-38.2% zone of wave 3, gives entry / stop / two wave-5 targets.
Elliott guidelines used: W2 50/61.8/76.4/85.4% of W1, W3 >= 161.8% of W1, W4 14.6/23.6/38.2% of W3,
W5 = 123.6-161.8% of W4, or equal to W1, or 61.8% of waves 0-3.
"""
import io, os, sys, textwrap
from datetime import timezone
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

from universe import UNIVERSE, SECTOR_OF  # Nifty 100 list + sectors live in universe.py

CFG = {
    "zone_lo": 0.146,         # wave 4 retraces 14.6% ... 
    "zone_hi": 0.382,         # ... to 38.2% of wave 3 (the entry zone)
    "zone_tol": 0.03,         # tolerance around the zone edges (3 percentage points)
    "w2_levels": (0.50, 0.618, 0.764, 0.854),   # wave 2 as a fraction of wave 1
    "w2_tol": 0.06,
    "w3_min": 1.5,            # wave 3 >= 161.8% of wave 1 (1.5 = a little tolerance)
    "w4_levels": (0.146, 0.236, 0.382),         # wave 4 as a fraction of wave 3
    "w4_tol": 0.04,
    "w5_w4": (1.236, 1.618),  # wave 5 = 123.6% / 161.8% of wave 4 ...
    "w5_w123": 0.618,         # ... or 61.8% of waves 0-3 ... (and "equal to wave 1")
    "min_score": 2,           # at least this many of the 3 guidelines (W2, W3, W4 level) must fit
    "min_rr": 1.5,            # reward : risk to the first target
    "min_stop_pct": 0.3,      # stop tighter than this is rejected (just noise, costs would eat it)
    "max_stop_pct": 4.5,      # stop wider than this is rejected
    "h1_lookback": 10,        # H1 breakout = close beyond the previous N candles' high/low
    "h1_window": 4,           # ... within the last N hours
    "atr_mult": (7, 5, 3.5),  # zig-zag swing sizes tried (x ATR); big swings first
    "max_charts": 8,
    "max_list": 12,           # rows per TRADE / WATCH list in the summary message
}

BG, GRID, TXT, MUTED = "#1c1c1f", "#2c2c30", "#e8e8ea", "#8d8d93"
GREEN, RED, BLUE, GOLD = "#0ecb81", "#f6465d", "#4da3ff", "#f5b800"


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


def _near(x, levels, tol):
    """nearest guideline level within tol, else None"""
    lv = min(levels, key=lambda l: abs(x - l))
    return lv if abs(x - lv) <= tol else None


def analyse(df, pts):
    """Test five pivots as impulse 0-1-2-3-4. Everything is done in 'long space'
    (prices x -1 for a downtrend) so one set of rules serves both directions."""
    s = 1 if pts[0][1] < pts[1][1] else -1
    side = "LONG" if s == 1 else "SHORT"
    q = [s * p for _, p in pts]
    O, H, L, C = (df[c].values for c in ("Open", "High", "Low", "Close"))
    qH, qC = (H, C) if s == 1 else (-L, -C)
    close_q, n = qC[-1], len(C)
    w1, w3, w4 = q[1] - q[0], q[3] - q[2], q[3] - q[4]
    zl, zh, tol = CFG["zone_lo"], CFG["zone_hi"], CFG["zone_tol"]
    r2 = (q[1] - q[2]) / w1 if w1 > 0 else 9.0
    ext3 = w3 / w1 if w1 > 0 else 0.0
    r4 = w4 / w3 if w3 > 0 else 9.0
    r_now = (q[3] - close_q) / w3 if w3 > 0 else 9.0
    fails = []
    # --- hard rules (never broken in an impulse)
    if q[2] <= q[0]:
        fails.append("Wave 2 retraced past the start of wave 1")
    if q[3] <= q[1]:
        fails.append("Wave 3 did not pass the end of wave 1")
    elif w3 < w1:
        fails.append("Wave 3 is shorter than wave 1")
    if q[4] <= q[1]:
        fails.append("Wave 4 overlaps wave 1")
    # --- wave 4 entry zone: 14.6-38.2% of wave 3 (plus tolerance)
    if r4 > zh + tol:
        fails.append(f"Wave 4 ({side.lower()}) has pulled back {r4 * 100:.0f}% of wave 3, past the "
                     f"{(zh + tol) * 100:.0f}% limit: no entry on this count.")
    elif r4 < zl - tol:
        fails.append(f"Wave 4 ({side.lower()}) pulled back only {r4 * 100:.0f}% of wave 3, under {(zl - tol) * 100:.0f}%.")
    elif not (zl - tol <= r_now <= zh + tol):
        fails.append(f"Price sits at {r_now * 100:.0f}% of wave 3, outside the "
                     f"{zl * 100:.1f}-{zh * 100:.1f}% entry zone.")
    # --- guidelines (often, not always): score out of 3
    l2, l4 = _near(r2, CFG["w2_levels"], CFG["w2_tol"]), _near(r4, CFG["w4_levels"], CFG["w4_tol"])
    g2, g3, g4 = l2 is not None, ext3 >= CFG["w3_min"], l4 is not None
    score = int(g2) + int(g3) + int(g4)
    if score < CFG["min_score"]:
        fails.append(f"Guideline fit {score}/3 (W2 {r2 * 100:.0f}%, W3 {ext3 * 100:.0f}% of W1, "
                     f"W4 {r4 * 100:.0f}% of W3): need {CFG['min_score']}.")
    # --- stop, wave-5 targets, reward:risk
    risk = close_q - q[4]
    stop_pct = risk / abs(close_q) * 100 if close_q else 99
    projs = {
        f"{CFG['w5_w4'][0] * 100:.1f}% of W4": q[4] + CFG["w5_w4"][0] * w4,
        f"{CFG['w5_w4'][1] * 100:.1f}% of W4": q[4] + CFG["w5_w4"][1] * w4,
        "W5 = W1": q[4] + w1,
        f"{CFG['w5_w123'] * 100:.1f}% of W0-3": q[4] + CFG["w5_w123"] * (q[3] - q[0]),
    }
    ok_t = sorted((v, k) for k, v in projs.items() if v > max(q[3], close_q))
    t1_q, t1_lbl = ok_t[0] if ok_t else (close_q, "-")
    t2_q, t2_lbl = ok_t[-1] if ok_t else (close_q, "-")
    rr1 = (t1_q - close_q) / risk if risk > 0 else 0.0
    if risk <= 0:
        fails.append("Price is already beyond the wave-4 extreme (stop).")
    else:
        if stop_pct > CFG["max_stop_pct"]:
            fails.append(f"Stop width {stop_pct:.1f}% is above the {CFG['max_stop_pct']:g}% maximum.")
        elif stop_pct < CFG["min_stop_pct"]:
            fails.append(f"Stop width {stop_pct:.2f}% is under the {CFG['min_stop_pct']:g}% minimum (too tight).")
        if not ok_t:
            fails.append("No wave-5 projection clears the end of wave 3.")
        elif rr1 < CFG["min_rr"]:
            fails.append(f"Reward:risk to target 1 is only {rr1:.1f} (need {CFG['min_rr']:g}).")
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
        "r2": r2, "ext3": ext3, "r4": r4, "r_now": r_now, "stop_pct": stop_pct, "h1_ago": h1_ago,
        "g": (g2, g3, g4), "score": score, "rr1": rr1, "t1_lbl": t1_lbl, "t2_lbl": t2_lbl,
        "entry": s * close_q, "stop": s * q[4], "target": s * t1_q, "target2": s * t2_q,
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
    cands.sort(key=lambda c: (not c["valid"], len(c["fails"]), -c["score"], c["h1_ago"] is None, -c["pts"][4][0]))
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
        levels += [c["stop"], c["target"], c["target2"]]
        lo_z, hi_z = sorted(c["zone"])
        ax.add_patch(Rectangle((xs[3], lo_z), right - xs[3], hi_z - lo_z, fc=GOLD, alpha=0.16, ec="none", zorder=1))
        ax.text(right, lo_z, f"entry zone {CFG['zone_lo'] * 100:.1f}-{CFG['zone_hi'] * 100:.1f}%", color=GOLD, fontsize=8.5, ha="right", va="top")
        ax.hlines(c["stop"], xs[4], right, colors=RED, linestyles="--", lw=1.3, zorder=4)
        ax.text(right, c["stop"], f"stop {fp(c['stop'])}", color=RED, fontsize=8.5, ha="right",
                va="bottom" if s == -1 else "top")
        ax.hlines(c["target"], xs[3], right, colors=GREEN, linestyles="--", lw=1.3, zorder=4)
        ax.text(right, c["target"], f"target 1 {fp(c['target'])}  ({c['t1_lbl']})", color=GREEN, fontsize=8.5,
                ha="right", va="bottom")
        if abs(c["target2"] - c["target"]) > 1e-9:
            ax.hlines(c["target2"], xs[3], right, colors=GREEN, linestyles=":", lw=1.1, zorder=4, alpha=0.8)
            ax.text(right, c["target2"], f"target 2 {fp(c['target2'])}  ({c['t2_lbl']})", color=GREEN,
                    fontsize=8, ha="right", va="bottom", alpha=0.9)
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
    right_txt = (f"entry {fp(c['entry'])} · stop {fp(c['stop'])} · T1 {fp(c['target'])} · T2 {fp(c['target2'])}"
                 if ok else "entry now: none")
    fig.text(0.975, 0.897, right_txt, color=TXT, fontsize=10, ha="right")

    if ok:
        mk = lambda b: "ok" if b else "-"
        l1 = (f"W2 {c['r2'] * 100:.0f}% of W1 ({mk(c['g'][0])}) · W3 {c['ext3'] * 100:.0f}% of W1 ({mk(c['g'][1])}) · "
              f"W4 {c['r4'] * 100:.0f}% of W3 ({mk(c['g'][2])}) · fit {c['score']}/3")
    else:
        l1 = c["fails"][0] if c["fails"] else ""
    if c["h1_ago"] is not None:
        l2 = f"H1 breakout the same way {c['h1_ago']} candle(s) ago: confirmed."
    else:
        l2 = (f"No H1 breakout the same way in the last {CFG['h1_window']}h. "
              f"H1 comes first (at the wave-3 push), so this count is unlikely to trade.")
    l3 = (f"Stop width {c['stop_pct']:.1f}% (max {CFG['max_stop_pct']:g}%) · reward:risk to T1 = {c['rr1']:.1f}"
          if c["valid"] else (f"Stop width now {c['stop_pct']:.1f}% (max {CFG['max_stop_pct']:g}%)" if c["stop_pct"] > 0 else ""))
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


def grade(c):
    return {3: "A ⭐⭐⭐", 2: "B ⭐⭐", 1: "C ⭐"}.get(c["score"], "-")


def caption_for(name, c, ctx=None):
    tag = "TRADE (H1 पक्का)" if c["status"] == "TRADE" else "WATCH (H1 breakout अभी नहीं)"
    mk = lambda b: "✓" if b else "✗"
    note = ctx_note(name, c, ctx)
    return (f"{'📈' if c['side'] == 'LONG' else '📉'} {name} — {c['side']} — {tag}\n"
            f"Entry {fp(c['entry'])} | SL {fp(c['stop'])}\n"
            f"Target 1 {fp(c['target'])} ({c['t1_lbl']})\nTarget 2 {fp(c['target2'])} ({c['t2_lbl']})\n"
            f"Reward:Risk (T1) 1:{c['rr1']:.1f} | Stop चौड़ाई {c['stop_pct']:.1f}%\n"
            f"W2 {c['r2'] * 100:.0f}% {mk(c['g'][0])} | W3 {c['ext3'] * 100:.0f}% {mk(c['g'][1])} | "
            f"W4 {c['r4'] * 100:.0f}% {mk(c['g'][2])} | ग्रेड {grade(c)}\n"
            + (note + "\n" if note else "")
            + "केवल जानकारी, निवेश सलाह नहीं।")


def row(name, c, ctx=None):
    note = ctx_note(name, c, ctx)
    return (f"{c['side']:<6}{name}  ({grade(c)[0]}, w4 {c['r4'] * 100:.0f}%)\n"
            f" E {fp(c['entry'])}  SL {fp(c['stop'])}\n T1 {fp(c['target'])}  T2 {fp(c['target2'])}\n"
            f" RR 1:{c['rr1']:.1f}  stop {c['stop_pct']:.1f}%"
            + (f"\n {note}" if note else ""))


FOOTER = ("_नियम: W2 = W1 का 50/61.8/76.4/85.4%, W3 ≥ W1 का 161.8%, W4 = W3 का 14.6/23.6/38.2% "
          f"(entry पट्टी 14.6–38.2%); ग्रेड = इन 3 में से कितने मिले (कम से कम {CFG['min_score']})। "
          f"Stop = wave-4 का सिरा (अधिकतम {CFG['max_stop_pct']:g}%); Target 1/2 = wave 5 के अनुमान "
          f"(W4 का 123.6/161.8%, W1 के बराबर, W0-3 का 61.8%); Reward:Risk ≥ {CFG['min_rr']:g}। "
          "1h गिनती अनुमान है, पक्की नहीं। केवल जानकारी, निवेश सलाह नहीं।_")


def tile_color(chg):
    t = max(-3.0, min(3.0, chg)) / 3.0
    neutral, tgt = (0x3a, 0x3a, 0x40), ((0x0f, 0x9d, 0x62) if t >= 0 else (0xc0, 0x2a, 0x3a))
    return tuple((n + (g - n) * abs(t)) / 255 for n, g in zip(neutral, tgt))
