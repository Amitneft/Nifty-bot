"""Strategy C: Opening Range Breakout (ORB) - back-test, Rs 1000 per trade, intraday only.
Rules (from the user):
  * Opening range = high / low of the first 15 minutes (3 five-minute candles, 9:15-9:30); variant: first candle only.
  * LONG : a 5-minute candle CLOSES above the range high.  SHORT: closes below the range low (a wick is not enough).
  * Nifty 50 confirms : LONG only if Nifty's own 5-min close is above its own opening-range high (SHORT: below its low).
  * VWAP confirms     : LONG only if price is above the day's VWAP, SHORT only if below.
  * Volume confirms   : breakout candle volume >= 1.5 x the average volume of the opening-range candles.
  * Range filter      : range width 0.5 % - 2.5 % of price.
  * Stock selection   : opening-range volume >= 2 x the usual (same slot, previous 10 sessions), best 10 per day;
                        a clear trigger = gap >= 0.5 % either way, or the stock's sector moving >= 0.3 % in the trade direction;
                        breakout must also be beyond yesterday's close (LONG above, SHORT below).
  * Stop = the opposite end of the range (variants: range midpoint, breakout-candle low/high).
  * Target = 2 x the risk (1:2) (variant: 1 x range width).  Square-off at 15:15 (last candle 15:10-15:15).
My choices where the rules were open: entry = OPEN of the candle after the breakout candle; entries only until 14:00;
one trade per stock per day (the first qualifying breakout); risk (entry-stop) must be at least 0.2 %.
Costs 0.10 % per trade, stop first if one candle touches both.
Run:  python orb.py [--adays 365] [--dry]
"""
import sys
from datetime import time as dtime

import numpy as np
import pandas as pd

import backtest as bt
import ew_scan
import filters
import variants as vr

COST_NOTE = "लागत 0.10% प्रति ट्रेड"
BASE = dict(orb_n=3, stop="end", target="2R", nifty="orb", vwap=True, vol_mult=1.5, relvol_min=2.0, top_n=10,
            trigger=True, prev="close", range_min=0.5, range_max=2.5, until=dtime(14, 0), min_risk=0.2,
            gap_min=0.5, sector_min=0.3)
VARIANTS = [
    ("आधार (15मि)", {}),
    ("5मि range", dict(orb_n=1)),
    ("stop=mid", dict(stop="mid")),
    ("stop=candle", dict(stop="candle")),
    ("target=range", dict(target="range")),
    ("Nifty mid", dict(nifty="mid")),
    ("बिना Nifty", dict(nifty="off")),
    ("बिना VWAP", dict(vwap=False)),
    ("बिना volume", dict(vol_mult=0.0)),
    ("बिना relvol", dict(relvol_min=0.0, top_n=999)),
    ("बिना trigger", dict(trigger=False)),
    ("prev high/low", dict(prev="hl")),
    ("सब ढीला", dict(nifty="off", vwap=False, vol_mult=0.0, relvol_min=0.0, top_n=999, trigger=False, prev="off")),
]
bt.STRAT["C"] = "C: Opening Range Breakout"


def prepare(data5):
    """name -> {date: (index, O, H, L, C, V, prev_close, prev_high, prev_low)}"""
    out = {}
    for name, df in data5.items():
        days, prev = {}, None
        v = df["Volume"].to_numpy(float) if "Volume" in df.columns else np.zeros(len(df))
        dates = np.array(df.index.date)
        O, H, L, C = (df[c].to_numpy(float) for c in ("Open", "High", "Low", "Close"))
        for d in sorted(set(dates)):
            m = np.where(dates == d)[0]
            a, b = m[0], m[-1] + 1
            days[d] = (df.index[a:b], O[a:b], H[a:b], L[a:b], C[a:b], v[a:b],
                       prev[0] if prev else None, prev[1] if prev else None, prev[2] if prev else None)
            prev = (C[b - 1], H[a:b].max(), L[a:b].min())
        out[name] = days
    return out


def open_volume_ratio(days, n):
    """date -> opening volume (first n candles) / average of the same on the previous 10 sessions (min 5)."""
    ds = sorted(days)
    sums = [float(np.nansum(days[d][5][:n])) for d in ds]
    res = {}
    for i, d in enumerate(ds):
        base = sums[max(0, i - 10): i]
        res[d] = (sums[i] / np.mean(base)) if len(base) >= 5 and np.mean(base) > 0 else np.nan
    return res


def scan(prep, ctx, p):
    """All ORB trades for one parameter set -> list of trade dicts."""
    p = {**BASE, **p}
    n = p["orb_n"]
    nifty = prep.get("NIFTY 50", {})
    stocks = [k for k in prep if k not in filters.INDEXES]
    rv = {k: open_volume_ratio(prep[k], n) for k in stocks}
    trades = []
    for d in sorted(nifty):
        ni = nifty[d]
        if len(ni[0]) < n + 3:
            continue
        nhi, nlo = ni[2][:n].max(), ni[3][:n].min()
        nmid = (nhi + nlo) / 2
        npos = {t: i for i, t in enumerate(ni[0])}
        cands = [(rv[k].get(d, np.nan), k) for k in stocks if d in prep[k]]
        cands = [(r, k) for r, k in cands if not np.isnan(r) and r >= p["relvol_min"]]
        cands.sort(reverse=True)
        for _, name in cands[: p["top_n"]]:
            idx, O, H, L, C, V, pc, ph, pl = prep[name][d]
            if len(idx) < n + 3 or pc is None:
                continue
            hi, lo = H[:n].max(), L[:n].min()
            width = (hi - lo) / lo * 100
            if not (p["range_min"] <= width <= p["range_max"]):
                continue
            mean_or = float(np.nanmean(V[:n]))
            gap = (O[0] / pc - 1) * 100
            tp = (H + L + C) / 3
            cumv = np.cumsum(V)
            vwap = np.where(cumv > 0, np.cumsum(tp * V) / np.where(cumv > 0, cumv, 1), np.nan)
            end = int(np.where(np.array([t.time() for t in idx]) <= dtime(15, 10))[0][-1])
            for k in range(n, end):
                if idx[k].time() > p["until"]:
                    break
                c = C[k]
                side = "LONG" if c > hi else ("SHORT" if c < lo else None)
                if side is None:
                    continue
                L_ = side == "LONG"
                if p["nifty"] != "off":
                    j = npos.get(idx[k])
                    if j is None:
                        continue
                    ref = nhi if p["nifty"] == "orb" else nmid
                    nlev = nlo if p["nifty"] == "orb" else nmid
                    if (L_ and not ni[4][j] > ref) or (not L_ and not ni[4][j] < nlev):
                        continue
                if p["vwap"] and not np.isnan(vwap[k]) and ((L_ and c <= vwap[k]) or (not L_ and c >= vwap[k])):
                    continue
                if p["vol_mult"] and not (mean_or > 0 and V[k] >= p["vol_mult"] * mean_or):
                    continue
                if p["prev"] == "close" and ((L_ and c <= pc) or (not L_ and c >= pc)):
                    continue
                if p["prev"] == "hl" and ((L_ and c <= ph) or (not L_ and c >= pl)):
                    continue
                if p["trigger"] and abs(gap) < p["gap_min"]:
                    sec = ctx.market_at(name, idx[k])[1]
                    if sec is None or (L_ and sec < p["sector_min"]) or (not L_ and sec > -p["sector_min"]):
                        continue
                entry = O[k + 1]
                stop = {"end": lo if L_ else hi, "mid": (hi + lo) / 2, "candle": L[k] if L_ else H[k]}[p["stop"]]
                risk = abs(entry - stop)
                tgt_d = 2 * risk if p["target"] == "2R" else (hi - lo)
                t1 = entry + tgt_d if L_ else entry - tgt_d
                if risk / entry * 100 < p["min_risk"] or not ((stop < entry < t1) if L_ else (t1 < entry < stop)):
                    break                                             # no valid trade from this breakout
                m = end + 1
                r = bt.run_trade(idx[:m], O[:m], H[:m], L[:m], C[:m], k + 1, entry, side, stop, t1, "intraday", d,
                                 gap_first=False, final=True)
                trades.append({"strat": "C", "name": name, "side": side, "mode": "intraday",
                               "entry_time": idx[k + 1], "exit_time": idx[r["j"]] + pd.Timedelta(minutes=5),
                               "entry": entry, "exit": r["exit"], "why": r["why"],
                               "net": bt.net_pct(side, entry, r["exit"]), "candles": r["j"] - k})
                break                                                 # one trade per stock per day
    return trades


def compact_row(label, a, b):
    ok = a["n"] >= 20 and b["n"] >= 20 and a["avg"] > 0 and b["avg"] > 0
    return (f"{'✅' if ok else '  '}{label:<11}{a['n']:4d}{a['win']:4.0f}{a['avg']:+6.2f}|"
            f"{b['n']:4d}{b['win']:4.0f}{b['avg']:+6.2f}")


def report(data5):
    prep = prepare(data5)
    ctx = filters.Ctx(data5)
    cut, first, last = vr.cut_of(data5)
    base_trades, rows = None, []
    for label, ch in VARIANTS:
        tr = scan(prep, ctx, ch)
        if base_trades is None:
            base_trades = tr
        a, b = vr.half_stats(tr, cut)
        rows.append(compact_row(label, a, b))
    taken = {("C", "intraday"): base_trades}
    s = bt.stats1k(base_trades)
    head = (f"*C: Opening Range Breakout — आधार नियम* (हर ट्रेड ₹{bt.PER_TRADE:,.0f}, केवल इंट्राडे)\n"
            f"_{first:%d %b %Y} – {last:%d %b %Y}; {COST_NOTE}_\n\n" + bt.block("C", "intraday", s))
    msgs = [head]
    yt = bt.year_text(taken)
    if yt:
        msgs.append(yt)
    msgs.append(f"*C: वेरिएंट टेस्ट* (बीच की तारीख़ {cut:%d %b %Y})\n```\n"
                f"{'':13}ट्रेन      |टेस्ट\n{'नाम':13}n  W% avg%| n  W% avg%\n" + "\n".join(rows) + "\n```\n"
                "_n = ट्रेड, W% = Win%, avg% = लागत के बाद हर ट्रेड का औसत। ✅ = दोनों हिस्सों में औसत > 0 और हर हिस्से में ≥20 ट्रेड। "
                "'बिना…' वाली पंक्तियाँ बताती हैं कि वह शर्त हटाने पर क्या बदलता है। 13 वेरिएंट आज़माए हैं, इसलिए ✅ भी पक्का सबूत नहीं। "
                "केवल जानकारी, निवेश सलाह नहीं।_")
    return msgs


def main():
    adays, dry = 365, "--dry" in sys.argv
    if "--adays" in sys.argv:
        adays = max(30, min(1200, int(sys.argv[sys.argv.index("--adays") + 1])))
    data5 = bt.load_history5(adays)
    if len(data5) < 40 or "NIFTY 50" not in data5:
        msg = "⚠️ ORB का टेस्ट नहीं हो पाया (5-मिनट का डेटा या Nifty 50 का डेटा कम मिला)।"
        print(msg) if dry else ew_scan.send_text(msg)
        return
    for m in report(data5):
        print(m, "\n") if dry else ew_scan.send_text(m)


if __name__ == "__main__":
    main()
      
