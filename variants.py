"""Variant test: does a small change to the rules make strategy A or B better - on data the rule was NOT picked on?
The history is cut in two halves of equal length. Every variant is shown on the first half (TRAIN) and on the
second half (TEST). A variant is only interesting if it is positive in BOTH halves.
Same engine as backtest.py: Rs 1000 per trade, cost 0.10 %, stop first if both are touched, no overlapping trades.
Run:  python variants.py [--days 1095] [--adays 365] [--dry]
"""
import os, sys
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd

import backtest as bt
import ew_scan
import filters
from ew_scan import IST

# (label, config changes, ideas per day and block)
A_VARIANTS = [
    ("आधार SL0.5 T1", {}, 8),
    ("SL 0.7", {"stop_pct": 0.7}, 8),
    ("SL 1.0", {"stop_pct": 1.0}, 8),
    ("SL1.0 T1.5", {"stop_pct": 1.0, "target_pct": 1.5}, 8),
    ("volume 2x", {"min_volume_ratio": 2.0}, 8),
    ("volume 3x", {"min_volume_ratio": 3.0}, 8),
    ("टॉप 3 रोज़", {}, 3),
    ("vol2x + टॉप3", {"min_volume_ratio": 2.0}, 3),
    ("move 1.5%+", {"min_move_pct": 1.5}, 8),
    ("vol3x SL1.0", {"min_volume_ratio": 3.0, "stop_pct": 1.0}, 8),
]
B_VARIANTS = ["सभी filter (आधार)", "सिर्फ़ volume", "सिर्फ़ Nifty+सेक्टर", "कोई filter नहीं",
              "कोई filter नहीं, सिर्फ़ LONG", "कोई filter नहीं, सिर्फ़ SHORT"]


class NoMarket:
    """Context that never blocks (no Nifty / sector rule)."""
    def market_at(self, name, t):
        return None, None


def half_stats(trades, cut):
    out = []
    for part in ([t for t in trades if t["entry_time"] < cut], [t for t in trades if t["entry_time"] >= cut]):
        n = len(part)
        w = sum(t["net"] > 0 for t in part)
        out.append({"n": n, "win": 100 * w / n if n else 0.0,
                    "avg": float(np.mean([t["net"] for t in part])) if n else 0.0,
                    "pnl": float(sum(bt.PER_TRADE * t["net"] / 100 for t in part))})
    return out


def row(label, tr, te):
    both = tr["n"] >= 20 and te["n"] >= 20 and tr["avg"] > 0 and te["avg"] > 0
    return (f"{'✅' if both else '  '}{label:<18}{tr['n']:5d} {tr['win']:4.0f}% {tr['avg']:+.2f} |"
            f"{te['n']:5d} {te['win']:4.0f}% {te['avg']:+.2f}")


HEAD = "   (नाम)              ट्रेन: ट्रेड Win% औसत% | टेस्ट: ट्रेड Win% औसत%"


def cut_of(data):
    first = min(df.index[0] for df in data.values())
    last = max(df.index[-1] for df in data.values())
    return first + (last - first) / 2, first, last


def run_A(data5):
    cfg0 = bt.load_criteria()
    ctx = filters.Ctx(data5)
    vols = {n: filters.slot_ratio(df) for n, df in data5.items()}
    cut, first, last = cut_of(data5)
    lines = {"intraday": [], "swing": []}
    for label, chg, topn in A_VARIANTS:
        cfg = {**cfg0, **chg}
        old, bt.A_MAX_IDEAS = bt.A_MAX_IDEAS, topn
        try:
            sigs = bt.scan_movers(data5, cfg, ctx, vols, True)
        finally:
            bt.A_MAX_IDEAS = old
        for mode in lines:
            tr = bt.no_overlap(bt.trades_A(data5, sigs, mode))
            a, b = half_stats(tr, cut)
            lines[mode].append(row(label, a, b))
    head = f"*A: टॉप गेनर/लूज़र + volume — वेरिएंट टेस्ट*\n_{first:%d %b %Y} – {last:%d %b %Y}; बीच की तारीख़ {cut:%d %b %Y}_"
    return [head + f"\n*{bt.LABEL[m]}*\n```\n{HEAD}\n" + "\n".join(lines[m]) + "\n```" for m in lines]


def run_B(data):
    items = list(data.items())
    try:
        with Pool(max(1, (os.cpu_count() or 2))) as pool:
            scanned = pool.map(bt.scan_symbol, items, chunksize=4)
    except Exception as e:
        print(f"pool failed ({e!r}); single process", file=sys.stderr)
        scanned = [bt.scan_symbol(x) for x in items]
    states = [s for _, lst in scanned for s in lst]
    ctx, nomkt = filters.Ctx(data), NoMarket()
    vols = {n: filters.slot_ratio(df) for n, df in data.items()}
    novol = {n: pd.Series(99.0, index=df.index) for n, df in data.items()}
    cut, first, last = cut_of(data)
    sets = {
        "सभी filter (आधार)": bt.signals_from_states(data, states, ctx, vols, True),
        "सिर्फ़ volume": bt.signals_from_states(data, states, nomkt, vols, True),
        "सिर्फ़ Nifty+सेक्टर": bt.signals_from_states(data, states, ctx, novol, True),
        "कोई filter नहीं": bt.signals_from_states(data, states, ctx, vols, False),
    }
    sets["कोई filter नहीं, सिर्फ़ LONG"] = [s for s in sets["कोई filter नहीं"] if s["side"] == "LONG"]
    sets["कोई filter नहीं, सिर्फ़ SHORT"] = [s for s in sets["कोई filter नहीं"] if s["side"] == "SHORT"]
    lines = {"intraday": [], "swing": []}
    for label in B_VARIANTS:
        for mode in lines:
            tr = bt.no_overlap(bt.trades_B(data, sets[label], mode))
            a, b = half_stats(tr, cut)
            lines[mode].append(row(label, a, b))
    head = f"*B: Elliott Wave — वेरिएंट टेस्ट*\n_{first:%d %b %Y} – {last:%d %b %Y}; बीच की तारीख़ {cut:%d %b %Y}_"
    return [head + f"\n*{bt.LABEL[m]}*\n```\n{HEAD}\n" + "\n".join(lines[m]) + "\n```" for m in lines]


NOTE = ("_औसत% = हर ट्रेड पर लागत (0.10%) घटाकर औसत नतीजा। ✅ = दोनों हिस्सों में औसत > 0 और हर हिस्से में कम से कम 20 ट्रेड। "
        "इतने वेरिएंट आज़माने पर कुछ संयोग से भी अच्छे दिखेंगे; ✅ वाला भी पक्का सबूत नहीं, आगे paper trading में परखना है। "
        "केवल जानकारी, निवेश सलाह नहीं।_")


def main():
    days, adays, dry = 1095, 365, "--dry" in sys.argv
    if "--days" in sys.argv:
        days = max(120, min(1200, int(sys.argv[sys.argv.index("--days") + 1])))
    if "--adays" in sys.argv:
        adays = max(30, min(1200, int(sys.argv[sys.argv.index("--adays") + 1])))
    data = bt.load_history(days)
    data5 = bt.load_history5(adays)
    msgs = []
    if len(data5) >= 40:
        msgs += run_A(data5)
    else:
        msgs.append("⚠️ A का वेरिएंट टेस्ट नहीं हो पाया (5-मिनट का डेटा कम मिला)।")
    if len(data) >= 40:
        msgs += run_B(data)
    else:
        msgs.append("⚠️ B का वेरिएंट टेस्ट नहीं हो पाया (घंटे का डेटा कम मिला)।")
    msgs.append(NOTE)
    for m in msgs:
        print(m, "\n") if dry else ew_scan.send_text(m)


def run_with(data, data5, dry=True):
    """For tests: same as main() but with data in hand."""
    msgs = run_A(data5) + run_B(data) + [NOTE]
    for m in msgs:
        print(m, "\n")
    return msgs


if __name__ == "__main__":
    main()
                                      
