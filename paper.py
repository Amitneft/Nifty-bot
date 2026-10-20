"""Paper trading (live forward test) of BOTH strategies, each in two styles. Every trade = Rs 1000.
  A : the morning "top gainer / loser + volume" ideas (opened by the 9:15 report)
  B : the Elliott Wave TRADE alerts (opened by the hourly alert)
Each signal is "taken" on paper twice: as an intraday trade and as a swing trade (4 books in all),
and followed until stop / target / time exit. A trade only opens when a signal really appears, so the
numbers stay at 0 until then - that is normal, not an error.
State lives in paper_trades.json (the workflow commits it back to the repo after the runs).
  python paper.py            -> send the current paper-trading summary to Telegram
"""
import json, os, sys
from datetime import datetime

import numpy as np
import pandas as pd

import ew_scan
from backtest import A_SWING_MAX, PER_TRADE, SWING_MAX, net_pct, run_trade

STATE = "paper_trades.json"
VERSION = 2
STRATS = ("A", "B")
MODES = ("intraday", "swing")
NAME = {"A": "A: टॉप गेनर/लूज़र + volume", "B": "B: Elliott Wave"}
LABEL = {"intraday": "इंट्राडे", "swing": "स्विंग"}


def key(s, m):
    return f"{s}_{m}"


def new_state(now):
    return {"version": VERSION, "since": now.strftime("%Y-%m-%d"), "per_trade": PER_TRADE,
            "books": {key(s, m): {"open": [], "closed": []} for s in STRATS for m in MODES}}


def load(now):
    try:
        st = json.load(open(STATE))
        if st.get("version") != VERSION:
            return new_state(now)
        for s in STRATS:
            for m in MODES:
                st["books"].setdefault(key(s, m), {"open": [], "closed": []})
        return st
    except Exception:
        return new_state(now)


def save(st):
    json.dump(st, open(STATE, "w"), ensure_ascii=False, indent=1)


def market_open(now):
    """Paper trades are opened only while the market is open (not in forced night-time tests)."""
    return now.weekday() < 5 and (9, 15) <= (now.hour, now.minute) <= (15, 30)


def open_names(st, strat):
    return sorted({p["name"] for m in MODES for p in st["books"][key(strat, m)]["open"]})


def settle(st, hourly, five, now):
    """Close paper trades that hit stop / target / time exit. A uses 5-minute candles, B 1-hour candles.
    Returns a list of Hindi lines."""
    lines = []
    for s in STRATS:
        data = five if s == "A" else hourly
        for m in MODES:
            bk = st["books"][key(s, m)]
            keep = []
            for p in bk["open"]:
                df = (data or {}).get(p["name"])
                if df is None:
                    keep.append(p); continue
                day0 = pd.Timestamp(p["sig_time"]).date()
                eod = (m == "intraday" and now.date() == day0 and (now.hour, now.minute) >= (15, 15))
                if m == "intraday" and now.date() > day0:      # missed the end of its day: close it there
                    df = df[np.array(df.index.date) == day0]
                    eod = True
                idx = df.index
                pos = np.where(idx > pd.Timestamp(p["sig_time"]))[0]
                if len(pos) == 0:
                    keep.append(p); continue
                r = run_trade(idx, df.Open.values, df.High.values, df.Low.values, df.Close.values, int(pos[0]),
                              p["entry"], p["side"], p["stop"], p["t1"], m, day0, gap_first=True, final=eod,
                              max_candles=A_SWING_MAX if s == "A" else SWING_MAX)
                if r is None:
                    keep.append(p); continue
                net = net_pct(p["side"], p["entry"], r["exit"])
                pnl = PER_TRADE * net / 100
                bk["closed"].append({"name": p["name"], "side": p["side"], "entry": p["entry"],
                                     "exit": float(r["exit"]), "why": r["why"], "net": net, "pnl": pnl,
                                     "opened": p["sig_time"], "closed": str(idx[r["j"]])})
                lines.append(f"{'✅' if net > 0 else '❌'} [{s} {LABEL[m]}] {p['name']} {p['side']} — {r['why']} — "
                             f"Entry {p['entry']:.2f} → {float(r['exit']):.2f} ({net:+.2f}%), ₹{pnl:+.1f}")
            bk["open"] = keep
    return lines


def open_new(st, signals, now):
    """signals: dicts(strat, name, side, entry, stop, t1, sig_time). Returns Hindi lines."""
    lines = []
    if not market_open(now):
        return lines
    for s in signals:
        today = pd.Timestamp(s["sig_time"]).date()
        for m in MODES:
            bk = st["books"][key(s["strat"], m)]
            if any(p["name"] == s["name"] for p in bk["open"]) or \
               any(c["name"] == s["name"] and pd.Timestamp(c["opened"]).date() == today for c in bk["closed"]):
                continue                                         # one trade per stock, style and day
            bk["open"].append({**s, "mode": m, "size": PER_TRADE})
            lines.append(f"📒 [{s['strat']} {LABEL[m]}] पेपर ट्रेड खुला: {s['name']} {s['side']} @ {s['entry']:.2f}, "
                         f"SL {s['stop']:.2f}, T1 {s['t1']:.2f}")
    return lines


def summary(st):
    out = [f"📒 *पेपर ट्रेडिंग (लाइव फॉरवर्ड टेस्ट)* — {st['since']} से, हर ट्रेड ₹{PER_TRADE:,.0f}"]
    total = 0
    for s in STRATS:
        for m in MODES:
            bk = st["books"][key(s, m)]
            cl = bk["closed"]
            n = len(cl)
            total += n + len(bk["open"])
            w = sum(c["net"] > 0 for c in cl)
            pnl = sum(c["pnl"] for c in cl)
            avg = (sum(c["net"] for c in cl) / n) if n else 0.0
            out.append(f"*{NAME[s]} — {LABEL[m]}*\n```\n"
                       f"बंद ट्रेड      : {n}\n"
                       f"जीत / हार      : {w} / {n - w}\n"
                       f"Win % / Loss % : {100 * w / n if n else 0:.1f}% / {100 * (n - w) / n if n else 0:.1f}%\n"
                       f"कुल लाभ/हानि   : ₹{pnl:+,.1f}  (औसत {avg:+.2f}% प्रति ट्रेड)\n"
                       f"खुले ट्रेड     : {len(bk['open'])}"
                       + ("\n" + "\n".join(f"  {p['name']} {p['side']} @ {p['entry']:.2f}" for p in bk["open"])
                          if bk["open"] else "")
                       + "\n```")
    if not total:
        out.append("_अभी तक कोई पेपर ट्रेड खुला ही नहीं, इसलिए सब 0 है। ट्रेड तब खुलता है जब सच में कोई "
                   "टॉप गेनर/लूज़र आइडिया (A) या Elliott TRADE अलर्ट (B) फ़िल्टर पास करके आए।_")
    out.append("_यह असली पैसे का ट्रेड नहीं, सिर्फ़ परख है: हर ट्रेड ₹1000 का, खर्च 0.10%। "
               "केवल जानकारी, निवेश सलाह नहीं।_")
    return "\n".join(out)


def main():
    now = datetime.now(ew_scan.IST)
    st = load(now)
    ew_scan.send_text(summary(st))


if __name__ == "__main__":
    main()
  
