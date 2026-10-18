"""Paper trading (forward test): every NEW TRADE alert is also "taken" on paper with Rs 1000,
and followed until stop / target / time exit. Two books run side by side: intraday and swing.
State lives in paper_trades.json (the workflow commits it back to the repo after each hourly run).
  python paper.py            -> send the current paper-trading summary to Telegram
"""
import json, os, sys
from datetime import datetime

import numpy as np
import pandas as pd

import ew_scan
from backtest import COST, MAX_OPEN, SLOT, START, SWING_MAX, net_pct, run_trade

STATE = "paper_trades.json"
MODES = ("intraday", "swing")
LABEL = {"intraday": "इंट्राडे", "swing": "स्विंग"}


def new_state(now):
    return {"since": now.strftime("%Y-%m-%d"), "start": START,
            "books": {m: {"equity": START, "open": [], "closed": []} for m in MODES}}


def load(now):
    try:
        st = json.load(open(STATE))
        for m in MODES:
            st["books"].setdefault(m, {"equity": START, "open": [], "closed": []})
        return st
    except Exception:
        return new_state(now)


def save(st):
    json.dump(st, open(STATE, "w"), ensure_ascii=False, indent=1)


def settle(st, data, now):
    """Close paper trades that hit stop / target / time exit. Returns list of Hindi lines."""
    lines = []
    for m in MODES:
        bk = st["books"][m]
        keep = []
        for p in bk["open"]:
            df = data.get(p["name"])
            if df is None:
                keep.append(p); continue
            idx = df.index
            pos = np.where(idx > pd.Timestamp(p["sig_time"]))[0]
            if len(pos) == 0:
                keep.append(p); continue
            i0 = int(pos[0])
            day0 = pd.Timestamp(p["sig_time"]).date()
            eod = (m == "intraday" and now.date() == day0 and (now.hour, now.minute) >= (15, 15))
            r = run_trade(idx, df.Open.values, df.High.values, df.Low.values, df.Close.values, i0,
                          p["entry"], p["side"], p["stop"], p["t1"], m, day0, gap_first=True, final=eod)
            if r is None:
                keep.append(p); continue
            net = net_pct(p["side"], p["entry"], r["exit"])
            pnl = p["size"] * net / 100
            bk["equity"] += pnl
            bk["closed"].append({"name": p["name"], "side": p["side"], "entry": p["entry"], "exit": float(r["exit"]),
                                 "why": r["why"], "net": net, "pnl": pnl, "opened": p["sig_time"],
                                 "closed": str(idx[r["j"]])})
            icon = "✅" if net > 0 else "❌"
            lines.append(f"{icon} [{LABEL[m]}] {p['name']} {p['side']} — {r['why']} — "
                         f"Entry {p['entry']:.2f} → {float(r['exit']):.2f} ({net:+.2f}%), ₹{pnl:+.1f}")
        bk["open"] = keep
    return lines


def open_new(st, signals, now):
    """signals: list of dict(name, side, entry, stop, t1, t2, sig_time). Returns Hindi lines."""
    lines = []
    for m in MODES:
        bk = st["books"][m]
        for s in signals:
            if len(bk["open"]) >= MAX_OPEN or any(p["name"] == s["name"] for p in bk["open"]):
                continue
            bk["open"].append({**s, "size": bk["equity"] * SLOT, "mode": m})
            lines.append(f"📒 [{LABEL[m]}] पेपर ट्रेड खुला: {s['name']} {s['side']} @ {s['entry']:.2f}, "
                         f"SL {s['stop']:.2f}, T1 {s['t1']:.2f}")
    return lines


def summary(st):
    out = [f"📒 *पेपर ट्रेडिंग (लाइव फॉरवर्ड टेस्ट)* — {st['since']} से, शुरुआती पूंजी ₹{st['start']:,.0f}"]
    for m in MODES:
        bk = st["books"][m]
        cl = bk["closed"]
        w = sum(c["net"] > 0 for c in cl)
        n = len(cl)
        out.append(f"*{LABEL[m]}*\n```\n"
                   f"पूंजी अभी     : ₹{bk['equity']:,.1f} ({(bk['equity'] / START - 1) * 100:+.1f}%)\n"
                   f"बंद ट्रेड     : {n}\n"
                   f"जीत / हार     : {w} / {n - w}\n"
                   f"Win % / Loss %: {100 * w / n if n else 0:.1f}% / {100 * (n - w) / n if n else 0:.1f}%\n"
                   f"खुले ट्रेड    : {len(bk['open'])}"
                   + ("\n" + "\n".join(f"  {p['name']} {p['side']} @ {p['entry']:.2f}" for p in bk["open"]) if bk["open"] else "")
                   + "\n```")
    out.append("_यह असली पैसे का ट्रेड नहीं, सिर्फ़ परख है: हर अलर्ट पर पूंजी का "
               f"{SLOT * 100:.0f}% (अधिकतम {MAX_OPEN} ट्रेड), खर्च {COST:.2f}%। केवल जानकारी, निवेश सलाह नहीं।_")
    return "\n".join(out)


def main():
    now = datetime.now(ew_scan.IST)
    st = load(now)
    ew_scan.send_text(summary(st))


if __name__ == "__main__":
    main()
