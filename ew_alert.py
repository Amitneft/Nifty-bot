"""Hourly intraday alert: message only when a brand-new Elliott TRADE setup appears
(Nifty 50 / Nifty Next 50 index or any of their stocks). Also runs the paper-trading book:
every alert is taken on paper (Rs 1000) and followed to stop / target.
Run at ~:20 past each hour, 10:20-15:20 IST:   python ew_alert.py [--dry] [--force]
"""
import os, sys
from datetime import datetime

import ew_scan as ew
import paper
from ew_scan import CFG, IST, best_count, caption_for, chart_png, send_photo, send_text
from report import load_ew
from universe import NEXT50


def grp(name):
    return "Nifty Next 50" if name in NEXT50 or name == "NIFTY NEXT 50" else "Nifty 50"


def new_trades(data, results):
    """Names whose TRADE setup is NEW this hour: TRADE now, but not TRADE (same side) one candle ago.
    This needs no saved state, so the same setup is never alerted twice."""
    out = []
    for name, c in results.items():
        if not (c and c["valid"] and c["status"] == "TRADE"):
            continue
        prev = best_count(data[name].iloc[:-1])
        if prev and prev["valid"] and prev["status"] == "TRADE" and prev["side"] == c["side"]:
            continue
        out.append(name)
    out.sort(key=lambda k: (k not in ("NIFTY 50", "NIFTY NEXT 50"), -results[k]["score"], results[k]["stop_pct"]))
    return out


def alert_run(data, results, now, dry=False):
    """Hourly intraday mode: message only when a brand-new TRADE setup appears. Returns the names."""
    names = new_trades(data, results)
    print(f"new TRADE setups: {names}")
    if not names:
        return names
    head = (f"🚨 *नया TRADE setup* — {now.strftime('%H:%M IST')}\n"
            + "\n".join(f"{'📈' if results[k]['side'] == 'LONG' else '📉'} {k} {results[k]['side']} ({grp(k)})"
                        for k in names)
            + "\n_केवल जानकारी, निवेश सलाह नहीं।_")
    if dry:
        os.makedirs("out", exist_ok=True)
        print(head)
    else:
        send_text(head)
    for nm in names[: CFG["max_charts"]]:
        c = results[nm]
        png = chart_png(nm, data[nm], c, now)
        if dry:
            open(f"out/alert_{nm.replace(' ', '_').replace('&', 'n')}.png", "wb").write(png)
        else:
            send_photo(png, caption_for(nm, c).replace(" — ", f" [{grp(nm)}] — ", 1))
    return names


def paper_run(data, results, names, now, dry=False):
    """Settle open paper trades, open the new ones, report. Never stops the alert on error."""
    try:
        st = paper.load(now)
        lines = paper.settle(st, data, now)
        sigs = [{"name": n, "side": results[n]["side"], "entry": float(results[n]["entry"]),
                 "stop": float(results[n]["stop"]), "t1": float(results[n]["target"]),
                 "t2": float(results[n]["target2"]), "sig_time": str(data[n].index[-1])} for n in names]
        lines += paper.open_new(st, sigs, now)
        if not dry:
            paper.save(st)
        text = "\n".join(lines)
        if now.hour >= 15:                                   # last run of the day: full summary
            text = (text + "\n\n" if text else "") + paper.summary(st)
        if text:
            print(text) if dry else send_text(text)
    except Exception as e:
        print(f"paper trading step failed: {e!r}", file=sys.stderr)


def main():
    now = datetime.now(IST)
    if now.weekday() >= 5 and "--force" not in sys.argv:
        print("Weekend - market closed."); return
    if ew.yf is None:
        sys.exit("yfinance not installed")
    data = load_ew(now)
    if len(data) < 40:
        sys.exit(f"Only {len(data)} symbols loaded; aborting.")
    results = {name: best_count(df) for name, df in data.items()}
    dry = "--dry" in sys.argv
    names = alert_run(data, results, now, dry=dry)
    paper_run(data, results, names, now, dry=dry)


if __name__ == "__main__":
    main()
