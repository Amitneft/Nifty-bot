"""Hourly intraday alert: message only when a brand-new Elliott TRADE setup appears
(index or any Nifty 100 stock). Reuses all rules from ew_scan.py.
Run at ~:20 past each hour, 10:20-15:20 IST:   python ew_alert.py [--dry] [--force]
"""
import os, sys
from datetime import datetime

import ew_scan as ew
from ew_scan import CFG, IST, best_count, caption_for, chart_png, load_all, send_photo, send_text


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
    out.sort(key=lambda k: (k != "NIFTY 50", results[k]["stop_pct"]))
    return out


def alert_run(data, results, now, dry=False):
    """Hourly intraday mode: message only when a brand-new TRADE setup appears."""
    names = new_trades(data, results)
    print(f"new TRADE setups: {names}")
    if not names:
        return
    head = (f"🚨 *नया TRADE setup* — {now.strftime('%H:%M IST')}\n"
            + "\n".join(f"{'📈' if results[k]['side'] == 'LONG' else '📉'} {k} {results[k]['side']}"
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
            send_photo(png, caption_for(nm, c))


def main():
    now = datetime.now(IST)
    if now.weekday() >= 5 and "--force" not in sys.argv:
        print("Weekend - market closed."); return
    if ew.yf is None:
        sys.exit("yfinance not installed")
    data = load_all(now)
    if len(data) < 40:
        sys.exit(f"Only {len(data)} symbols loaded; aborting.")
    results = {name: best_count(df) for name, df in data.items()}
    alert_run(data, results, now, dry="--dry" in sys.argv)


if __name__ == "__main__":
    main()
