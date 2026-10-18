"""Back-test of the Elliott Wave strategy (same rules as the live scanner) on 1h candles.

Walk-forward: at every completed candle the wave counter sees only the past, exactly like the live bot.
A signal = a NEW TRADE setup (the same rule the hourly alert uses). Entry = open of the next candle,
Stop = wave-4 extreme, Target = target 1. If a candle touches both stop and target, the stop counts first.
Two exit styles are tested side by side:
  intraday : any open trade is closed at the last candle of the entry day
  swing    : held until stop / target, at most SWING_MAX candles (~5 trading days)
Capital: Rs 1000, each trade uses 20% of the current capital, max 5 trades open at once, costs 0.10% per trade.
Run:  python backtest.py [--days 365] [--dry]
"""
import os, sys
from datetime import datetime, timedelta
from multiprocessing import Pool

import numpy as np
import pandas as pd

import ew_scan
from ew_scan import IST, best_count

try:
    import yfinance as yf
except ImportError:
    yf = None
from universe import NEXT50, UNIVERSE

WINDOW = 300        # candles of history shown to the wave counter at each step
WARM = 120          # first candle that is evaluated
COST = 0.10         # round-trip cost in % (brokerage + taxes + slippage)
SLOT = 0.20         # share of capital used per trade
MAX_OPEN = 5        # trades open at the same time
START = 1000.0      # starting capital (Rs)
SWING_MAX = 35      # candles (about 5 trading days)
INDEX_TICKS = {"NIFTY 50": "^NSEI", "NIFTY NEXT 50": "^NSMIDCP"}


# ------------------------------------------------------------------ trade engine
def run_trade(idx, O, H, L, C, i0, entry, side, stop, t1, mode, day0, gap_first, final):
    """Walk candles from i0. Returns dict(exit, why, j) or None if still open (final=False)."""
    n, long_ = len(C), side == "LONG"
    for j in range(i0, n):
        o, h, l, c = O[j], H[j], L[j], C[j]
        if j > i0 or gap_first:                       # opening gap beyond stop / target
            if (long_ and o <= stop) or (not long_ and o >= stop):
                return {"exit": o, "why": "SL", "j": j}
            if (long_ and o >= t1) or (not long_ and o <= t1):
                return {"exit": o, "why": "TARGET", "j": j}
        if long_:
            if l <= stop:
                return {"exit": stop, "why": "SL", "j": j}
            if h >= t1:
                return {"exit": t1, "why": "TARGET", "j": j}
        else:
            if h >= stop:
                return {"exit": stop, "why": "SL", "j": j}
            if l <= t1:
                return {"exit": t1, "why": "TARGET", "j": j}
        if mode == "intraday":
            if (j + 1 < n and idx[j + 1].date() != idx[j].date()) or idx[j].date() != day0:
                return {"exit": c, "why": "TIME", "j": j}
        elif j - i0 + 1 >= SWING_MAX:
            return {"exit": c, "why": "TIME", "j": j}
    if final:
        return {"exit": C[n - 1], "why": "TIME", "j": n - 1}
    return None


def net_pct(side, entry, exit_):
    g = (exit_ / entry - 1) * 100 if side == "LONG" else (entry / exit_ - 1) * 100
    return g - COST


# ------------------------------------------------------------------ signals
def scan_symbol(args):
    """Walk one symbol through history -> list of new-TRADE signals."""
    name, df = args
    sigs, prev, n = [], None, len(df)
    for t in range(WARM, n - 1):
        c = best_count(df.iloc[max(0, t - WINDOW + 1): t + 1])
        cur = c["side"] if (c and c["valid"] and c["status"] == "TRADE") else None
        if cur and cur != prev:
            sigs.append({"name": name, "t": t, "side": cur, "stop": c["stop"], "t1": c["target"],
                         "score": c["score"], "stop_pct": c["stop_pct"], "rr1": c["rr1"]})
        prev = cur
    return name, sigs


def trades_from_signals(data, sigs, mode):
    out = []
    for s in sigs:
        df = data[s["name"]]
        idx, O, H, L, C = df.index, df.Open.values, df.High.values, df.Low.values, df.Close.values
        i0 = s["t"] + 1
        entry = O[i0]
        if (s["side"] == "LONG" and not (s["stop"] < entry < s["t1"])) or \
           (s["side"] == "SHORT" and not (s["t1"] < entry < s["stop"])):
            continue                                   # opened beyond stop / target: no trade
        r = run_trade(idx, O, H, L, C, i0, entry, s["side"], s["stop"], s["t1"], mode,
                      idx[i0].date(), gap_first=False, final=True)
        out.append({"name": s["name"], "side": s["side"], "score": s["score"], "mode": mode,
                    "entry_time": idx[i0], "exit_time": idx[r["j"]] + timedelta(hours=1),
                    "entry": entry, "exit": r["exit"], "why": r["why"],
                    "net": net_pct(s["side"], entry, r["exit"]), "candles": r["j"] - i0 + 1})
    return out


# ------------------------------------------------------------------ capital simulation
def portfolio(trades):
    """Rs 1000 compounding: 20% of capital per trade, max 5 open, one per symbol."""
    eq, open_, taken, curve, skipped = START, [], [], [(None, START)], 0
    trades = sorted(trades, key=lambda x: (x["entry_time"], -x["score"]))

    def close_until(t):
        nonlocal eq
        for p in sorted([p for p in open_ if t is None or p["exit_time"] <= t], key=lambda p: p["exit_time"]):
            open_.remove(p)
            eq += p["size"] * p["net"] / 100
            p["pnl"] = p["size"] * p["net"] / 100
            curve.append((p["exit_time"], eq))

    for tr in trades:
        close_until(tr["entry_time"])
        if len(open_) >= MAX_OPEN or any(p["name"] == tr["name"] for p in open_):
            skipped += 1
            continue
        p = dict(tr, size=eq * SLOT)
        open_.append(p)
        taken.append(p)
    close_until(None)
    return taken, curve, skipped


def stats(taken, curve, skipped):
    n = len(taken)
    wins = [t for t in taken if t["net"] > 0]
    loss = [t for t in taken if t["net"] <= 0]
    gw, gl = sum(t["pnl"] for t in wins), -sum(t["pnl"] for t in loss)
    eqs = np.array([e for _, e in curve])
    dd = float(((eqs - np.maximum.accumulate(eqs)) / np.maximum.accumulate(eqs)).min() * 100) if len(eqs) else 0.0
    final = float(eqs[-1]) if len(eqs) else START
    by = lambda f: [t for t in taken if f(t)]
    wp = lambda lst: (100 * sum(t["net"] > 0 for t in lst) / len(lst)) if lst else 0.0
    return {
        "n": n, "wins": len(wins), "losses": len(loss), "skipped": skipped,
        "win_pct": 100 * len(wins) / n if n else 0.0, "loss_pct": 100 * len(loss) / n if n else 0.0,
        "target": sum(t["why"] == "TARGET" for t in taken), "sl": sum(t["why"] == "SL" for t in taken),
        "time": sum(t["why"] == "TIME" for t in taken),
        "avg_win": float(np.mean([t["net"] for t in wins])) if wins else 0.0,
        "avg_loss": float(np.mean([t["net"] for t in loss])) if loss else 0.0,
        "pf": (gw / gl) if gl > 0 else (float("inf") if gw > 0 else 0.0),
        "final": final, "profit": final - START, "ret": (final / START - 1) * 100, "maxdd": dd,
        "long_n": len(by(lambda t: t["side"] == "LONG")), "long_w": wp(by(lambda t: t["side"] == "LONG")),
        "short_n": len(by(lambda t: t["side"] == "SHORT")), "short_w": wp(by(lambda t: t["side"] == "SHORT")),
        "n50_n": len(by(lambda t: not (t["name"] in NEXT50 or t["name"] == "NIFTY NEXT 50"))),
        "n50_w": wp(by(lambda t: not (t["name"] in NEXT50 or t["name"] == "NIFTY NEXT 50"))),
        "nx_n": len(by(lambda t: t["name"] in NEXT50 or t["name"] == "NIFTY NEXT 50")),
        "nx_w": wp(by(lambda t: t["name"] in NEXT50 or t["name"] == "NIFTY NEXT 50")),
        "best": max((t["net"] for t in taken), default=0.0), "worst": min((t["net"] for t in taken), default=0.0),
    }


# ------------------------------------------------------------------ report
LABEL = {"intraday": "इंट्राडे (उसी दिन बंद)", "swing": f"स्विंग (अधिकतम {SWING_MAX // 7} दिन)"}


def block(mode, s):
    pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return (f"*{LABEL[mode]}*\n```\n"
            f"कुल ट्रेड      : {s['n']}  (छूटे: {s['skipped']})\n"
            f"जीत / हार      : {s['wins']} / {s['losses']}\n"
            f"Win %  / Loss %: {s['win_pct']:.1f}% / {s['loss_pct']:.1f}%\n"
            f"Target/SL/Time : {s['target']} / {s['sl']} / {s['time']}\n"
            f"औसत जीत / हार  : {s['avg_win']:+.2f}% / {s['avg_loss']:+.2f}%\n"
            f"Profit factor  : {pf}\n"
            f"₹1000 से अंत में: ₹{s['final']:,.0f}  ({s['profit']:+,.0f}, {s['ret']:+.1f}%)\n"
            f"Max drawdown   : {s['maxdd']:.1f}%\n"
            f"Long  : {s['long_n']} ट्रेड, win {s['long_w']:.0f}%\n"
            f"Short : {s['short_n']} ट्रेड, win {s['short_w']:.0f}%\n"
            f"Nifty 50     : {s['n50_n']} ट्रेड, win {s['n50_w']:.0f}%\n"
            f"Nifty Next 50: {s['nx_n']} ट्रेड, win {s['nx_w']:.0f}%\n"
            f"सबसे अच्छा / बुरा: {s['best']:+.2f}% / {s['worst']:+.2f}%\n```")


def report_text(res, span, n_sym):
    head = (f"🧪 *बैक-टेस्ट — Elliott Wave रणनीति*\n{span}, {n_sym} शेयर/इंडेक्स, 1 घंटे की candles\n"
            f"पूंजी ₹{START:,.0f}; हर ट्रेड में पूंजी का {SLOT * 100:.0f}%, एक साथ अधिकतम {MAX_OPEN} ट्रेड, "
            f"खर्च {COST:.2f}% प्रति ट्रेड।")
    note = ("_Entry = सिग्नल के बाद वाली candle का open। एक ही candle में stop और target दोनों छुएँ तो stop पहले माना है। "
            "आज के Nifty 100 की सूची पिछले समय पर लगाई है (survivorship bias), इसलिए असली नतीजे कुछ कमज़ोर हो सकते हैं। "
            "पिछला प्रदर्शन आगे की गारंटी नहीं। केवल जानकारी, निवेश सलाह नहीं।_")
    return head, [block(m, res[m]) for m in ("intraday", "swing")], note


def curve_png(curves, span):
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100, facecolor=ew_scan.BG)
    ax.set_facecolor(ew_scan.BG)
    for mode, col in (("intraday", ew_scan.GOLD), ("swing", ew_scan.BLUE)):
        pts = [(t, e) for t, e in curves[mode] if t is not None]
        if pts:
            ax.plot([t for t, _ in pts], [e for _, e in pts], color=col, lw=1.8, label=LABEL[mode])
    ax.axhline(START, color=ew_scan.MUTED, lw=0.8, ls="--")
    ax.set_title(f"₹{START:,.0f} की पूंजी: Elliott रणनीति ({span})", color="white", fontsize=13)
    ax.tick_params(colors=ew_scan.MUTED)
    ax.grid(color=ew_scan.GRID, lw=0.8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(facecolor=ew_scan.BG, labelcolor="white", frameon=False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=ew_scan.BG)
    plt.close(fig)
    return buf.getvalue()


# ------------------------------------------------------------------ main
def load_history(days):
    tick = dict(INDEX_TICKS)
    tick.update({s: s + ".NS" for s in UNIVERSE})
    data = yf.download(list(tick.values()), period=f"{days}d", interval="1h", group_by="ticker",
                       progress=False, threads=True, auto_adjust=False)
    now, out = datetime.now(IST), {}
    for name, t in tick.items():
        try:
            df = data[t][["Open", "High", "Low", "Close"]].dropna().copy()
            df.index = df.index.tz_convert(IST) if df.index.tz is not None else df.index.tz_localize(IST)
            if len(df) and df.index[-1] + timedelta(hours=1) > now:
                df = df.iloc[:-1]
            if len(df) >= WARM + 60:
                out[name] = df
        except Exception as e:
            print(f"skip {name}: {e}", file=sys.stderr)
    return out


def run(data, dry=False, csv_path="backtest_trades.csv"):
    items = list(data.items())
    try:
        with Pool(max(1, (os.cpu_count() or 2))) as pool:
            scanned = pool.map(scan_symbol, items, chunksize=4)
    except Exception as e:                                   # no multiprocessing: run in one process
        print(f"pool failed ({e!r}); single process", file=sys.stderr)
        scanned = [scan_symbol(x) for x in items]
    sigs = [s for _, lst in scanned for s in lst]
    print(f"signals: {len(sigs)}")
    res, curves, rows = {}, {}, []
    for mode in ("intraday", "swing"):
        trades = trades_from_signals(data, sigs, mode)
        taken, curve, skipped = portfolio(trades)
        res[mode], curves[mode] = stats(taken, curve, skipped), curve
        rows += taken
    first = min(df.index[0] for df in data.values())
    last = max(df.index[-1] for df in data.values())
    span = f"{first:%d %b %Y} से {last:%d %b %Y}"
    head, blocks, note = report_text(res, span, len(data))
    if rows:
        pd.DataFrame(rows).drop(columns=["size"], errors="ignore").to_csv(csv_path, index=False)
    if dry:
        print(head, "\n"); [print(b, "\n") for b in blocks]; print(note)
        return res
    ew_scan.send_text(head + "\n\n" + blocks[0])
    ew_scan.send_text(blocks[1] + "\n\n" + note)
    try:
        ew_scan.send_photo(curve_png(curves, span), f"₹{START:,.0f} की पूंजी का ग्राफ")
    except Exception as e:
        print(f"chart failed: {e!r}", file=sys.stderr)
    return res


def main():
    days = 365
    if "--days" in sys.argv:
        days = max(120, min(729, int(sys.argv[sys.argv.index("--days") + 1])))
    if yf is None:
        sys.exit("yfinance not installed")
    data = load_history(days)
    if len(data) < 40:
        sys.exit(f"Only {len(data)} symbols loaded; aborting.")
    print(f"loaded {len(data)} symbols, {days} days")
    run(data, dry="--dry" in sys.argv)


if __name__ == "__main__":
    main()
