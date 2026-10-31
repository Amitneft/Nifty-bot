"""Back-test of BOTH strategies, each in two styles (intraday / swing), every trade = Rs 1000.

 A : Top gainer / loser - stock already moved >= 1% today AND volume >= 1.5x the normal for that time,
     in the direction of Nifty 50 and of its sector (filters.py). Target 1%, stop 0.5%.
     5-minute candles; the decision is taken when the first candle (9:15-9:20) closes; at most 8 ideas
     per day and per block (Nifty 50 / Next 50), like the live report.
 B : Elliott Wave (1h candles). A signal = a NEW TRADE setup (same rule as the hourly alert) after the
     Nifty-direction, sector and volume filters. Entry = open of the next candle, stop = wave-4 extreme,
     target = target 1.
If one candle touches both stop and target, the stop counts first.
 intraday : an open trade is closed at the last candle of the entry day
 swing    : held until stop / target, at most 5 trading days
Every trade is independent and uses Rs 1000 (so no compounding); costs 0.10% per trade.
The result is also shown without the filters, to see what they change.
Run:  python backtest.py [--days 365] [--adays 90] [--dry]     (--days = history for B, --adays = for A)
"""
import json, os, sys
from datetime import datetime, timedelta
from multiprocessing import Pool

import numpy as np
import pandas as pd

import ew_scan
import filters
from ew_scan import IST, best_count

try:
    import yfinance as yf
except ImportError:
    yf = None
from universe import NEXT50, UNIVERSE

WINDOW = 300        # candles of history shown to the wave counter at each step (B)
WARM = 120          # first candle that is evaluated (B)
COST = 0.10         # round-trip cost in % (brokerage + taxes + slippage)
PER_TRADE = 1000.0  # Rs invested in every trade
SWING_MAX = 35      # swing = 5 trading days = 35 hourly candles
A_SWING_MAX = 375   # ... = 375 five-minute candles
A_DAYS = 90         # history for strategy A (Angel keeps ~100 days of 5-minute candles)
A_MAX_IDEAS = 8     # ideas per day and per block, like the live report
INDEX_TICKS = {"NIFTY 50": "^NSEI", "NIFTY NEXT 50": "^NSMIDCP"}
STRAT = {"A": "A: टॉप गेनर/लूज़र + volume", "B": "B: Elliott Wave"}
LABEL = {"intraday": "इंट्राडे (उसी दिन बंद)", "swing": "स्विंग (अधिकतम 5 दिन)"}
CRITERIA = {"min_move_pct": 1.0, "min_volume_ratio": 1.5, "target_pct": 1.0, "stop_pct": 0.5}


def load_criteria():
    cfg = dict(CRITERIA)
    try:
        cfg.update(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "criteria.json"))))
    except Exception:
        pass
    return cfg


# ------------------------------------------------------------------ trade engine
def run_trade(idx, O, H, L, C, i0, entry, side, stop, t1, mode, day0, gap_first, final, max_candles=SWING_MAX):
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
        elif j - i0 + 1 >= max_candles:
            return {"exit": c, "why": "TIME", "j": j}
    if final:
        return {"exit": C[n - 1], "why": "TIME", "j": n - 1}
    return None


def net_pct(side, entry, exit_):
    g = (exit_ / entry - 1) * 100 if side == "LONG" else (entry / exit_ - 1) * 100
    return g - COST


# ------------------------------------------------------------------ strategy B: Elliott
def scan_symbol(args):
    """Walk one symbol through history -> every candle where the raw count is a TRADE setup."""
    name, df = args
    states, n = [], len(df)
    for t in range(WARM, n - 1):
        c = best_count(df.iloc[max(0, t - WINDOW + 1): t + 1])
        if c and c["valid"] and c["status"] == "TRADE":
            states.append({"name": name, "t": t, "side": c["side"], "stop": c["stop"], "t1": c["target"],
                           "score": c["score"], "stop_pct": c["stop_pct"], "rr1": c["rr1"],
                           "h1_ago": c["h1_ago"]})
    return name, states


def signals_from_states(data, states, ctx, vols, use_filters=True):
    """NEW signals: TRADE now, but not TRADE (same side) on the previous candle - after the filters."""
    sigs, by = [], {}
    for s_ in states:
        by.setdefault(s_["name"], []).append(s_)
    for name, lst in by.items():
        df, last_ok = data[name], None
        for s_ in sorted(lst, key=lambda x: x["t"]):
            ok = True
            if use_filters:
                nifty, sec = ctx.market_at(name, df.index[s_["t"]])
                ok = filters.passes_trade(name, s_["side"], s_["h1_ago"], vols[name], s_["t"], nifty, sec)
            if ok and not (last_ok == (s_["t"] - 1, s_["side"])):
                sigs.append(s_)
            if ok:
                last_ok = (s_["t"], s_["side"])
    return sigs


def trades_B(data, sigs, mode):
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
        out.append({"strat": "B", "name": s["name"], "side": s["side"], "mode": mode,
                    "entry_time": idx[i0], "exit_time": idx[r["j"]] + timedelta(hours=1),
                    "entry": entry, "exit": r["exit"], "why": r["why"],
                    "net": net_pct(s["side"], entry, r["exit"]), "candles": r["j"] - i0 + 1})
    return out


# ------------------------------------------------------------------ strategy A: top gainer / loser + volume
def scan_movers(data5, cfg, ctx, vols, use_filters=True):
    """Ideas of every day: move >= min_move%, volume >= min_volume_ratio x, (Nifty + sector filters),
    measured when the first 5-minute candle (9:15-9:20) has closed."""
    sigs = []
    for name, df in data5.items():
        if name in filters.INDEXES:
            continue
        slot = np.array(df.index.strftime("%H:%M"))
        chg_s = ctx.tbl[name].reindex(df.index).to_numpy()
        vr = vols[name].to_numpy()
        C = df["Close"].to_numpy()
        for p in np.where(slot == "09:15")[0]:
            if p + 1 >= len(df):
                continue
            chg, v = chg_s[p], vr[p]
            if np.isnan(chg) or np.isnan(v) or abs(chg) < cfg["min_move_pct"] or v < cfg["min_volume_ratio"]:
                continue
            side = "LONG" if chg > 0 else "SHORT"
            if use_filters:
                nifty, sec = ctx.market_at(name, df.index[p])
                if not filters.check(side, nifty, sec)[0]:
                    continue
            e, tp, sp = C[p], cfg["target_pct"] / 100, cfg["stop_pct"] / 100
            sigs.append({"name": name, "t": int(p), "day": df.index[p].date(), "side": side, "chg": float(chg),
                         "block": name in NEXT50, "entry": float(e),
                         "t1": float(e * (1 + tp) if side == "LONG" else e * (1 - tp)),
                         "stop": float(e * (1 - sp) if side == "LONG" else e * (1 + sp))})
    best = {}
    for s in sigs:                                       # at most A_MAX_IDEAS per day and block, biggest move first
        best.setdefault((s["day"], s["block"]), []).append(s)
    out = []
    for lst in best.values():
        out += sorted(lst, key=lambda x: -abs(x["chg"]))[:A_MAX_IDEAS]
    return out


def trades_A(data5, sigs, mode):
    out = []
    for s in sigs:
        df = data5[s["name"]]
        idx, O, H, L, C = df.index, df.Open.values, df.High.values, df.Low.values, df.Close.values
        i0 = s["t"] + 1
        r = run_trade(idx, O, H, L, C, i0, s["entry"], s["side"], s["stop"], s["t1"], mode,
                      idx[s["t"]].date(), gap_first=False, final=True, max_candles=A_SWING_MAX)
        out.append({"strat": "A", "name": s["name"], "side": s["side"], "mode": mode,
                    "entry_time": idx[i0], "exit_time": idx[r["j"]] + timedelta(minutes=5),
                    "entry": s["entry"], "exit": r["exit"], "why": r["why"],
                    "net": net_pct(s["side"], s["entry"], r["exit"]), "candles": r["j"] - i0 + 1})
    return out


# ------------------------------------------------------------------ results at Rs 1000 per trade
def no_overlap(trades):
    """A new signal is skipped while an earlier trade in the same stock (same strategy and style) is still open."""
    out, busy = [], {}
    for t in sorted(trades, key=lambda x: x["entry_time"]):
        if busy.get(t["name"]) is not None and t["entry_time"] < busy[t["name"]]:
            continue
        busy[t["name"]] = t["exit_time"]
        out.append(t)
    return out


def peak_open(trades):
    ev = sorted([(t["entry_time"], 1) for t in trades] + [(t["exit_time"], -1) for t in trades],
                key=lambda e: (e[0], e[1]))
    cur = peak = 0
    for _, d in ev:
        cur += d
        peak = max(peak, cur)
    return peak


def stats1k(trades):
    n = len(trades)
    pnl = [PER_TRADE * t["net"] / 100 for t in trades]
    wins = [t for t in trades if t["net"] > 0]
    loss = [t for t in trades if t["net"] <= 0]
    gw = sum(p for p in pnl if p > 0)
    gl = -sum(p for p in pnl if p <= 0)
    by = lambda f: [t for t in trades if f(t)]
    wp = lambda lst: (100 * sum(t["net"] > 0 for t in lst) / len(lst)) if lst else 0.0
    return {
        "n": n, "wins": len(wins), "losses": len(loss),
        "win_pct": 100 * len(wins) / n if n else 0.0, "loss_pct": 100 * len(loss) / n if n else 0.0,
        "avg_win": float(np.mean([t["net"] for t in wins])) if wins else 0.0,
        "avg_loss": float(np.mean([t["net"] for t in loss])) if loss else 0.0,
        "pnl": float(sum(pnl)), "avg_pnl": float(sum(pnl) / n) if n else 0.0,
        "avg_pct": float(np.mean([t["net"] for t in trades])) if n else 0.0,
        "best": max((t["net"] for t in trades), default=0.0), "worst": min((t["net"] for t in trades), default=0.0),
        "target": sum(t["why"] == "TARGET" for t in trades), "sl": sum(t["why"] == "SL" for t in trades),
        "time": sum(t["why"] == "TIME" for t in trades),
        "pf": (gw / gl) if gl > 0 else (float("inf") if gw > 0 else 0.0),
        "long_n": len(by(lambda t: t["side"] == "LONG")), "long_w": wp(by(lambda t: t["side"] == "LONG")),
        "short_n": len(by(lambda t: t["side"] == "SHORT")), "short_w": wp(by(lambda t: t["side"] == "SHORT")),
        "peak": peak_open(trades),
    }


# ------------------------------------------------------------------ report
def block(strat, mode, s):
    head = f"*{STRAT[strat]} — {LABEL[mode]}* (हर ट्रेड ₹{PER_TRADE:,.0f})"
    if not s["n"]:
        return head + "\n```\nइस अवधि में कोई ट्रेड नहीं बना\n```"
    pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    k = PER_TRADE / 100
    return (head + "\n```\n"
            f"कुल ट्रेड        : {s['n']}\n"
            f"जीत / हार        : {s['wins']} / {s['losses']}\n"
            f"Win % / Loss %   : {s['win_pct']:.1f}% / {s['loss_pct']:.1f}%\n"
            f"औसत जीत          : {s['avg_win']:+.2f}%  (₹{s['avg_win'] * k:+.1f})\n"
            f"औसत हार          : {s['avg_loss']:+.2f}%  (₹{s['avg_loss'] * k:+.1f})\n"
            f"कुल लाभ/हानि     : ₹{s['pnl']:+,.0f}\n"
            f"प्रति ट्रेड औसत  : ₹{s['avg_pnl']:+.1f} ({s['avg_pct']:+.2f}%)\n"
            f"सबसे अच्छा/बुरा  : {s['best']:+.2f}% / {s['worst']:+.2f}%\n"
            f"Target/SL/Time   : {s['target']} / {s['sl']} / {s['time']}\n"
            f"Long  : {s['long_n']} ट्रेड, win {s['long_w']:.0f}%\n"
            f"Short : {s['short_n']} ट्रेड, win {s['short_w']:.0f}%\n"
            f"Profit factor    : {pf}\n"
            f"एक साथ अधिकतम खुले: {s['peak']} (पूंजी ≈ ₹{s['peak'] * PER_TRADE:,.0f})\n```")


def year_text(taken):
    """Year-by-year table (by the year the trade was entered): trades, win %, Rs P&L, for each strategy and style."""
    out = ["*साल-वार नतीजा* (हर ट्रेड ₹%d)" % PER_TRADE]
    for (st, mode), trades in sorted(taken.items()):
        if not trades:
            continue
        rows = []
        for y in sorted({t["entry_time"].year for t in trades}):
            ts = [t for t in trades if t["entry_time"].year == y]
            w = sum(t["net"] > 0 for t in ts)
            pnl = sum(PER_TRADE * t["net"] / 100 for t in ts)
            rows.append(f"{y}  {len(ts):5d}  {100 * w / len(ts):5.1f}%  {pnl:+9,.0f}")
        out.append(f"*{STRAT[st]} — {LABEL[mode]}*\n```\nसाल  ट्रेड    Win%      ₹ P&L\n" + "\n".join(rows) + "\n```")
    return "\n".join(out) if len(out) > 1 else ""


def compare_text(res, base):
    out = ["*फ़िल्टर का असर* (Nifty दिशा + सेक्टर; B में volume भी)\n```"]
    for st in ("A", "B"):
        for m in ("intraday", "swing"):
            if (st, m) not in res or (st, m) not in base:
                continue
            f, b = res[(st, m)], base[(st, m)]
            out.append(f"{st} {LABEL[m].split(' ')[0]}\n"
                       f" बिना फ़िल्टर : {b['n']:>4} ट्रेड, win {b['win_pct']:.0f}%, ₹{b['pnl']:+,.0f}\n"
                       f" फ़िल्टर संग  : {f['n']:>4} ट्रेड, win {f['win_pct']:.0f}%, ₹{f['pnl']:+,.0f}")
    out.append("```")
    return "\n".join(out)


def head_text(spanB, spanA, nB, nA):
    return (f"🧪 *बैक-टेस्ट — दोनों रणनीतियाँ, हर ट्रेड ₹{PER_TRADE:,.0f}*\n"
            f"A (टॉप गेनर/लूज़र + volume): {spanA}, {nA} शेयर, 5-मिनट candles\n"
            f"B (Elliott Wave): {spanB}, {nB} शेयर/इंडेक्स, 1 घंटे की candles\n"
            f"हर ट्रेड अलग है (पूंजी बढ़ाकर दोबारा नहीं लगाई), खर्च {COST:.2f}% प्रति ट्रेड।")


NOTE = ("_₹1000 की जगह किसी भी रकम पर % वही रहेगा, ₹ उसी अनुपात में बढ़ेगा/घटेगा। "
        "A में फ़ैसला पहली 5-मिनट candle (9:15–9:20) के अंत पर लिया है; Entry उसी का बंद भाव। "
        "एक ही candle में stop और target दोनों छुएँ तो stop पहले माना है। "
        "आज के Nifty 100 की सूची पिछले समय पर लगाई है (survivorship bias), इसलिए असली नतीजे कुछ कमज़ोर हो सकते हैं। "
        "Nifty/सेक्टर की दिशा उस समय तक की चाल से ली है (पिछले बंद भाव के मुक़ाबले)। "
        "पिछला प्रदर्शन आगे की गारंटी नहीं। केवल जानकारी, निवेश सलाह नहीं।_")


def curve_png(trades_by, span):
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100, facecolor=ew_scan.BG)
    ax.set_facecolor(ew_scan.BG)
    cols = {("A", "intraday"): ew_scan.GOLD, ("A", "swing"): "#ff8a3d",
            ("B", "intraday"): ew_scan.BLUE, ("B", "swing"): ew_scan.GREEN}
    for key, trades in trades_by.items():
        if not trades:
            continue
        tr = sorted(trades, key=lambda t: t["exit_time"])
        cum = np.cumsum([PER_TRADE * t["net"] / 100 for t in tr])
        ax.plot([t["exit_time"] for t in tr], cum, color=cols[key], lw=1.6, label=f"{key[0]} {key[1]}")
    ax.axhline(0, color=ew_scan.MUTED, lw=0.8, ls="--")
    ax.set_title(f"Cumulative P&L (Rs, Rs {PER_TRADE:,.0f} per trade): {span}", color="white", fontsize=12)
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
    import candles                      # Angel One first, Yahoo Finance as backup
    return candles.load(tick, days, WARM + 60, datetime.now(IST))


def load_history5(days=A_DAYS):
    tick = {"NIFTY 50": "^NSEI"}
    tick.update({s: s + ".NS" for s in UNIVERSE})
    import candles
    return candles.load(tick, days, 300, datetime.now(IST), interval="5m")


def _span(data):
    if not data:
        return "-"
    first = min(df.index[0] for df in data.values())
    last = max(df.index[-1] for df in data.values())
    return f"{first:%d %b %Y} से {last:%d %b %Y}"


def run(data, data5=None, dry=False, csv_path="backtest_trades.csv"):
    cfg = load_criteria()
    res, base, taken = {}, {}, {}
    # ---------------- B: Elliott (1h)
    if data:
        items = list(data.items())
        try:
            with Pool(max(1, (os.cpu_count() or 2))) as pool:
                scanned = pool.map(scan_symbol, items, chunksize=4)
        except Exception as e:                               # no multiprocessing: run in one process
            print(f"pool failed ({e!r}); single process", file=sys.stderr)
            scanned = [scan_symbol(x) for x in items]
        states = [s for _, lst in scanned for s in lst]
        ctx = filters.Ctx(data)
        vols = {n: filters.slot_ratio(df) for n, df in data.items()}
        sig_f = signals_from_states(data, states, ctx, vols, True)
        sig_r = signals_from_states(data, states, ctx, vols, False)
        print(f"B signals: {len(sig_f)} with filters, {len(sig_r)} without")
        for mode in ("intraday", "swing"):
            taken[("B", mode)] = no_overlap(trades_B(data, sig_f, mode))
            res[("B", mode)] = stats1k(taken[("B", mode)])
            base[("B", mode)] = stats1k(no_overlap(trades_B(data, sig_r, mode)))
    # ---------------- A: top gainer / loser + volume (5 min)
    if data5:
        ctx5 = filters.Ctx(data5)
        vols5 = {n: filters.slot_ratio(df) for n, df in data5.items()}
        a_f = scan_movers(data5, cfg, ctx5, vols5, True)
        a_r = scan_movers(data5, cfg, ctx5, vols5, False)
        print(f"A signals: {len(a_f)} with filters, {len(a_r)} without")
        for mode in ("intraday", "swing"):
            taken[("A", mode)] = no_overlap(trades_A(data5, a_f, mode))
            res[("A", mode)] = stats1k(taken[("A", mode)])
            base[("A", mode)] = stats1k(no_overlap(trades_A(data5, a_r, mode)))
    rows = [t for lst in taken.values() for t in lst]
    if rows:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    head = head_text(_span(data), _span(data5), len(data or {}), len(data5 or {}) - (1 if data5 else 0))
    msgs = []
    if data5:
        msgs.append(head + "\n\n" + block("A", "intraday", res[("A", "intraday")]) + "\n\n"
                    + block("A", "swing", res[("A", "swing")]))
    else:
        msgs.append(head + "\n\n⚠️ A का test नहीं हो पाया (5-मिनट का डेटा नहीं मिला)।")
    if data:
        msgs.append(block("B", "intraday", res[("B", "intraday")]) + "\n\n" + block("B", "swing", res[("B", "swing")]))
    yt = year_text(taken)
    if yt:
        msgs.append(yt + "\n_अधूरा साल (पहला/आख़िरी) कम दिनों का होता है, इसलिए ट्रेड कम दिख सकते हैं।_")
    msgs.append(compare_text(res, base) + "\n\n" + NOTE)
    if dry:
        for m in msgs:
            print(m, "\n")
        return res
    for m in msgs:
        ew_scan.send_text(m)
    try:
        ew_scan.send_photo(curve_png(taken, _span(data5 or data)), f"₹{PER_TRADE:,.0f} प्रति ट्रेड: कुल लाभ/हानि का ग्राफ")
    except Exception as e:
        print(f"chart failed: {e!r}", file=sys.stderr)
    return res


def main():
    days = 365
    if "--days" in sys.argv:
        days = max(120, min(1200, int(sys.argv[sys.argv.index("--days") + 1])))   # >729 days needs Angel One
    adays = A_DAYS
    if "--adays" in sys.argv:
        adays = max(30, min(1200, int(sys.argv[sys.argv.index("--adays") + 1])))
    if yf is None:
        sys.exit("yfinance not installed")
    data = load_history(days)
    if len(data) < 40:
        print(f"Only {len(data)} hourly symbols loaded; B skipped.", file=sys.stderr)
        data = None
    data5 = load_history5(adays)
    if len(data5) < 40:
        print(f"Only {len(data5)} five-minute symbols loaded; A skipped.", file=sys.stderr)
        data5 = None
    if not data and not data5:
        sys.exit("No data loaded; aborting.")
    print(f"loaded B: {len(data or {})} symbols, A: {len(data5 or {})} symbols")
    run(data, data5, dry="--dry" in sys.argv)


if __name__ == "__main__":
    main()
 
