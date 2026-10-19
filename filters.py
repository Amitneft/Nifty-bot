"""Trade filters shared by the 9:15 report, the hourly alert and the back-test.

 1. Nifty 50 direction : Nifty up   -> only LONG / BUY ideas
                         Nifty down -> only SHORT / SELL ideas
                         |move| below NEUTRAL % = sideways -> both sides allowed
 2. Sector             : LONG only if the stock's sector is up today (average move of its sector
                         peers > 0), SHORT only if the sector is down (< 0)
 3. Volume (Elliott)   : the H1 breakout candle must trade >= VOL_MIN x the average volume of the
                         same hour on the previous VOL_DAYS days. No volume data -> only WATCH.
                         (The two indices have no volume, so this rule does not apply to them.)
Everything else (wave rules, stops, targets) is unchanged.
"""
import numpy as np
import pandas as pd

from universe import NEXT50, SECTOR_OF

NEUTRAL = 0.10        # Nifty move (%) below this = sideways
VOL_MIN = 1.5         # same 1.5x as the movers strategy
VOL_DAYS = 10         # days of history for the "normal volume at this hour"
NIFTY = "NIFTY 50"
INDEXES = ("NIFTY 50", "NIFTY NEXT 50")


def _bad(x):
    return x is None or (isinstance(x, float) and np.isnan(x))


def nifty_state(chg):
    if _bad(chg):
        return "?"
    return "UP" if chg > NEUTRAL else ("DOWN" if chg < -NEUTRAL else "FLAT")


def check(side, nifty_chg, sector_avg=None, is_index=False):
    """(ok, reason). side = LONG / SHORT."""
    st = nifty_state(nifty_chg)
    if st == "UP" and side == "SHORT":
        return False, "Nifty 50 ऊपर है: SHORT नहीं"
    if st == "DOWN" and side == "LONG":
        return False, "Nifty 50 नीचे है: LONG नहीं"
    if not is_index and not _bad(sector_avg):
        if side == "LONG" and sector_avg <= 0:
            return False, f"सेक्टर नीचे/सपाट ({sector_avg:+.1f}%): LONG नहीं"
        if side == "SHORT" and sector_avg >= 0:
            return False, f"सेक्टर ऊपर/सपाट ({sector_avg:+.1f}%): SHORT नहीं"
    return True, ""


def market_line(nifty_chg, have_sector=True):
    st = nifty_state(nifty_chg)
    if st == "?":
        return "फ़िल्टर: Nifty 50 का आँकड़ा नहीं मिला, इसलिए दिशा की जाँच बंद।"
    word = {"UP": f"ऊपर ({nifty_chg:+.2f}%) → सिर्फ़ LONG/BUY",
            "DOWN": f"नीचे ({nifty_chg:+.2f}%) → सिर्फ़ SHORT/SELL",
            "FLAT": f"सपाट ({nifty_chg:+.2f}%) → दोनों तरफ़"}[st]
    return f"फ़िल्टर: Nifty 50 {word}; सेक्टर उसी दिशा में" + ("" if have_sector else " (सेक्टर आँकड़ा नहीं)")


# ------------------------------------------------------------------ volume
def slot_ratio(df, n=VOL_DAYS):
    """Per candle: volume / average volume of the same clock hour over the previous n sessions."""
    if "Volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    v = df["Volume"].astype(float)
    slot = pd.Series(df.index.strftime("%H:%M"), index=df.index)
    base = v.groupby(slot.values).transform(lambda s: s.shift(1).rolling(n, min_periods=5).mean())
    return v / base.where(base > 0)


def judge(name, c, vr, pos, nifty_chg, sector_avg):
    """Apply the volume rule and the market rules to a wave count seen at candle position `pos`.
    Returns a copy: volume added, TRADE -> WATCH if volume is low, or valid=False if the market is against it."""
    if not c or not c["valid"]:
        return c
    c = dict(c)
    is_index = name in INDEXES
    if is_index:
        c["vol"] = None
    else:
        k = pos - c["h1_ago"] if (c["status"] == "TRADE" and c.get("h1_ago") is not None) else pos
        v = float(vr.iloc[k]) if 0 <= k < len(vr) else float("nan")
        c["vol"] = None if np.isnan(v) else v
        if c["status"] == "TRADE" and (c["vol"] is None or c["vol"] < VOL_MIN):
            c["status"] = "WATCH"
            c["demoted"] = ("volume का डेटा नहीं" if c["vol"] is None
                            else f"breakout पर volume {c['vol']:.1f}x, चाहिए {VOL_MIN:g}x")
    ok, why = check(c["side"], nifty_chg, sector_avg, is_index)
    if not ok:
        c.update(valid=False, status=None, fails=[why], blocked=why)
    return c


def passes_trade(name, side, h1_ago, vr, pos, nifty_chg, sector_avg):
    """Back-test helper: is a raw TRADE state still a TRADE after the filters?"""
    c = judge(name, {"valid": True, "status": "TRADE", "side": side, "h1_ago": h1_ago},
              vr, pos, nifty_chg, sector_avg)
    return bool(c["valid"] and c["status"] == "TRADE")


# ------------------------------------------------------- market context from candles
def chg_table(data):
    """% move of every symbol versus the previous session's close, at every candle."""
    cols = {}
    for name, df in data.items():
        c = df["Close"]
        day = np.array(c.index.date)
        last_close = c.groupby(day).last()
        prev = pd.Series(last_close.values, index=last_close.index).shift(1)
        pc = pd.Series(day, index=c.index).map(prev)
        cols[name] = (c / pc.values - 1) * 100
    return pd.DataFrame(cols).sort_index()


class Ctx:
    """Nifty direction and sector direction at any candle time, built from the hourly candles."""

    def __init__(self, data):
        self.tbl = chg_table(data)
        self.arr = self.tbl.to_numpy(dtype=float)
        self.col = {n: i for i, n in enumerate(self.tbl.columns)}
        self.peers = {}
        for n in data:
            if n in INDEXES:
                continue
            key = (n in NEXT50, SECTOR_OF.get(n))
            self.peers.setdefault(key, []).append(self.col[n])

    def market_at(self, name, t):
        """(nifty % move, sector average % move) as of the last candle at or before time t."""
        i = int(self.tbl.index.searchsorted(t, side="right")) - 1
        if i < 0:
            return None, None
        nifty = self.arr[i, self.col[NIFTY]] if NIFTY in self.col else float("nan")
        sec = None
        if name not in INDEXES:
            idx = self.peers.get((name in NEXT50, SECTOR_OF.get(name)))
            if idx and SECTOR_OF.get(name):
                vals = self.arr[i, idx]
                vals = vals[~np.isnan(vals)]
                sec = float(vals.mean()) if len(vals) else None
        return (None if np.isnan(nifty) else float(nifty)), sec


# ------------------------------------------------------------------ text helpers
def vol_text(name, c):
    """One short Hindi line about the volume of a count."""
    if name in INDEXES:
        return "Volume: सूचकांक पर लागू नहीं"
    if c.get("vol") is None:
        return "Volume: डेटा नहीं"
    lab = "breakout candle" if (c["status"] == "TRADE" or c.get("demoted")) and c.get("h1_ago") is not None else "पिछली candle"
    return f"Volume ({lab}): {c['vol']:.1f}x {'✓' if c['vol'] >= VOL_MIN else '✗'}"


def caption(name, c, ctx_live=None):
    """Telegram caption: the usual one plus a volume line (and the real reason for WATCH)."""
    import ew_scan
    base = ew_scan.caption_for(name, c, ctx_live)
    if c.get("demoted"):
        base = base.replace("WATCH (H1 breakout अभी नहीं)", f"WATCH ({c['demoted']})")
    head, last = base.rsplit("\n", 1)
    return head + "\n" + vol_text(name, c) + "\n" + last
