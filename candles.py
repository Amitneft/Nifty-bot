"""Hourly candles for the Elliott scan and the back-test.
Angel One SmartAPI first; Yahoo Finance is the backup (per symbol, so one missing
symbol never stops the run).
    load(tick, days, min_len, now) -> {name: DataFrame[Open, High, Low, Close, Volume]}
    tick = {name: yahoo ticker}, e.g. {"RELIANCE": "RELIANCE.NS", "NIFTY 50": "^NSEI"}
"""
import sys
import time
from datetime import timedelta, timezone

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

IST = timezone(timedelta(hours=5, minutes=30))
INDEX_TOKENS = {"^NSEI": "99926000"}     # Nifty 50 index; other indices come from Yahoo
CHUNK_DAYS = 360                         # Angel allows up to 400 days of 1h candles per request
PAUSE = 0.4                              # stay under Angel's historical-data rate limit


# ----------------------------------------------------------------- Angel One
def _login():
    import bot
    return bot.angel_login()


def _tokens(symbols):
    """{symbol: Angel token} for NSE equities, from the public scrip master."""
    import bot
    import requests
    last = None
    for url in bot.ANGEL_MASTERS:
        try:
            r = requests.get(url, timeout=90)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            last = e
    else:
        raise RuntimeError(f"scrip master not reachable: {last}")
    want = {s + "-EQ": s for s in symbols}
    return {want[d["symbol"]]: d["token"] for d in data
            if d.get("exch_seg") == "NSE" and d.get("symbol") in want}


def _candle_call(api, token, start, end):
    p = {"exchange": "NSE", "symboltoken": token, "interval": "ONE_HOUR",
         "fromdate": start.strftime("%Y-%m-%d 09:15"), "todate": end.strftime("%Y-%m-%d %H:%M")}
    for attempt in (1, 2):
        try:
            return ((api.getCandleData(p) or {}).get("data")) or []
        except Exception as e:
            if attempt == 2:
                raise
            print(f"candle call retry ({e})", file=sys.stderr)
            time.sleep(1.5)


def _angel_one(api, token, days, now):
    frames, end, left = [], now, days
    while left > 0:
        span = min(CHUNK_DAYS, left)
        start = end - timedelta(days=span)
        rows = _candle_call(api, token, start, end)
        if rows:
            idx = pd.to_datetime([r[0] for r in rows])
            idx = idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)
            frames.append(pd.DataFrame(
                {"Open": [float(r[1]) for r in rows], "High": [float(r[2]) for r in rows],
                 "Low": [float(r[3]) for r in rows], "Close": [float(r[4]) for r in rows],
                 "Volume": [float(r[5]) if len(r) > 5 and r[5] is not None else float("nan") for r in rows]},
                index=idx))
        end, left = start - timedelta(minutes=1), left - span
        time.sleep(PAUSE)
    if not frames:
        return None
    df = pd.concat(frames)
    return df[~df.index.duplicated()].sort_index()


def _from_angel(tick, days, now):
    api = _login()
    syms = [t[:-3] for t in tick.values() if t.endswith(".NS")]
    toks = _tokens(syms)
    out = {}
    for name, t in tick.items():
        token = INDEX_TOKENS.get(t) or toks.get(t[:-3] if t.endswith(".NS") else "")
        if not token:
            continue
        try:
            df = _angel_one(api, token, days, now)
            if df is not None:
                out[name] = df
        except Exception as e:
            print(f"angel candles {name}: {e}", file=sys.stderr)
    return out


# --------------------------------------------------------------------- Yahoo
def _from_yahoo(tick, days):
    if yf is None or not tick:
        return {}
    data = yf.download(list(tick.values()), period=f"{days}d", interval="1h", group_by="ticker",
                       progress=False, threads=True, auto_adjust=False)
    out = {}
    for name, t in tick.items():
        try:
            try:
                d = data[t]
            except KeyError:
                d = data                                  # single-ticker download: flat columns
            cols = ["Open", "High", "Low", "Close"] + (["Volume"] if "Volume" in d.columns else [])
            df = d[cols].dropna(subset=["Open", "High", "Low", "Close"]).copy()
            df.index = df.index.tz_convert(IST) if df.index.tz is not None else df.index.tz_localize(IST)
            out[name] = df
        except Exception as e:
            print(f"yahoo candles {name}: {e}", file=sys.stderr)
    return out


# ---------------------------------------------------------------------- main
def _clean(df, min_len, now):
    cols = ["Open", "High", "Low", "Close"] + (["Volume"] if "Volume" in df.columns else [])
    df = df[cols].dropna(subset=["Open", "High", "Low", "Close"])
    if len(df) and df.index[-1] + timedelta(hours=1) > now:
        df = df.iloc[:-1]                                  # candle still forming
    return df if len(df) >= min_len else None


def load(tick, days, min_len, now):
    out, n_angel = {}, 0
    try:
        for name, df in _from_angel(tick, days, now).items():
            df = _clean(df, min_len, now)
            if df is not None:
                out[name] = df
        n_angel = len(out)
    except Exception as e:
        print(f"Angel One candles failed: {e}; using Yahoo Finance", file=sys.stderr)
    missing = {n: t for n, t in tick.items() if n not in out}
    if missing:
        for name, df in _from_yahoo(missing, days).items():
            df = _clean(df, min_len, now)
            if df is not None:
                out[name] = df
    print(f"candles: Angel One {n_angel}, Yahoo Finance {len(out) - n_angel}, "
          f"missing {len(tick) - len(out)}")
    return out
