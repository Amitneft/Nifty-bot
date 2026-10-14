"""Nifty 50 morning scanner -> Telegram. Run at ~09:15 IST on trading days."""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests

try:
    import yfinance as yf
except ImportError:  # only needed for backup prices and volume
    yf = None

IST = ZoneInfo("Asia/Kolkata")

# Verify against NSE's current constituents list (changes semi-annually).
NIFTY50 = """ADANIENT ADANIPORTS APOLLOHOSP ASIANPAINT AXISBANK BAJAJ-AUTO BAJFINANCE BAJAJFINSV
BEL BHARTIARTL CIPLA COALINDIA DRREDDY EICHERMOT ETERNAL GRASIM HCLTECH HDFCBANK HDFCLIFE
HINDALCO HINDUNILVR ICICIBANK INDIGO INFY ITC JIOFIN JSWSTEEL KOTAKBANK LT M&M MARUTI
MAXHEALTH NESTLEIND NTPC ONGC POWERGRID RELIANCE SBILIFE SBIN SHRIRAMFIN SUNPHARMA
TATACONSUM TMPV TATASTEEL TCS TECHM TITAN TRENT ULTRACEMCO WIPRO""".split()

SECTORS = {
    "IT": "TCS INFY HCLTECH WIPRO TECHM",
    "Banks": "HDFCBANK ICICIBANK SBIN KOTAKBANK AXISBANK",
    "Finance": "BAJFINANCE BAJAJFINSV SHRIRAMFIN JIOFIN HDFCLIFE SBILIFE",
    "Auto": "MARUTI M&M BAJAJ-AUTO EICHERMOT TMPV",
    "Metals": "TATASTEEL JSWSTEEL HINDALCO COALINDIA",
    "Energy": "RELIANCE ONGC",
    "Power": "NTPC POWERGRID",
    "Pharma": "SUNPHARMA CIPLA DRREDDY APOLLOHOSP MAXHEALTH",
    "FMCG": "HINDUNILVR ITC NESTLEIND TATACONSUM",
    "Infra": "LT BEL ADANIENT ADANIPORTS BHARTIARTL",
    "Cement": "ULTRACEMCO GRASIM",
    "Consumer": "TITAN TRENT ASIANPAINT ETERNAL INDIGO",
}
SECTOR_OF = {s: sec for sec, names in SECTORS.items() for s in names.split()}

DEFAULT_CFG = {"min_move_pct": 1.0, "min_volume_ratio": 1.5,
               "target_pct": 1.0, "stop_pct": 0.5, "max_ideas": 6}

# ---------------------------------------------------------------- data: NSE
NSE_HOME = "https://www.nseindia.com"
NSE_API = NSE_HOME + "/api/equity-stockIndices?index=NIFTY%2050"
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": NSE_HOME + "/market-data/live-market-indices/heatmap",
}


def parse_nse(payload):
    """NSE JSON -> (stock rows, index dict or None)."""
    rows, index = [], None
    for d in payload.get("data", []):
        if d.get("priority") == 1 or d.get("symbol") == "NIFTY 50":
            index = {"ltp": float(d["lastPrice"]), "chg": float(d["pChange"])}
            continue
        rows.append({"sym": d["symbol"], "ltp": float(d["lastPrice"]),
                     "prev": float(d["previousClose"]), "chg": float(d["pChange"])})
    return rows, index


def fetch_nse():
    s = requests.Session()
    s.headers.update(NSE_HEADERS)
    s.get(NSE_HOME, timeout=15)  # cookies NSE requires
    s.get(NSE_HOME + "/market-data/live-market-indices/heatmap", timeout=15)
    r = s.get(NSE_API, timeout=15)
    r.raise_for_status()
    return parse_nse(r.json())


# -------------------------------------------------------------- data: Yahoo
def quote(sym, yahoo_sym=None):
    try:
        fi = yf.Ticker(yahoo_sym or sym + ".NS").fast_info
        last, prev = fi["last_price"], fi["previous_close"]
        return {"sym": sym, "ltp": last, "prev": prev, "chg": (last - prev) / prev * 100}
    except Exception as e:
        print(f"skip {sym}: {e}", file=sys.stderr)
        return None


def fetch_yahoo():
    if yf is None:
        return [], None
    with ThreadPoolExecutor(10) as ex:
        rows = [q for q in ex.map(quote, NIFTY50) if q]
    idx = quote("NIFTY 50", "^NSEI")
    return rows, ({"ltp": idx["ltp"], "chg": idx["chg"]} if idx else None)


def fetch_all(now):
    """-> rows, index, source, volume_ratios (None = compute from Yahoo)."""
    try:
        rows, index, vr = fetch_angel(now)
        return rows, index, "Angel One SmartAPI", vr
    except Exception as e:
        print(f"Angel One failed: {e}; trying NSE", file=sys.stderr)
    try:
        rows, index = fetch_nse()
        if len(rows) >= 40:
            return rows, index, "NSE India", None
        print(f"NSE returned {len(rows)} rows; trying backup", file=sys.stderr)
    except Exception as e:
        print(f"NSE failed: {e}; trying backup", file=sys.stderr)
    rows, index = fetch_yahoo()
    return rows, index, "Yahoo Finance (backup)", None


# ------------------------------------------------------- data: Angel One
ANGEL_MASTERS = [
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json",
    "https://margincalculator.angelbroking.com/OpenAPI_Files/files/OpenAPIScripMaster.json",
]
ANGEL_INDEX_TOKEN = "99926000"  # Nifty 50 index
ANGEL_SECRETS = ["ANGEL_API_KEY", "ANGEL_CLIENT_CODE", "ANGEL_PIN", "ANGEL_TOTP_SECRET"]


def angel_login():
    import pyotp
    from SmartApi import SmartConnect
    api = SmartConnect(api_key=os.environ["ANGEL_API_KEY"].strip())
    totp = pyotp.TOTP("".join(os.environ["ANGEL_TOTP_SECRET"].split())).now()
    d = api.generateSession(os.environ["ANGEL_CLIENT_CODE"].strip(),
                            os.environ["ANGEL_PIN"].strip(), totp)
    if not d or not d.get("status"):
        raise RuntimeError(f"login failed: {(d or {}).get('message')}")
    return api


def angel_tokens():
    last = None
    for url in ANGEL_MASTERS:
        try:
            r = requests.get(url, timeout=90)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            last = e
    else:
        raise RuntimeError(f"scrip master not reachable: {last}")
    want = {s + "-EQ": s for s in NIFTY50}
    return {want[d["symbol"]]: d["token"] for d in data
            if d.get("exch_seg") == "NSE" and d.get("symbol") in want}


def angel_quote_row(q, sym):
    ltp = float(q["ltp"])
    chg = float(q["percentChange"])
    return {"sym": sym, "ltp": ltp, "prev": ltp / (1 + chg / 100), "chg": chg}


def angel_volume_ratios(api, tokens, now):
    import pandas as pd
    frm = (now - timedelta(days=9)).strftime("%Y-%m-%d 09:15")
    to = now.strftime("%Y-%m-%d %H:%M")
    out = {}
    for sym, tok in tokens.items():
        time.sleep(0.4)  # stay under the historical-data rate limit
        try:
            d = api.getCandleData({"exchange": "NSE", "symboltoken": tok,
                                   "interval": "ONE_MINUTE", "fromdate": frm, "todate": to})
            c = (d or {}).get("data") or []
            idx = pd.to_datetime([x[0] for x in c])
            idx = idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)
            out[sym] = vol_ratio(pd.Series([float(x[5]) for x in c], index=idx), now.date()) if c else None
        except Exception as e:
            print(f"candles {sym}: {e}", file=sys.stderr)
            out[sym] = None
    return out


def fetch_angel(now):
    if any(not os.environ.get(k) for k in ANGEL_SECRETS):
        raise RuntimeError("Angel One secrets not set")
    api = angel_login()
    tokens = angel_tokens()
    if len(tokens) < 40:
        raise RuntimeError(f"only {len(tokens)} Angel tokens found")
    by_token = {v: k for k, v in tokens.items()}
    ids, rows = list(by_token), []
    for i in range(0, len(ids), 25):
        resp = api.getMarketData("FULL", {"NSE": ids[i:i + 25]})
        for q in ((resp or {}).get("data") or {}).get("fetched", []):
            sym = by_token.get(str(q.get("symbolToken")))
            if sym:
                try:
                    rows.append(angel_quote_row(q, sym))
                except Exception as e:
                    print(f"quote {sym}: {e}", file=sys.stderr)
    index = None
    try:
        q = api.getMarketData("FULL", {"NSE": [ANGEL_INDEX_TOKEN]})["data"]["fetched"][0]
        index = {"ltp": float(q["ltp"]), "chg": float(q["percentChange"])}
    except Exception as e:
        print(f"index quote failed: {e}", file=sys.stderr)
    if len(rows) < 40:
        raise RuntimeError(f"only {len(rows)} Angel quotes")
    return rows, index, angel_volume_ratios(api, tokens, now)

# ------------------------------------------------------------------- volume
def vol_ratio(vol, today):
    """Volume so far today / average volume over the same number of minutes
    on previous days. `vol` = 1-minute volume Series with an IST DatetimeIndex."""
    vol = vol.dropna()
    if vol.empty:
        return None
    groups = {d: g for d, g in vol.groupby(vol.index.date)}
    if today not in groups:
        return None
    n = len(groups[today])
    prev = [g.iloc[:n].sum() for d, g in groups.items() if d < today]
    prev = [p for p in prev if p > 0]
    if len(prev) < 3:
        return None
    return float(groups[today].sum() / (sum(prev) / len(prev)))


def fetch_volume_ratios(today):
    if yf is None:
        return {}
    try:
        data = yf.download([s + ".NS" for s in NIFTY50], period="7d", interval="1m",
                           group_by="ticker", progress=False, threads=True)
    except Exception as e:
        print(f"volume download failed: {e}", file=sys.stderr)
        return {}
    out = {}
    for s in NIFTY50:
        try:
            v = data[s + ".NS"]["Volume"].copy()
            v.index = v.index.tz_convert(IST)
            out[s] = vol_ratio(v, today)
        except Exception:
            out[s] = None
    return out


# ----------------------------------------------------------------- analysis
def tick(x):
    return round(round(x / 0.05) * 0.05, 2)  # NSE tick size 0.05


def sector_heat(rows):
    acc = {}
    for r in rows:
        acc.setdefault(SECTOR_OF.get(r["sym"], "Other"), []).append(r["chg"])
    return sorted(((s, sum(v) / len(v)) for s, v in acc.items()), key=lambda x: -x[1])


def build_ideas(rows, vr, cfg):
    ideas = []
    for r in rows:
        v = vr.get(r["sym"])
        if abs(r["chg"]) < cfg["min_move_pct"] or v is None or v < cfg["min_volume_ratio"]:
            continue
        buy = r["chg"] > 0
        e, t, s = r["ltp"], cfg["target_pct"] / 100, cfg["stop_pct"] / 100
        ideas.append({
            "sym": r["sym"], "side": "BUY" if buy else "SELL", "chg": r["chg"], "vol": v,
            "entry": tick(e),
            "target": tick(e * (1 + t) if buy else e * (1 - t)),
            "sl": tick(e * (1 - s) if buy else e * (1 + s)),
        })
    ideas.sort(key=lambda i: abs(i["chg"]), reverse=True)
    return ideas[: cfg["max_ideas"]]


def build_message(rows, index, vr, cfg, source):
    ts = datetime.now(IST).strftime("%a %d %b %Y, %H:%M IST")
    adv = sum(r["chg"] > 0 for r in rows)
    movers = sum(abs(r["chg"]) >= cfg["min_move_pct"] for r in rows)
    nifty = (f"Nifty 50: {index['ltp']:,.2f} ({index['chg']:+.2f}%)"
             if index else "Nifty 50: डेटा उपलब्ध नहीं")
    heat = "\n".join(f"{('▲' if c >= 0 else '▼')} {s:<9}{c:+.2f}%" for s, c in sector_heat(rows))
    ideas = build_ideas(rows, vr, cfg)
    if ideas:
        body = "\n".join(
            f"{i['side']:<4} {i['sym']}  {i['chg']:+.2f}%  Vol {i['vol']:.1f}x\n"
            f" Entry {i['entry']:.2f}  SL {i['sl']:.2f}\n Target {i['target']:.2f}"
            for i in ideas)
    else:
        body = "कोई शेयर शर्तें पूरी नहीं करता"
    vol_known = sum(v is not None for v in vr.values())
    vol_note = "" if vol_known else "\n⚠️ Volume डेटा नहीं मिला, इसलिए कोई सिफ़ारिश नहीं।"
    return (
        f"📊 *Nifty 50 स्कैनर* — {ts}\n{nifty}\n"
        f"बढ़त {adv} | गिरावट {len(rows) - adv} | ±{cfg['min_move_pct']:g}% से ऊपर: {movers}\n\n"
        f"🏭 *सेक्टर हीट* (शेयरों की औसत चाल)\n```\n{heat}\n```\n"
        f"✅ *ट्रेड आइडिया* (चाल ≥ ±{cfg['min_move_pct']:g}% और Volume ≥ {cfg['min_volume_ratio']:g}x)\n"
        f"```\n{body}\n```{vol_note}\n"
        f"_Entry = मौजूदा भाव, Target {cfg['target_pct']:g}%, SL {cfg['stop_pct']:g}%। "
        f"Vol = अब तक का volume ÷ पिछले दिनों का उसी समय तक का औसत। "
        f"शुरुआती मिनटों में चाल अस्थिर होती है। स्रोत: {source}। "
        f"केवल जानकारी, निवेश सलाह नहीं।_"
    )


def send(text):
    token = "".join(os.environ["TELEGRAM_BOT_TOKEN"].split())
    chat = "".join(os.environ["TELEGRAM_CHAT_ID"].split())
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": text, "parse_mode": "Markdown"}, timeout=20)
    if not r.ok:
        sys.exit(f"Telegram error {r.status_code}: {r.text}")


if __name__ == "__main__":
    cfg = dict(DEFAULT_CFG)
    try:
        cfg.update(json.load(open(os.path.join(os.path.dirname(__file__), "criteria.json"))))
    except Exception as e:
        print(f"criteria.json not read ({e}); using defaults", file=sys.stderr)
    now = datetime.now(IST)
    if now.weekday() >= 5 and "--force" not in sys.argv:
        print("Weekend - market closed."); sys.exit(0)
    rows, index, source, vr = fetch_all(now)
    if len(rows) < 40:
        sys.exit(f"Only {len(rows)} quotes fetched; aborting.")
    if vr is None or sum(v is not None for v in vr.values()) < 25:
        vr = fetch_volume_ratios(now.date())
    msg = build_message(rows, index, vr, cfg, source)
    print(msg) if "--dry" in sys.argv else send(msg)
