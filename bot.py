"""Nifty 50 top gainers/losers -> Telegram. Run at 09:15 IST on trading days."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

try:
    import yfinance as yf
except ImportError:  # only needed for the backup source
    yf = None

# Verify against NSE's current constituents list (changes semi-annually).
NIFTY50 = """ADANIENT ADANIPORTS APOLLOHOSP ASIANPAINT AXISBANK BAJAJ-AUTO BAJFINANCE BAJAJFINSV
BEL BHARTIARTL CIPLA COALINDIA DRREDDY EICHERMOT ETERNAL GRASIM HCLTECH HDFCBANK HDFCLIFE
HINDALCO HINDUNILVR ICICIBANK INDIGO INFY ITC JIOFIN JSWSTEEL KOTAKBANK LT M&M MARUTI
MAXHEALTH NESTLEIND NTPC ONGC POWERGRID RELIANCE SBILIFE SBIN SHRIRAMFIN SUNPHARMA
TATACONSUM TMPV TATASTEEL TCS TECHM TITAN TRENT ULTRACEMCO WIPRO""".split()

IST = ZoneInfo("Asia/Kolkata")


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
    """NSE equity-stockIndices JSON -> list of stock rows (index row excluded)."""
    rows = []
    for d in payload.get("data", []):
        if d.get("priority") == 1 or d.get("symbol") == "NIFTY 50":
            continue  # first row is the index itself
        rows.append({"sym": d["symbol"], "ltp": float(d["lastPrice"]),
                     "prev": float(d["previousClose"]), "chg": float(d["pChange"])})
    return rows


def fetch_nse():
    s = requests.Session()
    s.headers.update(NSE_HEADERS)
    s.get(NSE_HOME, timeout=15)  # sets the cookies NSE requires
    s.get(NSE_HOME + "/market-data/live-market-indices/heatmap", timeout=15)
    r = s.get(NSE_API, timeout=15)
    r.raise_for_status()
    return parse_nse(r.json())


def quote(sym):
    try:
        fi = yf.Ticker(sym + ".NS").fast_info
        last, prev = fi["last_price"], fi["previous_close"]
        return {"sym": sym, "ltp": last, "prev": prev, "chg": (last - prev) / prev * 100}
    except Exception as e:
        print(f"skip {sym}: {e}", file=sys.stderr)
        return None


def fetch_yahoo():
    if yf is None:
        return []
    with ThreadPoolExecutor(10) as ex:
        return [q for q in ex.map(quote, NIFTY50) if q]


def fetch_all():
    try:
        rows = fetch_nse()
        if len(rows) >= 40:
            return rows, "NSE India"
        print(f"NSE returned {len(rows)} rows; trying backup", file=sys.stderr)
    except Exception as e:
        print(f"NSE failed: {e}; trying backup", file=sys.stderr)
    return fetch_yahoo(), "Yahoo Finance (backup)"


def build_message(rows, cfg, source="NSE India"):
    n = cfg.get("top_n", 5)
    floor = cfg.get("min_abs_change_pct", 0)
    rows = sorted(rows, key=lambda r: r["chg"], reverse=True)
    gain = [r for r in rows if r["chg"] >= floor][:n]
    lose = [r for r in reversed(rows) if r["chg"] <= -floor][:n]
    adv = sum(r["chg"] > 0 for r in rows)
    fmt = lambda r: f"{r['sym']:<12} ₹{r['ltp']:>9,.2f}  {r['chg']:+.2f}%"
    ts = datetime.now(IST).strftime("%a %d %b %Y, %H:%M IST")
    return (
        f"📊 *Nifty 50 Heat Map* — {ts}\n"
        f"Advances {adv} | Declines {len(rows) - adv}\n\n"
        f"🟢 *Top {len(gain)} Gainers*\n```\n" + "\n".join(map(fmt, gain)) + "\n```\n"
        f"🔴 *Top {len(lose)} Losers*\n```\n" + "\n".join(map(fmt, lose)) + "\n```\n"
        f"_Source: {source}. Change vs previous close. For information only, not advice._"
    )


def send(text):
    token = "".join(os.environ["TELEGRAM_BOT_TOKEN"].split())
    chat = "".join(os.environ["TELEGRAM_CHAT_ID"].split())
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": text, "parse_mode": "Markdown"}, timeout=20)
    if not r.ok:
        sys.exit(f"Telegram error {r.status_code}: {r.text}")


if __name__ == "__main__":
    cfg = json.load(open(os.path.join(os.path.dirname(__file__), "criteria.json")))
    now = datetime.now(IST)
    if now.weekday() >= 5 and "--force" not in sys.argv:
        print("Weekend - market closed."); sys.exit(0)
    rows, source = fetch_all()
    if len(rows) < 40:
        sys.exit(f"Only {len(rows)} quotes fetched; aborting.")
    msg = build_message(rows, cfg, source)
    print(msg) if "--dry" in sys.argv else send(msg)
