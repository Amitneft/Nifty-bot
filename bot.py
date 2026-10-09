"""Nifty 50 top gainers/losers -> Telegram. Run at 09:15 IST on trading days."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import yfinance as yf

# Verify against NSE's current constituents list (changes semi-annually).
NIFTY50 = """ADANIENT ADANIPORTS APOLLOHOSP ASIANPAINT AXISBANK BAJAJ-AUTO BAJFINANCE BAJAJFINSV
BEL BHARTIARTL CIPLA COALINDIA DRREDDY EICHERMOT ETERNAL GRASIM HCLTECH HDFCBANK HDFCLIFE
HINDALCO HINDUNILVR ICICIBANK INDIGO INFY ITC JIOFIN JSWSTEEL KOTAKBANK LT M&M MARUTI
MAXHEALTH NESTLEIND NTPC ONGC POWERGRID RELIANCE SBILIFE SBIN SHRIRAMFIN SUNPHARMA
TATACONSUMER TMPV TATASTEEL TCS TECHM TITAN TRENT ULTRACEMCO WIPRO""".split()

IST = ZoneInfo("Asia/Kolkata")


def quote(sym):
    try:
        fi = yf.Ticker(sym + ".NS").fast_info
        last, prev = fi["last_price"], fi["previous_close"]
        return {"sym": sym, "ltp": last, "prev": prev, "chg": (last - prev) / prev * 100}
    except Exception as e:
        print(f"skip {sym}: {e}", file=sys.stderr)
        return None


def fetch_all():
    with ThreadPoolExecutor(10) as ex:
        return [q for q in ex.map(quote, NIFTY50) if q]


def build_message(rows, cfg):
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
        "_Change vs previous close. For information only, not advice._"
    )


def send(text):
    token, chat = os.environ["TELEGRAM_BOT_TOKEN"], os.environ["TELEGRAM_CHAT_ID"]
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": text, "parse_mode": "Markdown"}, timeout=20)
    r.raise_for_status()


if __name__ == "__main__":
    cfg = json.load(open(os.path.join(os.path.dirname(__file__), "criteria.json")))
    now = datetime.now(IST)
    if now.weekday() >= 5 and "--force" not in sys.argv:
        sys.exit("Weekend - market closed.")
    rows = fetch_all()
    if len(rows) < 40:
        sys.exit(f"Only {len(rows)} quotes fetched; aborting.")
    msg = build_message(rows, cfg)
    print(msg) if "--dry" in sys.argv else send(msg)
