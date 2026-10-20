"""Fast Angel One quotes for the 9:15 report.
All 100 live prices come in a few calls, but the slow volume check (one request per stock) is done only for
the stocks that have already moved by min_move % or more - the only ones the strategy can use.
This makes the report reach Telegram about 30 seconds sooner.   fetch(now, min_move) -> rows, index, volume_ratios
"""
import os
import sys


def fetch(now, min_move=1.0):
    import bot
    if any(not os.environ.get(k) for k in bot.ANGEL_SECRETS):
        raise RuntimeError("Angel One secrets not set")
    api = bot.angel_login()
    tokens = bot.angel_tokens()
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
                    rows.append(bot.angel_quote_row(q, sym))
                except Exception as e:
                    print(f"quote {sym}: {e}", file=sys.stderr)
    index = None
    try:
        q = api.getMarketData("FULL", {"NSE": [bot.ANGEL_INDEX_TOKEN]})["data"]["fetched"][0]
        index = {"ltp": float(q["ltp"]), "chg": float(q["percentChange"])}
    except Exception as e:
        print(f"index quote failed: {e}", file=sys.stderr)
    if len(rows) < 40:
        raise RuntimeError(f"only {len(rows)} Angel quotes")
    movers = {r["sym"]: tokens[r["sym"]] for r in rows if abs(r["chg"]) >= min_move}
    print(f"volume check for {len(movers)} movers (of {len(rows)} stocks)")
    return rows, index, bot.angel_volume_ratios(api, movers, now)
  
