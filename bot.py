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
TATACONSUMER TMPV TATASTEEL TCS TECHM TITAN TRENT ULTRACEMCO WIPRO""".split()

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
