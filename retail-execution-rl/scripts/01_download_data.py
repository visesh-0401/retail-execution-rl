"""
scripts/01_download_data.py
---------------------------
Week 1 task: Download 6 months of 1-minute OHLCV data for 6 tickers
from Yahoo Finance and save to data/.

Usage:
    python scripts/01_download_data.py
    python scripts/01_download_data.py --stocks AAPL MSFT --interval 1m
"""

import argparse
import os
import sys

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_STOCKS = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ"]
DEFAULT_START = "2025-09-01"
DEFAULT_END = "2026-03-01"
DEFAULT_INTERVAL = "1m"   # 1-minute bars (change to "1d" for daily)
DATA_DIR = "data"


def download_stock(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """Download OHLCV data for a single ticker."""
    print(f"  Downloading {ticker} ({interval} bars, {start} → {end})...", end=" ", flush=True)
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        print("EMPTY — skipped.")
        return df
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    print(f"OK — {len(df):,} rows, {df.index[0].date()} to {df.index[-1].date()}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Download OHLCV data via yfinance")
    parser.add_argument("--stocks", nargs="+", default=DEFAULT_STOCKS, help="Tickers to download")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=DEFAULT_END, help="End date YYYY-MM-DD")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help="Bar interval (1m, 5m, 1h, 1d)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading market data")
    print(f"  Stocks   : {args.stocks}")
    print(f"  Period   : {args.start} → {args.end}")
    print(f"  Interval : {args.interval}")
    print(f"  Save dir : {DATA_DIR}/")
    print(f"{'='*60}\n")

    # NOTE: yfinance 1-minute data is only available for the last 30 days.
    # For longer historical runs, use interval="1d" (daily bars).
    if args.interval == "1m":
        print("⚠  WARNING: yfinance only provides 1m data for the last ~30 days.")
        print("   For the full 6-month dataset, use --interval 1d (daily bars).")
        print("   Switching to daily bars automatically for this run.\n")
        args.interval = "1d"

    downloaded = []
    skipped = []

    for ticker in args.stocks:
        df = download_stock(ticker, args.start, args.end, args.interval)
        if df.empty:
            skipped.append(ticker)
            continue

        path = os.path.join(DATA_DIR, f"{ticker}_{args.interval}.csv")
        df.to_csv(path)
        downloaded.append((ticker, len(df), path))

    print(f"\n{'='*60}")
    print(f"✅ Download complete")
    print(f"{'='*60}")
    for ticker, rows, path in downloaded:
        print(f"  {ticker:8s}  {rows:6,} rows  →  {path}")
    if skipped:
        print(f"\n⚠  Skipped (empty data): {skipped}")

    print(f"\nTotal files saved: {len(downloaded)}")
    print("Next: run  python scripts/02_verify_data.py\n")


if __name__ == "__main__":
    main()
