"""
scripts/02_verify_data.py
--------------------------
Week 1 task: Sanity-check all downloaded CSVs in data/.

Checks:
  - Files are present
  - DatetimeIndex is valid
  - No missing OHLCV columns
  - Missing value counts
  - Date range and row counts
  - Basic price sanity (no zero/negative prices)

Usage:
    python scripts/02_verify_data.py
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

DATA_DIR = "data"
REQUIRED_COLS = {"Open", "High", "Low", "Close", "Volume"}


def verify_file(path: str) -> dict:
    ticker = os.path.basename(path).split("_")[0]
    result = {"Ticker": ticker, "File": path, "Status": "OK", "Issues": []}

    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception as e:
        result["Status"] = "ERROR"
        result["Issues"].append(f"Failed to load: {e}")
        return result

    # Column check
    missing_cols = REQUIRED_COLS - set(df.columns)
    if missing_cols:
        result["Issues"].append(f"Missing columns: {missing_cols}")

    # Row count
    result["Rows"] = len(df)
    if len(df) < 10:
        result["Issues"].append(f"Too few rows: {len(df)}")

    # Date range
    try:
        result["Start"] = str(df.index[0].date())
        result["End"] = str(df.index[-1].date())
    except Exception:
        result["Issues"].append("Invalid DatetimeIndex")

    # NaN check
    nan_counts = df[list(REQUIRED_COLS & set(df.columns))].isna().sum().sum()
    result["NaNs"] = int(nan_counts)
    if nan_counts > 0:
        result["Issues"].append(f"{nan_counts} NaN values")

    # Price sanity
    if "Close" in df.columns:
        if (df["Close"] <= 0).any():
            result["Issues"].append("Non-positive Close prices found")
        result["AvgClose"] = round(float(df["Close"].mean()), 2)

    if result["Issues"]:
        result["Status"] = "WARN"

    return result


def main():
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

    if not csv_files:
        print(f"\n❌ No CSV files found in '{DATA_DIR}/'.")
        print("   Run:  python scripts/01_download_data.py  first.\n")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"Data Verification Report  ({len(csv_files)} files in {DATA_DIR}/)")
    print(f"{'='*65}")

    results = [verify_file(f) for f in csv_files]

    # Summary table
    print(f"\n{'Ticker':<10} {'Rows':>7} {'Start':>12} {'End':>12} {'NaNs':>6} {'AvgClose':>10}  Status")
    print("-" * 65)
    for r in results:
        ticker  = r.get("Ticker", "?")
        rows    = r.get("Rows", "—")
        start   = r.get("Start", "—")
        end     = r.get("End", "—")
        nans    = r.get("NaNs", "—")
        avg_cls = r.get("AvgClose", "—")
        status  = r.get("Status", "?")
        icon    = "✅" if status == "OK" else ("⚠ " if status == "WARN" else "❌")
        print(f"{ticker:<10} {str(rows):>7} {str(start):>12} {str(end):>12} {str(nans):>6} {str(avg_cls):>10}  {icon} {status}")
        for issue in r.get("Issues", []):
            print(f"{'':>10}  ↳ {issue}")

    # Overall verdict
    n_ok   = sum(1 for r in results if r["Status"] == "OK")
    n_warn = sum(1 for r in results if r["Status"] == "WARN")
    n_err  = sum(1 for r in results if r["Status"] == "ERROR")

    print(f"\n{'='*65}")
    print(f"Result: {n_ok} OK  |  {n_warn} warnings  |  {n_err} errors")
    if n_err == 0 and n_warn == 0:
        print("✅ All data looks clean! Proceed to:  python scripts/03_run_baselines.py")
    elif n_err == 0:
        print("⚠  Warnings found but data is usable. Review above.")
    else:
        print("❌ Errors found. Fix before proceeding.")
    print()


if __name__ == "__main__":
    main()
