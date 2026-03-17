"""
scripts/03_run_baselines.py
---------------------------
Week 3-4 task: Evaluate all baseline strategies (TWAP, VWAP, MarketOrder,
Random) against loaded data and print a comparison table.

Usage:
    python scripts/03_run_baselines.py
    python scripts/03_run_baselines.py --qty 200 --rate-limit 3 --episodes 50
"""

import argparse
import glob
import os
import sys

import pandas as pd

# Add project root to path so 'src' is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines import run_all_baselines


def load_data_map(data_dir: str, interval: str = "1d") -> dict:
    """Load all CSVs matching interval suffix."""
    pattern = os.path.join(data_dir, f"*_{interval}.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        # Fallback: any CSV
        files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        return {}

    data_map = {}
    for f in files:
        ticker = os.path.basename(f).split("_")[0]
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        if not df.empty:
            data_map[ticker] = df
    return data_map


def main():
    parser = argparse.ArgumentParser(description="Run baseline execution strategies")
    parser.add_argument("--data-dir", default="data", help="Directory with CSV files")
    parser.add_argument("--interval", default="1d", help="Bar interval suffix (1d, 1m)")
    parser.add_argument("--qty", type=int, default=100, help="Target order size (shares)")
    parser.add_argument("--rate-limit", type=int, default=5, help="API rate limit (rps)")
    parser.add_argument("--window", type=int, default=30, help="Execution window (bars)")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per stock")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save", default=None, help="Save results to CSV path")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print("Baseline Execution Strategy Comparison")
    print(f"{'='*65}")
    print(f"  Order size      : {args.qty} shares")
    print(f"  API rate limit  : {args.rate_limit} req/sec")
    print(f"  Exec window     : {args.window} bars")
    print(f"  Episodes/stock  : {args.episodes}")
    print(f"  Seed            : {args.seed}")
    print(f"{'='*65}\n")

    # Load data
    data_map = load_data_map(args.data_dir, args.interval)
    if not data_map:
        print(f"❌ No data found in '{args.data_dir}/'.")
        print("   Run:  python scripts/01_download_data.py  first.\n")
        sys.exit(1)

    print(f"Loaded {len(data_map)} tickers: {list(data_map.keys())}\n")
    print("Running baselines...\n")

    results = run_all_baselines(
        data_map=data_map,
        target_qty=args.qty,
        rate_limit_rps=args.rate_limit,
        execution_window_steps=args.window,
        n_episodes=args.episodes,
        seed=args.seed,
    )

    if results.empty:
        print("❌ No results generated. Check data files.")
        sys.exit(1)

    # Pivot for a clean comparison table
    pivot = results.pivot_table(
        index="Baseline",
        values=["AvgCostBps", "AvgSlippageBps", "AvgApiRequestsUsed", "AvgRejects"],
        aggfunc="mean",
    ).round(3)

    pivot.columns = ["Avg Cost (bps)", "Avg Slippage (bps)", "Avg API Reqs", "Avg Rejects"]
    pivot = pivot.sort_values("Avg Cost (bps)")

    print("=" * 65)
    print("Summary (averaged across all tickers and episodes):")
    print("=" * 65)
    print(pivot.to_string())

    print(f"\n{'='*65}")
    print("Per-Ticker Detail:")
    print("=" * 65)
    print(results.to_string(index=False))

    # Best baseline
    best = pivot["Avg Cost (bps)"].idxmin()
    print(f"\n🏆 Best baseline: {best} ({pivot.loc[best, 'Avg Cost (bps)']:.3f} bps avg cost)")
    print("   ← This is the bar your PPO agent needs to beat!\n")

    if args.save:
        results.to_csv(args.save, index=False)
        print(f"Results saved → {args.save}\n")
    else:
        save_path = os.path.join("results", "baseline_results.csv")
        os.makedirs("results", exist_ok=True)
        results.to_csv(save_path, index=False)
        print(f"Results auto-saved → {save_path}\n")

    print("Next: train the PPO agent with  python scripts/04_train_ppo.py\n")


if __name__ == "__main__":
    main()
