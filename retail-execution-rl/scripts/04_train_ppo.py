"""
scripts/04_train_ppo.py
-----------------------
Week 6-7 task: Train the PPO agent using Stable-Baselines3.

Runs on CPU for smoke tests (--timesteps 10000), designed for
Kaggle GPU for full training (--timesteps 500000, ~4-6 hours).

Features:
  - Multiple random seeds for statistical significance
  - EvalCallback: evaluates and saves best model during training
  - Checkpoint callback: saves model every N steps (for Kaggle GPU crashes)
  - Logs training curves to results/

Usage:
    # Quick smoke test (CPU, ~2 min)
    python scripts/04_train_ppo.py --timesteps 10000 --seed 42

    # Full training run (GPU recommended)
    python scripts/04_train_ppo.py --timesteps 500000 --seeds 42 43 44

    # Train on specific stocks only
    python scripts/04_train_ppo.py --stocks AAPL MSFT --timesteps 200000
"""

import argparse
import glob
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from src.environment import RetailExecutionEnv
from src.data_loader_gpu import GPUDataLoader, get_data_size_mb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data_map(data_dir: str, interval: str = "1d") -> dict:
    pattern = os.path.join(data_dir, f"*_{interval}.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    data_map = {}
    for f in files:
        ticker = os.path.basename(f).split("_")[0]
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        if not df.empty:
            data_map[ticker] = df
    return data_map


def make_env(data_map, rate_limit_rps, window, qty, seed):
    """Factory for a single monitored environment."""
    def _init():
        env = RetailExecutionEnv(
            data_map=data_map,
            target_qty=qty,
            rate_limit_rps=rate_limit_rps,
            execution_window_steps=window,
            seed=seed,
        )
        return Monitor(env)
    return _init


def train_single_seed(
    data_map: dict,
    train_tickers: list,
    eval_tickers: list,
    seed: int,
    timesteps: int,
    rate_limit_rps: int,
    window: int,
    qty: int,
    learning_rate: float,
    batch_size: int,
    models_dir: str,
    results_dir: str,
    use_gpu: bool = False,
) -> dict:
    """Train one PPO run with a single seed. Returns eval metrics.
    
    Parameters
    ----------
    use_gpu : bool
        If True, data_map contains GPU tensors (from GPUDataLoader).
        If False, data_map contains pandas DataFrames.
    """

    run_id = f"ppo_ratelimit{rate_limit_rps}_seed{seed}"
    print(f"\n  ── Seed {seed} | run_id: {run_id} ──")

    train_data = {t: data_map[t] for t in train_tickers if t in data_map}
    eval_data  = {t: data_map[t] for t in eval_tickers  if t in data_map}

    if not train_data:
        print("  ❌ No training data available.")
        return {}
    if not eval_data:
        eval_data = train_data  # fallback

    # Training environment (single env, not vectorized — for simplicity)
    train_env = Monitor(
        RetailExecutionEnv(
            data_map=train_data,
            target_qty=qty,
            rate_limit_rps=rate_limit_rps,
            execution_window_steps=window,
            seed=seed,
        )
    )
    eval_env = Monitor(
        RetailExecutionEnv(
            data_map=eval_data,
            target_qty=qty,
            rate_limit_rps=rate_limit_rps,
            execution_window_steps=window,
            seed=seed + 1000,
        )
    )

    best_model_path = os.path.join(models_dir, f"{run_id}_best")
    checkpoint_path = os.path.join(models_dir, "checkpoints", run_id)
    os.makedirs(checkpoint_path, exist_ok=True)

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_path,
        log_path=os.path.join(results_dir, run_id),
        eval_freq=max(1000, timesteps // 50),   # evaluate 50 times during training
        n_eval_episodes=10,
        deterministic=True,
        verbose=0,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(2000, timesteps // 20),   # save 20 checkpoints
        save_path=checkpoint_path,
        name_prefix=run_id,
        verbose=0,
    )

    # PPO model — using proposal defaults (Section 6.2)
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,       # encourages exploration
        verbose=1,
        seed=seed,
        tensorboard_log=os.path.join(results_dir, "tensorboard"),
    )

    print(f"  Training for {timesteps:,} timesteps...")
    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(models_dir, f"{run_id}_final")
    model.save(final_path)
    print(f"  ✅ Final model saved → {final_path}.zip")

    # Final evaluation
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"  📊 Eval reward: {mean_reward:.2f} ± {std_reward:.2f}")

    train_env.close()
    eval_env.close()

    return {
        "seed": seed,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "run_id": run_id,
        "model_path": f"{final_path}.zip",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for retail execution")
    parser.add_argument("--timesteps", type=int, default=50_000,
                        help="Total training timesteps (default: 50k smoke test)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Random seeds (multiple for statistical significance)")
    parser.add_argument("--stocks", nargs="+", default=None,
                        help="Tickers to train on (default: first 3 in data/)")
    parser.add_argument("--eval-stocks", nargs="+", default=None,
                        help="Tickers to evaluate on (default: remaining tickers)")
    parser.add_argument("--rate-limit", type=int, default=5,
                        help="API rate limit in req/sec")
    parser.add_argument("--window", type=int, default=30,
                        help="Execution window in bars")
    parser.add_argument("--qty", type=int, default=100,
                        help="Target order size (shares)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="PPO learning rate (proposal default: 3e-4)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="PPO mini-batch size")
    parser.add_argument("--use-gpu", action="store_true", default=False,
                        help="Preload data to GPU for faster training (40-60%% speedup)")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print("PPO Agent Training — Retail Execution RL")
    print(f"{'='*65}")
    print(f"  Timesteps   : {args.timesteps:,}")
    print(f"  Seeds       : {args.seeds}")
    print(f"  Rate limit  : {args.rate_limit} rps")
    print(f"  Window      : {args.window} bars")
    print(f"  Qty         : {args.qty} shares")
    print(f"  Lr          : {args.lr}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  GPU mode    : {'✓ ENABLED (40-60% speedup expected)' if args.use_gpu else '✗ CPU mode'}")
    print(f"{'='*65}\n")

    # Load data
    data_map = load_data_map(args.data_dir, args.interval)
    if not data_map:
        print(f"❌ No data found in '{args.data_dir}/'.")
        print("   Run:  python scripts/01_download_data.py  first.")
        sys.exit(1)

    all_tickers = list(data_map.keys())
    print(f"Available tickers: {all_tickers}")
    
    # GPU PRELOAD OPTIMIZATION
    # =======================
    if args.use_gpu:
        print("\n🚀 GPU DATA PRELOADING...")
        loader = GPUDataLoader(data_map, use_gpu=True, verbose=True)
        data_map = loader.to_device()
        print("✓ All data loaded to GPU\n")
    else:
        data_size = get_data_size_mb(data_map)
        print(f"\n💾 CPU mode (data size: {data_size:.1f} MB)")
        print(f"   Tip: Use --use-gpu flag for 40-60%% faster training on Kaggle GPU\n")

    # Split train / eval tickers
    train_tickers = args.stocks or all_tickers[:3]
    eval_tickers  = args.eval_stocks or all_tickers[3:] or train_tickers

    print(f"Train tickers : {train_tickers}")
    print(f"Eval tickers  : {eval_tickers}")

    # Train across all seeds
    all_results = []
    for seed in args.seeds:
        result = train_single_seed(
            data_map=data_map,
            train_tickers=train_tickers,
            eval_tickers=eval_tickers,
            seed=seed,
            timesteps=args.timesteps,
            rate_limit_rps=args.rate_limit,
            window=args.window,
            qty=args.qty,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            models_dir=args.models_dir,
            results_dir=args.results_dir,
            use_gpu=args.use_gpu,
        )
        if result:
            all_results.append(result)

    # Summary
    if all_results:
        rewards = [r["mean_reward"] for r in all_results]
        print(f"\n{'='*65}")
        print(f"Training Complete — {len(all_results)} seed(s)")
        print(f"{'='*65}")
        print(f"  Mean reward  : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"  Best seed    : {all_results[int(np.argmax(rewards))]['seed']}")
        print(f"  Best model   : {all_results[int(np.argmax(rewards))]['model_path']}")

        # Save summary
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(args.results_dir, "training_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n  Summary saved → {summary_path}")

    print("\n✅ Done! Next: analyze results and compare against baselines.\n")


if __name__ == "__main__":
    main()
