"""
src/baselines.py
----------------
Baseline execution strategies for comparison against the PPO agent.

All baselines respect the API rate limit constraint — this is the key
difference from naive TWAP/VWAP in institutional literature.

Baselines implemented (from proposal Section 6.3):
  1. TWAPBaseline    — Time-Weighted Average Price (equal time slices)
  2. VWAPBaseline    — Volume-Weighted Average Price (volume-proportional splits)
  3. MarketOrderBaseline — Single order, full quantity at once
  4. RandomBaseline  — Uniformly random split fractions at each step
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.simulator import RetailExecutionSimulator, ExecutionResult


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaselineStrategy:
    """Abstract base for all baseline strategies."""

    name: str = "BaselineStrategy"

    def run(
        self,
        simulator: RetailExecutionSimulator,
        target_qty: int,
        start_idx: int,
    ) -> ExecutionResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# TWAP Baseline
# ---------------------------------------------------------------------------

class TWAPBaseline(BaselineStrategy):
    """
    Time-Weighted Average Price baseline.

    Splits the target quantity into equal slices across the execution window,
    then submits one slice per time step — subject to the API rate limit.

    If rate limit is at 1 rps, TWAP must space orders one per bar.
    """

    name = "TWAP"

    def run(
        self,
        simulator: RetailExecutionSimulator,
        target_qty: int,
        start_idx: int,
    ) -> ExecutionResult:
        window = simulator.execution_window_steps
        api_budget = simulator.rate_limit_rps * window

        # Equal splits. At most `api_budget` orders across the window.
        n_orders = min(window, api_budget)
        qty_per_order = max(1, round(target_qty / n_orders))

        action_sequence = []
        remaining = target_qty
        for i in range(n_orders):
            if remaining <= 0:
                break
            # Determine fraction of REMAINING quantity to submit
            slice_qty = min(qty_per_order, remaining)
            frac = slice_qty / max(remaining, 1)
            # Find closest action index
            diffs = [abs(frac - f) for f in simulator.SPLIT_FRACTIONS]
            action_sequence.append(int(np.argmin(diffs)))
            remaining -= slice_qty

        return simulator.execute(target_qty, start_idx, action_sequence)


# ---------------------------------------------------------------------------
# VWAP Baseline
# ---------------------------------------------------------------------------

class VWAPBaseline(BaselineStrategy):
    """
    Volume-Weighted Average Price baseline.

    Allocates order sizes proportionally to historical volume at each bar,
    respecting the API rate limit. Higher volume bars get larger slices.
    """

    name = "VWAP"

    def run(
        self,
        simulator: RetailExecutionSimulator,
        target_qty: int,
        start_idx: int,
    ) -> ExecutionResult:
        window = simulator.execution_window_steps
        api_budget = simulator.rate_limit_rps * window
        n_orders = min(window, api_budget)

        # Extract volume profile for the execution window
        end_idx = min(start_idx + n_orders, len(simulator.data))
        vol_slice = simulator.data["Volume"].iloc[start_idx:end_idx].values.astype(float)

        if vol_slice.sum() == 0:
            vol_weights = np.ones(len(vol_slice)) / len(vol_slice)
        else:
            vol_weights = vol_slice / vol_slice.sum()

        # Convert volume weights to action indices per step
        action_sequence = []
        remaining = float(target_qty)

        for w in vol_weights:
            if remaining <= 0:
                break
            slice_qty = max(1, round(target_qty * w))
            slice_qty = min(slice_qty, int(remaining))
            frac = slice_qty / max(remaining, 1)
            diffs = [abs(frac - f) for f in simulator.SPLIT_FRACTIONS]
            action_sequence.append(int(np.argmin(diffs)))
            remaining -= slice_qty

        return simulator.execute(target_qty, start_idx, action_sequence)


# ---------------------------------------------------------------------------
# Market Order Baseline
# ---------------------------------------------------------------------------

class MarketOrderBaseline(BaselineStrategy):
    """
    Market Order: submits the entire quantity in a single order at time 0.
    This is the worst-case reference (maximum market impact).
    """

    name = "MarketOrder"

    def run(
        self,
        simulator: RetailExecutionSimulator,
        target_qty: int,
        start_idx: int,
    ) -> ExecutionResult:
        # Action 5 = 100% of remaining. Single step.
        return simulator.execute(target_qty, start_idx, action_sequence=[5])


# ---------------------------------------------------------------------------
# Random Baseline
# ---------------------------------------------------------------------------

class RandomBaseline(BaselineStrategy):
    """
    Random splits: at each step, randomly picks one of the 6 split fractions.
    Represents a lower bound / uninformed agent.
    """

    name = "Random"

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def run(
        self,
        simulator: RetailExecutionSimulator,
        target_qty: int,
        start_idx: int,
    ) -> ExecutionResult:
        window = simulator.execution_window_steps
        api_budget = simulator.rate_limit_rps * window
        n_orders = min(window, api_budget)
        action_sequence = self.rng.integers(0, 6, size=n_orders).tolist()
        return simulator.execute(target_qty, start_idx, action_sequence)


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_all_baselines(
    data_map: dict[str, pd.DataFrame],
    target_qty: int = 100,
    rate_limit_rps: int = 5,
    execution_window_steps: int = 30,
    n_episodes: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run all four baselines across all stocks and return a summary DataFrame.

    Parameters
    ----------
    data_map : dict[str, pd.DataFrame]
        Mapping of ticker -> OHLCV DataFrame.
    target_qty : int
        Shares to execute per episode.
    rate_limit_rps : int
        API rate limit (requests per second).
    execution_window_steps : int
        Number of 1-min bars in the execution window.
    n_episodes : int
        How many random start points to average over per stock.
    seed : int
        Master random seed.

    Returns
    -------
    pd.DataFrame with columns [Baseline, Ticker, AvgCostBps, AvgSlippageBps,
                                AvgApiRequests, AvgRejects]
    """
    rng = np.random.default_rng(seed)
    strategies = [
        TWAPBaseline(),
        VWAPBaseline(),
        MarketOrderBaseline(),
        RandomBaseline(seed=seed),
    ]
    rows = []

    for ticker, df in data_map.items():
        sim = RetailExecutionSimulator(
            df,
            rate_limit_rps=rate_limit_rps,
            execution_window_steps=execution_window_steps,
            seed=seed,
        )
        # Valid start indices with enough bars remaining
        max_start = len(df) - execution_window_steps - 1
        if max_start <= 0:
            continue
        starts = rng.integers(0, max_start, size=n_episodes).tolist()

        for strategy in strategies:
            costs, slippages, reqs, rejects = [], [], [], []
            for start_idx in starts:
                result = strategy.run(sim, target_qty, int(start_idx))
                costs.append(result.total_cost_bps)
                slippages.append(result.slippage_bps)
                reqs.append(result.api_requests_used)
                rejects.append(result.rejected_orders)

            rows.append({
                "Baseline": strategy.name,
                "Ticker": ticker,
                "AvgCostBps": round(np.mean(costs), 3),
                "AvgSlippageBps": round(np.mean(slippages), 3),
                "AvgApiRequestsUsed": round(np.mean(reqs), 1),
                "AvgRejects": round(np.mean(rejects), 2),
            })

    return pd.DataFrame(rows)
