"""
src/simulator.py
----------------
RetailExecutionSimulator: simulates retail broker order execution under
realistic API rate limits, discrete order sizing, latency, and market impact.

This is the core of the research project — it models constraints that
institutional execution research ignores.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ExecutionResult:
    """Result of a single order execution via the simulator."""
    target_qty: int
    executed_qty: int
    total_cost_bps: float          # slippage + transaction costs (basis points)
    slippage_bps: float            # price impact vs mid-price (bps)
    transaction_cost_bps: float    # bid-ask spread cost (bps)
    api_penalty: float             # penalty incurred for exceeding API budget
    api_requests_used: int         # total API requests consumed
    api_requests_budget: int       # allowed API budget for this execution
    rejected_orders: int           # how many orders were rejected by broker
    fill_prices: list[float] = field(default_factory=list)  # price of each fill

    def __repr__(self):
        return (
            f"ExecutionResult("
            f"executed={self.executed_qty}/{self.target_qty} shares | "
            f"cost={self.total_cost_bps:.2f} bps | "
            f"slippage={self.slippage_bps:.2f} bps | "
            f"API={self.api_requests_used}/{self.api_requests_budget} requests | "
            f"rejects={self.rejected_orders})"
        )


class RetailExecutionSimulator:
    """
    Simulates order execution under retail broker constraints.

    Key constraints modeled (from proposal Section 6.1):
      - API rate limit: max N requests allowed per execution window
      - Discrete order sizing: rounds to whole shares
      - Random execution latency: modeled as price "drift" during delay
      - Market impact: linear model (qty / avg_volume) * spread
      - Bid-ask spread: from historical data average
      - Order rejection: 2-5% random rejection on resubmission

    Parameters
    ----------
    data : pd.DataFrame
        OHLCV DataFrame with columns [Open, High, Low, Close, Volume].
        Index must be DatetimeIndex.
    rate_limit_rps : int
        Maximum API requests allowed per second (simulated as per-step budget).
    execution_window_steps : int
        Number of 1-minute bars available to complete the execution.
    rejection_prob : float
        Probability an order gets rejected by the broker (0.02–0.05 typical).
    latency_impact_bps : float
        Price drift per unit of simulated latency (basis points).
    seed : Optional[int]
        Random seed for reproducibility.
    """

    SPLIT_FRACTIONS = [0.10, 0.20, 0.30, 0.50, 0.75, 1.00]  # action space

    def __init__(
        self,
        data: pd.DataFrame,
        rate_limit_rps: int = 5,
        execution_window_steps: int = 30,
        rejection_prob: float = 0.03,
        latency_impact_bps: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.data = data.copy()
        self.rate_limit_rps = rate_limit_rps
        self.execution_window_steps = execution_window_steps
        self.rejection_prob = rejection_prob
        self.latency_impact_bps = latency_impact_bps
        self.rng = np.random.default_rng(seed)

        # Precompute rolling average volume (used in market impact formula)
        self.data["avg_volume"] = (
            self.data["Volume"].rolling(20, min_periods=1).mean()
        )
        # Precompute bid-ask spread estimate: (High - Low) / Close in bps
        self.data["spread_bps"] = (
            (self.data["High"] - self.data["Low"]) / self.data["Close"] * 10_000
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        target_qty: int,
        start_idx: int,
        action_sequence: Optional[list[int]] = None,
    ) -> ExecutionResult:
        """
        Execute a full order of `target_qty` shares starting at bar `start_idx`.

        Parameters
        ----------
        target_qty : int
            Total shares to buy.
        start_idx : int
            Bar index (into self.data) where execution begins.
        action_sequence : list[int] or None
            If provided, use this fixed sequence of action indices (0–5).
            If None, uses greedy equal-split (TWAP-like) for testing.

        Returns
        -------
        ExecutionResult
        """
        target_qty = max(1, int(target_qty))
        api_budget = self.rate_limit_rps * self.execution_window_steps
        remaining_qty = target_qty
        api_requests_used = 0
        rejected_orders = 0
        fill_prices = []
        total_slippage = 0.0
        total_transaction_cost = 0.0
        step = 0

        action_iter = iter(action_sequence) if action_sequence else None

        while remaining_qty > 0 and step < self.execution_window_steps:
            bar_idx = min(start_idx + step, len(self.data) - 1)
            bar = self.data.iloc[bar_idx]

            # Check API budget (rate limit enforcement)
            if api_requests_used >= api_budget:
                # Out of API budget — forced to wait (no more orders)
                break

            # Choose split fraction
            if action_iter is not None:
                try:
                    action = next(action_iter)
                except StopIteration:
                    action = 5  # default: 100% of remaining
            else:
                action = 5  # greedy: take all remaining (naive baseline)

            fraction = self.SPLIT_FRACTIONS[action]
            order_qty = max(1, round(remaining_qty * fraction))  # discrete sizing
            order_qty = min(order_qty, remaining_qty)

            # Simulate broker rejection
            if api_requests_used > 0 and self.rng.random() < self.rejection_prob:
                rejected_orders += 1
                api_requests_used += 1  # still costs an API call
                step += 1
                continue

            # Compute execution price
            mid_price = (bar["High"] + bar["Low"]) / 2.0
            avg_vol = max(bar["avg_volume"], 1)
            spread_bps = bar["spread_bps"]

            # Market impact: linear in qty / avg_volume (in bps)
            impact_bps = (order_qty / avg_vol) * spread_bps

            # Latency impact: simulated price drift from delay
            latency_bps = self.latency_impact_bps * self.rng.uniform(0.5, 5.0)

            # Execution price = mid + impact + latency drift
            exec_price_bps_above_mid = impact_bps + latency_bps
            exec_price = mid_price * (1 + exec_price_bps_above_mid / 10_000)

            # Accumulate costs
            slippage_bps = (exec_price - mid_price) / mid_price * 10_000
            txn_cost_bps = spread_bps / 2.0  # pay half spread

            total_slippage += slippage_bps * order_qty
            total_transaction_cost += txn_cost_bps * order_qty

            fill_prices.append(exec_price)
            remaining_qty -= order_qty
            api_requests_used += 1
            step += 1

        executed_qty = target_qty - remaining_qty
        norm = max(executed_qty, 1)

        # API over-budget penalty (from reward function in proposal)
        api_overage = max(0, api_requests_used - api_budget)
        api_penalty = 10.0 * api_overage

        return ExecutionResult(
            target_qty=target_qty,
            executed_qty=executed_qty,
            total_cost_bps=(total_slippage + total_transaction_cost) / norm,
            slippage_bps=total_slippage / norm,
            transaction_cost_bps=total_transaction_cost / norm,
            api_penalty=api_penalty,
            api_requests_used=api_requests_used,
            api_requests_budget=api_budget,
            rejected_orders=rejected_orders,
            fill_prices=fill_prices,
        )

    # ------------------------------------------------------------------
    # State snapshot (used by environment.py)
    # ------------------------------------------------------------------

    def get_state_features(
        self,
        bar_idx: int,
        remaining_qty: int,
        target_qty: int,
        elapsed_steps: int,
        api_requests_used: int,
    ) -> np.ndarray:
        """
        Compute the 8-feature state vector from the proposal (Section 6.2).

        Features:
          0. remaining_quantity  (% of original order left)
          1. elapsed_time        (% of execution window used)
          2. bid_ask_spread_bps  (current bar spread in bps)
          3. volatility_5min     (rolling 5-bar close std, normalized)
          4. api_requests_used   (count, normalized by budget)
          5. api_requests_limit  (rate limit, normalized)
          6. market_momentum     (5-bar price momentum, normalized)
          7. time_until_deadline (fraction of window remaining)
        """
        bar_idx = min(bar_idx, len(self.data) - 1)
        bar = self.data.iloc[bar_idx]

        remaining_frac = remaining_qty / max(target_qty, 1)
        elapsed_frac = elapsed_steps / max(self.execution_window_steps, 1)
        spread_bps = float(bar["spread_bps"]) / 100.0  # normalize ~0-1

        # 5-bar rolling volatility
        lo = max(0, bar_idx - 4)
        closes = self.data["Close"].iloc[lo : bar_idx + 1].values
        vol = float(np.std(closes) / max(np.mean(closes), 1e-8))

        api_budget = self.rate_limit_rps * self.execution_window_steps
        api_used_frac = api_requests_used / max(api_budget, 1)
        api_limit_norm = self.rate_limit_rps / 10.0  # normalize by max (10 rps)

        # 5-bar momentum: (current close - close 5 bars ago) / close 5 bars ago
        past_idx = max(0, bar_idx - 4)
        past_close = float(self.data["Close"].iloc[past_idx])
        curr_close = float(bar["Close"])
        momentum = (curr_close - past_close) / max(past_close, 1e-8)

        time_remaining = 1.0 - elapsed_frac

        return np.array(
            [
                remaining_frac,
                elapsed_frac,
                spread_bps,
                vol,
                api_used_frac,
                api_limit_norm,
                momentum,
                time_remaining,
            ],
            dtype=np.float32,
        )
