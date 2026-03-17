"""
src/environment.py
------------------
RetailExecutionEnv: Gymnasium-compatible environment wrapping the
RetailExecutionSimulator for PPO training with Stable-Baselines3.

Follows the proposal's RL formulation (Section 6.2):
  - Observation space: Box(8,) — 8 normalized features
  - Action space:      Discrete(6) — [10%, 20%, 30%, 50%, 75%, 100%]
  - Reward:            -(slippage + transaction_cost + api_rate_penalty)
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.simulator import RetailExecutionSimulator


class RetailExecutionEnv(gym.Env):
    """
    Gymnasium environment for retail order execution under API constraints.

    Each *episode* simulates executing one order of `target_qty` shares
    in a randomly selected stock starting at a randomly selected bar.
    The agent controls how to split the order over time.

    Observation (8 features, all normalized to ~[0, 1]):
      0. remaining_quantity  — fraction of original order still pending
      1. elapsed_time        — fraction of execution window elapsed
      2. bid_ask_spread_bps  — current spread (normalized)
      3. volatility_5min     — rolling 5-bar close std (normalized)
      4. api_requests_used   — fraction of API budget consumed
      5. api_rate_limit_norm — rate limit normalized by 10 rps
      6. market_momentum     — 5-bar price momentum (clipped ±5%)
      7. time_until_deadline — fraction of window remaining

    Actions (Discrete 6):
      0 → submit 10% of remaining qty
      1 → submit 20% of remaining qty
      2 → submit 30% of remaining qty
      3 → submit 50% of remaining qty
      4 → submit 75% of remaining qty
      5 → submit 100% of remaining qty

    Reward (per step):
      reward = -slippage_bps_this_step - txn_cost_bps_this_step - api_penalty
      Final step also penalizes unexecuted quantity.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_map: dict[str, pd.DataFrame],
        target_qty: int = 100,
        rate_limit_rps: int = 5,
        execution_window_steps: int = 30,
        rejection_prob: float = 0.03,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        data_map : dict[str, pd.DataFrame]
            Mapping of ticker -> OHLCV DataFrame (DatetimeIndex).
        target_qty : int
            Number of shares to execute per episode.
        rate_limit_rps : int
            API rate limit in requests per second.
        execution_window_steps : int
            Number of 1-minute bars available per episode.
        rejection_prob : float
            Probability broker rejects an order (2–5%).
        seed : int or None
            Master seed for reproducibility.
        """
        super().__init__()

        self.data_map = data_map
        self.tickers = list(data_map.keys())
        self.target_qty = target_qty
        self.rate_limit_rps = rate_limit_rps
        self.execution_window_steps = execution_window_steps
        self.rejection_prob = rejection_prob

        # ---- Spaces --------------------------------------------------------
        # Observation: 8 normalized floats in [0, 1] (momentum clipped)
        self.observation_space = spaces.Box(
            low=np.array([-1.0] * 8, dtype=np.float32),
            high=np.array([1.0] * 8, dtype=np.float32),
            dtype=np.float32,
        )
        # Action: discrete split fraction index
        self.action_space = spaces.Discrete(6)

        # ---- Internal state (reset per episode) ----------------------------
        self._sim: Optional[RetailExecutionSimulator] = None
        self._bar_idx: int = 0
        self._remaining_qty: int = 0
        self._elapsed_steps: int = 0
        self._api_requests_used: int = 0
        self._total_cost_bps: float = 0.0
        self._np_random = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """Sample a random ticker and start bar, reset episode state."""
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        # Pick a random ticker
        ticker = self._np_random.choice(self.tickers)
        df = self.data_map[ticker]

        # Pick a valid start bar (leave room for the full window)
        max_start = len(df) - self.execution_window_steps - 1
        if max_start <= 0:
            # Fallback: use bar 0 if df is too short
            start_idx = 0
        else:
            start_idx = int(self._np_random.integers(0, max_start))

        # Build simulator for chosen ticker
        self._sim = RetailExecutionSimulator(
            df,
            rate_limit_rps=self.rate_limit_rps,
            execution_window_steps=self.execution_window_steps,
            rejection_prob=self.rejection_prob,
            seed=int(self._np_random.integers(0, 2**31)),
        )
        self._bar_idx = start_idx
        self._remaining_qty = self.target_qty
        self._elapsed_steps = 0
        self._api_requests_used = 0
        self._total_cost_bps = 0.0

        obs = self._get_obs()
        info = {"ticker": ticker, "start_idx": start_idx}
        return obs, info

    def step(self, action: int):
        """
        Execute one order submission.

        Parameters
        ----------
        action : int
            Index into SPLIT_FRACTIONS (0–5).

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        assert self._sim is not None, "Call reset() before step()."

        fraction = RetailExecutionSimulator.SPLIT_FRACTIONS[action]
        order_qty = max(1, round(self._remaining_qty * fraction))
        order_qty = min(order_qty, self._remaining_qty)

        api_budget = self.rate_limit_rps * self.execution_window_steps
        api_used_before = self._api_requests_used

        # Simulate this single order submission via the simulator
        # We call execute() with a 1-step window and 1-action sequence
        single_step_sim = RetailExecutionSimulator(
            self._sim.data,
            rate_limit_rps=self.rate_limit_rps,
            execution_window_steps=1,
            rejection_prob=self.rejection_prob,
            seed=int(self._np_random.integers(0, 2**31)),
        )
        result = single_step_sim.execute(
            target_qty=order_qty,
            start_idx=self._bar_idx,
            action_sequence=[5],  # submit all of the slice
        )

        # Update episode state
        self._api_requests_used += result.api_requests_used
        self._bar_idx += 1
        self._elapsed_steps += 1

        if result.executed_qty > 0:
            self._remaining_qty -= result.executed_qty

        # --- Reward ---------------------------------------------------------
        step_slippage = result.slippage_bps
        step_txn_cost = result.transaction_cost_bps
        # Penalty for exceeding API budget
        api_overage = max(0, self._api_requests_used - api_budget)
        api_penalty = 10.0 * api_overage if api_overage > 0 else 0.0

        reward = -(step_slippage + step_txn_cost + api_penalty)
        self._total_cost_bps += step_slippage + step_txn_cost

        # --- Episode termination --------------------------------------------
        window_done = self._elapsed_steps >= self.execution_window_steps
        order_complete = self._remaining_qty <= 0
        terminated = order_complete
        truncated = window_done and not order_complete

        if truncated:
            # Penalize for unexecuted quantity (missed execution is costly)
            unexecuted_frac = self._remaining_qty / max(self.target_qty, 1)
            reward -= 50.0 * unexecuted_frac

        obs = self._get_obs()
        info = {
            "remaining_qty": self._remaining_qty,
            "api_requests_used": self._api_requests_used,
            "total_cost_bps": self._total_cost_bps,
            "executed_qty": result.executed_qty,
            "rejected": result.rejected_orders > 0,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self, mode: str = "human"):
        """Print current episode state."""
        frac_done = (
            1.0 - self._remaining_qty / max(self.target_qty, 1)
        ) * 100
        print(
            f"[Env] Step {self._elapsed_steps}/{self.execution_window_steps} | "
            f"Executed {frac_done:.0f}% | "
            f"Remaining: {self._remaining_qty} shares | "
            f"API used: {self._api_requests_used}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Return the current 8-feature observation vector."""
        if self._sim is None:
            return np.zeros(8, dtype=np.float32)

        obs = self._sim.get_state_features(
            bar_idx=self._bar_idx,
            remaining_qty=self._remaining_qty,
            target_qty=self.target_qty,
            elapsed_steps=self._elapsed_steps,
            api_requests_used=self._api_requests_used,
        )
        # Clip to observation space bounds
        return np.clip(obs, -1.0, 1.0).astype(np.float32)
