# Retail Execution RL

**Optimizing Order Splitting Under API Rate Limits: A Reinforcement Learning Approach for Retail Traders**

> **Status**: Active Development — Month 1, Week 1  
> **Target**: ArXiv submission, Week 12

---

## 🧠 What This Is

A research project training a **PPO reinforcement learning agent** to execute stock orders optimally within **retail broker API rate limits** — a constraint ignored by all existing institutional execution research.

**The novel question**: Can RL learn order-splitting policies that beat TWAP/VWAP when you're limited to 1–10 API requests/second?

---

## ⚡ Quick Start

```bash
# 1. Create environment
conda create -n rl_trading python=3.10
conda activate rl_trading
pip install -r requirements.txt

# 2. Download data (one-time, ~10 min)
python scripts/01_download_data.py

# 3. Verify data
python scripts/02_verify_data.py

# 4. Run baselines (TWAP vs VWAP)
python scripts/03_run_baselines.py

# 5. Train PPO agent (requires GPU for full run, CPU works for smoke test)
python scripts/04_train_ppo.py --timesteps 10000 --seed 42
```

---

## 📁 Project Structure

```
retail-execution-rl/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                   # Downloaded OHLCV CSVs (git-ignored)
│   ├── AAPL_1m.csv
│   ├── MSFT_1m.csv
│   └── ...
├── scripts/
│   ├── 01_download_data.py     # Week 1: data collection
│   ├── 02_verify_data.py       # Week 1: sanity checks
│   ├── 03_run_baselines.py     # Week 3-4: TWAP/VWAP comparison
│   └── 04_train_ppo.py         # Week 6-7: PPO training
├── src/
│   ├── simulator.py            # Core execution simulator
│   ├── environment.py          # Gymnasium wrapper
│   └── baselines.py            # TWAP & VWAP implementations
├── models/                 # Saved PPO models (git-ignored)
├── results/                # Backtest results CSVs (git-ignored)
└── paper/                  # LaTeX paper (Week 10-12)
```

---

## 📊 Research Design

| Component | Design Choice | Reason |
|-----------|--------------|--------|
| **Algorithm** | PPO (Proximal Policy Optimization) | Stable, proven in finance, implementable in 3 months |
| **Data** | Yahoo Finance via yfinance | Free, no auth, 20-year history |
| **Baselines** | TWAP, VWAP, Random, Market Order | Standard execution benchmarks |
| **State space** | 8 features (qty, time, spread, volatility, API state...) | Minimal, interpretable |
| **Action space** | 6 discrete splits (10%–100% of remaining qty) | Simple, maps to real order submission |
| **Reward** | -(slippage + transaction_cost + api_penalty) | Directly minimizes execution cost |

---

## 📅 Timeline

| Phase | Weeks | Goal |
|-------|-------|------|
| Foundation | 1–4 | Simulator + baselines (CPU only) |
| RL Training | 5–8 | PPO agent on Kaggle GPU (12 hrs) |
| Paper | 9–12 | ArXiv submission |

---

## 📖 References

- Nevmyvaka et al. (2006) — Reinforced learning for optimal execution
- Kim et al. (2023) — PPO for financial execution
- Hafsi et al. (2024) — ABIDES-based execution simulation
- Almgren & Chriss (2001) — Optimal execution of portfolio transactions
