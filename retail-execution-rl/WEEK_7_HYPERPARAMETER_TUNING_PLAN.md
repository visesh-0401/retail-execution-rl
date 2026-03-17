# WEEK 7: HYPERPARAMETER TUNING PLAN

**Objective:** Optimize PPO hyperparameters to improve mean reward on AAPL & MSFT  
**Duration:** 6 hours GPU (Kaggle Session 2)  
**Baseline:** Week 6 results (AAPL: -81.96 ± 35.4, MSFT: similar range)

---

## 📊 Hyperparameter Candidates

### High-Impact Parameters (Test First)

| Parameter | Baseline | Test Range | Purpose |
|-----------|----------|-----------|---------|
| **Learning Rate** | 3e-4 | 1e-4, 5e-4, 1e-3 | Policy gradient magnitude |
| **Batch Size** | 64 | 32, 128, 256 | Gradient stability |
| **Entropy Coefficient** | Auto | 0.01, 0.001 | Exploration vs exploitation |

### Medium-Impact Parameters (If Time)

| Parameter | Baseline | Test Range | Purpose |
|-----------|----------|-----------|---------|
| **Gamma (Discount)** | 0.99 | 0.95, 0.99, 0.999 | Long-term credit assignment |
| **GAE Lambda** | 0.95 | 0.9, 0.95, 0.99 | Variance reduction |
| **Clip Range** | 0.2 | 0.1, 0.2, 0.3 | Policy update constraint |

### Low-Priority (Week 8+)

| Parameter | Baseline | Test Range | Purpose |
|-----------|----------|-----------|---------|
| Window size | 30 | 15, 30, 60 | Memory depth |
| Rate limit | 5 RPS | 2, 5, 10 | Order frequency |
| Quantity | 100 shares | 50, 100, 200 | Order size |

---

## 🎯 Ablation Study Design

### Phase 1: Learning Rate Sweep (60 min)

**Hypothesis:** Lower LR more stable, higher LR faster convergence

```
LR Tests:
├── 1e-4   (conservative)
├── 3e-4   (baseline)
├── 5e-4   (aggressive)
└── 1e-3   (risky)

Stock: AAPL (1 seed each = 4 seeds)
Timesteps: 25,000 (quick check)
Expected time: 15 min per model
```

### Phase 2: Batch Size Sweep (60 min)

**Hypothesis:** Larger batch = more stable, but slower per step

```
Batch Size Tests:
├── 32   (small, high variance)
├── 64   (baseline)
├── 128  (large, smooth)
└── 256  (very large, risk of underfitting)

Stock: AAPL (1 seed each = 4 seeds)
Timesteps: 25,000
Expected time: 15 min per model
```

### Phase 3: Entropy Sweep (30 min)

**Hypothesis:** Higher entropy = more exploration early

```
Entropy Tests:
├── 0.001  (low exploration)
├── 0.01   (baseline)
└── 0.1    (high exploration)

Stock: AAPL (1 seed each = 3 seeds)
Timesteps: 25,000
Expected time: 15 min per model
```

### Phase 4: Best Config Validation (120 min)

**Use best parameters from Phases 1-3**

```
Final Test:
├── 1 stock: AAPL or MSFT (whichever showed best)
├── 5 seeds: 42, 43, 44, 45, 46
├── Timesteps: 50,000 (full)
├── Expected time: ~20 min per model = 100 min total
└── Goal: Compare with Week 6 baseline
```

---

## 📈 Success Criteria

| Metric | Target | Success? |
|--------|--------|----------|
| Best new reward | > -60 | Improvement of ~20 points |
| Stability | Std < 35 | Better variance control |
| Fastest converging | < 15k steps | Smoother learning |

---

## 🔧 Configuration Template

```python
# Baseline (Week 6)
{
    'learning_rate': 3e-4,
    'batch_size': 64,
    'entropy_coef': 0.01,  # Auto in SB3
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
}

# Example: High LR Test
{
    'learning_rate': 1e-3,   # ← Changed
    'batch_size': 64,
    'entropy_coef': 0.01,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
}
```

---

## 📝 Tracking Sheet

### Learning Rate Results
```
| LR      | Best_Reward | Mean_Reward | Std_Dev | Time | Notes |
|---------|------------|-------------|---------|------|-------|
| 1e-4    |            |             |         |      |       |
| 3e-4    | (baseline) | -81.96      | 35.4    | 20m  | Ref   |
| 5e-4    |            |             |         |      |       |
| 1e-3    |            |             |         |      |       |
```

### Batch Size Results
```
| BS   | Best_Reward | Mean_Reward | Std_Dev | Time | Notes |
|------|------------|-------------|---------|------|-------|
| 32   |            |             |         |      |       |
| 64   | (baseline) | -81.96      | 35.4    | 20m  | Ref   |
| 128  |            |             |         |      |       |
| 256  |            |             |         |      |       |
```

### Entropy Results
```
| Entropy | Best_Reward | Mean_Reward | Std_Dev | Time | Notes |
|---------|------------|-------------|---------|------|-------|
| 0.001   |            |             |         |      |       |
| 0.01    | (baseline) | -81.96      | 35.4    | 20m  | Ref   |
| 0.1     |            |             |         |      |       |
```

---

## 🚀 Kaggle Session 2 Setup

**Same as Session 1, but with parameter sweep:**

```bash
# Modified training command (parametrized)
python scripts/04_train_ppo.py \
  --use-gpu \
  --num-gpus 2 \
  --timesteps 25000 \        # ← Reduced for quick tests
  --seeds 42 \               # ← Single seed for speed
  --stocks AAPL \            # ← Focus on AAPL
  --learning-rate 5e-4 \     # ← Sweep this
  --batch-size 128 \         # ← Sweep this
  --entropy-coef 0.01 \      # ← Sweep this
  --eval-stocks TSLA SPY QQQ
```

---

## 📊 Analysis After Session

1. **Quick Win:** Which parameter had biggest impact?
2. **Best Combo:** Top 3 parameter combinations
3. **Statistical Significance:** t-test vs baseline?
4. **Convergence Speed:** Which converges fastest?

---

## ⏱️ Timeline (6h Session)

```
9:00 AM - Setup + Clone repo (5 min)
9:05 AM - Install dependencies (5 min)
9:10 AM - Learning rate sweep (60 min) ← Phase 1
10:10 AM - Batch size sweep (60 min) ← Phase 2
11:10 AM - Entropy coefficient sweep (30 min) ← Phase 3
11:40 AM - Final validation (100 min) ← Phase 4
1:20 PM - Archive results (10 min)
1:30 PM - Download + Buffer time (4.5h remaining)
6:00 PM - Session ends
```

---

## 📚 References

- Baseline: [WEEK_6_RESULTS_SUMMARY.md](WEEK_6_RESULTS_SUMMARY.md)
- Implementation: [scripts/04_train_ppo.py](scripts/04_train_ppo.py)
- GPU Guide: [MULTI_GPU_SETUP.md](docs/MULTI_GPU_SETUP.md)

---

**Status:** Ready for Kaggle Session 2 ✅
