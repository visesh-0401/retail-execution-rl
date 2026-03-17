# Kaggle Session 2: Hyperparameter Tuning (Week 7)

**Date:** TBD (Schedule 6h GPU session)  
**Objective:** Optimize PPO hyperparameters using ablation studies  
**Baseline:** Week 6 AAPL mean reward: -81.96 ± 35.4  
**Expected Output:** 15-30 trained models with tuned hyperparameters  

---

## 📋 Kaggle Notebook Setup (Same as Session 1)

### Cell 1: Clone repo & install dependencies

```python
# Clone repository
import os
os.chdir('/kaggle/working')
os.system('git clone https://github.com/visesh-0401/retail-execution-rl.git')
os.chdir('retail-execution-rl')

# Install dependencies
os.system('pip install -q stable-baselines3 gymnasium pandas yfinance torch tensorboard')
```

### Cell 2-4: Create scripts & download data

```python
# Create scripts folder
os.makedirs('scripts', exist_ok=True)
os.makedirs('configs', exist_ok=True)

# Download data (AAPL + MSFT)
os.system('python scripts/01_download_data.py --stocks AAPL MSFT')
```

All Python files from repo are auto-cloned (04_train_ppo.py, environment.py, data_loader_gpu.py, etc.)

---

## 🧪 Phase 1: Learning Rate Sweep (60 min)

**Goal:** Quick test (25k timesteps) of 4 learning rates  

### Cell 5A: Learning Rate 1e-4

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['PYTHONPATH'] = '.'

# Test LR = 1e-4 (conservative)
os.system('''
python scripts/04_train_ppo.py \
  --use-gpu \
  --num-gpus 2 \
  --timesteps 25000 \
  --seeds 42 \
  --stocks AAPL \
  --lr 1e-4 \
  --batch-size 64 \
  --entropy-coef 0.01 \
  --eval-stocks TSLA SPY QQQ
''')
```

**Expected Time:** 6-8 min  
**Expected Reward:** Likely slower convergence, more stable

### Cell 5B: Learning Rate 3e-4 (Baseline)

```python
# Test LR = 3e-4 (baseline — should match Week 6)
os.system('''
python scripts/04_train_ppo.py \
  --use-gpu \
  --num-gpus 2 \
  --timesteps 25000 \
  --seeds 42 \
  --stocks AAPL \
  --lr 3e-4 \
  --batch-size 64 \
  --entropy-coef 0.01 \
  --eval-stocks TSLA SPY QQQ
''')
```

**Expected Time:** 6-8 min  
**Expected Reward:** ~-80 to -85 (partial training)

### Cell 5C: Learning Rate 5e-4

```python
# Test LR = 5e-4 (moderately aggressive)
os.system('''
python scripts/04_train_ppo.py \
  --use-gpu \
  --num-gpus 2 \
  --timesteps 25000 \
  --seeds 42 \
  --stocks AAPL \
  --lr 5e-4 \
  --batch-size 64 \
  --entropy-coef 0.01 \
  --eval-stocks TSLA SPY QQQ
''')
```

**Expected Time:** 6-8 min  
**Expected Reward:** Likely faster convergence, slightly noisier

### Cell 5D: Learning Rate 1e-3

```python
# Test LR = 1e-3 (very aggressive)
os.system('''
python scripts/04_train_ppo.py \
  --use-gpu \
  --num-gpus 2 \
  --timesteps 25000 \
  --seeds 42 \
  --stocks AAPL \
  --lr 1e-3 \
  --batch-size 64 \
  --entropy-coef 0.01 \
  --eval-stocks TSLA SPY QQQ
''')
```

**Expected Time:** 6-8 min  
**Expected Reward:** Risk of instability, noisy updates

---

## 🧪 Phase 2: Batch Size Sweep (60 min)

**Goal:** Quick test (25k timesteps) of 4 batch sizes  

### Cell 6A: Batch Size 32

```python
# Test BS = 32 (small, high variance)
os.system('''
python scripts/04_train_ppo.py \
  --use-gpu \
  --num-gpus 2 \
  --timesteps 25000 \
  --seeds 42 \
  --stocks AAPL \
  --lr 3e-4 \
  --batch-size 32 \
  --entropy-coef 0.01 \
  --eval-stocks TSLA SPY QQQ
''')
```

**Expected Time:** 6-8 min  
**Expected Reward:** Noisier but potentially faster to converge on some tasks

### Cell 6B: Batch Size 64 (Baseline)

```python
# Test BS = 64 (baseline)
os.system('''
python scripts/04_train_ppo.py \
  --use-gpu \
  --num-gpus 2 \
  --timesteps 25000 \
  --seeds 42 \
  --stocks AAPL \
  --lr 3e-4 \
  --batch-size 64 \
  --entropy-coef 0.01 \
  --eval-stocks TSLA SPY QQQ
''')
```

**Expected Time:** 6-8 min  
**Expected Reward:** Control — should match Phase 1 Cell 5B

### Cell 6C: Batch Size 128

```python
# Test BS = 128 (larger, smoother gradient)
os.system('''
python scripts/04_train_ppo.py \
  --use-gpu \
  --num-gpus 2 \
  --timesteps 25000 \
  --seeds 42 \
  --stocks AAPL \
  --lr 3e-4 \
  --batch-size 128 \
  --entropy-coef 0.01 \
  --eval-stocks TSLA SPY QQQ
''')
```

**Expected Time:** 6-8 min  
**Expected Reward:** Smoother, potentially better stability on noisy objectives

### Cell 6D: Batch Size 256

```python
# Test BS = 256 (very large)
os.system('''
python scripts/04_train_ppo.py \
  --use-gpu \
  --num-gpus 2 \
  --timesteps 25000 \
  --seeds 42 \
  --stocks AAPL \
  --lr 3e-4 \
  --batch-size 256 \
  --entropy-coef 0.01 \
  --eval-stocks TSLA SPY QQQ
''')
```

**Expected Time:** 6-8 min  
**Expected Reward:** Risk of underfitting (too few policy updates), overly smooth

---

## 🧪 Phase 3: Entropy Coefficient Sweep (40 min)

**Goal:** Quick test (25k timesteps) of 3 entropy coefficients  

### Cell 7A: Entropy = 0.001

```python
# Test entropy = 0.001 (low exploration)
os.system('''
python scripts/04_train_ppo.py \
  --use-gpu \
  --num-gpus 2 \
  --timesteps 25000 \
  --seeds 42 \
  --stocks AAPL \
  --lr 3e-4 \
  --batch-size 64 \
  --entropy-coef 0.001 \
  --eval-stocks TSLA SPY QQQ
''')
```

**Expected Time:** 6-8 min  
**Expected Reward:** More exploitation, risk of suboptimal local minima

### Cell 7B: Entropy = 0.01 (Baseline)

```python
# Test entropy = 0.01 (baseline)
os.system('''
python scripts/04_train_ppo.py \
  --use-gpu \
  --num-gpus 2 \
  --timesteps 25000 \
  --seeds 42 \
  --stocks AAPL \
  --lr 3e-4 \
  --batch-size 64 \
  --entropy-coef 0.01 \
  --eval-stocks TSLA SPY QQQ
''')
```

**Expected Time:** 6-8 min  
**Expected Reward:** Control — Week 6 baseline

### Cell 7C: Entropy = 0.1

```python
# Test entropy = 0.1 (high exploration)
os.system('''
python scripts/04_train_ppo.py \
  --use-gpu \
  --num-gpus 2 \
  --timesteps 25000 \
  --seeds 42 \
  --stocks AAPL \
  --lr 3e-4 \
  --batch-size 64 \
  --entropy-coef 0.1 \
  --eval-stocks TSLA SPY QQQ
''')
```

**Expected Time:** 6-8 min  
**Expected Reward:** More exploration, potentially slower convergence

---

## ✅ Phase 4: Final Validation (100 min)

**Goal:** Train best parameters from Phases 1-3 for full 50k timesteps with 5 seeds

**Before starting Phase 4:**
1. Check results from Phase 1 → pick best learning rate
2. Check results from Phase 2 → pick best batch size
3. Check results from Phase 3 → pick best entropy

### Cell 8: Best Config - 5 Seeds

```python
# Example: Assuming best config is LR=5e-4, BS=128, Entropy=0.01
# Adjust based on Phase 1-3 results!

os.system('''
python scripts/04_train_ppo.py \
  --use-gpu \
  --num-gpus 2 \
  --timesteps 50000 \
  --seeds 42 43 44 45 46 \
  --stocks AAPL \
  --lr 5e-4 \
  --batch-size 128 \
  --entropy-coef 0.01 \
  --eval-stocks TSLA SPY QQQ
''')
```

**Expected Time:** 20 min × 5 seeds = 100 min  
**Expected Reward:** Compare against baseline (-81.96 ± 35.4)

---

## 📈 Phase Results Analysis

### Cell 9: Load & Compare Results

```python
import pandas as pd
import numpy as np

# Load results
results = pd.read_csv('results/training_summary.csv')

# Group by configuration
print("\n=== PHASE 1: Learning Rate Comparison ===")
print(results[['lr', 'mean_reward', 'std_reward']].groupby('lr').mean())

print("\n=== PHASE 2: Batch Size Comparison ===")
print(results[['batch_size', 'mean_reward', 'std_reward']].groupby('batch_size').mean())

print("\n=== PHASE 3: Entropy Comparison ===")
print(results[['entropy_coef', 'mean_reward', 'std_reward']].groupby('entropy_coef').mean())

print("\n=== PHASE 4: Best Config (5 seeds) ===")
best_config = results[results['timesteps'] == 50000]
if len(best_config) > 0:
    print(f"Mean reward: {best_config['mean_reward'].mean():.2f} ± {best_config['mean_reward'].std():.2f}")
```

---

## 📊 Download Results

### Cell 10: Archive to Output

```python
import shutil

# Copy results to Kaggle output folder for download
shutil.copy('results/training_summary.csv', '/kaggle/output/week_7_results.csv')

# Optionally copy best models
best_model_dir = 'models'
if os.path.exists(best_model_dir):
    shutil.copytree(best_model_dir, '/kaggle/output/models', dirs_exist_ok=True)

print("✅ Results copied to /kaggle/output/")
```

---

## 🎯 Expected Timeline (6h Session)

```
[Min] Task                          Duration    Cumulative
---   Setup + Clone + Pip             10 min      10 min
10    Download Data (AAPL + MSFT)     5 min       15 min
15    ─── PHASE 1: Learning Rate ───
15    LR=1e-4 (Seed 42)               8 min       23 min
23    LR=3e-4 (Seed 42) [baseline]    8 min       31 min
31    LR=5e-4 (Seed 42)               8 min       39 min
39    LR=1e-3 (Seed 42)               8 min       47 min
47    ─── PHASE 2: Batch Size ───
47    BS=32 (Seed 42)                 8 min       55 min
55    BS=64 (Seed 42) [baseline]      8 min       63 min
63    BS=128 (Seed 42)                8 min       71 min
71    BS=256 (Seed 42)                8 min       79 min
79    ─── PHASE 3: Entropy ───
79    Entropy=0.001                   8 min       87 min
87    Entropy=0.01 [baseline]         8 min       95 min
95    Entropy=0.1                     8 min       103 min
103   ─── PHASE 4: Best Config ───
103   Best (5 seeds @ 50k) [Seed 42]  20 min      123 min
123   Best (5 seeds @ 50k) [Seed 43]  20 min      143 min
143   Best (5 seeds @ 50k) [Seed 44]  20 min      163 min
163   Best (5 seeds @ 50k) [Seed 45]  20 min      183 min
183   Best (5 seeds @ 50k) [Seed 46]  20 min      203 min
203   Analysis + Archive              5 min       208 min
208   ← 3h 28 min total (1h 32 min buffer left)
```

---

## 📋 Checklist

- [ ] Kaggle notebook created
- [ ] Cell 1: Repo cloned
- [ ] Cell 2-4: Dependencies installed, data downloaded
- [ ] Cell 5A-D: Phase 1 (LR sweep) completed
- [ ] Cell 6A-D: Phase 2 (BS sweep) completed
- [ ] Cell 7A-C: Phase 3 (Entropy sweep) completed
- [ ] **Decide Phase 4 best config before running**
- [ ] Cell 8: Phase 4 (5 seeds @ 50k) completed
- [ ] Cell 9: Results analyzed and compared
- [ ] Cell 10: Results archived to output folder
- [ ] Download results CSV to local machine
- [ ] Create WEEK_7_SESSION_2_RESULTS.md summary
- [ ] Commit to GitHub

---

## 🔗 References

- **Baseline:** [WEEK_6_RESULTS_SUMMARY.md](WEEK_6_RESULTS_SUMMARY.md)
- **Tuning Plan:** [WEEK_7_HYPERPARAMETER_TUNING_PLAN.md](WEEK_7_HYPERPARAMETER_TUNING_PLAN.md)
- **Training Script:** [scripts/04_train_ppo.py](scripts/04_train_ppo.py)
- **GPU Setup:** [MULTI_GPU_SETUP.md](docs/MULTI_GPU_SETUP.md)

---

**Status:** Ready for Kaggle Session 2 ✅
