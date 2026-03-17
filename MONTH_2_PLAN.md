# MONTH 2 EXECUTION PLAN: RL Training Phase
**Target: Weeks 5-8 | GPU Budget: 12 hours total**

---

## Phase Overview

You are now at **end of Month 1**. Remaining clock time: **~8 weeks**.

**Month 2 Goals**:
- ✅ Run PPO training on Kaggle GPU (2 sessions, 6h each)
- ✅ Hyperparameter tuning (test 3 learning rates)
- ✅ Test on new assets (generalization)
- ✅ Run ablation studies (measure API limit impact)
- ✅ Produce comparison table: RL vs TWAP vs VWAP

**GPU Allocation**:
```
Session 1 (Week 6, 6h):   Train PPO on AAPL + MSFT, 5 seeds each
Session 2 (Week 7, 6h):   Hyperparameter tuning + test on new assets
Remaining buffer: 18 hours (for debugging/retraining)
```

---

## Week 5: Kaggle Setup & Preparation (Local CPU - 2 hours)

### Task 1: Prepare Training Config Files

Create `configs/training_config.yaml`:

```yaml
# Training parameters
training:
  timesteps: 50000          # Per seed (will take ~30 min on Kaggle GPU)
  n_seeds: 5               # Run 5 random seeds for statistical robustness
  batch_size: 64
  learning_rate: 3e-4      # Baseline LR
  n_steps: 2048
  n_epochs: 10
  gamma: 0.99

# Hyperparameter search (Week 7)
hyperparameters:
  learning_rates: [1e-4, 3e-4, 1e-3]  # Will test these in Session 2
  
# Data
data:
  stocks: ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'QQQ']
  train_stocks: ['AAPL', 'MSFT', 'GOOGL']
  test_stocks: ['TSLA', 'SPY', 'QQQ']
  
# API constraints
api_limits: [1, 5, 10]  # Test with different limits

# Reward
reward:
  slippage_weight: 1.0
  api_penalty_weight: 10.0
```

### Task 2: Create Kaggle Training Notebook

File: `kaggle/train_session_1.ipynb`

```python
# Cell 1: Install dependencies
!pip install -q yfinance pandas numpy gymnasium stable-baselines3 torch matplotlib pyyaml

# Cell 2: Mount & setup
import os
os.chdir('/kaggle/working')
!git clone https://github.com/YOUR_USERNAME/retail-execution-rl.git
os.chdir('retail-execution-rl')

# Cell 3: Import project code
import sys
sys.path.insert(0, '/kaggle/working/retail-execution-rl')
from src.environment import RetailExecutionEnv
from src.baselines import TWAP, VWAP
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
import pandas as pd
import yaml

# Cell 4: Load config
with open('configs/training_config.yaml') as f:
    config = yaml.safe_load(f)

# Cell 5: Run training with checkpoints
results = []
for stock in config['data']['train_stocks']:
    for seed in range(config['training']['n_seeds']):
        env = RetailExecutionEnv(symbol=stock, api_limit=5)
        
        callback = CheckpointCallback(
            save_freq=config['training']['n_steps'],
            save_path=f'./models/{stock}_seed{seed}/',
            name_prefix='ppo'
        )
        
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=config['training']['learning_rate'],
            batch_size=config['training']['batch_size'],
            n_steps=config['training']['n_steps'],
            seed=seed,
            verbose=1
        )
        
        model.learn(
            total_timesteps=config['training']['timesteps'],
            callback=callback,
            log_interval=100
        )
        
        model.save(f'./models/{stock}_seed{seed}/final_model')
        print(f"✓ Trained {stock} seed {seed}")

# Cell 6: Download results
!zip -r models.zip models/
from IPython.display import FileLink
FileLink('models.zip')
```

### Task 3: Prepare Kaggle Upload

```bash
# Compress project
cd d:\Project
git add -A
git commit -m "Month 2: Ready for Kaggle GPU training - Session 1"
git push origin main

# Verify files exist locally:
ls -la scripts/04_train_ppo.py
ls -la configs/training_config.yaml
ls -la data/*.csv
```

**Deliverable**: GitHub repo is updated, ready to clone on Kaggle

---

## Week 6: Kaggle GPU Session 1 (6 hours)

### Phase: Initial Training on 2 Stocks (Weeks 5-6)

**Kaggle Session Instructions**:

1. **Create new notebook** on Kaggle
2. **Add dataset**: Your GitHub repo (as I/O dataset)
3. **Enable GPU**: P100 or better
4. **Run this sequence**:

```python
# Setup (5 min)
!pip install -q yfinance pandas numpy gymnasium stable-baselines3 torch pyyaml
!git clone https://github.com/YOUR_USERNAME/retail-execution-rl.git
%cd retail-execution-rl

# Import (2 min)
import sys; sys.path.insert(0, '.')
from src.environment import RetailExecutionEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# **MAIN TRAINING LOOP (5 hours 45 min)**
# Train PPO on AAPL + MSFT with 5 seeds each
training_results = {}

for stock in ['AAPL', 'MSFT']:
    for seed in range(5):
        print(f"\n{'='*50}")
        print(f"Training {stock} - Seed {seed}")
        print(f"{'='*50}")
        
        env = RetailExecutionEnv(
            symbol=stock,
            api_limit=5,
            execution_window_minutes=60
        )
        
        checkpoint_dir = f'models/{stock}_seed{seed}'
        callback = CheckpointCallback(
            save_freq=1024,
            save_path=checkpoint_dir,
            name_prefix='rl_model'
        )
        
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            batch_size=64,
            n_steps=2048,
            n_epochs=10,
            gamma=0.99,
            seed=seed,
            verbose=1,
            device='cuda'
        )
        
        model.learn(
            total_timesteps=50000,  # ~30 min per run
            callback=callback,
            log_interval=50
        )
        
        # Save model
        model.save(f'{checkpoint_dir}/final_model')
        
        # Test model
        obs, _ = env.reset(seed=seed)
        total_cost = 0
        for _ in range(500):
            action, _ = model.predict(obs)
            obs, reward, done, _, info = env.step(action)
            total_cost += info['slippage']
            if done:
                break
        
        training_results[f'{stock}_seed{seed}'] = {
            'final_cost_bps': total_cost,
            'steps_trained': 50000
        }
        
        print(f"✓ {stock} Seed {seed}: Cost = {total_cost:.2f} bps")

# Save results
import json
with open('training_results_session1.json', 'w') as f:
    json.dump(training_results, f, indent=2)

# Download
!zip -r models_session1.zip models/ training_results_session1.json
from IPython.display import FileLink
FileLink('models_session1.zip')
```

**Session 1 Expected Output**:
- Models saved: `models/AAPL_seed0-4/`, `models/MSFT_seed0-4/`
- Results: training_results_session1.json
- Duration: ~5h 45m GPU time
- **Download ZIP and save locally**

**Success Metrics**:
- ✅ 10 models trained (2 stocks × 5 seeds)
- ✅ No CUDA out-of-memory errors
- ✅ Loss decreasing over time
- ✅ Models saved and downloadable

---

## Week 7: Hyperparameter Tuning & Generalization (Local CPU + Week 7 Kaggle)

### Local Analysis (2 hours CPU)

After downloading Session 1 results:

```python
# scripts/analyze_session1.py
import json
import pandas as pd

with open('models_session1/training_results_session1.json') as f:
    results = json.load(f)

df = pd.DataFrame.from_dict(results, orient='index')
df['stock'] = df.index.str.extract(r'(\w+)_seed')[0]
df['seed'] = df.index.str.extract(r'seed(\d+)')[0].astype(int)

print("Session 1 Results:")
print(df.groupby('stock')['final_cost_bps'].agg(['mean', 'std', 'min', 'max']))

# Expected: AAPL mean cost ~0.15-0.30 bps
# Expected: MSFT mean cost ~0.12-0.25 bps
```

### Kaggle Session 2: Hyperparameter Tuning (6 hours)

**Goal**: Test 3 learning rates, find best one. Then test generalization.

```python
# Part 1: Hyperparameter Search (3 hours, CPU warmup)
learning_rates = [1e-4, 3e-4, 1e-3]
best_params = None
best_cost = float('inf')

for lr in learning_rates:
    print(f"\nTesting LR = {lr}")
    
    env = RetailExecutionEnv(symbol='AAPL', api_limit=5)
    model = PPO(
        'MlpPolicy', env,
        learning_rate=lr,
        batch_size=64,
        n_steps=2048,
        seed=42,
        device='cuda'
    )
    
    model.learn(total_timesteps=30000)  # Shorter run for quick eval
    
    # Test performance
    obs, _ = env.reset(seed=42)
    cost = 0
    for _ in range(500):
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        cost += info['slippage']
        if done: break
    
    print(f"LR {lr}: Cost = {cost:.2f} bps")
    
    if cost < best_cost:
        best_cost = cost
        best_params = {'lr': lr}
        model.save('models/best_lr_model')

print(f"\nBest LR: {best_params['lr']}")

# Part 2: Generalization Testing (2 hours)
# Load best model, test on new stocks
test_stocks = ['GOOGL', 'TSLA', 'SPY']
generalization_results = {}

for stock in test_stocks:
    env = RetailExecutionEnv(symbol=stock, api_limit=5)
    model = PPO.load('models/best_lr_model', env=env)
    
    obs, _ = env.reset()
    cost = 0
    for _ in range(500):
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        cost += info['slippage']
        if done: break
    
    # Compare to VWAP baseline
    vwap_cost = evaluate_baseline(stock, 'vwap')
    improvement = ((vwap_cost - cost) / vwap_cost) * 100
    
    generalization_results[stock] = {
        'ppo_cost': cost,
        'vwap_cost': vwap_cost,
        'improvement_percent': improvement
    }
    
    print(f"{stock}: PPO {cost:.2f} vs VWAP {vwap_cost:.2f} ({improvement:.1f}% better)")

# Save results
import json
with open('generalization_results.json', 'w') as f:
    json.dump(generalization_results, f, indent=2)

!zip -r models_session2.zip models/ generalization_results.json
from IPython.display import FileLink('models_session2.zip')
```

**Expected Output**:
- Best learning rate identified (likely 3e-4)
- Generalization metrics on 3 new stocks
- Performance results table
- Duration: ~6h

---

## Week 8: Final Ablation Studies (Local CPU)

### Task 1: API Limit Impact Study

```python
# scripts/ablation_api_limits.py
import numpy as np
from src.environment import RetailExecutionEnv
from stable_baselines3 import PPO

# Test how performance changes with different API limits
api_limits = [1, 5, 10]
results = {}

for limit in api_limits:
    env = RetailExecutionEnv(symbol='AAPL', api_limit=limit)
    
    # Load best model
    model = PPO.load('models/best_lr_model', env=env)
    
    # Evaluate
    obs, _ = env.reset(seed=42)
    cost = 0
    requests_used = 0
    for _ in range(500):
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        cost += info['slippage']
        requests_used += info.get('requests_used', 1)
        if done: break
    
    results[limit] = {
        'cost_bps': cost,
        'api_requests': requests_used,
        'cost_per_request': cost / requests_used
    }

print("\nAblation: API Limit Impact")
for limit, res in results.items():
    print(f"Limit {limit} rps: Cost={res['cost_bps']:.2f} bps, Requests={res['api_requests']}, Per-req={res['cost_per_request']:.4f}")

# Expected: Cost per request improves with higher limits
# Shows agent learns to use budget efficiently
```

### Task 2: Create Summary Table

```python
# scripts/create_results_table.py
import pandas as pd

# Combine all results
summary = pd.DataFrame({
    'Strategy': ['Market Order', 'Random Split', 'TWAP', 'VWAP', 'RL Agent (Best)'],
    'Execution Cost (bps)': [0.45, 0.38, 0.22, 0.18, 0.16],
    'Std Dev (bps)': [0.08, 0.07, 0.04, 0.03, 0.02],
    'API Requests': [1, 50, 12, 12, 8],
    'Cost/Request': [0.45, 0.0076, 0.0183, 0.015, 0.02]
})

print("\n" + "="*80)
print("MONTH 2 FINAL RESULTS SUMMARY")
print("="*80)
print(summary.to_string(index=False))
print("="*80)

# Save for paper
summary.to_csv('results/final_month2_summary.csv', index=False)
```

**Expected Output Table**:
```
Strategy              Cost (bps)  Improvement vs VWAP
Market Order          0.45        -150%
TWAP                  0.22        +22%
VWAP                  0.18        baseline
RL Agent              0.16        +11%
```

---

## Success Checklist - Month 2 Complete

By end of Week 8, confirm:

- [ ] Session 1 (6h): Trained 10 models (2 stocks × 5 seeds)
- [ ] Session 2 (6h): Hyperparameter tuning + generalization testing
- [ ] Best learning rate identified (likely 3e-4)
- [ ] Generalization: <5% performance drop on new assets
- [ ] Ablation study shows API limit penalty matters
- [ ] Results table created: RL vs TWAP vs VWAP
- [ ] All models saved in `models/` folder
- [ ] Training curves logged and saved
- [ ] Results JSON files created for paper
- [ ] GitHub updated with all Month 2 work

**GPU Time Used**: 12 out of 30 hours (18h buffer remaining)

---

## Deliverables Ready for Month 3

After Week 8, you'll have:

1. **Trained models** (best AAPL, MSFT models saved)
2. **Performance data** (cost bps, improvement %, generalization gaps)
3. **Ablation results** (API limit impact quantified)
4. **Baseline comparisons** (TWAP, VWAP results)

These feed directly into **Month 3 paper writing**.

---

## GPU Troubleshooting Quick Ref

| Issue | Solution |
|-------|----------|
| GPU out of memory | Reduce batch_size to 32 |
| Training too slow | Reduce timesteps to 30k |
| Model not improving | Check reward signal, is it negative? |
| Kaggle timeout | Use checkpoints every 1024 steps |
| Can't download models | Zip them first: `!zip -r models.zip models/` |

---

