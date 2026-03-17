# WEEK 6: KAGGLE SESSION 1 EXECUTION CHECKLIST
**Duration**: 6 hours  
**GPU Budget**: 6 hours (6 remaining after)  
**Expected Output**: 10 trained models + results JSON

---

## BEFORE SESSION (Friday Evening - 30 min prep)

- [ ] Create Kaggle notebook: https://kaggle.com/code/new
  - Name: `retail-execution-rl-session-1`
  - Enable GPU (default P100 is fine)
  - Save it (navigate away + back)

- [ ] Quick GPU test in Kaggle:
  ```python
  import torch
  print(f"GPU: {torch.cuda.get_device_name(0)}")  # Should print P100 or V100
  ```
  - [ ] Expected: P100, V100, or Tesla GPU
  - [ ] If CPU only: Restart notebook and toggle GPU again

- [ ] Download code locally:
  ```bash
  cd downloads/
  git clone https://github.com/YOUR_USERNAME/retail-execution-rl.git
  # Or just: git pull in existing folder
  ```

- [ ] Verify these files exist LOCALLY:
  - [ ] `scripts/04_train_ppo.py`
  - [ ] `configs/training_config.yaml`
  - [ ] `data/AAPL_1m.csv` (and other CSVs)
  - [ ] `src/environment.py`

---

## SESSION DAY (Monday ~8 AM)

### Pre-Session Checklist (8:00 AM - 8:15 AM)

- [ ] Open Kaggle notebook in browser
- [ ] Print this checklist (you'll need it)
- [ ] Have 2 terminal windows open:
  - Terminal 1: For monitoring notebooks
  - Terminal 2: For any debugging
- [ ] Have these files nearby:
  - GitHub repo open
  - MONTH_2_PLAN.md open
  - Python interpreter running locally (optional)

---

## SESSION TIMELINE (8:15 AM - 2:15 PM)

### CELL 1: Install Dependencies (8:15 - 8:20 AM, ~5 min)

**In Kaggle notebook, copy this**:
```python
# CELL 1: Dependencies
!pip install -q yfinance pandas numpy gymnasium stable-baselines3 torch pyyaml
print("✓ Dependencies installed")
```

**What to watch**:
- [ ] No red errors
- [ ] Ends with "✓ Dependencies installed"
- [ ] Takes ~5 minutes max

**If slow**: GPU is maybe queued, wait

---

### CELL 2: Clone Repo (8:20 - 8:25 AM, ~5 min)

**Copy this**:
```python
# CELL 2: Clone & Setup
import os
os.chdir('/kaggle/working')

# Clone repo
!git clone https://github.com/YOUR_USERNAME/retail-execution-rl.git
os.chdir('retail-execution-rl')

# Verify files
import os
print("Files in repo:")
for f in os.listdir('.')[:10]:
    print(f"  {f}")

print("✓ Repo cloned successfully")
```

**What to watch**:
- [ ] No authentication errors
- [ ] Files listed include: `src`, `scripts`, `configs`, `data`
- [ ] Ends with "✓ Repo cloned successfully"

**If git clone fails**:
- [ ] Download as ZIP from GitHub instead
- [ ] Unzip in Kaggle working directory

---

### CELL 3: Load Config & Test (8:25 - 8:30 AM, ~5 min)

**Copy this**:
```python
# CELL 3: Load config & verify setup
import sys
sys.path.insert(0, '/kaggle/working/retail-execution-rl')

import yaml
import torch
from pathlib import Path

# Load config
with open('configs/training_config.yaml') as f:
    config = yaml.safe_load(f)

print("Configuration:")
print(f"  Timesteps: {config['training']['timesteps']}")
print(f"  N Seeds: {config['training']['n_seeds']}")
print(f"  Learning Rate: {config['training']['learning_rate']}")
print(f"  Training on: {config['data']['train_stocks'][:2]}")

print(f"\nCompute:")
print(f"  GPU Available: {torch.cuda.is_available()}")
print(f"  GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"  CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n✓ Config loaded and GPU ready")
```

**What to watch**:
- [ ] Timesteps = 50000
- [ ] N Seeds = 5
- [ ] Learning Rate = 0.0003
- [ ] GPU Available = True
- [ ] GPU Name shows (P100, V100, etc.)

**If GPU not available**:
- [ ] STOP - don't continue
- [ ] Restart notebook: Menu > Restart session
- [ ] Toggle GPU off/on in notebook settings
- [ ] Re-run Cell 3

---

### CELL 4: MAIN TRAINING (8:30 AM - 2:10 PM, ~5h 40m) 

**⚠️ THIS IS THE BIG CELL - COPY EXACTLY**:

```python
# CELL 4: MAIN TRAINING LOOP (5+ hours)
# This trains 10 models: AAPL × 5 seeds + MSFT × 5 seeds

from src.environment import RetailExecutionEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from datetime import datetime
import json

# Initialize tracker
training_results = {}
start_time = datetime.now()
STOCKS = ['AAPL', 'MSFT']  # Only these 2 for Session 1
SEEDS = 5

print("\n" + "="*70)
print("STARTING PPO TRAINING SESSION 1")
print(f"Start time: {start_time.strftime('%H:%M:%S')}")
print(f"Stocks: {STOCKS}")
print(f"Seeds per stock: {SEEDS}")
print(f"Total models: {len(STOCKS) * SEEDS}")
print("="*70)

model_counter = 0
total_models = len(STOCKS) * SEEDS

# Train each stock × seed combination
for stock in STOCKS:
    print(f"\n{'='*70}")
    print(f"STOCK: {stock}")
    print(f"{'='*70}")
    
    for seed in range(SEEDS):
        model_counter += 1
        print(f"\n[{model_counter}/{total_models}] {stock} - Seed {seed}")
        print("-" * 70)
        
        try:
            # Create environment
            env = RetailExecutionEnv(
                symbol=stock,
                api_limit=5,
                execution_window_minutes=60
            )
            
            # Create checkpoint directory
            checkpoint_dir = f'models/{stock}_seed{seed}'
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            
            # Callback to save checkpoints
            callback = CheckpointCallback(
                save_freq=2048,  # Save every n_steps
                save_path=checkpoint_dir,
                name_prefix='rl_model',
                save_replay_buffer=False,
                save_vecnorm_wrapper=False
            )
            
            # Create PPO model
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
                device='cuda',
                tensorboard_log='./logs/'
            )
            
            # Train
            print(f"Training {config['training']['timesteps']} timesteps...")
            model.learn(
                total_timesteps=config['training']['timesteps'],
                callback=callback,
                log_interval=50
            )
            
            # Save final model
            final_path = f'{checkpoint_dir}/final_model'
            model.save(final_path)
            print(f"✓ Model saved: {final_path}")
            
            # Quick evaluation
            obs, info = env.reset(seed=seed)
            episode_cost = 0
            steps = 0
            
            for _ in range(500):
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_cost += info['slippage']
                steps += 1
                if terminated or truncated:
                    break
            
            # Store result
            training_results[f'{stock}_seed{seed}'] = {
                'stock': stock,
                'seed': seed,
                'final_cost_bps': float(episode_cost),
                'steps_evaluated': steps,
                'training_time_min': (datetime.now() - start_time).total_seconds() / 60
            }
            
            elapsed_total = (datetime.now() - start_time).total_seconds() / 3600
            rate_per_hour = model_counter / elapsed_total if elapsed_total > 0 else 0
            eta_total = total_models / rate_per_hour if rate_per_hour > 0 else 0
            
            print(f"✓ Evaluation cost: {episode_cost:.3f} bps")
            print(f"  Elapsed: {elapsed_total:.2f}h | ETA total: {eta_total:.2f}h")
            
        except Exception as e:
            print(f"❌ ERROR on {stock} seed {seed}: {str(e)}")
            training_results[f'{stock}_seed{seed}'] = {
                'ERROR': str(e)
            }
            continue

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)

# Save results
with open('training_results_session1.json', 'w') as f:
    json.dump(training_results, f, indent=2)

total_time = (datetime.now() - start_time).total_seconds() / 3600
print(f"Total time: {total_time:.2f} hours")
print(f"Models trained: {len(training_results)}")
print(f"✓ Results saved to training_results_session1.json")
```

**What to watch**:
- [ ] First model starts training (loss values printed)
- [ ] Every 50 iterations: loss updates visible
- [ ] After ~30 min: AAPL seed 0 completes
- [ ] After ~1h: AAPL seed 1 completes
- [ ] After ~2h30m: All AAPL seeds done
- [ ] Starts MSFT at ~2h35m
- [ ] After ~5h45m: All done
- [ ] Final message: "✓ Results saved to training_results_session1.json"

**Expected Loss Pattern**:
```
Loss decreasing:
[Seed 0] step 0    loss: 0.85
[Seed 0] step 100  loss: 0.72
[Seed 0] step 500  loss: 0.45
[Seed 0] step 1000 loss: 0.38
...
[Seed 0] step 50000 loss: ~0.15-0.20  ← Final
```

**If training too slow** (>40 min per seed):
- Could be CPU-bound (GPU not being used)
- Check: `torch.cuda.memory_allocated()` in new cell
- Might be Kaggle competition for resources (can wait)

**If CUDA error**:
- Likely out of memory
- Reduce batch_size to 32 in code
- Restart notebook and retry

---

### CELL 5: Results Summary (2:10 - 2:15 PM, ~5 min)

**Copy this**:
```python
# CELL 5: Summary
import pandas as pd

with open('training_results_session1.json') as f:
    results = json.load(f)

print("\n" + "="*70)
print("SESSION 1 RESULTS")
print("="*70)

# Create DataFrame
df = pd.DataFrame.from_dict(results, orient='index')
df = df[df['cost_bps'].notna()]  # Remove errors

# Summary by stock
print("\nSummary by Stock:")
summary = df.groupby('stock')['final_cost_bps'].agg([
    'mean', 'std', 'min', 'max'
]).round(4)
print(summary)

print("\nIndividual Results:")
for key in sorted(results.keys()):
    val = results[key]
    if 'final_cost_bps' in val:
        print(f"  {key:20s}: {val['final_cost_bps']:7.4f} bps")

print("\n✓ Training session complete!")
print(f"  Total models: {len(df)}")
print(f"  Average cost: {df['final_cost_bps'].mean():.4f} bps")
```

**What to watch**:
- [ ] 10 models listed (AAPL 5 + MSFT 5)
- [ ] Average cost: ~0.15-0.25 bps (reasonable)
- [ ] No "ERROR" entries

---

### CELL 6: Download Results (2:15 - 2:20 PM, ~5 min)

**Copy this**:
```python
# CELL 6: Package for download
import os

# List what we're zipping
print("Creating backup ZIP with:")
print("  - 10 trained models (final weights)")
print("  - Checkpoint files (if needed to resume)")
print("  - Results JSON")
print("  - Training logs")

# Create zip
!zip -r -q models_session1.zip models/ training_results_session1.json logs/ && echo "✓ ZIP created"

# Show file size
import os
size_mb = os.path.getsize('models_session1.zip') / (1024**2)
print(f"\nFile size: {size_mb:.1f} MB")
print("Ready to download!")

# Also show direct links
print("\nTo download:")
print("  1. Click 'output' tab on right")
print("  2. Click 'models_session1.zip'")
print("  3. Click download icon")
```

**What to watch**:
- [ ] ZIP created successfully
- [ ] File size ~200-300 MB (normal)
- [ ] Can see "models_session1.zip" in output tab

---

## AFTER SESSION (2:20 PM - 3:00 PM)

### Step 1: Download ZIP (5 min)
- [ ] Click "Output" tab in Kaggle
- [ ] Find `models_session1.zip`
- [ ] Click download icon
- [ ] Wait for download (might be slow)
- [ ] Verify file size: ~200-300 MB

### Step 2: Extract Locally (5 min)
```bash
cd d:\Project
# Navigate to downloads folder
unzip ~/Downloads/models_session1.zip -d Month2_Session1_Results/

# Verify
ls -la Month2_Session1_Results/models/
# Should show: AAPL_seed0, AAPL_seed1, ..., MSFT_seed4
```

### Step 3: Copy to Project (5 min)
```bash
cd d:\Project

# Move results into project
cp Month2_Session1_Results/models/* models/
cp Month2_Session1_Results/training_results_session1.json results/

# Verify
ls -la models/AAPL_seed0/final_model.zip
ls results/training_results_session1.json
```

### Step 4: Push to GitHub (5 min)
```bash
cd d:\Project

git add models/ results/training_results_session1.json
git commit -m "Month 2 Week 6: Session 1 complete - 10 models trained"
git push origin main

# Verify on GitHub:
# https://github.com/YOUR_USERNAME/retail-execution-rl
# Should see: models/ folder with AAPL/MSFT subfolders
```

### Step 5: Quick Analysis (10 min)
```bash
# View results
python -c "import json; r = json.load(open('results/training_results_session1.json')); print(json.dumps(r, indent=2))" | head -50

# Expected output:
# {
#   "AAPL_seed0": {"final_cost_bps": 0.162, ...},
#   "AAPL_seed1": {"final_cost_bps": 0.158, ...},
#   ...
# }
```

---

## SUCCESS CRITERIA

✅ **Session 1 is successful if**:

- [x] All 10 models trained without error
- [x] No CUDA out-of-memory crashes
- [x] Results show reasonable costs (0.12-0.25 bps)
- [x] Models saved in: models/AAPL_seed0-4/, models/MSFT_seed0-4/
- [x] training_results_session1.json created
- [x] ZIP downloaded and extracted locally
- [x] Results pushed to GitHub

---

## IF SOMETHING GOES WRONG

| Issue | Solution |
|-------|----------|
| Training stops mid-way | Models are already saved via checkpoints. Download what you have. |
| CUDA out-of-memory | Reduce batch_size to 32 in CELL 4 and restart notebook. |
| ZIP won't download | Kill notebook, wait 5 min, try again. |
| Can't extract ZIP | Try: `unzip -l models_session1.zip` to verify it's not corrupted. |
| GitHub push fails | `git status` to see what's blocking, then `git pull`, then push. |

---

## TIMELINE BUFFER

**If session takes ~7h instead of 6h**: 
- No problem! Still have 23h GPU budget remaining
- Session 2 scheduled still has plenty of time

**If training very slow**:
- Kaggle GPU might be under high load
- Not your fault, just wait (it will finish)

---

## AFTER SESSION 1: WHAT'S NEXT

**By end of Wednesday**: 
- [ ] Session 1 results analyzed locally
- [ ] Models verified to work
- [ ] Results table created

**For Session 2 (Week 7)**:
- [ ] Will test hyperparameters on same models
- [ ] Will test generalization on new stocks
- [ ] Another 6-hour Kaggle run

---

## 📞 SUPPORT

If stuck during session:
1. Check MONTH_2_PLAN.md (section on troubleshooting)
2. Check error message carefully (often gives hint)
3. Check GPU memory: `torch.cuda.memory_allocated()`
4. Can always pause and resume (models saved via checkpoints)

**You got this! 🚀**

---

