# Kaggle Session 2: Week 7 Hyperparameter Tuning - READY TO EXECUTE

**Date:** Today (March 17, 2026)  
**Expected Duration:** 6 hours  
**Objective:** Test 11 hyperparameter configurations + validate best (Phase 4 pending)  
**Baseline to Beat:** AAPL -81.96 ± 35.4 (Week 6)

---

## 🚀 START HERE: Copy-Paste Cells into Kaggle Notebook

Go to **https://www.kaggle.com/code** → **New Notebook** → Create notebook  
Then copy each cell below in order.

---

## ✅ CELL 1: Setup & Clone Repository

Copy this entire block:

```python
# Setup: Clone repo, install dependencies
import os
import subprocess
import sys

os.chdir('/kaggle/working')

# Clone repository
print("📦 Cloning repository...")
result = subprocess.run(['git', 'clone', 'https://github.com/visesh-0401/retail-execution-rl.git'], 
                       capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("⚠️ Clone warning:", result.stderr)

os.chdir('retail-execution-rl')

# Install dependencies
print("\n📥 Installing dependencies...")
os.system('pip install -q stable-baselines3 gymnasium pandas yfinance torch tensorboard')

print("\n✅ Setup complete!")
print(f"Current directory: {os.getcwd()}")
```

**Expected Output:**
```
📦 Cloning repository...
Cloning into 'retail-execution-rl'...
📥 Installing dependencies...
✅ Setup complete!
Current directory: /kaggle/working/retail-execution-rl
```

---

## ✅ CELL 2: Download Data (AAPL + MSFT)

```python
# Download data for training & evaluation
import os
os.chdir('/kaggle/working/retail-execution-rl')

print("📥 Downloading market data...")
os.system('python scripts/01_download_data.py --stocks AAPL MSFT')

print("\n✅ Data download complete!")
```

**Expected Output:**
```
📥 Downloading market data...
✅ AAPL data downloaded: data/AAPL_1d.csv
✅ MSFT data downloaded: data/MSFT_1d.csv
✅ Data download complete!
```

---

## ✅ CELL 3: Verify Environment & GPU Setup

```python
import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['PYTHONPATH'] = '.'

print("🔧 Environment Setup Check")
print("=" * 50)

# Check Python
print(f"✓ Python: {os.sys.version.split()[0]}")

# Check PyTorch
print(f"✓ PyTorch version: {torch.__version__}")

# Check CUDA
num_gpus = torch.cuda.device_count()
print(f"✓ GPUs available: {num_gpus}")

if num_gpus > 0:
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")
    print(f"\n✓ CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
else:
    print("⚠️ No GPUs detected!")

# Check data
import glob
data_files = glob.glob('data/*.csv')
print(f"✓ Data files: {len(data_files)}")
for f in sorted(data_files):
    print(f"  - {os.path.basename(f)}")

print("\n✅ Environment verified!")
```

**Expected Output:**
```
🔧 Environment Setup Check
==================================================
✓ Python: 3.12.0
✓ PyTorch version: 2.X.X
✓ GPUs available: 2
  GPU 0: Tesla T4, 16.0 GB
  GPU 1: Tesla T4, 16.0 GB

✓ CUDA_VISIBLE_DEVICES: 0,1
✓ Data files: 2
  - AAPL_1d.csv
  - MSFT_1d.csv

✅ Environment verified!
```

---

## 🧪 PHASE 1: Learning Rate Sweep (60 min)

### ✅ CELL 4A: Test LR = 1e-4

```python
import os
os.chdir('/kaggle/working/retail-execution-rl')
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['PYTHONPATH'] = '.'

print("\n" + "="*70)
print("PHASE 1: Learning Rate Sweep")
print("="*70)
print("\n[1/4] Testing LR = 1e-4 (Conservative)")
print("-"*70)

os.system('''
python scripts/04_train_ppo.py \\
  --use-gpu \\
  --num-gpus 2 \\
  --timesteps 25000 \\
  --seeds 42 \\
  --stocks AAPL \\
  --lr 1e-4 \\
  --batch-size 64 \\
  --entropy-coef 0.01 \\
  --eval-stocks TSLA SPY QQQ
''')

print("\n✅ LR=1e-4 test complete")
```

**Expected Time:** 7-9 min  
**Note:** Check `results/training_summary.csv` after this cell completes

---

### ✅ CELL 4B: Test LR = 3e-4 (Baseline)

```python
print("\n[2/4] Testing LR = 3e-4 (Baseline)")
print("-"*70)

os.system('''
python scripts/04_train_ppo.py \\
  --use-gpu \\
  --num-gpus 2 \\
  --timesteps 25000 \\
  --seeds 42 \\
  --stocks AAPL \\
  --lr 3e-4 \\
  --batch-size 64 \\
  --entropy-coef 0.01 \\
  --eval-stocks TSLA SPY QQQ
''')

print("\n✅ LR=3e-4 test complete (should be similar to Week 6)")
```

**Expected Time:** 7-9 min  
**Expected Reward:** ~-80 to -85 (partial training)

---

### ✅ CELL 4C: Test LR = 5e-4

```python
print("\n[3/4] Testing LR = 5e-4 (Aggressive)")
print("-"*70)

os.system('''
python scripts/04_train_ppo.py \\
  --use-gpu \\
  --num-gpus 2 \\
  --timesteps 25000 \\
  --seeds 42 \\
  --stocks AAPL \\
  --lr 5e-4 \\
  --batch-size 64 \\
  --entropy-coef 0.01 \\
  --eval-stocks TSLA SPY QQQ
''')

print("\n✅ LR=5e-4 test complete")
```

**Expected Time:** 7-9 min

---

### ✅ CELL 4D: Test LR = 1e-3

```python
print("\n[4/4] Testing LR = 1e-3 (Very Aggressive)")
print("-"*70)

os.system('''
python scripts/04_train_ppo.py \\
  --use-gpu \\
  --num-gpus 2 \\
  --timesteps 25000 \\
  --seeds 42 \\
  --stocks AAPL \\
  --lr 1e-3 \\
  --batch-size 64 \\
  --entropy-coef 0.01 \\
  --eval-stocks TSLA SPY QQQ
''')

print("\n✅ LR=1e-3 test complete")
print("\n" + "="*70)
print("PHASE 1 COMPLETE - 4 LR tests done")
print("="*70)
```

**Expected Time:** 7-9 min  
**Total Phase 1:** ~32 min

---

## 🧪 PHASE 2: Batch Size Sweep (60 min)

### ✅ CELL 5A: Test BS = 32

```python
print("\n" + "="*70)
print("PHASE 2: Batch Size Sweep")
print("="*70)
print("\n[1/4] Testing Batch Size = 32 (Small)")
print("-"*70)

os.system('''
python scripts/04_train_ppo.py \\
  --use-gpu \\
  --num-gpus 2 \\
  --timesteps 25000 \\
  --seeds 42 \\
  --stocks AAPL \\
  --lr 3e-4 \\
  --batch-size 32 \\
  --entropy-coef 0.01 \\
  --eval-stocks TSLA SPY QQQ
''')

print("\n✅ BS=32 test complete")
```

**Expected Time:** 7-9 min

---

### ✅ CELL 5B: Test BS = 64 (Baseline)

```python
print("\n[2/4] Testing Batch Size = 64 (Baseline)")
print("-"*70)

os.system('''
python scripts/04_train_ppo.py \\
  --use-gpu \\
  --num-gpus 2 \\
  --timesteps 25000 \\
  --seeds 42 \\
  --stocks AAPL \\
  --lr 3e-4 \\
  --batch-size 64 \\
  --entropy-coef 0.01 \\
  --eval-stocks TSLA SPY QQQ
''')

print("\n✅ BS=64 test complete (baseline control)")
```

**Expected Time:** 7-9 min

---

### ✅ CELL 5C: Test BS = 128

```python
print("\n[3/4] Testing Batch Size = 128 (Large)")
print("-"*70)

os.system('''
python scripts/04_train_ppo.py \\
  --use-gpu \\
  --num-gpus 2 \\
  --timesteps 25000 \\
  --seeds 42 \\
  --stocks AAPL \\
  --lr 3e-4 \\
  --batch-size 128 \\
  --entropy-coef 0.01 \\
  --eval-stocks TSLA SPY QQQ
''')

print("\n✅ BS=128 test complete")
```

**Expected Time:** 7-9 min

---

### ✅ CELL 5D: Test BS = 256

```python
print("\n[4/4] Testing Batch Size = 256 (Very Large)")
print("-"*70)

os.system('''
python scripts/04_train_ppo.py \\
  --use-gpu \\
  --num-gpus 2 \\
  --timesteps 25000 \\
  --seeds 42 \\
  --stocks AAPL \\
  --lr 3e-4 \\
  --batch-size 256 \\
  --entropy-coef 0.01 \\
  --eval-stocks TSLA SPY QQQ
''')

print("\n✅ BS=256 test complete")
print("\n" + "="*70)
print("PHASE 2 COMPLETE - 4 batch size tests done")
print("="*70)
```

**Expected Time:** 7-9 min  
**Total Phase 2:** ~32 min

---

## 🧪 PHASE 3: Entropy Coefficient Sweep (40 min)

### ✅ CELL 6A: Test Entropy = 0.001

```python
print("\n" + "="*70)
print("PHASE 3: Entropy Coefficient Sweep")
print("="*70)
print("\n[1/3] Testing Entropy = 0.001 (Low Exploration)")
print("-"*70)

os.system('''
python scripts/04_train_ppo.py \\
  --use-gpu \\
  --num-gpus 2 \\
  --timesteps 25000 \\
  --seeds 42 \\
  --stocks AAPL \\
  --lr 3e-4 \\
  --batch-size 64 \\
  --entropy-coef 0.001 \\
  --eval-stocks TSLA SPY QQQ
''')

print("\n✅ Entropy=0.001 test complete")
```

**Expected Time:** 7-9 min

---

### ✅ CELL 6B: Test Entropy = 0.01 (Baseline)

```python
print("\n[2/3] Testing Entropy = 0.01 (Baseline)")
print("-"*70)

os.system('''
python scripts/04_train_ppo.py \\
  --use-gpu \\
  --num-gpus 2 \\
  --timesteps 25000 \\
  --seeds 42 \\
  --stocks AAPL \\
  --lr 3e-4 \\
  --batch-size 64 \\
  --entropy-coef 0.01 \\
  --eval-stocks TSLA SPY QQQ
''')

print("\n✅ Entropy=0.01 test complete (baseline control)")
```

**Expected Time:** 7-9 min

---

### ✅ CELL 6C: Test Entropy = 0.1

```python
print("\n[3/3] Testing Entropy = 0.1 (High Exploration)")
print("-"*70)

os.system('''
python scripts/04_train_ppo.py \\
  --use-gpu \\
  --num-gpus 2 \\
  --timesteps 25000 \\
  --seeds 42 \\
  --stocks AAPL \\
  --lr 3e-4 \\
  --batch-size 64 \\
  --entropy-coef 0.1 \\
  --eval-stocks TSLA SPY QQQ
''')

print("\n✅ Entropy=0.1 test complete")
print("\n" + "="*70)
print("PHASE 3 COMPLETE - 3 entropy tests done")
print("="*70)
```

**Expected Time:** 7-9 min  
**Total Phase 3:** ~25 min

---

## 📊 CELL 7: Analyze Phases 1-3 Results

```python
import pandas as pd
import numpy as np
import os

os.chdir('/kaggle/working/retail-execution-rl')

print("\n" + "="*70)
print("ANALYZING PHASES 1-3 RESULTS")
print("="*70)

# Load results
try:
    results = pd.read_csv('results/training_summary.csv')
    print(f"\n✅ Loaded {len(results)} training results")
    
    # Display all Phase 1-3 results
    print("\n📊 All Test Results (25k timesteps, seed 42):")
    print(results[['seed', 'mean_reward', 'std_reward', 'run_id']].tail(11).to_string())
    
    # Analyze by configuration
    print("\n" + "-"*70)
    print("PHASE 1: Learning Rate Impact")
    print("-"*70)
    phase1_results = results[results['timesteps'] == 25000].copy()
    for lr in [1e-4, 3e-4, 5e-4, 1e-3]:
        matching = results[(results['mean_reward'] > -200) & (results['timesteps'] == 25000)]
        if len(matching) > 0:
            avg_reward = matching['mean_reward'].mean()
            print(f"LR {lr:>6}: {avg_reward:>7.2f}")
    
    print("\n" + "-"*70)
    print("SUMMARY: Best Configuration So Far")
    print("-"*70)
    best_idx = results['mean_reward'].idxmax()
    best_result = results.loc[best_idx]
    print(f"Best Reward: {best_result['mean_reward']:.2f}")
    print(f"Config: {best_result['run_id']}")
    
except FileNotFoundError:
    print("⚠️ Results file not found yet. Run training cells first.")

print("\n" + "="*70)
print("READY FOR PHASE 4")
print("="*70)
print("\n📋 INSTRUCTIONS FOR PHASE 4:")
print("1. Review the best configurations from Phases 1-3 above")
print("2. Identify best: Learning Rate, Batch Size, and Entropy Coefficient")
print("3. In Cell 8, update the parameters based on results above")
print("4. Run Cell 8 with best 5-seed configuration")
```

**Expected Output:**
```
✅ Loaded 11 training results

📊 All Test Results (25k timesteps, seed 42):
...results table...

PHASE 1: Learning Rate Impact
LR  1e-04:  XXXX.XX
LR  3e-04:  XXXX.XX
...

SUMMARY: Best Configuration So Far
Best Reward: XXX.XX
Config: ppo_ratelimit5_seedXX

📋 INSTRUCTIONS FOR PHASE 4:
...
```

---

## ✅ PHASE 4: Final Validation - MANUALLY ADJUST THESE VALUES

### ⚠️ CELL 8: Best Config (5 Seeds @ 50k timesteps)

**BEFORE RUNNING THIS CELL:**
1. Look at Cell 7 output above
2. Find the best Learning Rate, Batch Size, and Entropy from Phases 1-3
3. **Replace the values in the cell below** with your best parameters

Example command (modify values based on your results):

```python
import os
os.chdir('/kaggle/working/retail-execution-rl')
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['PYTHONPATH'] = '.'

# ⚠️ EDIT THESE BASED ON CELL 7 RESULTS ⚠️
BEST_LR = 3e-4          # ← Change if needed
BEST_BATCH_SIZE = 64    # ← Change if needed
BEST_ENTROPY = 0.01     # ← Change if needed

print("\n" + "="*70)
print("PHASE 4: Final Validation with Best Configuration")
print("="*70)
print(f"\n📊 Best Configuration Found:")
print(f"  Learning Rate: {BEST_LR}")
print(f"  Batch Size: {BEST_BATCH_SIZE}")
print(f"  Entropy Coef: {BEST_ENTROPY}")
print(f"\n🚀 Training 5 seeds with best config (50k timesteps each)...")
print("="*70)

for seed in [42, 43, 44, 45, 46]:
    print(f"\n[Seed {seed}/46]")
    os.system(f'''
python scripts/04_train_ppo.py \\
  --use-gpu \\
  --num-gpus 2 \\
  --timesteps 50000 \\
  --seeds {seed} \\
  --stocks AAPL \\
  --lr {BEST_LR} \\
  --batch-size {BEST_BATCH_SIZE} \\
  --entropy-coef {BEST_ENTROPY} \\
  --eval-stocks TSLA SPY QQQ
''')

print("\n✅ All 5 seeds complete!")
```

**Expected Time:** 20 min per seed = ~100 min total  
**Total Phase 4:** ~100 min

---

## 📈 CELL 9: Final Results Analysis

```python
import pandas as pd
import numpy as np

print("\n" + "="*70)
print("WEEK 7 SESSION 2 - FINAL RESULTS")
print("="*70)

# Load all results
results = pd.read_csv('results/training_summary.csv')

# Filter Phase 4 results (50k timesteps, seeds 42-46)
phase4 = results[results['timesteps'] == 50000].tail(5)

print("\n📊 PHASE 4 FINAL RESULTS (50k timesteps, 5 seeds):")
print(phase4[['seed', 'mean_reward', 'std_reward', 'run_id']].to_string())

if len(phase4) >= 5:
    mean_reward = phase4['mean_reward'].mean()
    std_reward = phase4['mean_reward'].std()
    
    print("\n" + "="*70)
    print("🏆 PHASE 4 SUMMARY")
    print("="*70)
    print(f"Mean Reward:        {mean_reward:>7.2f}")
    print(f"Std Dev:            {std_reward:>7.2f}")
    print(f"\nWeek 6 Baseline:    -81.96 ± 35.4")
    improvement = mean_reward - (-81.96)
    print(f"Improvement:        {improvement:+.2f} points")
    
    if mean_reward > -81.96:
        print(f"\n✅ IMPROVEMENT! Week 7 is better than Week 6")
    else:
        print(f"\n⚠️ No improvement vs baseline")
    
    print("="*70)
else:
    print("\n⚠️ Phase 4 not complete yet. Run Cell 8 first.")
```

**Expected Output:**
```
WEEK 7 SESSION 2 - FINAL RESULTS
======================================================================

📊 PHASE 4 FINAL RESULTS (50k timesteps, 5 seeds):
        seed  mean_reward  std_reward              run_id
...results...

======================================================================
🏆 PHASE 4 SUMMARY
======================================================================
Mean Reward:           XX.XX
Std Dev:               XX.XX

Week 6 Baseline:    -81.96 ± 35.4
Improvement:        XX.XX points

✅ IMPROVEMENT! Week 7 is better than Week 6
======================================================================
```

---

## ✅ CELL 10: Archive Results to Output Folder

```python
import shutil
import os

os.chdir('/kaggle/working/retail-execution-rl')

print("\n📦 Archiving results...")

# Copy training summary
shutil.copy('results/training_summary.csv', '/kaggle/output/week_7_training_summary.csv')
print("✅ Copied training_summary.csv")

# Create results info file
info = """# Week 7 Hyperparameter Tuning - Session 2 Results

## Test Phases
- Phase 1: Learning Rate sweep (4 tests)
- Phase 2: Batch Size sweep (4 tests)
- Phase 3: Entropy sweep (3 tests)
- Phase 4: Best config validation (5 seeds @ 50k)

## Files
- week_7_training_summary.csv: All 12+ training runs
- models/: Final trained models
- results/tensorboard/: TensorBoard logs

## Next Steps
1. Download week_7_training_summary.csv
2. Load into local machine
3. Create WEEK_7_SESSION_2_RESULTS.md summary
4. Commit to GitHub
"""

with open('/kaggle/output/README.txt', 'w') as f:
    f.write(info)
print("✅ Created README.txt")

# Optional: Copy some models
try:
    best_model_path = 'models'
    if os.path.exists(best_model_path):
        shutil.copytree(best_model_path, '/kaggle/output/models', 
                       dirs_exist_ok=True)
        print("✅ Copied best models")
except Exception as e:
    print(f"⚠️ Skipped model copy: {e}")

print("\n✅ All results archived to /kaggle/output/")
print("\n📥 Download these files:")
print("  1. week_7_training_summary.csv")
print("  2. README.txt")
print("  3. (Optional) models/ folder")
```

---

## 📋 Kaggle Session 2 Execution Checklist

- [ ] **Cell 1:** Setup & clone repo ← **START HERE**
- [ ] **Cell 2:** Download data (AAPL + MSFT)
- [ ] **Cell 3:** Verify environment & GPU
- [ ] **Cell 4A:** LR=1e-4 test
- [ ] **Cell 4B:** LR=3e-4 test (baseline)
- [ ] **Cell 4C:** LR=5e-4 test
- [ ] **Cell 4D:** LR=1e-3 test
  - ⏱️ **After Phase 1:** ~32 min elapsed
- [ ] **Cell 5A:** BS=32 test
- [ ] **Cell 5B:** BS=64 test (baseline)
- [ ] **Cell 5C:** BS=128 test
- [ ] **Cell 5D:** BS=256 test
  - ⏱️ **After Phase 2:** ~64 min elapsed
- [ ] **Cell 6A:** Entropy=0.001 test
- [ ] **Cell 6B:** Entropy=0.01 test (baseline)
- [ ] **Cell 6C:** Entropy=0.1 test
  - ⏱️ **After Phase 3:** ~90 min elapsed
- [ ] **Cell 7:** Analyze 1-3 results  ← **CRITICAL: Decide Phase 4 params here**
- [ ] **Cell 8:** EDIT PARAMETERS, then run Phase 4 (5 seeds @ 50k)
  - ⏱️ **After Phase 4:** ~190 min elapsed
- [ ] **Cell 9:** Final results analysis
- [ ] **Cell 10:** Archive to output folder
- [ ] **Download:** week_7_training_summary.csv from Kaggle output

---

## 🎯 Timeline Reference

```
- 0-5 min:    Cell 1 (Setup)
- 5-10 min:   Cell 2 (Data)
- 10-15 min:  Cell 3 (Verify)
- 15-90 min:  Cells 4A-D (Phase 1: 32 min)
- 90-155 min: Cells 5A-D (Phase 2: 32 min)
- 155-185 min: Cells 6A-C (Phase 3: 25 min)
- 185 min:    Cell 7 (Analyze & decide best config)
- 185-295 min: Cell 8 (Phase 4: 100 min for 5 seeds)
- 295-300 min: Cells 9-10 (Analysis & archive)

BUFFER: 270 min available - ~300 min used = Ready!
```

---

## ✅ Next: Download & Create Summary (On Local Machine)

After Kaggle session completes:

1. **Download** `week_7_training_summary.csv` from Kaggle output folder
2. **Copy** to `d:\Project\retail-execution-rl\results\`
3. **Run** locally to create `WEEK_7_SESSION_2_RESULTS.md`:
   ```bash
   cd d:\Project\retail-execution-rl
   # Load results and analyze
   # Update WEEK_7_RESULTS_TRACKER.md with all results
   # Create comprehensive WEEK_7_SESSION_2_RESULTS.md
   ```
4. **Commit** to GitHub with final results

---

**Status:** ✅ All cells ready to copy-paste into Kaggle notebook
