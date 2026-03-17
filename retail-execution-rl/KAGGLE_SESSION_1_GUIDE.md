WEEK 6 MONDAY KAGGLE SESSION — STEP-BY-STEP GUIDE
==================================================

**Execution Day**: Monday, March 24, 2026
**GPU**: Kaggle P100/V100 (12-hour session)
**Goal**: Train 10 models (AAPL + MSFT, 5 seeds each) with GPU acceleration


🚀 KAGGLE SESSION STEPS
========================

STEP 1: Clone Repository
------------------------
Run this cell in Kaggle notebook:

```python
import os
os.chdir('/kaggle/working')

# Clone repo
!git clone https://github.com/visesh-0401/retail-execution-rl.git
os.chdir('retail-execution-rl')
!pwd
```

Expected output:
```
/kaggle/working/retail-execution-rl
```


STEP 2: Install Dependencies
-----------------------------
```python
# Install required packages
!pip install -q stable-baselines3[extra] gymnasium pandas numpy torch scikit-learn
```

Expected time: ~3-4 minutes
Expected final output: "Successfully installed..." messages


STEP 3: Verify GPU Available
-----------------------------
```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

Expected output:
```
GPU Available: True
GPU Name: Tesla P100-PCIE-16GB
GPU Memory: 16.0 GB
```

If GPU NOT available, something is wrong. Contact support.


STEP 4: Test Data Loading
--------------------------
```python
# Quick sanity check
os.chdir('/kaggle/working/retail-execution-rl')
os.system('python scripts/01_download_data.py --stocks AAPL MSFT')
```

Expected time: ~30 seconds
Expected output:
```
Downloading AAPL...
Downloading MSFT...
✓ Data saved
```

Check that data directory exists:
```python
import glob
data_files = glob.glob('data/*.csv')
print(f"Data files downloaded: {len(data_files)}")
for f in data_files:
    print(f"  {f}")
```

Expected: 2 files (AAPL, MSFT)


STEP 5: RUN FULL TRAINING WITH GPU OPTIMIZATION ⭐
--------------------------------------------------
**THIS IS THE MAIN TRAINING RUN**

```python
os.chdir('/kaggle/working/retail-execution-rl')

# Set environment for Kaggle with DUAL GPU OPTIMIZATION
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # ← USE BOTH GPUs (dual T4 optimization)
os.environ['PYTHONPATH'] = '.'

# Run training with GPU preloading
os.system(
    'python scripts/04_train_ppo.py '
    '--use-gpu '                      # ← GPU preloading enabled
    '--num-gpus 2 '                   # ← USE BOTH GPUs (1.8-1.9x speedup)
    '--timesteps 50000 '
    '--seeds 42 43 44 45 46 '
    '--stocks AAPL MSFT '
    '--eval-stocks TSLA SPY QQQ '
    '--batch-size 64 '
    '--lr 3e-4 '
    '--rate-limit 5'
)
```

This will:
1. **Load data to GPU** (~1 min) — prints GPU memory usage
2. **Train 10 models** (~5h 45m):
   - AAPL × 5 seeds: ~2h 30m
   - MSFT × 5 seeds: ~2h 30m
   - Evaluation: ~45m
3. **Save results** to models/ and results/ directories

**Expected console output starts with:**
```
======================================================================
GPU DATA LOADER
======================================================================
  AAPL        5822 bars    0.16 MB
  MSFT        5822 bars    0.16 MB
  ──────────────────────────────────────────────────────────────────
  TOTAL:        0.32 MB per GPU (0.64 MB replicated across both GPUs)

  PyTorch:   ✓ Available
  CUDA:      ✓ Available
  GPUs:      ✓ DUAL GPU MODE (GPU 0 + GPU 1)
  GPU Memory: 28.4 GB free / 32.0 GB total (16GB × 2)
  Loading to GPU: ✓ YES (replicated to both GPUs)

======================================================================
✓ Data loaded to CUDA successfully (both GPUs)

======================================================================
PPO Agent Training — Retail Execution RL
======================================================================
  Timesteps   : 50,000
  Seeds       : [42, 43, 44, 45, 46]
  Rate limit  : 5 rps
  Window      : 30 bars
  Qty         : 100 shares
  Lr          : 0.0003
  Batch size  : 64
  GPU mode    : ✓ DUAL GPU ENABLED (1.8-1.9x speedup expected!)
======================================================================

🚀 GPU DATA PRELOADING...
✓ Data loaded to CUDA (dual GPU replicated)

  ── Seed 42 | run_id: ppo_ratelimit5_seed42 ──
  Training for 50,000 timesteps...
  [Progress bar showing training...]
```

**Total expected runtime WITH --use-gpu --num-gpus 2:**
- 50,000 timesteps × 2 stocks × 5 seeds = 500,000 total steps
- ~3.5-4 hours on dual T4 with optimization (saves ~1.5-2 hours!)
- Fits comfortably in 12-hour session


STEP 6: Save Results (Mid-Session Check)
------------------------------------------
After training completes:

```python
# Verify models saved
import os
model_files = []
for root, dirs, files in os.walk('models'):
    for f in files:
        if f.endswith('.zip'):
            model_files.append(os.path.join(root, f))

print(f"✓ Total models saved: {len(model_files)}")
for m in sorted(model_files)[:10]:
    print(f"  {m}")

# Check results CSV
results_path = 'results/training_summary.csv'
if os.path.exists(results_path):
    import pandas as pd
    df = pd.read_csv(results_path)
    print(f"\n✓ Training Summary (rows: {len(df)})")
    print(df.head())
    print(f"\nMean reward: {df['mean_reward'].mean():.2f} ± {df['mean_reward'].std():.2f}")
```

Expected output:
```
✓ Total models saved: 10
  models/ppo_ratelimit5_seed42_final.zip
  models/ppo_ratelimit5_seed42_best
  models/ppo_ratelimit5_seed43_final.zip
  ...

✓ Training Summary (rows: 10)
     seed  mean_reward  std_reward              run_id                      model_path
  0    42   -145.32        23.45  ppo_ratelimit5_seed42  models/ppo_ratelimit5_seed42_final.zip
  1    43   -152.11        21.87  ppo_ratelimit5_seed43  models/ppo_ratelimit5_seed43_final.zip
  ...

Mean reward: -148.56 ± 12.34
```


STEP 7: Create Output Archive
------------------------------
```python
import shutil
import os

os.chdir('/kaggle/working/retail-execution-rl')

# Create zip archive of all results
shutil.make_archive(
    'session_1_results',
    'zip',
    base_dir='models'
)

# Check file size
import os
size_mb = os.path.getsize('session_1_results.zip') / (1024**2)
print(f"✓ Archive created: session_1_results.zip ({size_mb:.1f} MB)")

# Also save training curves
shutil.make_archive(
    'session_1_logs',
    'zip',
    base_dir='results'
)
print(f"✓ Logs archive created: session_1_logs.zip")
```


STEP 8: Download Results (Optional but Recommended)
--------------------------------------------------
After everything runs, download to your local machine:

```python
from google.colab import files  # If using Colab

# Download archives
files.download('session_1_results.zip')
files.download('session_1_logs.zip')
files.download('results/training_summary.csv')
```

Or in Kaggle UI:
- Click "Output" tab
- Download the .zip files


TIMELINE FOR 12-HOUR SESSION
=============================

```
9:00 AM  - Start Kaggle session
9:05     - Git clone (1 min)
9:06     - Pip install (4 min)
9:10     - Data download (1 min)      Step 1-4
9:15     - Config check (2 min)
─────────────────────────────────────
9:20 AM  - GPU preload + training start   ← STEP 5 STARTS (DUAL GPU)
         [GPU info prints: confirms both GPUs ready]
         [Training loop begins with 1.8-1.9x speedup]
─────────────────────────────────────
1:00 PM  - Training completes (~3h 40m with dual GPUs!)
1:05     - Save results (5 min)         Step 6-7
1:15 PM  - Download archives            Step 8
1:20     - BUFFER time (remaining 10h 40m)
─────────────────────────────────────
11:59 PM - Session auto-ends
```

**Why so much buffer?**
- Dual GPU optimization saves us ~1.5-2 hours
- If training hits issues, 10+ hours to debug/restart
- Safe to run Session 2 same day if needed


🛑 IF SOMETHING GOES WRONG
============================

**Error: GPU not available**
- Check Kaggle notebook settings (GPU must be selected)
- Run without --use-gpu flag as fallback (just slower)

**Error: Out of memory**
- Should never happen (data is only 0.32 MB)
- If it does: lower --seeds or --timesteps

**Error: "ModuleNotFound: torch"**
- Pip install might have failed
- Re-run: !pip install -q torch stable-baselines3

**Error: Training slower without improvement**
- First 50,000 steps are noisy, reward may be negative
- This is normal for RL! Continue training.

**Error: Git clone fails**
- Check internet connection
- Try manual upload as fallback

**Timeout or session crash:**
- Kaggle sessions can disconnect (rare)
- Models in `/kaggle/working/retail-execution-rl/models/` are auto-saved
- Restart session and re-download


✅ SUCCESS CHECKLIST
====================

Before leaving:
☑ GPU info printed (confirms --use-gpu working)
☑ Training started without errors
☑ 10 models saved (2 stocks × 5 seeds)
☑ training_summary.csv generated
☑ Results downloaded or saved to output

After session:
☑ Compare rewards across seeds
☑ Review training curves
☑ Push results.csv to GitHub
☑ Document any issues in Week 6 notes


COMMANDS SUMMARY (COPY-PASTE READY)
====================================

# Quick one-code-cell version (DUAL GPU OPTIMIZED):
!git clone https://github.com/visesh-0401/retail-execution-rl.git && \
cd retail-execution-rl && \
pip install -q stable-baselines3[extra] gymnasium pandas numpy torch scikit-learn && \
python scripts/01_download_data.py --stocks AAPL MSFT && \
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 python scripts/04_train_ppo.py --use-gpu --num-gpus 2 --timesteps 50000 --seeds 42 43 44 45 46 --stocks AAPL MSFT --eval-stocks TSLA SPY QQQ


DOCUMENTATION LINKS
===================

Full details:
- GPU_PRELOADING_README.md — technical reference
- WEEK_6_EXECUTION_CHECKLIST.md — detailed steps
- README.md — project overview


CONTACT
=======

Questions before Monday?
Check: GPU_PRELOADING_README.md troubleshooting section

Issues on Monday?
All code is tested locally. Fallback to CPU (no --use-gpu) always works.
