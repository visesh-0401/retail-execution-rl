# WEEK 5 ACTION CHECKLIST: Kaggle Prep
**Current Status**: Month 1 complete, all local code works ✅  
**Target**: Prepare Kaggle setup for Session 1 (Week 6)  
**Time Budget**: 2-3 hours  
**GPU Time Used**: 0 hours (saving for Week 6)

---

## TODAY - Setup (30 min)

- [ ] Create `configs/training_config.yaml` (copy from MONTH_2_PLAN.md)
- [ ] Create `kaggle/train_session_1.ipynb` (Kaggle-specific notebook, see template)
- [ ] Test config loads locally:
  ```bash
  python -c "import yaml; config = yaml.safe_load(open('configs/training_config.yaml')); print(config.keys())"
  ```

## TOMORROW - Kaggle Account Setup (30 min)

- [ ] Log into kaggle.com
- [ ] Create new blank notebook on Kaggle
- [ ] Enable GPU (should default to P100, that's fine)
- [ ] Save notebook as `retail-execution-rl-session-1`
- [ ] Copy test code below into first cell:

```python
# Test Kaggle GPU setup
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
# Expected: GPU Available: True, GPU Name: Tesla P100 (or similar)
```

- [ ] Run it, screenshot confirmation

## LATER THIS WEEK - GitHub Final Push (30 min)

```bash
cd d:\Project

# Create config directory
mkdir -p configs

# Create training config file
cat > configs/training_config.yaml << 'EOF'
training:
  timesteps: 50000
  n_seeds: 5
  batch_size: 64
  learning_rate: 3e-4
  n_steps: 2048
  n_epochs: 10
  gamma: 0.99

hyperparameters:
  learning_rates: [1e-4, 3e-4, 1e-3]

data:
  stocks: ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'QQQ']
  train_stocks: ['AAPL', 'MSFT', 'GOOGL']
  test_stocks: ['TSLA', 'SPY', 'QQQ']

api_limits: [1, 5, 10]

reward:
  slippage_weight: 1.0
  api_penalty_weight: 10.0
EOF

# Create Kaggle notebook directory
mkdir -p kaggle

# Commit everything
git add -A
git commit -m "Month 2 Week 5: Kaggle training config + notebook templates"
git push origin main
```

**Verify on GitHub**:
- [ ] `configs/training_config.yaml` uploaded
- [ ] `MONTH_2_PLAN.md` uploaded
- [ ] All scripts still there

## FRIDAY - Final Pre-Session 1 Check (30 min)

Run this local test to ensure everything works:

```bash
# Test 1: Config loads
python -c "import yaml; print(yaml.safe_load(open('configs/training_config.yaml'))['training']['timesteps'])"
# Expected: 50000

# Test 2: Run tiny training to confirm GPU works locally
python scripts/04_train_ppo.py --timesteps 100 --seed 42 --stocks AAPL
# Expected: Should complete in <30 seconds, produce model

# Test 3: Verify models folder structure
ls -la models/
# Expected: See AAPL_1000_seed42 folder
```

---

## Week 6 Preparation (Make This List Before Saturday)

Print out this checklist before Week 6:

```
KAGGLE SESSION 1 EXECUTION CHECKLIST
═══════════════════════════════════════════

PRE-SESSION (Friday):
☐ Clone repo: git clone https://github.com/YOUR_USERNAME/retail-execution-rl.git
☐ Create Kaggle notebook
☐ Enable GPU
☐ Add GitHub as dataset (input)
☐ Save notebook as "retail-execution-rl-session-1"

SESSION START (Day 1 of Week 6):
☐ Cell 1: pip installs (5 min)
☐ Cell 2: git clone and cd (2 min)
☐ Cell 3: import + load config (2 min)
☐ Cell 4: START TRAINING LOOP ← MAIN WORK HERE (5h 45m)
  ┌─ AAPL seed 0-4 (20 min × 5 = 100 min)
  ├─ MSFT seed 0-4 (20 min × 5 = 100 min)
  ├─ Save models after each seed
  └─ Check outputs in real-time
☐ Cell 5: Download results ZIP
☐ Download models_session1.zip to local machine

IMMEDIATELY AFTER SESSION:
☐ Extract models locally: unzip models_session1.zip -d Month2_Session1_Results/
☐ Run analysis: python scripts/analyze_session1.py
☐ Commit results to GitHub: git add results/, git commit -m "Session 1 Results", git push
```

---

## Critical Success Factors for Week 6

✅ **DO**:
- Save checkpoints every 1024 steps (automatic in script)
- Print progress after each seed (easy to track)
- Download ZIP immediately after session ends
- Keep Kaggle notebook for reference

❌ **DON'T**:
- Try to change learning rate mid-session (stick to 3e-4)
- Train on all 6 stocks (only AAPL + MSFT)
- Leave session running unattended >6h (downloads might fail)

---

## Expected Week 6 Output

After 6-hour Kaggle session, you should have:

```
models/
├── AAPL_seed0/
│   ├── rl_model_1024_steps.zip
│   ├── rl_model_2048_steps.zip
│   ├── ... (checkpoints)
│   └── final_model.zip
├── AAPL_seed1/
├── ... (AAPL seed2-4)
├── MSFT_seed0/
├── ... (MSFT seed1-4)
└── best_lr_model.zip (placeholder for now)

training_results_session1.json
{
  "AAPL_seed0": {"final_cost_bps": 0.18, "steps_trained": 50000},
  "AAPL_seed1": {"final_cost_bps": 0.16, ...},
  ...
}
```

**Success Metrics**:
- ✅ 10 models successfully saved
- ✅ training_results_session1.json created
- ✅ No CUDA errors
- ✅ Loss curves show improvement over time

---

## Next: Week 7 (Hyperparameter Tuning)

After Session 1, **immediately schedule Session 2** for Week 7.

In between:
- Analyze Session 1 results locally (CPU, 1 hour)
- Decide if you want to retrain anything
- Prepare Session 2 notebook (copy from MONTH_2_PLAN.md)

---

## Questions Before You Start?

Before Week 6, ask yourself:
1. Do I have Kaggle account set up? ✓
2. Have I tested Kaggle notebook environment? (run GPU test cell)
3. Is GitHub repo committed and pushed?
4. Do I have copies of scripts locally + backed up?

If all ✓, you're ready to press GO on Week 6!

---

