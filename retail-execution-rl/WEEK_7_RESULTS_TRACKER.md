# WEEK 7: Results Tracking Sheet

Fill in results as each phase completes. Compare against **Week 6 Baseline: -81.96 ± 35.4**

---

## Phase 1: Learning Rate Sweep (25k timesteps, 1 seed)

| LR | Seed | Best Reward | Mean Reward | Std Dev | Time | Notes |
|----|------|------------|-------------|---------|------|-------|
| 1e-4 | 42 | | | | | Conservative |
| 3e-4 | 42 | **-81.96** | **-81.96** | **35.4** | 8m | **BASELINE** |
| 5e-4 | 42 | | | | | Aggresive |
| 1e-3 | 42 | | | | | Very aggressive |

**Analysis:**
- Best LR from Phase 1: ________________
- Learning rate impact: ________________
- Recommendation: ________________

---

## Phase 2: Batch Size Sweep (25k timesteps, 1 seed)

| BS | Seed | Best Reward | Mean Reward | Std Dev | Time | Notes |
|----|------|------------|-------------|---------|------|-------|
| 32 | 42 | | | | | Small, high variance |
| 64 | 42 | **-81.96** | **-81.96** | **35.4** | 8m | **BASELINE** |
| 128 | 42 | | | | | Large, smooth |
| 256 | 42 | | | | | Very large, risk of underfitting |

**Analysis:**
- Best BS from Phase 2: ________________
- Batch size impact: ________________
- Recommendation: ________________

---

## Phase 3: Entropy Coefficient Sweep (25k timesteps, 1 seed)

| Entropy | Seed | Best Reward | Mean Reward | Std Dev | Time | Notes |
|---------|------|------------|-------------|---------|------|-------|
| 0.001 | 42 | | | | | Low exploration |
| 0.01 | 42 | **-81.96** | **-81.96** | **35.4** | 8m | **BASELINE** |
| 0.1 | 42 | | | | | High exploration |

**Analysis:**
- Best Entropy from Phase 3: ________________
- Entropy coefficient impact: ________________
- Recommendation: ________________

---

## Phase 4: Best Config Validation (50k timesteps, 5 seeds)

**Selected Configuration:**
```
LR: __________ (from Phase 1)
Batch Size: __________ (from Phase 2)
Entropy: __________ (from Phase 3)
Gamma: 0.99 (unchanged)
GAE Lambda: 0.95 (unchanged)
Clip Range: 0.2 (unchanged)
```

| Seed | Best Reward | Mean Reward | Std Dev | Model Path | Notes |
|------|------------|-------------|---------|------------|-------|
| 42 | | | | ppo_ratelimit5_seed42_final.zip | |
| 43 | | | | ppo_ratelimit5_seed43_final.zip | |
| 44 | | | | ppo_ratelimit5_seed44_final.zip | |
| 45 | | | | ppo_ratelimit5_seed45_final.zip | |
| 46 | | | | ppo_ratelimit5_seed46_final.zip | |

**Final Result:**
- **Mean Reward:** ________________ (vs baseline -81.96)
- **Std Dev:** ________________ (vs baseline 35.4)
- **Improvement:** ________________ points
- **Success?** YES / NO

---

## Summary: Best Hyperparameters Found

| Parameter | Baseline | Week 7 Best | Improvement |
|-----------|----------|------------|-------------|
| Learning Rate | 3e-4 | __________ | __________ |
| Batch Size | 64 | __________ | __________ |
| Entropy Coef | 0.01 | __________ | __________ |
| **Reward** | **-81.96** | **__________** | **__________** |
| **Std Dev** | **35.4** | **__________** | **__________** |

---

## Statistical Significance Test

If running t-test between Week 6 (5 seeds) and Week 7 Best (5 seeds):

```
Week 6 rewards: -81.96, -73.02, -90.43, ?, ?
Week 7 rewards: ?, ?, ?, ?, ?

T-test result: 
p-value: __________ (< 0.05 = significant improvement)
Conclusion: ________________
```

---

## Lessons Learned

**What worked well:**
1. ________________
2. ________________
3. ________________

**What didn't work:**
1. ________________
2. ________________
3. ________________

**Next steps (Week 8):**
- [ ] Test ablation on different stocks (MSFT, TSLA)
- [ ] Vary window size and rate limit
- [ ] Run on CPU for cost-effective experiments
- [ ] Compare final model vs different RL algorithms

---

## Delivery Checklist

- [ ] All 12 Phase 1-3 quick tests completed
- [ ] Phase 4 best config (5 seeds @ 50k) completed
- [ ] Results CSV downloaded from Kaggle
- [ ] This tracker filled in with all results
- [ ] Analysis completed and lessons learned documented
- [ ] WEEK_7_SESSION_2_RESULTS.md created
- [ ] All results committed to GitHub (commit message: "Add Week 7 hyperparameter tuning results")

---

**Session Completion Date:** ________________  
**Total GPU Time Used:** ________________ hours  
**Status:** ☐ In Progress | ☑ Ready to Start | ☐ Complete
