# Kaggle Notebook: PPO Training Session 1
# Copy this entire script into a Kaggle notebook, execute cell by cell
# Estimated duration: 6 hours
# GPU required: P100 or better

# ============================================================================
# CELL 1: INSTALL DEPENDENCIES (5 minutes)
# ============================================================================
# !pip install -q yfinance pandas numpy gymnasium stable-baselines3 torch pyyaml

# ============================================================================
# CELL 2: CLONE REPO & SETUP (5 minutes)
# ============================================================================
# import os
# os.chdir('/kaggle/working')
# !git clone https://github.com/YOUR_USERNAME/retail-execution-rl.git
# os.chdir('retail-execution-rl')

# ============================================================================
# CELL 3: IMPORTS & CONFIGURATION (2 minutes)
# ============================================================================
# import sys
# sys.path.insert(0, '/kaggle/working/retail-execution-rl')

# from src.environment import RetailExecutionEnv
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import CheckpointCallback
# import yaml
# import json
# from datetime import datetime
# import torch
# import numpy as np

# print(f"GPU Available: {torch.cuda.is_available()}")
# print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# # Load config
# with open('configs/training_config.yaml') as f:
#     config = yaml.safe_load(f)

# print("\nConfiguration loaded:")
# print(f"  Timesteps per seed: {config['training']['timesteps']}")
# print(f"  Number of seeds: {config['training']['n_seeds']}")
# print(f"  Learning rate: {config['training']['learning_rate']}")
# print(f"  Training on stocks: {config['data']['train_stocks'][:2]}  # Only AAPL, MSFT")

# ============================================================================
# CELL 4: MAIN TRAINING LOOP (5.5 hours) - THIS IS THE MAIN WORK
# ============================================================================
"""
import os
from pathlib import Path

training_results = {}
start_time = datetime.now()

# Train PPO on AAPL and MSFT (2 stocks × 5 seeds = 10 models)
for stock in config['data']['train_stocks'][:2]:  # Only AAPL, MSFT
    print(f"\n{'='*70}")
    print(f"STARTING TRAINING ON {stock}")
    print(f"{'='*70}")
    
    for seed in range(config['training']['n_seeds']):
        print(f"\n[{stock} - Seed {seed}] Starting...")
        print(f"  Timesteps: {config['training']['timesteps']}")
        print(f"  Batch size: {config['training']['batch_size']}")
        print(f"  Learning rate: {config['training']['learning_rate']}")
        
        # Create environment
        env = RetailExecutionEnv(
            symbol=stock,
            api_limit=5,  # Use 5 rps for baseline training
            execution_window_minutes=config['environment']['execution_window_minutes']
        )
        
        # Create checkpoint callback
        checkpoint_dir = f'models/{stock}_seed{seed}'
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        callback = CheckpointCallback(
            save_freq=config['training']['n_steps'],  # Save every n_steps
            save_path=checkpoint_dir,
            name_prefix='rl_model',
            save_replay_buffer=False
        )
        
        # Create and train model
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=config['training']['learning_rate'],
            batch_size=config['training']['batch_size'],
            n_steps=config['training']['n_steps'],
            n_epochs=config['training']['n_epochs'],
            gamma=config['training']['gamma'],
            seed=seed,
            verbose=1,
            device='cuda',
            tensorboard_log='./logs/'
        )
        
        # Train
        model.learn(
            total_timesteps=config['training']['timesteps'],
            callback=callback,
            log_interval=50  # Print progress every 50 iterations
        )
        
        # Save final model
        final_path = f'{checkpoint_dir}/final_model'
        model.save(final_path)
        print(f"  ✓ Saved final model to {final_path}")
        
        # Quick evaluation on training stock
        obs, info = env.reset()
        episode_cost = 0
        episode_steps = 0
        
        for step in range(500):  # Quick eval: ~10% of episode
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_cost += info['slippage']
            episode_steps += 1
            
            if terminated or truncated:
                break
        
        # Store results
        result_key = f'{stock}_seed{seed}'
        training_results[result_key] = {
            'stock': stock,
            'seed': seed,
            'final_cost_bps': float(episode_cost),
            'steps_evaluated': episode_steps,
            'steps_trained': config['training']['timesteps'],
            'timestamp': datetime.now().isoformat()
        }
        
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        print(f"[{stock} - Seed {seed}] ✓ Complete")
        print(f"  Cost: {episode_cost:.3f} bps")
        print(f"  Elapsed time: {elapsed:.1f} minutes")
        print(f"  Est. total time: {elapsed / ((len(config['data']['train_stocks'][:2]) * seed + seed + 1) / 10):.1f} minutes")

print(f"\n{'='*70}")
print(f"TRAINING COMPLETE")
print(f"{'='*70}")
print(f"Total models trained: {len(training_results)}")
print(f"Total time: {(datetime.now() - start_time).total_seconds() / 3600:.2f} hours")

# Save results
with open('training_results_session1.json', 'w') as f:
    json.dump(training_results, f, indent=2)

print(f"\n✓ Results saved to training_results_session1.json")
"""

# ============================================================================
# CELL 5: RESULTS SUMMARY (5 minutes)
# ============================================================================
"""
import pandas as pd

# Load results
with open('training_results_session1.json') as f:
    results = json.load(f)

# Create summary
df = pd.DataFrame.from_dict(results, orient='index')
df['stock'] = df['stock'].astype(str)

print("\n" + "="*70)
print("SESSION 1 RESULTS SUMMARY")
print("="*70)

summary = df.groupby('stock')['final_cost_bps'].agg([
    ('Mean Cost (bps)', 'mean'),
    ('Std Dev (bps)', 'std'),
    ('Min (bps)', 'min'),
    ('Max (bps)', 'max'),
    ('Count', 'count')
]).round(4)

print(summary)

print(f"\nOverall average cost: {df['final_cost_bps'].mean():.4f} bps")
print(f"Overall std dev: {df['final_cost_bps'].std():.4f} bps")

# Show individual results
print("\n" + "-"*70)
print("Individual Results:")
print("-"*70)
for key, val in results.items():
    print(f"{key:20s}: {val['final_cost_bps']:8.4f} bps")
"""

# ============================================================================
# CELL 6: DOWNLOAD RESULTS (2 minutes)
# ============================================================================
"""
# Create zip with all models and results
!zip -r -q models_session1.zip models/ training_results_session1.json logs/ 2>/dev/null || true

# Display download link
from IPython.display import FileLink
print("\n✓ Download ZIP file:")
FileLink('models_session1.zip')
"""

# ============================================================================
# NOTES FOR EXECUTION
# ============================================================================
"""
TIMING BREAKDOWN:
- Cell 1 (pip install): ~5 min
- Cell 2 (git clone): ~2 min
- Cell 3 (load config): ~2 min
- Cell 4 (main training): ~5h 45m
  └─ AAPL 5 seeds: ~2h 30m (30 min × 5)
  └─ MSFT 5 seeds: ~2h 30m (30 min × 5)
  └─ Evaluation: ~15 min (3 min × 5)
- Cell 5 (summary): ~3 min
- Cell 6 (download): ~1 min
TOTAL: ~6 hours

EXPECTED GPU USAGE:
- GPU utilization: 70-85%
- Memory: 4-6 GB out of 16 GB available
- No out-of-memory errors expected

WHAT TO WATCH FOR:
1. Loss values decreasing over training (good sign)
2. Evaluation costs reasonable (<1 bps reasonable)
3. No CUDA errors
4. Checkpoints saved every 2048 steps

IF SOMETHING FAILS:
- Check GPU is available: torch.cuda.is_available()
- Check memory: torch.cuda.memory_allocated() / 1e9
- Save models manually: model.save('path/to/model')
- Download ZIP even if training incomplete

AFTER SESSION:
1. Download models_session1.zip
2. Extract locally: unzip models_session1.zip -d Month2_Session1/
3. Push results to GitHub
4. Analyze Session 1 results before Session 2
"""
