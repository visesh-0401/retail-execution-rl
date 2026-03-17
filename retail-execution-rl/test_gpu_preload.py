#!/usr/bin/env python3
"""
test_gpu_preload.py
-------------------
Quick test to verify GPU data preloading works with the simulator.

Run with:
    python test_gpu_preload.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader_gpu import GPUDataLoader
from src.simulator import RetailExecutionSimulator


def create_sample_data(num_days=60):
    """Create sample OHLCV data for testing."""
    dates = pd.date_range("2025-01-01", periods=num_days*390, freq="1min")
    
    df = pd.DataFrame({
        'Open': np.random.randn(num_days*390).cumsum() + 100,
        'High': np.random.randn(num_days*390).cumsum() + 101,
        'Low': np.random.randn(num_days*390).cumsum() + 99,
        'Close': np.random.randn(num_days*390).cumsum() + 100,
        'Volume': np.random.randint(1000, 10000, num_days*390),
    }, index=dates)
    
    # Ensure High > Open/Close > Low
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1) + 0.5
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1) - 0.5
    
    return df


def test_cpu_path():
    """Test simulator with CPU DataFrames (baseline)."""
    print("\n" + "="*70)
    print("TEST 1: CPU Path (baseline)")
    print("="*70)
    
    data = create_sample_data(5)
    
    try:
        sim = RetailExecutionSimulator(
            data=data,
            rate_limit_rps=5,
            execution_window_steps=30,
            seed=42,
        )
        
        result = sim.execute(target_qty=100, start_idx=100)
        print(f"✓ CPU path works: {result}")
        return True
    except Exception as e:
        print(f"✗ CPU path failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_preload():
    """Test simulator with GPU-preloaded DataFrames."""
    print("\n" + "="*70)
    print("TEST 2: GPU Preload Path")
    print("="*70)
    
    data_map = {
        'AAPL': create_sample_data(5),
        'MSFT': create_sample_data(5),
        'GOOGL': create_sample_data(5),
    }
    
    try:
        # Load to GPU
        loader = GPUDataLoader(data_map, use_gpu=True, verbose=True)
        gpu_data = loader.to_device()
        
        # Test simulator with GPU data
        print("\nTesting simulator with GPU data...")
        sim = RetailExecutionSimulator(
            data=gpu_data['AAPL'],
            rate_limit_rps=5,
            execution_window_steps=30,
            seed=42,
        )
        
        result = sim.execute(target_qty=100, start_idx=100)
        print(f"✓ GPU path works: {result}")
        return True
    except Exception as e:
        print(f"✗ GPU path failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_with_gpu():
    """Test environment with GPU-preloaded data."""
    print("\n" + "="*70)
    print("TEST 3: Environment with GPU Data")
    print("="*70)
    
    try:
        from src.environment import RetailExecutionEnv
        
        data_map = {
            'AAPL': create_sample_data(5),
            'MSFT': create_sample_data(5),
        }
        
        # Load to GPU
        loader = GPUDataLoader(data_map, use_gpu=True, verbose=False)
        gpu_data = loader.to_device()
        
        print("Creating environment with GPU data...")
        env = RetailExecutionEnv(
            data_map=gpu_data,
            target_qty=100,
            rate_limit_rps=5,
            execution_window_steps=30,
        )
        
        # Run a few steps
        obs, info = env.reset()
        print(f"  Initial observation shape: {obs.shape}")
        print(f"  Initial observation: {obs[:5]}...")
        
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                break
        
        print(f"✓ Environment works with GPU data")
        env.close()
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("GPU Data Preloading Tests")
    print("="*70)
    
    results = {}
    results['cpu'] = test_cpu_path()
    results['gpu'] = test_gpu_preload()
    results['env'] = test_environment_with_gpu()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:20s}  {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✅ All tests passed! GPU preloading is ready.\n")
    else:
        print("\n❌ Some tests failed. Check output above.\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
