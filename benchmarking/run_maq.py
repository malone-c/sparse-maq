import numpy as np
from maq import MAQ
import time
import tracemalloc

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--base-path', type=str)
    args = parser.parse_args()

    print("\n=== MAQ BASELINE PROFILING ===")

    # Start profiling
    tracemalloc.start()
    t0 = time.perf_counter()

    # Load data
    rewards = np.load(f'{args.base_path}/reward.npy')
    costs = np.load(f'{args.base_path}/cost.npy')

    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    print(f"Load data: {t1-t0:.2f}s, Peak Memory: {peak/1024**3:.2f} GB")
    print(f"  rewards shape: {rewards.shape}, size: {rewards.nbytes/1024**3:.2f} GB")
    print(f"  costs shape: {costs.shape}, size: {costs.nbytes/1024**3:.2f} GB")

    # Fit solver
    solver = MAQ()
    t2 = time.perf_counter()
    solver.fit(rewards, costs, rewards)
    t3 = time.perf_counter()

    current, peak = tracemalloc.get_traced_memory()
    print(f"Fit solver: {t3-t2:.2f}s, Peak Memory: {peak/1024**3:.2f} GB")
    print(f"\nTotal time: {t3-t0:.2f}s")
    print(f"Total peak memory: {peak/1024**3:.2f} GB")
    tracemalloc.stop()

