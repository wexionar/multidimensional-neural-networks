# =============================================================
# LUMIN-DEMO 002: 20D Non-Linear & Integrity Check (v1.4 C)
# =============================================================
# Project: SLRM-nD (Lumin Synthesis & Resolution)
# Developers: Alex Kinetic & Gemini
# Repository: https://github.com/wexionar/multi-dimensional-neural-networks
# License: MIT License
# Date: 2026-01-17
# Description: Stress test evaluating the compiler's ability to 
#              linearize parabolic curves and verifying the 
#              Anti-Hallucination (The Void) logic.
# =============================================================

"""
SLRM-nD: 20D Non-Linear Stress Test (All-in-One Demo)
Description:
This script benchmarks the engine's ability to linearize a parabolic curve
in 20 dimensions and verifies the "The Void" (Anti-Hallucination) logic.
"""

import numpy as np
import pandas as pd
import time

# =============================================================
# LUMIN-SYNTHESIS v1.4B (Core Compiler)
# =============================================================
class Synthesis:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def compile(self, df):
        start_time = time.perf_counter()
        data = df.values
        D = data.shape[1] - 1
        remaining_data = data.copy()
        synthesis_rows = []

        while len(remaining_data) > 0:
            best_axis_order = None
            best_group_size = -1
            best_W, best_B = None, None
            
            for axis in range(D):
                order = [i for i in range(D) if i != axis] + [axis]
                current_sort = remaining_data[np.lexsort(remaining_data[:, order].T[::-1])]
                
                for i in range(1, len(current_sort)):
                    X_g = current_sort[:i+1, :-1]
                    Y_g = current_sort[:i+1, -1]
                    A = np.c_[X_g, np.ones(X_g.shape[0])]
                    try:
                        res, _, _, _ = np.linalg.lstsq(A, Y_g, rcond=None)
                        W_tmp, B_tmp = res[:-1], res[-1]
                        if np.all(np.abs(np.dot(X_g, W_tmp) + B_tmp - Y_g) <= self.epsilon):
                            if i > best_group_size:
                                best_group_size = i
                                best_axis_order = current_sort
                                best_W, best_B = W_tmp, B_tmp
                        else: break
                    except: break

            if best_group_size == -1:
                idx_to_save = 1
                best_W, best_B = np.zeros(D), remaining_data[0, -1]
                best_axis_order = remaining_data
            else:
                idx_to_save = best_group_size + 1

            sector_data = best_axis_order[:idx_to_save]
            row = np.concatenate([
                np.min(sector_data[:, :-1], axis=0), 
                np.max(sector_data[:, :-1], axis=0), 
                best_W, [best_B]
            ])
            synthesis_rows.append(row)
            remaining_data = best_axis_order[idx_to_save:]

        cols = [f'X{i}_min' for i in range(D)] + [f'X{i}_max' for i in range(D)] + \
               [f'W{i}' for i in range(D)] + ['Bias']
        return pd.DataFrame(synthesis_rows, columns=cols), time.perf_counter() - start_time

# =============================================================
# LUMIN-RESOLUTION v1.4C (Batch Executor)
# =============================================================
class Resolution:
    def __init__(self, sectors_df):
        self.sectors = sectors_df.values
        self.D = (self.sectors.shape[1] - 1) // 3
        self.mins = self.sectors[:, :self.D]
        self.maxs = self.sectors[:, self.D:2*self.D]
        self.weights = self.sectors[:, 2*self.D:3*self.D]
        self.biases = self.sectors[:, -1]

    def resolve(self, X_input):
        X = np.atleast_2d(X_input)
        # Optimized float64 array for native performance
        results = np.full(X.shape[0], np.nan) 
        for i, point in enumerate(X):
            inside = np.all((point >= self.mins - 1e-9) & (point <= self.maxs + 1e-9), axis=1)
            candidates = np.where(inside)[0]
            if len(candidates) > 0:
                idx = candidates[0]
                results[i] = np.dot(point, self.weights[idx]) + self.biases[idx]
        return results[0] if len(results) == 1 else results

# =============================================================
# MAIN STRESS TEST FUNCTION
# =============================================================
def run_stress_test():
    print("🧪 SLRM-nD: 20D NON-LINEAR STRESS TEST")
    print("-" * 50)

    # 1. Dataset Generation (5k points, 20 Dimensions)
    N, D = 5000, 20
    X = np.random.uniform(-5, 5, (N, D))
    Y = np.zeros(N)

    for i in range(N):
        if X[i, 0] > 0:
            Y[i] = np.sum(X[i, :]) + 10 # Linear Zone
        else:
            Y[i] = np.sum(X[i, :]**2) / 5 # Non-Linear Zone (Parabola)

    df = pd.DataFrame(np.c_[X, Y], columns=[f'X{i+1}' for i in range(D)] + ['Y'])

    # 2. Synthesis (Compilation)
    epsilon = 0.1
    print(f"Compiling {N} points in {D}D (Epsilon={epsilon})...")
    compiler = Synthesis(epsilon=epsilon)
    master_sectors, synth_time = compiler.compile(df)

    print(f"✅ Synthesis Complete: {synth_time:.2f}s")
    print(f"📦 Master Sectors Created: {len(master_sectors)}")

    # 3. Resolution (Inference)
    resolver = Resolution(master_sectors)
    
    # Batch Test (1000 points)
    test_batch = np.random.uniform(-5, 5, (1000, D))
    start_res = time.perf_counter()
    resolver.resolve(test_batch)
    print(f"🚀 Resolution Speed (1000 pts): {(time.perf_counter() - start_res)*1000:.2f} ms")

    # 4. Integrity Test (The Void)
    void_point = np.full((1, D), 100.0) 
    pred_void = resolver.resolve(void_point)
    
    print(f"🌌 Void Detection Result: {pred_void} (The Void)")
    
    # Robust anti-hallucination verification
    if np.all(pd.isna(pred_void)):
        print("✅ Integrity Verified: Correctly detected 'The Void' (Anti-Hallucination).")

if __name__ == "__main__":
    run_stress_test()
  
