# =============================================================
# LUMIN-DEMO 001: 50D Galactic Stress Test (v1.4 B)
# =============================================================
# Project: SLRM-nD (Lumin Core)
# Developers: Alex Kinetic & Gemini
# Repository: https://github.com/wexionar/multi-dimensional-neural-networks
# License: MIT License
# Date: 2026-01-16
# Description: Stress test demonstrating the Axis-Pivot Compiler 
#              solving 50-dimensional synthesis in seconds.
# =============================================================

import numpy as np
import pandas as pd
import time

# Note: This demo uses the classes defined in the repository.
# For standalone testing, we include the LuminSynthesis logic below.

class LuminSynthesis:
    """
    Hierarchical Axis-Pivot Compiler (v1.4 B).
    Deduces the optimal axis order to maximize data synthesis.
    """
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.master_sectors = None

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

            # Hierarchical Pivot Search
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

            # Master Sector Consolidation
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
        self.master_sectors = pd.DataFrame(synthesis_rows, columns=cols)
        return self.master_sectors, time.perf_counter() - start_time

# --- EXECUTION: 50D GALACTIC STRESS TEST ---
if __name__ == "__main__":
    print("🌌 LUMIN-DEMO 001: 50-Dimensional Stress Test")
    print("Target: Synthesize a 50D universe into a single Master Sector.")
    
    D, N = 50, 1000
    X = np.random.uniform(-10, 10, (N, D))
    
    # Define a single 50D hyper-law: Y = Sum(Xi) + 10
    # A single law should result in 100% synthesis (1 Master Sector)
    Y = np.sum(X, axis=1) + 10
    
    # Add minimal noise to simulate real-world precision
    Y += np.random.normal(0, 1e-6, N)
    
    df_galactic = pd.DataFrame(np.c_[X, Y], columns=[f'X{i+1}' for i in range(D)] + ['Y'])

    epsilon_test = 0.01
    compiler = LuminSynthesis(epsilon=epsilon_test)

    print(f"🚀 Launching compilation: {N} points in {D} dimensions...")
    master_df, duration = compiler.compile(df_galactic)

    print("-" * 60)
    print(f"Synthesis Result: {len(master_df)} Master Sector(s)")
    print(f"Compression Rate: {((N - len(master_df))/N)*100:.2f}%")
    print(f"Execution Time: {duration:.4f} seconds")
    print("-" * 60)
    
    if len(master_df) == 1:
        print("SUCCESS: The Axis-Pivot Compiler has unified the 50D universe.")
    else:
        print("RESULT: Complexity detected. Data partitioned into master sectors.")
