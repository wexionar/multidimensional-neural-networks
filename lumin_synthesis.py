# =============================================================
# LUMIN-SYNTHESIS: nD Knowledge Compiler (v1.4 B)
# =============================================================
# Project: SLRM-nD (Lumin Core)
# Developers: Alex Kinetic & Gemini
# License: MIT License
# Date: 2026-01-16
# Description: Synthesizes massive datasets into minimal 
#              Master Sectors using hierarchical deductive 
#              geometry and Axis-Pivot Compilation.
# =============================================================

import numpy as np
import pandas as pd
import time

class LuminSynthesis:
    """
    Hierarchical Axis-Pivot Compiler.
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

            # Search for the axis pivot that maximizes compression
            for axis in range(D):
                # Re-order prioritizing the current axis (Hierarchical Pivot)
                # Sort by all axes, but the current one is the primary key
                order = [i for i in range(D) if i != axis] + [axis]
                current_sort = remaining_data[np.lexsort(remaining_data[:, order].T[::-1])]

                # Attempt to expand the sector
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

            # Consolidate the Master Sector
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

    def predict(self, X_input):
        X_input = np.array(X_input)
        for _, s in self.master_sectors.iterrows():
            D = (len(s)-1)//3
            # Check if the point falls within the sector's bounding box
            if np.all(X_input >= s.iloc[:D].values - 1e-9) and \
               np.all(X_input <= s.iloc[D:2*D].values + 1e-9):
                return np.dot(X_input, s.iloc[2*D:3*D].values) + s.iloc[-1]
        return None

# --- EXECUTION: STRESS TEST ---
if __name__ == "__main__":
    print("LUMIN-SYNTHESIS v1.4 B: High-Dimensional Segmented Synthesis")
    D, N = 5, 1000
    X = np.random.uniform(-10, 10, (N, D))

    # Two distinct laws separated by X1=0
    Y = np.where(X[:,0] > 0,
                 5*X[:,0] - 2*X[:,1] + X[:,4] + 10,
                 -3*X[:,0] + 4*X[:,2] - 0.5*X[:,3] - 5)

    Y += np.random.normal(0, 0.0001, N) # Minimal noise
    df_fire = pd.DataFrame(np.c_[X, Y], columns=[f'X{i+1}' for i in range(D)] + ['Y'])

    epsilon_fire = 0.01
    compiler = LuminSynthesis(epsilon=epsilon_fire)

    print(f"Starting compilation of {N} points in {D}D (Epsilon={epsilon_fire})...")
    master_df, duration = compiler.compile(df_fire)

    print("-" * 50)
    print(f"Duced Master Sectors: {len(master_df)}")
    print(f"Compression Rate: {((N - len(master_df))/N)*100:.2f}%")
    print(f"Execution Time: {duration:.4f} s")
    print("-" * 50)

    # Spot checks
    p_a = [5, 2, 0, 0, 1]  # Theoretical A: 32
    p_b = [-5, 0, 2, 4, 0] # Theoretical B: 16

    print(f"Validation Sector A (X1>0): Real ~32 | Pred: {compiler.predict(p_a)}")
    print(f"Validation Sector B (X1<0): Real ~16 | Pred: {compiler.predict(p_b)}")
