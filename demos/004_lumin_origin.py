# =============================================================
# LUMIN-DEMO 004: The Industrial Origin Engine (High-Speed)
# =============================================================
# Project: SLRM-nD (Lumin Core)
# Developers: Alex Kinetic & Gemini
# Repository: https://github.com/wexionar/multi-dimensional-neural-networks
# License: MIT License
# Date: 2026-01-22
# Description: High-speed Symmetric Compression Engine with 
#              Auto-Sorting and Dual Epsilon Logic.
# =============================================================

import numpy as np
import pandas as pd
import time

# --- USER CONFIGURATION ---
EPSILON_VAL = 0.02    # Fracture sensitivity (Gradient/Delta-Y)
EPSILON_TYPE = 'abs'  # 'abs' (Absolute) | 'rel' (Relative %)
STRUCT_MODE = 1       # 1: DIVERSITY (Continuous) | 2: PURITY (Isolated)
PRINT_SUMMARY = 0     # 1: Print sector laws | 0: Silent mode
# ===========================

def run_origin_004(input_csv, output_npy):
    start_time = time.perf_counter()
    
    # 1. DATA LOADING & AUTO-SORTING (Critical Bug Fix)
    # We ensure X0 is ordered to maintain geometric continuity
    df = pd.read_csv(input_csv)
    df = df.sort_values(by=df.columns[0]).reset_index(drop=True)
    
    data = df.values
    X, Y = data[:, :-1], data[:, -1]
    n_rows, n_dims = X.shape

    # 2. SYMMETRIC NORMALIZATION
    # Keeps zero-center integrity in [-1, 1] domain
    max_abs_y = np.max(np.abs(Y)) if np.max(np.abs(Y)) != 0 else 1.0
    Y_norm = Y / max_abs_y

    # 3. COMPRESSION ENGINE (Vectorized Logic)
    sectors = []
    start_idx = 0

    while start_idx < n_rows:
        end_idx = start_idx + 1
        found_end = False
        
        while end_idx < n_rows:
            if end_idx == start_idx + 1:
                end_idx += 1
                continue
            
            # Linear Prediction (Interpolation) between boundaries
            x_s, x_e = X[start_idx], X[end_idx]
            y_s, y_e = Y_norm[start_idx], Y_norm[end_idx]
            
            test_idx = np.arange(start_idx + 1, end_idx)
            denom = (x_e[0] - x_s[0]) if (x_e[0] - x_s[0]) != 0 else 1e-9
            t = (X[test_idx, 0] - x_s[0]) / denom
            y_pred = y_s + t * (y_e - y_s)
            
            # Error Detection (Delta-Y Projection vs Reality)
            errors = np.abs(Y_norm[test_idx] - y_pred)
            
            # Dynamic vs Fixed Margin logic
            if EPSILON_TYPE == 'rel':
                margins = np.abs(Y_norm[test_idx]) * EPSILON_VAL
            else:
                margins = np.full_like(errors, EPSILON_VAL)
            
            if np.any(errors > margins):
                found_end = True
                break
            else:
                end_idx += 1
        
        actual_end = end_idx - 1 if found_end else n_rows - 1
        if actual_end <= start_idx: actual_end = start_idx + 1
        
        # Sector Law Synthesis
        #mins = np.min(X[start_idx:actual_end+1], axis=0) # (Optional in v4.1 for speed)
        #maxs = np.max(X[start_idx:actual_end+1], axis=0)
        mins, maxs = X[start_idx], X[actual_end] # Faster for sorted data

        w = np.zeros(n_dims)
        dist_x = (X[actual_end, 0] - X[start_idx, 0])
        if dist_x != 0:
            w[0] = (Y_norm[actual_end] - Y_norm[start_idx]) / dist_x
            
        bias = Y_norm[start_idx] - w[0] * X[start_idx, 0]
        
        if PRINT_SUMMARY:
            print(f"Sector {len(sectors)}: W0={w[0]:.4f} | Size={actual_end - start_idx}")

        sectors.append(np.concatenate([mins, maxs, w, [bias]]))
        
        # 4. STRUCTURAL FLOW (Diversity vs Purity)
        start_idx = actual_end if STRUCT_MODE == 1 else actual_end + 1
        if not found_end or start_idx >= n_rows: break

    # 5. BINARY EXPORT WITH METADATA
    sectors_matrix = np.array(sectors)
    metadata = np.zeros(sectors_matrix.shape[1])
    metadata[0] = max_abs_y
    metadata[1] = 1 if EPSILON_TYPE == 'abs' else 0
    
    np.save(output_npy, np.vstack([metadata, sectors_matrix]))
    
    exec_time = time.perf_counter() - start_time
    print(f"✅ [LUMIN-ORIGIN 004] Pipeline Finished.")
    print(f"Sectors: {len(sectors)} | Time: {exec_time:.4f}s")

if __name__ == "__main__":
    # run_origin_004("input.csv", "output.npy")
    pass
    
