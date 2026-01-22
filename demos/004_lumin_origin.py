# =============================================================
# LUMIN-DEMO 004: The Symmetric Origin Engine (High-Speed)
# =============================================================
# Project: SLRM-nD (Lumin Core)
# Developers: Alex Kinetic & Gemini
# Repository: https://github.com/wexionar/multi-dimensional-neural-networks
# License: MIT License
# Date: 2026-01-22
# Description: High-speed Symmetric Compression Engine.
#              Implements Max-Abs Scaling and Relative Epsilon
#              to preserve geometric integrity across nD space.
# =============================================================

import numpy as np
import pandas as pd
import time

def run_origin_004(input_csv, output_npy, epsilon_rel=0.01):
    print(f"🚀 [LUMIN-ORIGIN 004] Starting compression pipeline...")

    # 1. DATA LOADING
    df = pd.read_csv(input_csv)
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1]
    n_rows, n_dims = X.shape

    # 2. SYMMETRIC NORMALIZATION (The Core Principle)
    # We find the Max Absolute to keep the zero centered
    max_abs_y = np.max(np.abs(Y))
    if max_abs_y == 0: max_abs_y = 1.0

    # Work strictly in the [-1, 1] domain
    Y_norm = Y / max_abs_y

    # 3. COMPRESSION ENGINE (Vectorized logic)
    # Epsilon is already relative, so 0.01 means 1% of the current normalized value
    sectors = []
    start_idx = 0
    start_time = time.perf_counter()

    print(f"📦 Processing {n_rows} rows in {n_dims}D space...")

    while start_idx < n_rows:
        end_idx = start_idx + 1
        found_end = False

        # Initial sector law (from start to the next point)
        while end_idx < n_rows:
            # Multi-dimensional Linear Interpolation
            # We predict Y based on the line between start_idx and current end_idx
            if end_idx == start_idx + 1:
                end_idx += 1
                continue

            # Current Law Weights (W) and Bias (B)
            # Y = X*W + B
            # Simplified for segments:
            x_start = X[start_idx]
            x_end = X[end_idx]
            y_start = Y_norm[start_idx]
            y_end = Y_norm[end_idx]

            # Check intermediate points
            test_indices = np.arange(start_idx + 1, end_idx)
            # Vectorized prediction for all intermediate points
            # This is where we check if the sector "absorbs" the points

            # Prediction logic (Linear)
            # t = ratio of distance (0 to 1)
            # Use only the first dimension for ratio if X is ordered
            denom = (x_end[0] - x_start[0]) if (x_end[0] - x_start[0]) != 0 else 1.0
            t = (X[test_indices, 0] - x_start[0]) / denom
            y_pred = y_start + t * (y_end - y_start)

            errors = np.abs(Y_norm[test_indices] - y_pred)
            # Dynamic Relative Margin
            margins = np.abs(Y_norm[test_indices]) * epsilon_rel

            # If any point exceeds its margin, the sector breaks at end_idx - 1
            if np.any(errors > margins):
                found_end = True
                break
            else:
                end_idx += 1

        # Save Sector Data
        actual_end = end_idx - 1 if found_end else n_rows - 1
        if actual_end <= start_idx: actual_end = start_idx + 1

        # Store: Mins, Maxs, Weights (simplified), Bias
        # [mins(D), maxs(D), weights(D), bias(1)]
        mins = np.min(X[start_idx:actual_end+1], axis=0)
        maxs = np.max(X[start_idx:actual_end+1], axis=0)

        # Simple Law Calculation for the sector
        # (In v4, we use the line between boundaries for the law)
        w = np.zeros(n_dims)
        denom = (X[actual_end, 0] - X[start_idx, 0])
        if denom != 0:
            w[0] = (Y_norm[actual_end] - Y_norm[start_idx]) / denom
        bias = Y_norm[start_idx] - w[0] * X[start_idx, 0]

        sector_row = np.concatenate([mins, maxs, w, [bias]])
        sectors.append(sector_row)

        start_idx = actual_end
        if not found_end: break

    # 4. BINARY OUTPUT WITH METADATA HEADER
    sectors_matrix = np.array(sectors)
    # Metadata Row: [MaxAbs, 0, 0, ... 0]
    metadata = np.zeros(sectors_matrix.shape[1])
    metadata[0] = max_abs_y

    final_output = np.vstack([metadata, sectors_matrix])
    np.save(output_npy, final_output)

    total_time = time.perf_counter() - start_time
    print(f"✅ Finished. Sectors created: {len(sectors)} | Time: {total_time:.4f}s")
    print(f"💾 Saved to: {output_npy}")

if __name__ == "__main__":
    # Test with dummy data or real csv
    # run_origin_004("input_data.csv", "origin_map.npy")
    pass
  
