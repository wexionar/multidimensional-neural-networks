# =============================================================
# LUMIN-DEMO 004: The Symmetric Origin Engine (High-Speed)
# =============================================================
# Project: SLRM-nD (Lumin Core)
# Developers: Alex Kinetic & Gemini
# Repository: https://github.com/wexionar/multi-dimensional-neural-networks
# License: MIT License
# Date: 2026-01-22
# Description: High-speed Symmetric Compression Engine.
#              Implements Max-Abs Scaling and Dual-Epsilon logic
#              to preserve geometric integrity across nD space.
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
    
    # 1. DATA LOADING & AUTO-SORTING
    df = pd.read_csv(input_csv)
    df = df.sort_values(by=df.columns[0]).reset_index(drop=True)
    
    data = df.values
    X, Y = data[:, :-1], data[:, -1]
    n_rows, n_dims = X.shape

    # 2. SYMMETRIC NORMALIZATION (Preserving Origin at 0.0)
    max_abs_y = np.max(np.abs(Y)) if np.max(np.abs(Y)) != 0 else 1.0
    Y_norm = Y / max_abs_y

    # 3. COMPRESSION ENGINE (nD Hyperplane Synthesis)
    sectors = []
    start_idx = 0

    while start_idx < n_rows:
        end_idx = start_idx + 2 
        if end_idx > n_rows: break
        
        W_final, B_final = None, None
        
        while end_idx <= n_rows:
            X_slice = X[start_idx:end_idx]
            Y_slice = Y_norm[start_idx:end_idx]
            
            # Solve local hyperplane using Least Squares
            A = np.c_[X_slice, np.ones(X_slice.shape[0])]
            try:
                res, _, _, _ = np.linalg.lstsq(A, Y_slice, rcond=None)
                W_current, B_current = res[:-1], res[-1]
                
                # Structural Integrity Check
                Y_pred = np.dot(X_slice, W_current) + B_current
                errors = np.abs(Y_slice - Y_pred)
                
                margins = (np.abs(Y_slice) * EPSILON_VAL) if EPSILON_TYPE == 'rel' else EPSILON_VAL
                
                if np.any(errors > margins):
                    break
                else:
                    W_final, B_final = W_current, B_current
                    end_idx += 1
            except:
                break
        
        actual_end = end_idx - 1
        
        # Bounding Box Jurisdiction
        mins = np.min(X[start_idx:actual_end+1], axis=0)
        maxs = np.max(X[start_idx:actual_end+1], axis=0)
        
        if W_final is None:
            W_final, B_final = np.zeros(n_dims), Y_norm[start_idx]

        sectors.append(np.concatenate([mins, maxs, W_final, [B_final]]))
        
        if PRINT_SUMMARY:
            print(f"Sector {len(sectors)}: Points {actual_end - start_idx + 1}")

        start_idx = actual_end if STRUCT_MODE == 1 else actual_end + 1
        if start_idx >= n_rows - 1: break

    # 4. BINARY EXPORT WITH COMPLETE DNA (Row 0)
    sectors_matrix = np.array(sectors)
    metadata = np.zeros(sectors_matrix.shape[1])
    metadata[0] = max_abs_y                       # Real-world scale factor
    metadata[1] = 1 if EPSILON_TYPE == 'abs' else 0 # Logic flag
    metadata[2] = EPSILON_VAL                     # Stored threshold
    
    np.save(output_npy, np.vstack([metadata, sectors_matrix]))
    
    exec_time = time.perf_counter() - start_time
    print(f"✅ [LUMIN-ORIGIN 004] DNA Saved: Scale={max_abs_y} | Logic={EPSILON_TYPE} | Val={EPSILON_VAL}")

if __name__ == "__main__":
    pass
    
