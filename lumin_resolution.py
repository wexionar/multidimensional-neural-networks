# =============================================================
# LUMIN-RESOLUTION: nD Inference Engine (v1.4 C)
# =============================================================
# Project: SLRM-nD (Lumin Core)
# Developers: Alex Kinetic & Gemini
# License: MIT License
# Description: Batch-optimized inference engine with Bounding Box
# logic. Supports single-point and matrix-based resolution.
# =============================================================

import numpy as np
import pandas as pd
import time

class LuminResolution:
    """
    Resolution Engine.
    Vectorized to find Master Sectors and compute laws for
    single points or massive batches.
    """
    def __init__(self, sectors_df=None):
        self.sectors = None
        self.D = 0
        if sectors_df is not None:
            self.set_sectors(sectors_df)

    def set_sectors(self, df):
        """Injects sectors and prepares internal matrices."""
        self.sectors = df.values
        # D is (total_columns - 1) / 3 -> [mins, maxs, weights] + 1 bias
        self.D = (self.sectors.shape[1] - 1) // 3

        # Pre-separate to avoid repetitive slicing in the loop
        self.mins = self.sectors[:, :self.D]
        self.maxs = self.sectors[:, self.D:2*self.D]
        self.weights = self.sectors[:, 2*self.D:3*self.D]
        self.biases = self.sectors[:, -1]

    def resolve(self, X_input):
        """
        Main Resolver (Batch Support).
        X_input can be a list [x1, x2...] or a matrix [[x1, x2...], [...]]
        """
        if self.sectors is None:
            raise ValueError("No sectors loaded.")

        # Convert to 2D for uniform handling
        X = np.atleast_2d(X_input)
        results = np.full(X.shape[0], None) # 'None' for the empty space (The Void)

        # Optimized inference
        for i, point in enumerate(X):
            # Verify inclusion in all hyperboxes at once
            inside = np.all((point >= self.mins - 1e-9) & (point <= self.maxs + 1e-9), axis=1)
            candidate_indices = np.where(inside)[0]

            if len(candidate_indices) > 0:
                # If there is overlap, the first one is the owner (v1.4C logic)
                idx = candidate_indices[0]
                results[i] = np.dot(point, self.weights[idx]) + self.biases[idx]

        # If a single point was entered, return a scalar value, otherwise an array
        return results[0] if len(results) == 1 else results

# --- STRESS TEST: BATCH RESOLUTION ---
if __name__ == "__main__":
    print("🚀 LUMIN-RESOLUTION v1.4C: Batch Processing Test")

    # 1. Simulate master sectors (e.g., Law Y = 2*X1 + 5)
    # 50 Dimensions, only 1 Master Sector for the test
    D_test = 50
    mock_row = np.concatenate([
        np.full(D_test, -10), # mins
        np.full(D_test, 10),  # maxs
        np.full(D_test, 2),   # weights (W1...W50 = 2)
        [5]                   # bias
    ])

    df_mock = pd.DataFrame([mock_row])
    resolver = LuminResolution(df_mock)

    # 2. Generate a massive batch of points to resolve (10,000 points in 50D)
    N_batch = 10000
    batch_points = np.random.uniform(-5, 5, (N_batch, D_test))

    print(f"Resolving {N_batch} points in {D_test}D...")
    start = time.perf_counter()
    results = resolver.resolve(batch_points)
    end = time.perf_counter()

    print("-" * 50)
    print(f"Total time: {end - start:.4f} s")
    print(f"Speed: {N_batch / (end - start):.2f} pts/sec")
    print(f"Result Example (Point 0): {results[0]}")
    print("-" * 50)
    
