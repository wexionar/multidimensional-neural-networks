# =============================================================
# LUMIN-RESOLUTION: nD Inference Engine (v1.4 B)
# =============================================================
# Project: SLRM-nD (Lumin Core)
# Developers: Alex Kinetic & Gemini
# License: MIT License
# Date: 2026-01-16
# Description: Ultra-fast inference engine using Bounding Box 
#              logic for Master Sector inclusion.
# =============================================================

import numpy as np
import pandas as pd

class LuminResolution:
    """
    Resolution Engine.
    Uses vectorization to find the corresponding Master Sector
    and computes the linear law instantly.
    """
    def __init__(self, master_sectors_path=None):
        self.sectors = None
        self.D = 0
        if master_sectors_path:
            self.load_sectors(master_sectors_path)

    def load_sectors(self, path):
        """Loads synthesized sectors from a CSV file."""
        df = pd.read_csv(path)
        self.sectors = df.values
        # D is (total_columns - 1) / 3 -> [mins, maxs, weights] + 1 bias
        self.D = (self.sectors.shape[1] - 1) // 3
        
    def set_sectors_from_df(self, df):
        """Allows direct injection of a DataFrame."""
        self.sectors = df.values
        self.D = (self.sectors.shape[1] - 1) // 3

    def resolve(self, X_input):
        """
        Core Inference: Finds the sector and applies the law.
        Uses NumPy broadcasting for maximum speed.
        """
        if self.sectors is None:
            raise ValueError("No sectors loaded. Compile or load data first.")

        X = np.array(X_input)
        
        # 1. Extract Bounds (Mins and Maxs)
        mins = self.sectors[:, :self.D]
        maxs = self.sectors[:, self.D:2*self.D]
        
        # 2. Bounding Box Check (Vectorized)
        # Numerical stability epsilon added for floating point comparisons
        inside = np.all((X >= mins - 1e-9) & (X <= maxs + 1e-9), axis=1)
        
        # 3. Filter candidate sectors
        candidate_indices = np.where(inside)[0]
        
        if len(candidate_indices) == 0:
            return None # Unknown territory (The Void)
        
        # 4. Compute Linear Law (Deterministic execution)
        idx = candidate_indices[0]
        weights = self.sectors[idx, 2*self.D:3*self.D]
        bias = self.sectors[idx, -1]
        
        return np.dot(X, weights) + bias

# --- DEMO: RESOLUTION SPEED TEST ---
if __name__ == "__main__":
    print("🚀 LUMIN-RESOLUTION: Testing Instant Inference...")
    
    # Mock data to simulate a Master Sector (Law: Y = X1 + X2 + 10)
    # Format: [MinX1, MinX2, MaxX1, MaxX2, W1, W2, Bias]
    mock_data = {
        'X0_min': [0], 'X1_min': [0],
        'X0_max': [10], 'X1_max': [10],
        'W0': [1], 'W1': [1],
        'Bias': [10]
    }
    df_mock = pd.DataFrame(mock_data)
    
    resolver = LuminResolution()
    resolver.set_sectors_from_df(df_mock)
    
    test_point = [5, 5]
    result = resolver.resolve(test_point)
    
    print(f"Point: {test_point}")
    print(f"Resolved Value: {result} (Expected: 20)")
    
    # Testing "The Void" (Out of bounds)
    void_point = [15, 15]
    print(f"Void Point: {void_point} -> Result: {resolver.resolve(void_point)}")
  
