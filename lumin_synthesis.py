# =============================================================
# LUMIN-SYNTHESIS: n-Dimensional Lexicographical Compiler
# =============================================================
# Project: SLRM-nD (Lumin Core v1.4)
# Developers: Alex & Gemini
# License: MIT License
# Date: 2026-01-15
# Description: Synthesizes massive datasets into minimal 
#              Master Sectors using deductive geometry.
# =============================================================

import numpy as np
import pandas as pd
import time

class LuminSynthesis:
    """
    Lumin Synthesis Engine: Converts raw data into master 
    geometric rules (Hyperplanes) via lexicographical sweep.
    """
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.master_sectors = None

    def compile(self, df):
        """
        Deduces mathematical laws from the dataset and compresses them.
        """
        start_time = time.perf_counter()
        data = df.values
        
        # Lexicographical Sort: Establishes the logical sequence for sector growth
        # Sorts by X1, then X2... up to XD
        sorted_data = data[np.lexsort(data[:, :-1].T[::-1])]
        D = data.shape[1] - 1
        
        synthesis_rows = []
        base_idx = 0
        
        while base_idx < len(sorted_data):
            last_valid_idx = base_idx
            best_W, best_B = None, None
            
            # Sector Expansion (Logos-nD Strategy)
            for i in range(base_idx + 1, len(sorted_data)):
                current_group = sorted_data[base_idx : i + 1]
                X_g = current_group[:, :-1]
                Y_g = current_group[:, -1]
                
                # Deductive Training (Direct Least Squares)
                A = np.c_[X_g, np.ones(X_g.shape[0])]
                try:
                    # Solving for the optimal hyperplane for this group
                    result, _, _, _ = np.linalg.lstsq(A, Y_g, rcond=None)
                    W_tmp, B_tmp = result[:-1], result[-1]
                    
                    # Fidelity Validation via Epsilon
                    Y_pred = np.dot(X_g, W_tmp) + B_tmp
                    if np.all(np.abs(Y_g - Y_pred) <= self.epsilon):
                        best_W, best_B = W_tmp, B_tmp
                        last_valid_idx = i
                    else:
                        break
                except:
                    break
            
            # Case for isolated points or end of sequence
            if best_W is None:
                best_W = np.zeros(D)
                best_B = sorted_data[base_idx][-1]

            # Register Master Sector: [Boundaries, Weights, Bias]
            row = np.concatenate([
                sorted_data[base_idx][:-1], 
                sorted_data[last_valid_idx][:-1], 
                best_W, 
                [best_B]
            ])
            synthesis_rows.append(row)
            base_idx = last_valid_idx + 1

        # Naming columns for the Master CSV
        cols = [f'X{i}_start' for i in range(1, D+1)] + \
               [f'X{i}_end' for i in range(1, D+1)] + \
               [f'W{i}' for i in range(1, D+1)] + ['Bias']
        
        self.master_sectors = pd.DataFrame(synthesis_rows, columns=cols)
        duration = time.perf_counter() - start_time
        return self.master_sectors, duration

    def predict(self, X_input):
        """
        Instant Inference using synthesized knowledge.
        """
        X_input = np.array(X_input)
        for _, sector in self.master_sectors.iterrows():
            D = (len(sector) - 1) // 3
            x_start = sector.iloc[0:D].values
            x_end = sector.iloc[D:2*D].values
            
            # Check if input point falls within sector boundaries
            if np.all(X_input >= x_start) and np.all(X_input <= x_end):
                W = sector.iloc[2*D:3*D].values
                B = sector.iloc[-1]
                return np.dot(X_input, W) + B
        return None

# --- Validation and Example Execution ---
if __name__ == "__main__":
    print("Initializing Lumin-Synthesis engine...")
    
    # Example: Y = 3*X1 + 5*X2 + 10 + noise
    N = 500
    X_test = np.random.uniform(0, 100, (N, 2))
    Y_test = 3*X_test[:,0] + 5*X_test[:,1] + 10 + np.random.uniform(-0.05, 0.05, N)
    df_example = pd.DataFrame(np.c_[X_test, Y_test], columns=['X1', 'X2', 'Y'])

    # Compile dataset
    compiler = LuminSynthesis(epsilon=0.1)
    master_df, duration = compiler.compile(df_example)

    print(f"\n[OK] Compilation finished in {duration:.4f}s")
    print(f"[INFO] Master Sectors created: {len(master_df)}")
    print(f"[INFO] Compression Rate: {((len(df_example)-len(master_df))/len(df_example))*100:.2f}%")
  
