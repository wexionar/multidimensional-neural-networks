# ==========================================
# Project: SLRM-nD (Lumin Core v1.4)
# Developers: Alex & Gemini
# License: MIT License
# ==========================================
import numpy as np
import time

class SLRMLumin:
    """
    SLRM Lumin Core v1.4
    Optimized for high-dimensional hyperspaces using Simplex Sectoring (D+1).
    F1-Vectorized Edition for minimal latency.
    """
    def __init__(self, dimensions):
        self.d = dimensions
        self.dataset = None

    def fit(self, data):
        """Clean and load the dataset into memory."""
        self.dataset = np.array(data)
        # Numerical stability: remove rows with NaNs
        self.dataset = self.dataset[~np.isnan(self.dataset).any(axis=1)]
        print(f"Lumin Core v1.4 (F1): {len(self.dataset)} points loaded.")

    def predict(self, input_point):
        """
        Prediction via Simplex Reconstruction.
        Discards 'opposite' points on each axis to form a minimal surrounding sector.
        """
        if self.dataset is None:
            return "Error: Dataset not loaded."
        
        input_point = np.array(input_point)
        X = self.dataset[:, :-1]
        Y = self.dataset[:, -1]

        # --- VECTORIZED F1 SEARCH ---
        # Calculate differences in a single pass across all points/dimensions
        diffs = input_point - X # Positive values indicate INF nodes
        
        # Binary masks for sectoring
        inf_mask = diffs >= 0
        sup_mask = diffs < 0
        
        # Use extreme values to find the nearest candidates without slow loops
        inf_data = np.where(inf_mask, diffs, -np.inf)
        sup_data = np.where(sup_mask, diffs, np.inf)

        # Retrieve indices for the closest candidates per axis
        idx_inf = np.argmax(inf_data, axis=0)
        idx_sup = np.argmin(sup_data, axis=0)

        # Select candidate nodes
        nodes_inf = self.dataset[idx_inf]
        nodes_sup = self.dataset[idx_sup]

        # --- THE MAGIC OF DISCARDING (Vectorized) ---
        # Compare absolute distances on each axis to determine the winner
        dist_inf = np.abs(input_point - nodes_inf[:, :-1]).diagonal()
        dist_sup = np.abs(nodes_sup[:, :-1] - input_point).diagonal()
        
        # Select node based on proximity: "The closest survives"
        choice_mask = (dist_inf < dist_sup).reshape(-1, 1)
        simplex_nodes = np.where(choice_mask, nodes_inf, nodes_sup)

        # --- SECTOR CLOSURE ---
        # Append the global nearest neighbor to ensure a closed geometric volume
        global_dists = np.linalg.norm(X - input_point, axis=1)
        closest_node = self.dataset[np.argmin(global_dists)]
        
        # Consolidate nodes and remove duplicates
        final_nodes = np.unique(np.vstack([simplex_nodes, closest_node]), axis=0)

        # --- GEOMETRIC INTERPOLATION (Linear Projection) ---
        node_coords = final_nodes[:, :-1]
        node_values = final_nodes[:, -1]
        
        local_dists = np.linalg.norm(node_coords - input_point, axis=1)
        local_dists = np.maximum(local_dists, 1e-10) # Avoid division by zero
        
        inv_dists = 1.0 / local_dists
        weights = inv_dists / np.sum(inv_dists)
        
        return np.dot(weights, node_values)

# --- MILLENNIUM TEST (1,000 Dimensions) ---
if __name__ == "__main__":
    D, P = 1000, 1500
    print(f"Launching Test: Lumin v1.4 in {D}D with {P} points...")

    # Generate Synthetic Data: Sum of Squares
    X_test = np.random.rand(P, D)
    Y_test = np.sum(X_test**2, axis=1).reshape(-1, 1)
    data = np.hstack((X_test, Y_test))

    engine = SLRMLumin(D)
    engine.fit(data)

    target = np.random.rand(D)
    real_val = np.sum(target**2)

    start = time.perf_counter()
    pred = engine.predict(target)
    end = time.perf_counter()

    print("-" * 50)
    print(f"RESULTS F1 EDITION:")
    print(f"REAL: {real_val:.4f} | PRED: {pred:.4f}")
    print(f"ABS ERROR: {abs(real_val - pred):.4f}")
    print(f"LATENCY: {(end - start)*1000:.2f} ms")
    print("-" * 50)
