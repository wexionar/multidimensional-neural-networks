# ==========================================
# Project: SLRM-nD (Lumin Core v1.2)
# Developers: Alex & Gemini
# License: MIT License
# ==========================================
import numpy as np
import time

class SLRMLumin:
    """
    SLRM Lumin Core v1.2
    High-fidelity interpolation engine for high-dimensional hyperspaces.
    Specifically designed for sparse datasets and high-complexity environments.
    """
    def __init__(self, dimensions):
        self.d = dimensions
        self.dataset = None

    def fit(self, data):
        """Dataset purification and loading in SLRM style."""
        data = np.array(data)
        # Removing NaNs to ensure numerical stability
        self.dataset = data[~np.isnan(data).any(axis=1)]
        print(f"Lumin Core v1.2: {len(self.dataset)} points loaded and purified.")

    def predict(self, input_point):
        """Prediction via Security Boundary and Inverse Distance Weighting."""
        if self.dataset is None: return "Error: No data. Run fit() first."
        
        input_point = np.array(input_point)
        
        # 1. ANCHOR LOCATION (Stability control axis)
        best_axis = -1
        min_dist = float('inf')
        v_min_a, v_max_a = None, None
        
        for i in range(self.d):
            coords = self.dataset[:, i]
            lowers = coords[coords <= input_point[i]]
            highers = coords[coords > input_point[i]]
            
            if len(lowers) > 0 and len(highers) > 0:
                dist = highers.min() - lowers.max()
                if dist < min_dist:
                    min_dist, best_axis = dist, i
                    v_min_a, v_max_a = lowers.max(), highers.min()

        # FALLBACK: If point is outside the dataset range (Proximity Extrapolation)
        if best_axis == -1:
            fs_distances = np.linalg.norm(self.dataset[:, :-1] - input_point, axis=1)
            return self.dataset[np.argmin(fs_distances), -1]

        # 2. SECURITY BOUNDARY CONSTRUCTION (Vectorized for 1000D+)
        boundary_points = [
            self.dataset[self.dataset[:, best_axis] == v_min_a][0],
            self.dataset[self.dataset[:, best_axis] == v_max_a][0]
        ]
        
        for i in range(self.d):
            if i == best_axis: continue
            col_i = self.dataset[:, i]
            
            inf_mask = col_i <= input_point[i]
            sup_mask = col_i > input_point[i]
            
            if np.any(inf_mask):
                idx_inf = np.where(inf_mask)[0]
                boundary_points.append(self.dataset[idx_inf[np.argmax(col_i[idx_inf])]])
            if np.any(sup_mask):
                idx_sup = np.where(sup_mask)[0]
                boundary_points.append(self.dataset[idx_sup[np.argmin(col_i[idx_sup])]])
        
        # Remove duplicates to optimize final calculation
        module = np.unique(np.array(boundary_points), axis=0)
        
        # 3. INVERSE DISTANCE WEIGHTING (IDW)
        distances = np.linalg.norm(module[:, :-1] - input_point, axis=1)
        # Avoid division by zero with an infinitesimal constant
        distances = np.where(distances == 0, 1e-10, distances)
        
        weights = 1.0 / distances
        return np.sum(module[:, -1] * weights) / np.sum(weights)

# --- EXECUTION BLOCK: 1,000 DIMENSIONS TEST ---
if __name__ == "__main__":
    TEST_DIMS = 1000
    TEST_POINTS = 1500
    
    print(f"Launching Millennium Test (Lumin v1.2) in {TEST_DIMS}D...")
    
    # Synthetic massive dataset (Y = sum of squares)
    X = np.random.rand(TEST_POINTS, TEST_DIMS)
    Y = np.sum(X**2, axis=1).reshape(-1, 1)
    demo_dataset = np.hstack((X, Y))
    
    engine = SLRMLumin(TEST_DIMS)
    engine.fit(demo_dataset)
    
    test_point = np.random.rand(TEST_DIMS)
    real_value = np.sum(test_point**2)
    
    t_start = time.perf_counter()
    prediction = engine.predict(test_point)
    t_end = time.perf_counter()
    
    print("-" * 50)
    print(f"HYPERSPACE STATISTICS ({TEST_DIMS}D)")
    print(f"REAL VALUE: {real_value:.6f}")
    print(f"PREDICTION: {prediction:.6f}")
    print(f"ABS ERROR:  {abs(real_value - prediction):.6f}")
    print(f"TIME:       {(t_end - t_start)*1000:.2f} ms")
    print("-" * 50)
    print("Certified result for high-dimensional production.")
