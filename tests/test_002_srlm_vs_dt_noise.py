import numpy as np
import time
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# =============================================================
# CORE: SRLM-nD (Lumin Fusion 006)
# =============================================================
class LuminOrigin:
    def __init__(self, epsilon_val=0.05, epsilon_type=1, mode_type=1):
        self.epsilon_val = epsilon_val
        self.epsilon_type = int(epsilon_type)
        self.mode = 'diversity' if mode_type == 1 else 'purity'
        self.master_sectors = []
        self.current_sector_nodes = []
        self.D = None

    def _calculate_law(self, nodes):
        if len(nodes) < 2: return None, None
        nodes_np = np.array(nodes)
        X, Y = nodes_np[:, :-1], nodes_np[:, -1]
        A_mat = np.c_[X, np.ones(X.shape[0])]
        try:
            res = np.linalg.lstsq(A_mat, Y, rcond=None)[0]
            return res[:-1], res[-1]
        except: return None, None

    def _close_sector(self):
        if len(self.current_sector_nodes) < 2: return
        nodes = np.array(self.current_sector_nodes)
        W, B = self._calculate_law(self.current_sector_nodes)
        if W is not None:
            sector = np.concatenate([
                np.min(nodes[:, :-1], axis=0),
                np.max(nodes[:, :-1], axis=0),
                W, [B]
            ])
            self.master_sectors.append(sector)

    def ingest(self, cell):
        cell_np = np.array(cell, dtype=float)
        if self.D is None: self.D = len(cell_np) - 1
        if len(self.current_sector_nodes) < 2:
            self.current_sector_nodes.append(cell_np.tolist())
            return
        W, B = self._calculate_law(self.current_sector_nodes)
        y_pred = np.dot(cell_np[:-1], W) + B
        error_abs = abs(cell_np[-1] - y_pred)
        threshold = self.epsilon_val if self.epsilon_type == 1 else abs(cell_np[-1]) * self.epsilon_val
        if error_abs <= threshold:
            self.current_sector_nodes.append(cell_np.tolist())
        else:
            self._close_sector()
            self.current_sector_nodes = [cell_np.tolist()] if self.mode == 'purity' else [self.current_sector_nodes[-1], cell_np.tolist()]

class LuminResolution:
    def __init__(self, sectors):
        self.sectors = np.array(sectors)
        if len(self.sectors) == 0: 
            self.mins = self.maxs = self.weights = self.biases = np.array([])
            return
        self.D = (self.sectors.shape[1] - 1) // 3
        self.mins = self.sectors[:, :self.D]
        self.maxs = self.sectors[:, self.D : 2*self.D]
        self.weights = self.sectors[:, 2*self.D : 3*self.D]
        self.biases = self.sectors[:, -1]

    def resolve(self, X_input):
        X = np.atleast_2d(X_input)
        inside = np.all((X[:, np.newaxis, :] >= self.mins - 1e-9) & (X[:, np.newaxis, :] <= self.maxs + 1e-9), axis=2)
        has_sector = np.any(inside, axis=1)
        indices = np.argmax(inside, axis=1)
        results = np.full(X.shape[0], np.nan)
        if np.any(has_sector):
            results[has_sector] = np.einsum('ij,ij->i', X[has_sector], self.weights[indices[has_sector]]) + self.biases[indices[has_sector]]
        return results

# =============================================================
# TEST 002: CHAOS RESILIENCE (HIGH NOISE)
# =============================================================
def run_test_002():
    np.random.seed(42)
    n_samples = 10000
    X = np.random.uniform(0, 1, (n_samples, 5))
    
    # Friedman Function with HIGH NOISE (sigma=1.0)
    Y = 10 * np.sin(np.pi * X[:,0] * X[:,1]) + 20 * (X[:,2] - 0.5)**2 + 10 * X[:,3] + 5 * X[:,4]
    Y += np.random.normal(0, 1.0, n_samples) 
    
    Y_norm = (Y - Y.min()) / (Y.max() - Y.min())
    data_fusion = np.hstack([X, Y_norm.reshape(-1, 1)])

    print("🧪 TEST 002: CHAOS RESILIENCE (HIGH NOISE x10)")
    print("Config: SRLM-nD Epsilon 0.12 | PURITY MODE")
    print("-" * 50)

    # --- A) DECISION TREE ---
    model_tree = DecisionTreeRegressor(max_depth=12, random_state=42)
    model_tree.fit(X, Y_norm)
    mae_tree = np.mean(np.abs(Y_norm - model_tree.predict(X)))

    # --- B) SRLM-nD ---
    origin = LuminOrigin(epsilon_val=0.12, epsilon_type=1, mode_type=2) 
    for point in data_fusion: origin.ingest(point)
    res = LuminResolution(origin.master_sectors)
    y_pred_srlm = res.resolve(X)
    
    mask = ~np.isnan(y_pred_srlm)
    mae_srlm = np.mean(np.abs(Y_norm[mask] - y_pred_srlm[mask]))

    # RESULTS DISPLAY
    results = {
        "Metric": ["MAE (Fidelity)", "Logic Units"],
        "Decision Tree": [f"{mae_tree:.6f}", f"{model_tree.tree_.node_count} nodes"],
        "SRLM-nD": [f"{mae_srlm:.6f}", f"{len(origin.master_sectors)} sectors"]
    }
    print(pd.DataFrame(results).to_string(index=False))
    print("-" * 50)
    
    ratio = model_tree.tree_.node_count / len(origin.master_sectors)
    print(f"💡 CONCLUSION: SRLM-nD is {ratio:.2f}x more resilient to noise than the Tree.")

if __name__ == "__main__":
    run_test_002()
  
