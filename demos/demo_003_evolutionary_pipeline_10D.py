# =============================================================
# LUMIN-DEMO 003: The Evolutionary Pipeline (Origin + Resolution)
# =============================================================
# Project: SLRM-nD (Lumin Core)
# Developers: Alex Kinetic & Gemini
# Repository: https://github.com/wexionar/multi-dimensional-neural-networks
# License: MIT License
# Date: 2026-01-19
# Description: High-speed bridge between Origin (Sensory) 
#              and Resolution (Motor) modules. Simulates a 
#              live 10D stream with reality fractures.
# =============================================================

import numpy as np
import pandas as pd
import time

# --- REQUIRED CLASSES (Fast load for testing) ---

class LuminOrigin:
    def __init__(self, epsilon=0.05):
        self.epsilon, self.master_sectors, self.current_sector_nodes, self.D = epsilon, [], [], None
        
    def _calculate_law(self, nodes):
        nodes = np.array(nodes); X, Y = nodes[:, :-1], nodes[:, -1]
        A = np.c_[X, np.ones(X.shape[0])]
        try: 
            res = np.linalg.lstsq(A, Y, rcond=None)[0]; return res[:-1], res[-1]
        except: return None, None
        
    def _close_sector(self):
        if len(self.current_sector_nodes) < 2: return
        nodes = np.array(self.current_sector_nodes)
        W, B = self._calculate_law(nodes)
        if W is not None:
            sector = np.concatenate([np.min(nodes[:, :-1], axis=0), np.max(nodes[:, :-1], axis=0), W, [B]])
            self.master_sectors.append(sector)
            
    def ingest(self, cell):
        cell_np = np.array(cell, dtype=float)
        if self.D is None: self.D = len(cell_np) - 1
        if len(self.current_sector_nodes) < 2: 
            self.current_sector_nodes.append(cell_np.tolist()); return
        W, B = self._calculate_law(self.current_sector_nodes)
        if abs(cell_np[-1] - (np.dot(cell_np[:-1], W) + B)) <= self.epsilon: 
            self.current_sector_nodes.append(cell_np.tolist())
        else:
            self._close_sector()
            self.current_sector_nodes = [self.current_sector_nodes[-1], cell_np.tolist()]
            
    def get_master_df(self):
        self._close_sector()
        cols = [f'X{i}_min' for i in range(self.D)] + [f'X{i}_max' for i in range(self.D)] + [f'W{i}' for i in range(self.D)] + ['Bias']
        return pd.DataFrame(self.master_sectors, columns=cols)

class LuminResolution:
    def __init__(self, df):
        self.sectors = df.values; self.D = (self.sectors.shape[1] - 1) // 3
        self.mins, self.maxs = self.sectors[:, :self.D], self.sectors[:, self.D:2*self.D]
        self.weights, self.biases = self.sectors[:, 2*self.D:3*self.D], self.sectors[:, -1]
        
    def resolve(self, X_input):
        X = np.atleast_2d(X_input); results = np.full(X.shape[0], None)
        for i, point in enumerate(X):
            inside = np.all((point >= self.mins - 1e-9) & (point <= self.maxs + 1e-9), axis=1)
            idx = np.where(inside)[0]
            if len(idx) > 0: 
                results[i] = np.dot(point, self.weights[idx[0]]) + self.biases[idx[0]]
        return results[0] if len(results) == 1 else results

# --- FULL PIPELINE TEST ---

# Config: 5k points for stream training, 10k for inference batch, 10 Dimensions
N_train, N_query, D = 5000, 10000, 10
X = np.linspace(0, 100, N_train).reshape(-1, 1) * np.ones((1, D))
# Non-linear laws with 3 distinct regimes (fractures)
Y = np.where(X[:,0] < 30, X[:,0]*2, np.where(X[:,0] < 70, X[:,0]*(-1.5) + 100, X[:,0]*3 - 200))

print("🏗️  STEP 1: Origin ingesting data stream (Sensory Phase)...")
origin = LuminOrigin(epsilon=0.1)
t0 = time.perf_counter()
for row in np.c_[X, Y]: 
    origin.ingest(row)
master_df = origin.get_master_df()
t_origin = time.perf_counter() - t0
print(f"✅ Sectors Created: {len(master_df)}")
print(f"✅ Learning Speed: {N_train/t_origin:,.2f} pts/sec")

print("\n⚡ STEP 2: Resolution Batch Inference (Motor Phase)...")
resolver = LuminResolution(master_df)
t1 = time.perf_counter()
# Resolving 10,000 points in 10D space
predictions = resolver.resolve(np.random.uniform(0, 100, (N_query, D)))
t_res = time.perf_counter() - t1

print(f"🚀 Execution Throughput: {N_query/t_res:,.2f} pts/sec")
print("-" * 50)
