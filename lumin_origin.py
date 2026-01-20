# =============================================================
# LUMIN-ORIGIN: Real-Time nD Structural Knowledge Architect
# =============================================================
# Project: SLRM-nD (Lumin Core)
# Developers: Alex Kinetic & Gemini
# Version: 1.4
# License: MIT License
# Description: Self-structuring organism that generates Master 
#              Sectors from live data streams.
#              The "Source of Truth" for the Lumin ecosystem.
# =============================================================

import numpy as np
import pandas as pd
import time

# --- USER CONFIGURATION ---
MODE = 1       # 1: DIVERSITY (Embraces every fracture)
               # 2: PURITY (Seeks clean laws)
               # NOTE: Any other value will trigger MODE 1 by default.

EPSILON = 0.5  # Structural sensitivity threshold (Fracture)
# ===========================

class LuminOrigin:
    """
    Geometric Origin Engine.
    Detects reality fractures and synthesizes the fundamental 
    laws of data into Master Sectors.
    """
    def __init__(self, epsilon=0.05, mode_type=1):
        self.epsilon = epsilon
        # SAFETY: Pro-diversity shielding
        if mode_type == 2:
            self.mode = 'purity'
            self.mode_label = "PURITY (Mode 2)"
        else:
            self.mode = 'diversity'
            self.mode_label = "DIVERSITY (Mode 1)"
        
        self.master_sectors = []
        self.current_sector_nodes = []
        self.D = None

    def _calculate_law(self, nodes):
        if len(nodes) < 2: return None, None
        nodes_np = np.array(nodes)
        X, Y = nodes_np[:, :-1], nodes_np[:, -1]
        A = np.c_[X, np.ones(X.shape[0])]
        try:
            # Solving the local hyperplane (The sector law)
            res = np.linalg.lstsq(A, Y, rcond=None)[0]
            return res[:-1], res[-1]
        except:
            return None, None

    def _close_sector(self):
        if len(self.current_sector_nodes) < 2: return
        nodes = np.array(self.current_sector_nodes)
        W, B = self._calculate_law(nodes)
        if W is not None:
            # Encapsulating sector jurisdiction
            sector = np.concatenate([
                np.min(nodes[:, :-1], axis=0),
                np.max(nodes[:, :-1], axis=0),
                W, [B]
            ])
            self.master_sectors.append(sector)

    def ingest(self, cell):
        """Processes a new multidimensional node into the structural flow."""
        cell_np = np.array(cell, dtype=float)
        if self.D is None: self.D = len(cell_np) - 1
        
        if len(self.current_sector_nodes) < 2:
            self.current_sector_nodes.append(cell_np.tolist())
            return
            
        W, B = self._calculate_law(self.current_sector_nodes)
        y_pred = np.dot(cell_np[:-1], W) + B
        error = abs(cell_np[-1] - y_pred)
        
        if error <= self.epsilon:
            self.current_sector_nodes.append(cell_np.tolist())
        else:
            self._close_sector()
            if self.mode == 'diversity':
                # Mode A: Maintains last point to ensure continuity
                self.current_sector_nodes = [self.current_sector_nodes[-1], cell_np.tolist()]
            else:
                # Mode B: Resets without dragging previous error
                self.current_sector_nodes = [cell_np.tolist()]

    def get_master_df(self):
        """Returns the synthesized knowledge map."""
        self._close_sector()
        cols = [f'X{i}_min' for i in range(self.D)] + \
               [f'X{i}_max' for i in range(self.D)] + \
               [f'W{i}' for i in range(self.D)] + ['Bias']
        return pd.DataFrame(self.master_sectors, columns=cols)

# --- EXECUTION: STRESS TEST ---
if __name__ == "__main__":
    # Test data generation (50 Dimensions)
    D, N = 50, 2000
    X = np.cumsum(np.random.randn(N, D), axis=0)
    Y = np.where(np.arange(N) < N//2, 
                 np.sum(X, axis=1) * 2, 
                 np.sum(X, axis=1) * -1).reshape(-1, 1)
    stream = np.hstack((X, Y))
    
    # Initialization with user configuration
    origin = LuminOrigin(epsilon=EPSILON, mode_type=MODE)
    
    print(f"🚀 LUMIN-ORIGIN v1.4 | STARTING SYNTHESIS")
    print(f"CONFIG: {origin.mode_label} | EPSILON: {EPSILON}")
    print("-" * 50)
    
    start = time.perf_counter()
    for point in stream:
        origin.ingest(point)
    duration = time.perf_counter() - start
    
    master_df = origin.get_master_df()
    
    print(f"STATUS: Synthesis completed successfully.")
    print(f"DIMENSIONS PROCESSED: {D}D")
    print(f"POINTS ANALYZED: {N}")
    print(f"SECTORS IDENTIFIED: {len(master_df)}")
    print(f"STREAM SPEED: {N / duration:,.2f} pts/sec")
    print(f"TOTAL TIME: {duration:.4f} s")
    print("-" * 50)
    print("Master Sector Map Sample (Top 2):")
    print(master_df.head(2))
