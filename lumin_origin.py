# =============================================================
# LUMIN-ORIGIN: Real-Time nD Structural Knowledge Architect
# =============================================================
# Project: SLRM-nD (Lumin Core)
# Developers: Alex Kinetic & Gemini
# Version: 1.0 (Stable)
# License: MIT License
# Description: Self-structuring organism that generates Master 
#              Sectors from live data streams. 
#              The "Source of Truth" for the Lumin Ecosystem.
# =============================================================

import numpy as np
import pandas as pd

class LuminOrigin:
    """
    Geometric Origin Engine.
    Detects reality fractures and synthesizes the fundamental 
    laws of data into Master Sectors.
    """
    def __init__(self, epsilon=0.05):
        self.epsilon = epsilon
        self.master_sectors = []
        self.current_sector_nodes = []
        self.D = None

    def _calculate_law(self, nodes):
        nodes = np.array(nodes)
        X, Y = nodes[:, :-1], nodes[:, -1]
        A = np.c_[X, np.ones(X.shape[0])]
        try:
            # Solving the local hyperplane (The law of the sector)
            res = np.linalg.lstsq(A, Y, rcond=None)[0]
            return res[:-1], res[-1]
        except: return None, None

    def _close_sector(self):
        if len(self.current_sector_nodes) < 2: return
        nodes = np.array(self.current_sector_nodes)
        W, B = self._calculate_law(nodes)
        if W is not None:
            # Encapsulating the sector's jurisdiction
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
        
        # Stability check vs. Fracture detection
        if abs(cell_np[-1] - y_pred) <= self.epsilon:
            self.current_sector_nodes.append(cell_np.tolist())
        else:
            self._close_sector()
            # Originating a new sector from the last fracture point
            self.current_sector_nodes = [self.current_sector_nodes[-1], cell_np.tolist()]

    def get_master_df(self):
        """Returns the synthesized knowledge map."""
        self._close_sector()
        cols = [f'X{i}_min' for i in range(self.D)] + \
               [f'X{i}_max' for i in range(self.D)] + \
               [f'W{i}' for i in range(self.D)] + ['Bias']
        return pd.DataFrame(self.master_sectors, columns=cols)
      
