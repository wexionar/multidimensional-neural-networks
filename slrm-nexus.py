# ==========================================
# Project: SLRM-nD (Nexus Core v1.2)
# Developers: Alex & Gemini
# License: MIT License
# ==========================================

import numpy as np

class SLRMNexus:
    def __init__(self, dimensions):
        self.d = dimensions
        self.dataset = None

    def fit(self, data):
        """
        Cleans, purifies, and organizes the dataset for the Nexus engine.
        """
        # --- SHIELD v1.2: Data Cleaning ---
        # Convert to numpy array and remove rows with NaNs (null values)
        data = np.array(data)
        data = data[~np.isnan(data).any(axis=1)]
        
        # Remove exact duplicates to avoid redundancy
        _, idx = np.unique(data[:, :-1], axis=0, return_index=True)
        self.dataset = data[idx]
        
        # Algebraic Sorting (Nexus navigation map)
        # This sorts by all dimensions to allow recursive neighbor searching
        self.dataset = self.dataset[np.lexsort([self.dataset[:, i] for i in range(self.d-1, -1, -1)])]
        print(f"Nexus Core v1.2: {len(self.dataset)} points purified.")

    def _fold(self, point, data, dim):
        """
        Recursive core that performs the dimensional folding.
        """
        # BASE CASE: We reached the final Y value
        if dim == self.d: return data[0, -1]

        x_in = point[dim]
        coords = np.unique(data[:, dim])
        
        # ALEX'S RULE: Constant logic if no pair is found for slope calculation
        if len(coords) == 1: 
            return self._fold(point, data, dim + 1)
        
        # BORDER EXTRAPOLATION & NEIGHBORHOOD SEARCH
        if x_in <= coords[0]: 
            x0, x1 = coords[0], coords[1]
        elif x_in >= coords[-1]: 
            x0, x1 = coords[-2], coords[-1]
        else:
            x0 = coords[coords <= x_in].max()
            x1 = coords[coords > x_in].min()

        # NEXUS FOLDING ENGINE
        # Distance 'd' for linear interpolation or extrapolation
        d = (x_in - x0) / (x1 - x0)
        
        # Recursive branching
        y0 = self._fold(point, data[data[:, dim] == x0], dim + 1)
        y1 = self._fold(point, data[data[:, dim] == x1], dim + 1)

        # The fundamental folding equation
        return y0 * (1 - d) + y1 * d

    def predict(self, point):
        """
        Predicts the output for a given N-dimensional point.
        """
        if self.dataset is None: return "Error: No data loaded."
        return self._fold(point, self.dataset, 0)
