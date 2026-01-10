# ==========================================
# Proyecto: SLRM-nD (Nexus Core v1.2)
# Desarrolladores: Alex & Gemini
# Licencia: MIT License
# ==========================================

import numpy as np

class SLRMNexus:
    def __init__(self, dimensiones):
        self.d = dimensiones
        self.dataset = None

    def fit(self, data):
        # --- ARMADURA v1.2: Limpieza de porquería ---
        # Convertimos a array y quitamos filas con NaNs (nulos)
        data = np.array(data)
        data = data[~np.isnan(data).any(axis=1)]
        
        # Eliminamos duplicados y ordenamos algebraicamente
        _, idx = np.unique(data[:, :-1], axis=0, return_index=True)
        self.dataset = data[idx]
        self.dataset = self.dataset[np.lexsort([self.dataset[:, i] for i in range(self.d-1, -1, -1)])]
        print(f"Nexus Core v1.2: {len(self.dataset)} puntos purificados.")

    def _plegar(self, punto, data, dim):
        if dim == self.d: return data[0, -1]

        x_in = punto[dim]
        coords = np.unique(data[:, dim])
        
        # Regla Alex: Constante si no hay pareja
        if len(coords) == 1: 
            return self._plegar(punto, data, dim + 1)
        
        # Vecindarios y Extrapolación
        if x_in <= coords[0]: x0, x1 = coords[0], coords[1]
        elif x_in >= coords[-1]: x0, x1 = coords[-2], coords[-1]
        else:
            x0 = coords[coords <= x_in].max()
            x1 = coords[coords > x_in].min()

        d = (x_in - x0) / (x1 - x0)
        y0 = self._plegar(punto, data[data[:, dim] == x0], dim + 1)
        y1 = self._plegar(punto, data[data[:, dim] == x1], dim + 1)

        return y0 * (1 - d) + y1 * d

    def predict(self, punto):
        if self.dataset is None: return "Error: Sin datos."
        return self._plegar(punto, self.dataset, 0)
