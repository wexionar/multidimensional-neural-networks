# =============================================================
# LUMIN-ORIGIN: Arquitecto de Conocimiento Estructural nD en Tiempo Real
# =============================================================
# Proyecto: SLRM-nD (Lumin Core)
# Desarrolladores: Alex Kinetic & Gemini
# Versión: 1.0 (Estable)
# Licencia: MIT License
# Descripción: Organismo de auto-estructuración que genera Sectores
#              Maestros a partir de flujos de datos en vivo.
#              La "Fuente de la Verdad" del ecosistema Lumin.
# =============================================================

import numpy as np
import pandas as pd

class LuminOrigin:
    """
    Motor de Origen Geométrico.
    Detecta fracturas de la realidad y sintetiza las leyes fundamentales
    de los datos en Sectores Maestros.
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
            # Resolviendo el hiperplano local (La ley del sector)
            res = np.linalg.lstsq(A, Y, rcond=None)[0]
            return res[:-1], res[-1]
        except: return None, None

    def _close_sector(self):
        if len(self.current_sector_nodes) < 2: return
        nodes = np.array(self.current_sector_nodes)
        W, B = self._calculate_law(nodes)
        if W is not None:
            # Encapsulando la jurisdicción del sector
            sector = np.concatenate([
                np.min(nodes[:, :-1], axis=0),
                np.max(nodes[:, :-1], axis=0),
                W, [B]
            ])
            self.master_sectors.append(sector)

    def ingest(self, cell):
        """Procesa un nuevo nudo multidimensional en el flujo estructural."""
        cell_np = np.array(cell, dtype=float)
        if self.D is None: self.D = len(cell_np) - 1
        
        if len(self.current_sector_nodes) < 2:
            self.current_sector_nodes.append(cell_np.tolist())
            return
            
        W, B = self._calculate_law(self.current_sector_nodes)
        y_pred = np.dot(cell_np[:-1], W) + B
        
        # Verificación de estabilidad vs. Detección de fractura
        if abs(cell_np[-1] - y_pred) <= self.epsilon:
            self.current_sector_nodes.append(cell_np.tolist())
        else:
            self._close_sector()
            # Originando un nuevo sector desde el último punto de fractura
            self.current_sector_nodes = [self.current_sector_nodes[-1], cell_np.tolist()]

    def get_master_df(self):
        """Devuelve el mapa de conocimiento sintetizado."""
        self._close_sector()
        cols = [f'X{i}_min' for i in range(self.D)] + \
               [f'X{i}_max' for i in range(self.D)] + \
               [f'W{i}' for i in range(self.D)] + ['Bias']
        return pd.DataFrame(self.master_sectors, columns=cols)
      
