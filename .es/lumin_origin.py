# =============================================================
# LUMIN-ORIGIN: Arquitecto de Conocimiento Estructural nD en Tiempo Real
# =============================================================
# Proyecto: SLRM-nD (Lumin Core)
# Desarrolladores: Alex Kinetic & Gemini
# Versión: 1.4
# Licencia: MIT License
# Descripción: Organismo de auto-estructuración que genera Sectores 
#              Maestros a partir de flujos de datos en vivo.
#              La "Fuente de la Verdad" del ecosistema Lumin.
# =============================================================

import numpy as np
import pandas as pd
import time

# --- CONFIGURACIÓN DE USUARIO ---
MODO = 1       # 1: DIVERSIDAD (Abraza cada fractura)
               # 2: PUREZA (Busca leyes limpias)
               # NOTA: Cualquier otro valor activará el MODO 1 por defecto.

EPSILON = 0.5  # Umbral de sensibilidad estructural (Fractura)
# ================================

class LuminOrigin:
    """
    Geometric Origin Engine.
    Detects reality fractures and synthesizes the fundamental 
    laws of data into Master Sectors.
    """
    def __init__(self, epsilon=0.05, mode_type=1):
        self.epsilon = epsilon
        # SEGURIDAD: Blindaje pro-diversidad
        if mode_type == 2:
            self.mode = 'purity'
            self.mode_label = "PUREZA (Modo 2)"
        else:
            self.mode = 'diversity'
            self.mode_label = "DIVERSIDAD (Modo 1)"
        
        self.master_sectors = []
        self.current_sector_nodes = []
        self.D = None

    def _calculate_law(self, nodes):
        if len(nodes) < 2: return None, None
        nodes_np = np.array(nodes)
        X, Y = nodes_np[:, :-1], nodes_np[:, -1]
        A = np.c_[X, np.ones(X.shape[0])]
        try:
            res = np.linalg.lstsq(A, Y, rcond=None)[0]
            return res[:-1], res[-1]
        except:
            return None, None

    def _close_sector(self):
        if len(self.current_sector_nodes) < 2: return
        nodes = np.array(self.current_sector_nodes)
        W, B = self._calculate_law(nodes)
        if W is not None:
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
                # Modo A: Mantiene el último punto para asegurar continuidad
                self.current_sector_nodes = [self.current_sector_nodes[-1], cell_np.tolist()]
            else:
                # Modo B: Reinicia sin arrastrar el error anterior
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
    # Generación de datos de prueba (50 Dimensiones)
    D, N = 50, 2000
    X = np.cumsum(np.random.randn(N, D), axis=0)
    Y = np.where(np.arange(N) < N//2, 
                 np.sum(X, axis=1) * 2, 
                 np.sum(X, axis=1) * -1).reshape(-1, 1)
    stream = np.hstack((X, Y))
    
    # Inicialización con configuración del usuario
    origin = LuminOrigin(epsilon=EPSILON, mode_type=MODO)
    
    print(f"🚀 LUMIN-ORIGIN v1.4 | INICIANDO SÍNTESIS")
    print(f"CONFIGURACIÓN: {origin.mode_label} | EPSILON: {EPSILON}")
    print("-" * 50)
    
    start = time.perf_counter()
    for point in stream:
        origin.ingest(point)
    duration = time.perf_counter() - start
    
    master_df = origin.get_master_df()
    
    print(f"ESTADO: Síntesis completada con éxito.")
    print(f"DIMENSIONES PROCESADAS: {D}D")
    print(f"PUNTOS ANALIZADOS: {N}")
    print(f"SECTORES IDENTIFICADOS: {len(master_df)}")
    print(f"VELOCIDAD DE FLUJO: {N / duration:,.2f} pts/sec")
    print(f"TIEMPO TOTAL: {duration:.4f} s")
    print("-" * 50)
    print("Muestra del Mapa de Sectores (Top 2):")
    print(master_df.head(2))
