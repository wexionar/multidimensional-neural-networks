# =============================================================
# LUMIN-RESOLUTION: nD Inference Engine (v1.4 C)
# =============================================================
# Project: SLRM-nD (Lumin Core)
# Developers: Alex Kinetic & Gemini
# License: MIT License
# Description: Batch-optimized inference engine with Bounding Box
# logic. Supports single-point and matrix-based resolution.
# =============================================================

import numpy as np
import pandas as pd
import time

class LuminResolution:
    """
    Resolution Engine.
    Vectorized to find Master Sectors and compute laws for
    single points or massive batches.
    """
    def __init__(self, sectors_df=None):
        self.sectors = None
        self.D = 0
        if sectors_df is not None:
            self.set_sectors(sectors_df)

    def set_sectors(self, df):
        """Injects sectors and prepares internal matrices."""
        self.sectors = df.values
        # D is (total_columns - 1) / 3 -> [mins, maxs, weights] + 1 bias
        self.D = (self.sectors.shape[1] - 1) // 3

        # Pre-separar para evitar slicing repetitivo en el bucle
        self.mins = self.sectors[:, :self.D]
        self.maxs = self.sectors[:, self.D:2*self.D]
        self.weights = self.sectors[:, 2*self.D:3*self.D]
        self.biases = self.sectors[:, -1]

    def resolve(self, X_input):
        """
        Main Resolver (Batch Support).
        X_input can be a list [x1, x2...] or a matrix [[x1, x2...], [...]]
        """
        if self.sectors is None:
            raise ValueError("No sectors loaded.")

        # Convertir a 2D para manejo uniforme
        X = np.atleast_2d(X_input)
        results = np.full(X.shape[0], None) # 'None' para el vacío (The Void)

        # Inferencia optimizada
        for i, point in enumerate(X):
            # Verificamos inclusión en todas las hipercajas a la vez
            inside = np.all((point >= self.mins - 1e-9) & (point <= self.maxs + 1e-9), axis=1)
            candidate_indices = np.where(inside)[0]

            if len(candidate_indices) > 0:
                # Si hay solapamiento, el primero es el dueño (v1.4C logic)
                idx = candidate_indices[0]
                results[i] = np.dot(point, self.weights[idx]) + self.biases[idx]

        # Si entró un solo punto, devolver un valor escalar, sino un array
        return results[0] if len(results) == 1 else results

# --- TEST DE ESTRÉS: BATCH RESOLUTION ---
if __name__ == "__main__":
    print("🚀 LUMIN-RESOLUTION v1.4C: Batch Processing Test")

    # 1. Simular sectores maestros (Ej: Ley Y = 2*X1 + 5)
    # 50 Dimensiones, solo 1 Sector Maestro para el test
    D_test = 50
    mock_row = np.concatenate([
        np.full(D_test, -10), # mins
        np.full(D_test, 10),  # maxs
        np.full(D_test, 2),   # weights (W1...W50 = 2)
        [5]                   # bias
    ])

    df_mock = pd.DataFrame([mock_row])
    resolver = LuminResolution(df_mock)

    # 2. Generar un lote masivo de puntos para resolver (10,000 puntos en 50D)
    N_batch = 10000
    batch_points = np.random.uniform(-5, 5, (N_batch, D_test))

    print(f"Resolviendo {N_batch} puntos en {D_test}D...")
    start = time.perf_counter()
    results = resolver.resolve(batch_points)
    end = time.perf_counter()

    print("-" * 50)
    print(f"Tiempo total: {end - start:.4f} s")
    print(f"Velocidad: {N_batch / (end - start):.2f} pts/seg")
    print(f"Ejemplo Resultado (Punto 0): {results[0]}")
    print("-" * 50)
    
