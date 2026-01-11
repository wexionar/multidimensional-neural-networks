# ==========================================
# Proyecto: SLRM-nD (Nexus Core v1.4)
# Desarrolladores: Alex & Gemini
# Licencia: MIT License
# Lanzamiento: 2026 - Edición Benchmarking
# ==========================================
import numpy as np
import time

class SLRMNexus:
    """
    Nexus Core v1.4: Motor de Plegado Geométrico en Alta Dimensionalidad.
    Introduce la 'Estrategia Lumin' para la selección determinista de ejes.
    """
    def __init__(self, dimensions):
        self.d = dimensions
        self.dataset = None

    def fit(self, data):
        """Purifica y organiza el conjunto de datos para el motor Nexus."""
        data = np.array(data)
        # Escudo v1.2: Eliminar NaNs y Duplicados
        data = data[~np.isnan(data).any(axis=1)]
        _, idx = np.unique(data[:, :-1], axis=0, return_index=True)
        self.dataset = data[idx]
        # Ordenación lexicográfica para la estructura base
        self.dataset = self.dataset[np.lexsort([self.dataset[:, i] for i in range(self.d-1, -1, -1)])]
        print(f"DEBUG: Nexus cargado con {len(self.dataset)} puntos.")

    def _get_quality_roadmap(self, point):
        """
        CRITERIO LUMIN: Calcula la 'Hoja de Ruta de Calidad'.
        Prioriza las dimensiones con los deltas de proximidad más pequeños.
        """
        quality_scores = []
        for i in range(self.d):
            column = self.dataset[:, i]
            lower_bound = column[column <= point[i]]
            upper_bound = column[column >= point[i]]
            
            if len(lower_bound) > 0 and len(upper_bound) > 0:
                delta = upper_bound.min() - lower_bound.max()
            else:
                delta = column.max() - column.min()
            quality_scores.append((delta, i))
        
        # Ordenar por el delta más pequeño (mayor precisión primero)
        return [idx for delta, idx in sorted(quality_scores)]

    def _fold(self, point, data, roadmap, level):
        """Núcleo recursivo guiado por la Hoja de Ruta de Calidad."""
        # Caso Base: Todas las dimensiones plegadas o se alcanzó un solo punto
        if level == len(roadmap) or data.shape[0] <= 1:
            return np.mean(data[:, -1]) if data.size > 0 else 0

        current_dim = roadmap[level]
        x_in = point[current_dim]
        coords = np.unique(data[:, current_dim])

        if len(coords) == 1:
            return self._fold(point, data, roadmap, level + 1)

        # Lógica de Extrapolación de Bordes
        if x_in <= coords[0]:
            x0, x1 = coords[0], coords[1]
        elif x_in >= coords[-1]:
            x0, x1 = coords[-2], coords[-1]
        else:
            x0 = coords[coords <= x_in].max()
            x1 = coords[coords > x_in].min()

        # Peso de interpolación del plegado
        d = (x_in - x0) / (x1 - x0)
        
        # Ramificación usando la hoja de ruta priorizada
        y0 = self._fold(point, data[data[:, current_dim] == x0], roadmap, level + 1)
        y1 = self._fold(point, data[data[:, current_dim] == x1], roadmap, level + 1)

        return y0 * (1 - d) + y1 * d

    def predict(self, point):
        """Predice el resultado usando la Hoja de Ruta optimizada."""
        if self.dataset is None: return "Error: No hay datos"
        roadmap = self._get_quality_roadmap(point)
        return self._fold(point, self.dataset, roadmap, 0)

# --- BLOQUE DE PRUEBA (BENCHMARK) ---
if __name__ == "__main__":
    print("--- Iniciando Nexus v1.4 (Prueba de 1000 Dimensiones) ---")
    
    D = 1000  # Dimensiones
    P = 1000  # Puntos
    data_sim = np.random.rand(P, D + 1)

    engine = SLRMNexus(dimensions=D)
    engine.fit(data_sim)

    input_point = np.random.rand(D)

    start = time.time()
    res = engine.predict(input_point)
    end = time.time()

    print("-" * 40)
    print(f"RESULTADO: {res}")
    print(f"TIEMPO: {(end - start) * 1000:.2f} ms")
    print("-" * 40)
