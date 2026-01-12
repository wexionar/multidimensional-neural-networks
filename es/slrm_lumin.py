# ==========================================
# Proyecto: SLRM-nD (Lumin Core v1.4)
# Desarrolladores: Alex & Gemini
# Licencia: MIT License
# ==========================================
import numpy as np
import time

class SLRMLumin:
    def __init__(self, dimensions):
        self.d = dimensions
        self.dataset = None

    def fit(self, data):
        self.dataset = np.array(data)
        self.dataset = self.dataset[~np.isnan(self.dataset).any(axis=1)]
        print(f"Lumin Core v1.4 (F1): {len(self.dataset)} points loaded.")

    def predict(self, input_point):
        if self.dataset is None: return "Error"
        
        input_point = np.array(input_point)
        X = self.dataset[:, :-1]
        Y = self.dataset[:, -1]

        # --- VECTORIZED F1 SEARCH ---
        # Calculamos las diferencias de una sola pasada para todos los puntos y dimensiones
        diffs = input_point - X # Positivo si el punto del dataset es INF
        
        # Máscaras rápidas
        inf_mask = diffs >= 0
        sup_mask = diffs < 0
        
        # Usamos valores extremos para encontrar los más cercanos sin bucles lentos
        inf_data = np.where(inf_mask, diffs, -np.inf)
        sup_data = np.where(sup_mask, diffs, np.inf)

        # Encontramos los índices de los candidatos más cercanos por cada eje
        idx_inf = np.argmax(inf_data, axis=0)
        idx_sup = np.argmin(sup_data, axis=0)

        # Seleccionamos los candidatos
        nodes_inf = self.dataset[idx_inf]
        nodes_sup = self.dataset[idx_sup]

        # LA MAGIA DEL DESCARTE (Vectorizada)
        # Comparamos distancias absolutas en cada eje para elegir el ganador
        dist_inf = np.abs(input_point - nodes_inf[:, :-1]).diagonal()
        dist_sup = np.abs(nodes_sup[:, :-1] - input_point).diagonal()
        
        # Elegimos el nodo según tu lógica de "el más cercano sobrevive"
        choice_mask = (dist_inf < dist_sup).reshape(-1, 1)
        simplex_nodes = np.where(choice_mask, nodes_inf, nodes_sup)

        # SECTOR CLOSURE (Añadimos el vecino más cercano global)
        global_dists = np.linalg.norm(X - input_point, axis=1)
        closest_node = self.dataset[np.argmin(global_dists)]
        
        # Unimos todo y limpiamos duplicados
        final_nodes = np.unique(np.vstack([simplex_nodes, closest_node]), axis=0)

        # INTERPOLACIÓN GEOMÉTRICA (Linear Projection)
        node_coords = final_nodes[:, :-1]
        node_values = final_nodes[:, -1]
        
        local_dists = np.linalg.norm(node_coords - input_point, axis=1)
        local_dists = np.maximum(local_dists, 1e-10) # Evitar división por cero
        
        inv_dists = 1.0 / local_dists
        weights = inv_dists / np.sum(inv_dists)
        
        return np.dot(weights, node_values)

# --- TEST RÁPIDO ---
if __name__ == "__main__":
    D, P = 1000, 1500
    X_test = np.random.rand(P, D)
    Y_test = np.sum(X_test**2, axis=1).reshape(-1, 1)
    data = np.hstack((X_test, Y_test))

    engine = SLRMLumin(D)
    engine.fit(data)

    target = np.random.rand(D)
    real_val = np.sum(target**2)

    start = time.perf_counter()
    pred = engine.predict(target)
    end = time.perf_counter()

    print(f"\nRESULTS F1 EDITION:")
    print(f"REAL: {real_val:.4f} | PRED: {pred:.4f}")
    print(f"LATENCY: {(end - start)*1000:.2f} ms")
    
