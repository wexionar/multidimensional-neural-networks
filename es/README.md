# Modelo de Regresión Lineal Segmentada para Redes Neuronales Multi-dimensionales (SLRM-nD)

**Desarrolladores:** Alex & Gemini  
**Licencia:** MIT License  

---

## Parte A: Nexus Core (Motor de Plegado Geométrico)

**Descripción general:** SLRM-nD Nexus es un motor de alto rendimiento diseñado para resolver problemas **N-Dimensionales** mediante el plegado recursivo de vecindarios.

**Versión Actual:** 1.4 (Estrategia Lumin Integrada)

### Características Clave (Nexus)
* Precisión Absoluta: Tasa de error 0.0 en sistemas lineales estructurados.
* Plegado Determinista: Priorización de ejes de alta velocidad basada en la densidad de datos (Estrategia Lumin).

### Benchmark de Nexus (1000D)
```text
Launching Nexus v1.4 Test (1000D)...
Nexus loaded with 1500 points.
PREDICTION RESULT: 0.6227576180933394
EXECUTION TIME: 40.99 ms
```

### Inicio Rápido (Nexus)
```python
from slrm_nexus import SLRMNexus
import numpy as np

# Inicializar para 1000 dimensiones
model = SLRMNexus(dimensions=1000)

# Limpiar y cargar datos
model.fit(tus_datos)

# Predecir en milisegundos
result = model.predict(tu_punto)
```

## Anexo A: Nexus-Memory v1.4
Motor de computación de alta dimensión (1000D+) basado en deducción geométrica y almacenamiento persistente Memory-C.

**Script:** `nexus_memory.py`

### Rendimiento (Benchmarking 2026)
- Dataset: 240,000 puntos @ 1000 dimensiones.
- Deducción Inicial (Nexus Core): ~726.18 ms.
- Recobro (Memory-C): 0.01 ms.
- Eficiencia: Factor de aceleración constante >67,000x.

*Sin entrenamiento. Sin GPU. Geometría pura. Latencia cero.*

---

## Parte B: Lumin Core (Motor de Sectorización Simplex)

**Descripción general:** Lumin es la evolución especializada para hiperespacios dispersos de alta dimensión. Utiliza **Sectorización Simplex (D+1)** para descartar puntos opuestos en cada eje, garantizando el cierre geométrico incluso en entornos extremadamente dispersos.

**Versión Actual:** 1.4 (Vectorización F1)

### Millennium Benchmark (Lumin v1.4)
```text
Launching Test: Lumin v1.4 in 1000D with 1500 points...
Lumin Core v1.4 (F1): 1500 points loaded.
RESULTS F1 EDITION:
REAL: 336.6912
PRED: 333.6967
ABS ERROR: 2.9945
LATENCY: 89.17 ms
```

### Inicio Rápido (Lumin)
```python
from slrm_lumin import SLRMLumin
import numpy as np

# Inicializar para 1000 dimensiones
model = SLRMLumin(dimensions=1000)

# Cargar datos y Predecir
model.fit(datos_1000d)
result = model.predict(punto_1000d)
```

## Anexo B: Lumin-Memory v1.4
Motor de alta dimensión escalable que utiliza indexación espacial (cKDTree) y recobro persistente Memory-C.

**Script:** `lumin_memory.py`

### Rendimiento (Benchmarking 2026)
- Dataset: 240,000 puntos @ 1000 dimensiones.
- Deducción Inicial (Lumin Core): ~586.68 ms.
- Recobro (Memory-C): 0.01 ms.
- Eficiencia: Factor de aceleración constante >40,000x.

*Sin entrenamiento. Sin GPU. Geometría pura. Latencia cero.*

---

## Parte C: El Puente de Identidad (Lumin-to-ReLU)

**Descripción general:** El "Bridge" es un traductor matemático que convierte los sectores de Símplex geométricos de Lumin en arquitecturas estándar de Redes Neuronales. Demuestra la identidad entre un modelo lineal local basado en Símplex y una red ReLU de una sola capa.

**Script:** `lumin_to_relu.py`

### Benchmark del Puente (Identidad 1000D)
```text
--- LUMIN TO RELU BRIDGE ---
Dimensions: 1000
Latency: 2071.61 us

--- ReLU Equation (1000 terms) ---
Y = bias + (w1)*ReLU(x1) + ... + (w1000)*ReLU(x1000)

--- MATHEMATICAL TRUTH TEST ---
Result: -60.0137615537
Error:  0.0e+00
```

### Concepto Clave
El Puente demuestra que SLRM-nD no es solo una alternativa al Deep Learning, sino un método determinista para **inicializar y estabilizar** capas de Redes Neuronales de alta dimensión con cero error de aproximación.

---

## Parte D: Observaciones Técnicas y Comparativa

### Nexus vs. Lumin: ¿Cuál utilizar?

* **Nexus (Plegado Geométrico):** La complejidad se relaciona con el potencial de coordenadas (2^d). Es ideal para conjuntos de datos densos y estructurados donde se requiere perfección matemática. Optimizado para precisión absoluta en espacios de densidad media a alta.

* **Lumin (Sectorización Simplex):** La complejidad se relaciona con la sectorización lineal (d + 1). Es la opción especializada para la "Maldición de la Dimensionalidad" en entornos muy dispersos. Asegura un volumen geométrico cerrado seleccionando solo los nodos circundantes más relevantes por eje.

### Rendimiento General
Ambos motores están construidos sobre NumPy y Scipy, garantizando operaciones matriciales de alta velocidad. No requieren GPUs ni ciclos pesados de entrenamiento, lo que los convierte en la alternativa ligera perfecta al Deep Learning tradicional para tareas específicas de regresión en hiperespacios masivos.

---

## Apéndice: Estructura del Repositorio

* `slrm_nexus.py`: Motor principal para Plegado Geométrico (Parte A).
* `nexus_memory.py`: Almacenamiento persistente y recuperación rápida para Nexus (Anexo A).
* `slrm_lumin.py`: Motor principal para Sectorización de Símplex (Parte B).
* `lumin_memory.py`: Almacenamiento persistente y recuperación rápida para Lumin (Anexo B).
* `lumin_to_relu.py`: Puente de identidad hacia el formato de Redes Neuronales (Parte C).

---
*Desarrollado para la comunidad global de desarrolladores. Cerrando la brecha entre la lógica geométrica y las redes neuronales de alta dimensión.*
