# Modelo de Regresión Lineal Segmentada para Redes Neuronales Multi-dimensionales (SLRM-nD)

**Desarrolladores:** Alex & Gemini  
**Licencia:** MIT License

## Parte A: Nexus Core (Motor de Plegado Geométrico)

**Descripción General:** SLRM-nD Nexus es un motor de alto rendimiento diseñado para resolver problemas **N-Dimensionales** mediante el plegado recursivo de vecindades. 

**Versión Actual:** 1.4 (Estrategia Lumin Integrada)

### Características Clave (Nexus)
* **Precisión Absoluta:** Tasa de error 0.0 en sistemas lineales estructurados.
* **Plegado Determinista:** Priorización de ejes a alta velocidad basada en la densidad de datos (Estrategia Lumin).

### Benchmark Nexus (1000D)
```text
Iniciando Test Nexus v1.4 (1000D)...
Nexus cargado con 1500 puntos de datos.

RESULTADO DE PREDICCIÓN: 0.6227576180933394
TIEMPO DE EJECUCIÓN: 40.99 ms
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

## Parte B: Lumin Core (Motor para Altas Dimensiones)

**Descripción General:** Lumin es la evolución de "fuerza bruta" de la lógica SLRM para Redes Neuronales de Alta Dimensionalidad. Está especializado en rendimiento (1000D+) y conjuntos de datos extremadamente dispersos, utilizando Anclaje de Ejes y Fronteras de Seguridad.

**Versión Actual:** 1.2 (Estable)

### Millennium Benchmark (Lumin)
```text
Iniciando Test Millennium (Lumin v1.2) en 1000D...
Lumin Core v1.2: 1500 puntos cargados y purificados.

VALOR REAL: 334.674728
PREDICCIÓN: 333.324631
ERROR ABSOLUTO: 1.350097
TIEMPO: 194.94 ms
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

## Parte C: Observaciones Técnicas y Comparativa

### Nexus vs. Lumin: ¿Cuál utilizar?

* **Nexus (Plegado Geométrico):** La complejidad depende del potencial de coordenadas (2^d). Ideal para datasets densos y estructurados donde se requiere perfección matemática. Ahora optimizado para respuestas en sub-50ms en 1000D.
* **Lumin (Anclaje de Ejes):** La complejidad depende de un anclaje lineal (1 + d). Es la elección especializada para combatir la "Maldición de la Dimensionalidad" en entornos muy dispersos.

### Rendimiento General
Ambos motores están construidos sobre NumPy, asegurando operaciones matriciales de alta velocidad. No requieren GPUs ni ciclos pesados de entrenamiento, siendo la alternativa ligera perfecta al Deep Learning tradicional.

---
*Desarrollado para la comunidad global de desarrolladores. Cerrando la brecha entre la lógica geométrica y las Redes Neuronales de alta dimensionalidad.*
