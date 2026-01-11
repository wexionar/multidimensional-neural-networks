# Modelo de Regresión Lineal Segmentada para Redes Neuronales Multi-dimensionales (SLRM-nD)

**Versión:** 1.2 (Estable)  
**Desarrolladores:** Alex & Gemini  
**Licencia:** Licencia MIT

## Parte A: Nexus Core (Motor de Precisión)

**Descripción General:** SLRM-nD Nexus es un motor de alto rendimiento diseñado para resolver problemas **N-Dimensionales** mediante el plegado recursivo de vecindades. Utiliza lógica de Regresión Lineal Multidimensional Segmentada para proporcionar respuestas exactas basadas en puntos de datos reales sin tiempo de entrenamiento.

### Características Clave (Nexus)
* **Precisión Absoluta:** Tasa de error 0.0 en sistemas lineales estructurados.
* **Ultra-Ligero:** Optimizado para entornos de bajos recursos.
* **Escudo de Datos (v1.2):** Manejo automático de nulos (NaN), duplicados y sectores de datos dispersos (Regla de la Constante de Alex).

### Inicio Rápido (Nexus)
```python
from slrm_nexus import SLRMNexus
import numpy as np

# Inicializar para 10 dimensiones
modelo = SLRMNexus(dimensions=10)

# Limpiar y cargar datos
modelo.fit(tus_datos)

# Predecir un punto
resultado = modelo.predict(tu_punto)
```

## Parte B: Lumin Core (Motor de Alta Dimensionalidad)

**Descripción General:** Lumin es la evolución de "fuerza bruta" de la lógica SLRM para Redes Neuronales de alta dimensionalidad. Está especializado en rendimiento masivo (1000D+) y conjuntos de datos extremadamente dispersos, utilizando Anclaje de Ejes y Cercos de Seguridad.

### Benchmark Millennium (1000D)
```text
Lanzando Millennium Test (Lumin v1.2) en 1000D...
Lumin Core v1.2: 1500 puntos cargados y purificados.
--------------------------------------------------
ESTADÍSTICAS DEL HIPERESPACIO (1000D)
VALOR REAL: 334.674728
PREDICCIÓN: 333.324631
ERROR ABS:  1.350097 
TIEMPO:     194.94 ms
--------------------------------------------------
```

### Inicio Rápido (Lumin)
```python
from slrm_lumin import SLRMLumin
import numpy as np

# Inicializar para 1000 dimensiones
modelo = SLRMLumin(dimensions=1000)

# Cargar datos y predecir
modelo.fit(datos_1000d)
resultado = modelo.predict(punto_1000d)
```

## Parte C: Observaciones Técnicas y Comparativa

### Nexus vs. Lumin: ¿Cuál utilizar?
* **Usa Nexus** cuando tengas un dataset denso, bien estructurado y necesites perfección matemática (error 0.0 en tendencias lineales). Es el "Maestro del Plegado".
* **Usa Lumin** cuando te enfrentes a la "Maldición de la Dimensionalidad" en IA. Si tienes más de 50-100 dimensiones o datos muy dispersos, el Cerco de Seguridad de Lumin proporcionará resultados estables donde las redes tradicionales fallan.

### Rendimiento General
Ambos motores están construidos sobre NumPy, asegurando operaciones matriciales de alta velocidad. No requieren GPUs ni ciclos pesados de entrenamiento, lo que los hace la alternativa ligera perfecta al Deep Learning tradicional para tareas específicas de regresión.

---
*Desarrollado para la comunidad global de desarrolladores. Cerrando la brecha entre la lógica geométrica y las Redes Neuronales de alta dimensionalidad.*
