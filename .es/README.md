# Modelo de Regresión Lineal Segmentada para Redes Neuronales Multi-dimensionales (SLRM-nD)

**Licencia:** MIT License  
**Desarrolladores:** Alex & Gemini  

```text
SLRM-nD/
├── lumin_origin.py         # Ingesta Dinámica y Sistema Sensorial (Parte H)
├── lumin_synthesis.py      # Compilador de Conocimiento (Parte E)
├── lumin_resolution.py     # Ejecutor de Inferencia Ultra-rápido (Parte F)
├── lumin_to_relu.py        # Puente de Identidad (Parte C)
├── lumin_core.py           # Motor de Sectorizado Simplex (Parte B)
├── lumin_memory.py         # Soporte de Persistencia Lumin (Anexo B)
├── nexus_core.py           # Motor de Plegado Geométrico (Parte A)
├── nexus_memory.py         # Soporte de Persistencia Nexus (Anexo A)
├── demos/                  # Benchmarks y Pruebas de Estrés 50D
├── tests/                  # Benchmarks Comparativos (SRLM-nD vs Otros)
├── .es/                    # Laboratorio y documentación en español
├── LICENSE                 # Licencia MIT
└── README.md               # Documentación Principal
```

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
from nexus_core import SLRMNexus
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
from lumin_core import SLRMLumin
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

## Parte E: Motor de Síntesis de Conocimiento (El Compilador)

**Descripción General:** El Motor de Síntesis representa el nivel más alto de inteligencia en SLRM-nD. En lugar de buscar vecinos, "compila" los datos brutos en **Master Sectors** (Sectores Maestros): leyes geométricas puras que describen la verdad matemática subyacente del conjunto de datos.

**Script:** `lumin_synthesis.py`

### Características Clave
* **Deducción de Eje-Pivote:** Encuentra automáticamente el orden jerárquico óptimo de las dimensiones para maximizar la compresión de datos.
* **Compresión Extrema:** Capaz de unificar miles de puntos en sectores de un solo dígito (tasas de compresión del 99.9%).
* **Filtrado de Ruido:** Utiliza un umbral de tolerancia (Epsilon) para separar la señal del ruido.

---

## Parte F: Resolución Ultra-Rápida (El Ejecutor)

**Resumen:** El Motor de Resolución es el socio especializado de Synthesis. Realiza inferencias casi instantáneas al localizar el punto de entrada dentro de las "Hiper-Cajas" sintetizadas y aplicar la ley lineal correspondiente.

**Script:** `lumin_resolution.py`

### Rendimiento (Test de Estrés 50D)
* **Capacidad de procesamiento:** ~68,210 pts/seg (CPU única).
* **Latencia:** < 0.15s para 10,000 puntos.
* **Predictibilidad:** Devuelve `None` si el punto está en "El Vacío" (The Void), garantizando un 100% de integridad intelectual (sin alucinaciones).
* **Cero Fricción:** Sin pesos pesados ni tensores; solo una tabla ligera de coeficientes geométricos.

---

## Parte G: Modos de Operación y Selección Estratégica

SLRM-nD se adapta a su hardware y a la volatilidad de sus datos. Use esta guía para elegir su motor:

| Escenario | Script Recomendado | Modo | Lógica |
| :--- | :--- | :--- | :--- |
| **Datos en Tiempo Real / Volátiles** | `lumin_core.py` | **Directo** | Sin compilación. Ideal cuando la "verdad" de los datos cambia constantemente. |
| **Big Data / Altas Dimensiones** | `lumin_synthesis.py` | **Compilador** | Pague un costo único (Síntesis) para obtener una vida de respuestas instantáneas. |
| **Dispositivos Edge / Embebidos** | `lumin_resolution.py` | **Reflejo** | Huella de RAM ultra-baja utilizando Sectores Maestros pre-compilados. |

### **El "Pipeline Evolutivo" (Origin + Resolution)**
Esta es la forma más avanzada de utilizar SLRM-nD. En lugar de una compilación pesada por lotes (batch), utiliza **Origin** para construir el mapa mientras los datos llegan, e intercambia en caliente el `master_df` hacia **Resolution** para un despliegue instantáneo.

* **Ideal para:** Trading de alta frecuencia, monitoreo de sensores en tiempo real y sistemas de IA adaptativos.
* **Ventaja:** Tiempo de inactividad de entrenamiento cero. El sistema aprende y ejecuta en paralelo.

**Consejo Profesional:** Si su conjunto de datos es disperso pero estable (como leyes de mercados históricos o constantes físicas), use siempre **Synthesis + Resolution**. Si está procesando un flujo de señales impredecibles en vivo, manténgase con **Lumin/Nexus Core**.

---

## Parte H: Sinergia Evolutiva (La Vía Rápida)

El verdadero poder de SLRM-nD reside en la comunicación directa entre **Origin** y **Resolution**.

1. **Origin (v1.4)** actúa como el *Sistema Sensorial*, detectando estructuras y puntos de "mitosis" en flujos de datos en vivo.
   * **Modo 1 (Diversidad):** Seguimiento de alta fidelidad de cada fractura estructural (Continuo).
   * **Modo 2 (Pureza):** Síntesis de alta velocidad enfocada en leyes geométricas limpias (Independiente).
2. **Resolution** actúa como el *Sistema Motor*, ejecutando las leyes sintetizadas a velocidades ultra-altas.

**Benchmark en Vivo (10D):**
* **Velocidad de Aprendizaje:** ~480+ pts/seg (Crecimiento Estructural Continuo).
* **Velocidad de Ejecución:** ~58,000+ pts/seg (Inferencia en Tiempo Real).

*Conclusión: Esta sinergia permite una "IA Viva" que adapta su geometría interna manteniendo tiempos de respuesta de grado industrial. Es el equilibrio perfecto entre la adaptación en tiempo real y la ejecución de alto rendimiento.*

---

## Apéndice: Estructura del Repositorio y Laboratorio

### Sistema Central (Core)
* `lumin_origin.py`: Ingesta dinámica y Sistema Sensorial (Parte H).
* `lumin_synthesis.py`: Compilador de conocimiento en altas dimensiones (Parte E).
* `lumin_resolution.py`: Motor de inferencia vectorizado ultra-rápido (Parte F).
* `lumin_to_relu.py`: Puente de identidad hacia formato de Redes Neuronales (Parte C).
* `lumin_core.py`: Motor principal de sectorización Simplex (Parte B).
* `lumin_memory.py`: Almacenamiento persistente y recuperación rápida para Lumin (Anexo B).
* `nexus_core.py`: Motor principal de plegado geométrico (Parte A).
* `nexus_memory.py`: Almacenamiento persistente y recuperación rápida para Nexus (Anexo A).

### Laboratorio Experimental
* `/demos`: Benchmarks listos para ejecutar (Python y Jupyter Notebooks) que muestran SLRM-nD frente a la Maldición de la Dimensionalidad.
* `/tests`: Suite de validación científica y benchmarks comparativos (SRLM-nD vs. Otros Algoritmos).
* `.es`: Documentación en español y laboratorio de desarrollo.

---

## Citar

Si encuentras útil SLRM-nD en tu investigación, por favor cítalo como:

```bibtex
@misc{slrm-nd,
  author = {Alex Kinetic and Gemini},
  title = {SLRM-nD: Segmented Linear Regression Model for Multi-dimensional Neural Networks},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/wexionar/multi-dimensional-neural-networks}
}
```

## Licencia

MIT

---

## Proyecto Prometeo
SLRM-nD es un marco teórico fundamental entregado a la comunidad de código abierto. Nuestro objetivo es proporcionar el "fuego" de la síntesis geométrica para que los desarrolladores puedan construir, bifurcar (fork) y evolucionar esta lógica de manera independiente (por ejemplo, SLRM-fork). Nosotros establecemos la base; la comunidad global define el horizonte.

---

*Desarrollado para la comunidad global de desarrolladores. Cerrando la brecha entre la lógica geométrica y las Redes Neuronales multidimensionales. Parte de la iniciativa del Proyecto Prometeo.*
