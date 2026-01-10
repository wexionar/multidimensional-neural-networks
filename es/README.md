# Segmented Linear Regression Model for Multi-dimensional Neural Networks (SLRM-nD)

# SLRM-nD (Nexus Core)
**Versión:** 1.2 (Estable)  
**Desarrolladores:** Alex & Gemini  
**Licencia:** MIT License

## Descripción General (Overview)
**SLRM-nD** es un motor de predicción geométrica universal de alto rendimiento, diseñado para resolver problemas **N-Dimensionales** mediante el plegado recursivo de vecindarios (*recursive neighborhood folding*). A diferencia de las redes neuronales tradicionales, SLRM-nD no requiere tiempo de entrenamiento; utiliza una lógica de **Regresión Lineal Multidimensional Segmentada** para dar respuestas exactas basadas en puntos de datos reales.

## Características Clave (Key Features)
* **Precisión Absoluta:** Tasa de error de 0.0 en sistemas lineales.
* **Ultra-Ligero:** Optimizado para hardware móvil y entornos de bajos recursos.
* **Nexus Engine:** Un algoritmo recursivo que navega por el hiperespacio de manera eficiente.
* **Data Shield (v1.2):** Manejo automático de valores nulos (NaN), duplicados y sectores de datos dispersos (**Alex's Constant Rule**).
* **Extrapolación Natural:** Predice tendencias más allá de los límites del conjunto de datos utilizando las pendientes de los bordes.

## Inicio Rápido (Quick Start)
```python
from slrm_nexus import SLRMNexus
import numpy as np

# Inicializar para 10 dimensiones
model = SLRMNexus(dimensions=10)

# Ejemplo de conjunto de datos [Dim0, Dim1, ... DimN, Salida_Y]
data = np.array([
    [0, 0, 10],
    [1, 1, 20],
    [0.5, 0.5, 15]
])

# Limpiar y cargar datos
model.fit(data)

# Predecir un punto en el espacio
result = model.predict([0.7, 0.7])
print(f"Prediction: {result}")
```

## Cómo Funciona (How it Works)
El motor utiliza un proceso llamado **Dimensional Folding** (Plegado Dimensional). Reduce recursivamente un problema N-dimensional en una serie de segmentos lineales de 1D, "plegando" el espacio hasta alcanzar el valor escalar final. Si una dimensión carece de datos suficientes, aplica la **Constant Logic** (Lógica de Constantes) para mantener la estabilidad.

---
*Desarrollado en el nd-lab para la comunidad global de desarrolladores.*
