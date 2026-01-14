# Segmented Linear Regression Model for Multi-dimensional Neural Networks (SLRM-nD)

**Developers:** Alex & Gemini  
**License:** MIT License  

---

## Part A: Nexus Core (Geometric Folding Engine)

**Overview:** SLRM-nD Nexus is a high-performance engine designed to solve N-Dimensional problems through recursive neighborhood folding.

**Current Version:** 1.4 (Lumin-Strategy Integrated)

### Key Features (Nexus)
* Absolute Precision: 0.0 error rate in structured linear systems.
* Deterministic Folding: High-speed axis prioritization based on data density (Lumin-Strategy).

### Nexus Benchmark (1000D)
```text
Launching Nexus v1.4 Test (1000D)...
Nexus loaded with 1500 points.
PREDICTION RESULT: 0.6227576180933394
EXECUTION TIME: 40.99 ms
```

### Quick Start (Nexus)
```python
from slrm_nexus import SLRMNexus
import numpy as np

# Initialize for 1000 dimensions
model = SLRMNexus(dimensions=1000)

# Clean and load data
model.fit(your_data)

# Predict in milliseconds
result = model.predict(your_point)
```

## Annex A: Nexus-Memory v1.4
High-dimensional computing engine (1000D+) based on geometric deduction and persistent Memory-C storage.

**Script:** `nexus_memory.py`

### Performance (2026 Benchmarking)
- Dataset: 240,000 points @ 1000 dimensions.
- Initial Deduction (Nexus Core): ~726.18 ms.
- Recall (Memory-C): 0.01 ms.
- Efficiency: >67,000x constant acceleration factor.

*No training. No GPU. Pure geometry. Zero latency.*

---

## Part B: Lumin Core (Simplex Sectoring Engine)

**Overview:** Lumin is the specialized evolution for Sparse High-Dimensional Hyperspaces. It uses Simplex Sectoring (D+1) to discard opposite points on each axis, ensuring geometric closure even in extremely sparse environments.

**Current Version:** 1.4 (F1-Vectorized)

### Millennium Benchmark (Lumin v1.4)
```text
Launching Test: Lumin v1.4 in 1000D with 1500 points...
Lumin Core v1.4 (F1): 1500 points loaded.
RESULTS F1 EDITION:
REAL VALUE: 336.6912
PREDICTION: 333.6967
ABS ERROR: 2.9945
LATENCY: 89.17 ms
```

### Quick Start (Lumin)
```python
from slrm_lumin import SLRMLumin
import numpy as np

# Initialize for 1000 dimensions
model = SLRMLumin(dimensions=1000)

# Load data and Predict
model.fit(data_1000d)
result = model.predict(point_1000d)
```

## Annex B: Lumin-Memory v1.4
Scalable high-dimensional engine using spatial indexing (cKDTree) and Memory-C persistent recall.

**Script:** `lumin_memory.py`

### Performance (2026 Benchmarking)
- Dataset: 240,000 points @ 1000 dimensions.
- Initial Deduction (Lumin Core): ~586.68 ms.
- Recall (Memory-C): 0.01 ms.
- Efficiency: >40,000x constant acceleration factor.

*No training. No GPU. Pure geometry. Zero latency.*

---

## Part C: The Identity Bridge (Lumin-to-ReLU)

**Overview:** The Bridge is a mathematical translator that converts the geometric Simplex sectors of Lumin into standard Neural Network architectures. It proves the identity between a Simplex-based local linear model and a single-layer ReLU network.

**Script:** `lumin_to_relu.py`

### Bridge Benchmark (1000D Identity)
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

### Key Concept
The Bridge demonstrates that SLRM-nD is not just an alternative to Deep Learning, but a deterministic method to **initialize and stabilize** high-dimensional Neural Network layers with zero approximation error.

---

## Part D: Technical Observations & Comparison

### Nexus vs. Lumin: Which one to use?

* Nexus (Geometric Folding): Complexity relates to coordinate potential (2^d). Best for dense, structured datasets where mathematical perfection is required. Optimized for absolute precision in medium-to-high density spaces.

* Lumin (Simplex Sectoring): Complexity relates to linear sectoring (d + 1). The specialized choice for the "Curse of Dimensionality" in very sparse environments. It ensures a closed geometric volume by selecting only the most relevant surrounding nodes per axis.

### General Performance
Both engines are built on top of NumPy and Scipy, ensuring high-speed matrix operations. They do not require GPUs or heavy training cycles, making them the perfect lightweight alternative to traditional Deep Learning for specific regression tasks in massive hyperspaces.

---

## Appendix: Repository Structure

* `slrm_nexus.py`: Core engine for Geometric Folding (Part A).
* `nexus_memory.py`: Persistent storage and fast recall for Nexus (Annex A).
* `slrm_lumin.py`: Core engine for Simplex Sectoring (Part B).
* `lumin_memory.py`: Persistent storage and fast recall for Lumin (Annex B).
* `lumin_to_relu.py`: Identity bridge to Neural Network format (Part C).

---
*Developed for the global developer community. Bridging the gap between geometric logic and high-dimensional Neural Networks.*
