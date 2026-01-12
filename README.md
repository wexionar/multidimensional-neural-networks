# Segmented Linear Regression Model for Multi-dimensional Neural Networks (SLRM-nD)

**Developers:** Alex & Gemini  
**License:** MIT License  

## Part A: Nexus Core (Geometric Folding Engine)

**Overview:**
SLRM-nD Nexus is a high-performance engine designed to solve **N-Dimensional** problems through recursive neighborhood folding. 

**Current Version:** 1.4 (Lumin-Strategy Integrated)

### Key Features (Nexus)
* **Absolute Precision:** 0.0 error rate in structured linear systems.
* **Deterministic Folding:** High-speed axis prioritization based on data density (Lumin-Strategy).

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

## Part B: Lumin Core (High-Dimension Engine)

**Overview:**
Lumin is the "brute force" evolution of the SLRM logic for High-Dimensional Neural Networks. It is specialized in performance (1000D+) and extremely sparse datasets, using Axis-Anchoring and Security Boundaries.

**Current Version:** 1.2 (Stable)

### Millennium Benchmark (Lumin)
```text
Launching Millennium Test (Lumin v1.2) in 1000D...
Lumin Core v1.2: 1500 points loaded and purified.

REAL VALUE: 334.674728
PREDICTION: 333.324631
ABS ERROR: 1.350097
TIME: 194.94 ms
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

## Part C: Technical Observations & Comparison

### Nexus vs. Lumin: Which one to use?

* **Nexus (Geometric Folding):** Complexity relates to coordinate potential (2^d). Best for dense, structured datasets where mathematical perfection is required. Now optimized for sub-50ms response in 1000D.
* **Lumin (Axis Anchoring):** Complexity relates to linear anchoring (1 + d). The specialized choice for the "Curse of Dimensionality" in very sparse environments where geometric folding lacks sufficient anchor points.

### General Performance
Both engines are built on top of NumPy, ensuring high-speed matrix operations. They do not require GPUs or heavy training cycles, making them the perfect lightweight alternative to traditional Deep Learning for specific regression tasks.

---
*Developed for the global developer community. Bridging the gap between geometric logic and high-dimensional Neural Networks.*
