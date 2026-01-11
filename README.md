# Segmented Linear Regression Model for Multi-dimensional Neural Networks (SLRM-nD)

**Version:** 1.2 (Stable)  
**Developers:** Alex & Gemini  
**License:** MIT License

## Part A: Nexus Core (Precision Engine)

**Overview:** SLRM-nD Nexus is a high-performance engine designed to solve **N-Dimensional** problems through recursive neighborhood folding. It uses Segmented Multidimensional Linear Regression to provide exact answers based on real data points without training time.

### Key Features (Nexus)
* **Absolute Precision:** 0.0 error rate in structured linear systems.
* **Ultra-Lightweight:** Optimized for low-resource environments.
* **Data Shield (v1.2):** Handles Nulls (NaN), duplicates, and sparse sectors (Alex's Constant Rule).

### Quick Start (Nexus)
```python
from slrm_nexus import SLRMNexus
import numpy as np

# Initialize for 10 dimensions
model = SLRMNexus(dimensions=10)

# Clean and load data
model.fit(your_data)

# Predict a point
result = model.predict(your_point)
```

## Part B: Lumin Core (High-Dimension Engine)

**Overview:** Lumin is the "brute force" evolution of the SLRM logic for High-Dimensional Neural Networks. It is specialized in performance (1000D+) and extremely sparse datasets, using Axis-Anchoring and Security Boundaries.

### Millennium Benchmark (1000D)
```text
Launching Millennium Test (Lumin v1.2) in 1000D...
Lumin Core v1.2: 1500 points loaded and purified.
--------------------------------------------------
HYPERSPACE STATISTICS (1000D)
REAL VALUE: 334.674728
PREDICTION: 333.324631
ABS ERROR:  1.350097 
TIME:       194.94 ms
--------------------------------------------------
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
* **Use Nexus** when you have a dense, well-structured dataset and you need mathematical perfection (0.0 error in linear trends). It is the "Master of Folding".
* **Use Lumin** when dealing with "The Curse of Dimensionality" in AI. If you have more than 50-100 dimensions or very sparse data, Lumin's Security Boundary will provide stable results where traditional networks struggle.

### General Performance
Both engines are built on top of NumPy, ensuring high-speed matrix operations. They do not require GPUs or heavy training cycles, making them the perfect lightweight alternative to traditional Deep Learning for specific regression tasks.

---
*Developed for the global developer community. Bridging the gap between geometric logic and high-dimensional Neural Networks.*
