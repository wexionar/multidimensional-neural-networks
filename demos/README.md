# 📂 SLRM-nD Demos & Benchmarks

Welcome to the experimental laboratory of the **SLRM-nD (Lumin Core)** project. This repository contains stress tests and practical demonstrations designed to showcase the power of **Hierarchical Axis-Pivot Compilation** in high-dimensional environments.

The SLRM-nD architecture ensures that **Inference (Resolution)** remains near-instantaneous once the synthesis is complete, providing a deterministic alternative to traditional black-box models.

---

## 🚀 Demo 001: 50D Galactic Stress Test
This is our flagship benchmark. It challenges the compiler to synthesize a 50-dimensional universe governed by a single hyper-law into the smallest possible unit of knowledge: **one single Master Sector**.

### 📊 Latest Benchmark Results (v1.4 B)
| Metric | Value |
| :--- | :--- |
| **Dimensions** | 50D |
| **Data Points** | 1,000 |
| **Synthesis Result** | **1 Master Sector** |
| **Compression Rate** | **99.90%** |
| **Execution Time** | ~149 seconds |

### 🧠 Why this matters
Traditional neural networks often struggle with the "Curse of Dimensionality," leading to data fragmentation. **Lumin-Synthesis** uses deductive geometry to find the optimal axis pivot, allowing it to "see" the underlying law through the noise and unify the dataset without losing precision.

### 🛠️ How to run
* **Interactive:** Open `001_stress_test_50D.ipynb` in Google Colab for a zero-installation experience.
* **Local Script:** Ensure you have `numpy` and `pandas` installed:
  ```bash
  python 001_stress_test_50D.py
  ```

---

## 🚀 Demo 002: Non-Linearity & Integrity Check (20D)
This test evaluates the compiler's ability to approximate non-linear curves (parabolas) using piecewise linear sectors and verifies the **Anti-Hallucination** (The Void) logic.

### 📊 Benchmark Results (v1.4 C)
| Metric | Value |
| :--- | :--- |
| **Dimensions** | 20D |
| **Challenge** | Hybrid Linear/Parabolic Function |
| **Points Processed** | 5,000 |
| **Sectors Synthesized** | ~238 |
| **Inference Speed** | **~0.025 ms / point** |
| **Void Detection** | **Success (Zero Hallucination)** |

### 🧠 Why this matters
Real-world data is rarely perfectly linear. This demo shows how the engine "carves" a multidimensional curve into efficient linear laws. Furthermore, the **Resolution** engine proves it can detect unknown territory, providing `NaN` (The Void) instead of guessing—effectively solving the hallucination problem.

### 🛠️ How to run
* **Local Script:**
  ```bash
  python 002_non_linear_stress_test_20D.py
  ```
---

### 🚀 Demo 003: The Evolutionary Pipeline (Origin + Resolution)

**Description:** This demo showcases the high-speed bridge between **Lumin-Origin** and **Lumin-Resolution**. It simulates a live 10-Dimensional environment where the system learns the underlying mathematical laws on the fly and immediately provides a high-throughput inference service.

**Key Prototypical Benchmarks:**
* **Sensory Phase (Origin):** Automates the detection of "reality fractures" (sectoring) during live data ingestion.
* **Motor Phase (Resolution):** Executes pre-compiled laws at ultra-high speeds.
* **Measured Throughput:** ~58,000+ pts/sec (on standard single-CPU).

**Goal:** To prove that SLRM-nD can maintain 100% precision and zero-training downtime even in volatile, high-dimensional data streams.

### 🛠️ How to run
* **Local Script:**
  ```bash
  python 003_evolutionary_pipeline_10D.py
  ```
---

## 🚀 Demo 004: Symmetric Origin Engine (High-Speed Compression)

This laboratory marks the transition of the **Lumin-Origin** engine to industrial-grade standards, focusing on mathematical honesty and scale agnosticism.

### 🛠️ Architectural Breakthroughs

#### 1. The Symmetric Normalization Shield (Max-Abs Mapping)
By implementing **Symmetric Max-Absolute Scaling**, we keep the origin at `0.0`. This is a critical evolution from previous versions:
* **Stability:** Prevents floating-point errors and matrix ill-conditioning in high-dimensional spaces.
* **Integrity:** Preserves the geometric relationship between positive and negative values, ensuring that the "Relative Epsilon" (structural tolerance) remains consistent regardless of the raw data's magnitude.

#### 2. Scale Metadata Header (Auto-Contained Maps)
The output `.npy` file (binary format) now carries its own "DNA." The first row of the matrix is reserved for metadata:
* **Key [0,0]:** Contains the `max_abs_y` factor.
* **Utility:** This allows the **Resolution Engine** (Demo 005) to reconstruct reality without external configuration files, making the compressed knowledge map fully portable.

### 📊 Performance & Stability
| Metric | Specification |
| :--- | :--- |
| **Normalization** | Symmetric Max-Abs ([-1, 1] Range) |
| **Precision** | High-Fidelity Relative Epsilon |
| **I/O Format** | Binary NumPy (.npy) |
| **Compatibility** | Fully n-Dimensional (nD) |

### 🧠 Research Conclusion
By decoupling the **Real-World Scale** from the **Geometric Core**, SLRM-nD achieves "Scale Agnosticism." This ensures that the synthesized map remains a mathematically perfect representation of the underlying law, optimized for ultra-fast vectorized inference.

### 🛠️ How to run
* **Local Script:**
  ```bash
  python 004_lumin_origin.py
  ```
  
---
*Note: All tests were performed in a Google Colab environment using vectorized NumPy operations for data generation and LuminOrigin v1.4 for synthesis.*

**Note:** These demos are optimized for both Google Colab and local environments. For deep technical insights into the v1.4 C kernel, refer to the core documentation in the root folder.
