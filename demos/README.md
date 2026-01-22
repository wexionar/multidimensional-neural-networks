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

## 🚀 Demo 004: Multidimensional Origin Engine (High-Fidelity)

This laboratory marks the transition of the **Lumin-Origin** engine to industrial-grade standards, focusing on mathematical honesty, multidimensional integrity, and scale agnosticism.

### 🛠️ Architectural Breakthroughs

#### 1. Hybrid Epsilon Logic (Dual-Thresholding)
The engine now supports two distinct ways of detecting "reality fractures," critical for handling complex data topologies:
* **Absolute Mode (Type 1):** Constant error corridor. Prevents artificial sector fragmentation in data crossing the zero-origin (polar data).
* **Relative Mode (Type 2):** Threshold proportional to the signal's magnitude. Optimal for exponential trends where precision must scale with the data.

#### 2. The Symmetric Normalization Shield (Max-Abs Mapping)
By implementing **Symmetric Max-Absolute Scaling**, we keep the origin at `0.0`:
* **Stability:** Prevents floating-point errors and matrix ill-conditioning in high-dimensional spaces.
* **Integrity:** Preserves the geometric relationship between positive and negative values.

#### 3. Scale Metadata Header (Binary "DNA")
The output `.npy` file carries its own configuration in the first row, making the map fully self-contained:
* **Key [0,0]:** `max_abs_y` factor for real-world scale recovery.
* **Key [0,1]:** `Epsilon Type` flag (Absolute vs Relative) for Resolution consistency.
* **Utility:** This allows **Demo 005** to reconstruct reality without external config files.

#### 4. Multidimensional Hyperplane Synthesis
The engine uses **Least Squares (LSTSQ)** to solve the optimal law for every sector, considering all dimensions ($X_0$ to $X_n$) simultaneously. It also includes **Auto-Sorting** to ensure geometric continuity.

### 📊 Performance & Stability (v1.4 High-Fidelity)
| Metric | Specification |
| :--- | :--- |
| **Synthesis Logic** | nD Least Squares (Hyperplane) |
| **Normalization** | Symmetric Max-Abs ([-1, 1] Range) |
| **I/O Format** | Binary NumPy (.npy) with DNA Header |
| **Structure Modes** | Mode 1 (Diversity) / Mode 2 (Purity) |

### 🧠 Research Conclusion
By decoupling the **Real-World Scale** from the **Geometric Core**, SLRM-nD achieves "Scale Agnosticism." This ensures the synthesized map remains a mathematically perfect representation of the underlying law, optimized for ultra-fast vectorized inference.

### 🛠️ How to run
* **Local Script:**
  ```bash
  python 004_lumin_origin.py
  ```
---

## 🚀 Demo 005: F1-Vectorized Resolution (Massive Throughput)

The ultimate "Pilot" for the SLRM-nD architecture. Demo 005 introduces a **Zero-Loop Inference** engine designed to handle millions of points with near-zero latency by leveraging modern CPU SIMD capabilities.

### 🛠️ Key Architectural Innovations

#### 1. Zero-Loop Vectorization (The NumPy Engine)
Unlike traditional iterative resolvers, Demo 005 uses advanced linear algebra operations to process data batches:
* **Broadcasting:** Identifies point-to-sector ownership across the entire multidimensional map simultaneously.
* **Einstein Summation (`np.einsum`):** Executes the linear laws ($Y = XW + B$) for thousands of points in a single matrix operation, bypassing the overhead of Python `for` loops.

#### 2. Automatic Scale Recovery (Back to Reality)
By reading the **Metadata Header** generated by Origin 004, the Resolution engine operates in a protected, normalized mathematical space and only applies the scale factor at the final output stage. This ensures maximum floating-point precision.

### 📊 Benchmark Capabilities (Vectorized)
| Metric | Specification |
| :--- | :--- |
| **Inference Mode** | 100% Vectorized (Matrix-Based) |
| **Throughput** | Millions of points per second (Single-CPU) |
| **Precision** | Deterministic (Zero Hallucination) |
| **Memory Map** | Binary `.npy` load (Instantaneous) |

### 🧠 Why this matters
This engine proves that **SLRM-nD** is not just a theoretical model but a production-ready solution for **Edge Computing** and **Real-Time Digital Twins**. It solves the "Inference Bottleneck" common in high-dimensional AI by replacing complex neural weights with direct, deterministic geometric laws.

### 🛠️ How to run
* **Prerequisite:** Ensure you have a map file generated by `004_lumin_origin.py`.
* **Local Script:**
  ```bash
  python 005_lumin_resolution.py
  ```
---
*Note: All tests were performed in a Google Colab environment using vectorized NumPy operations for data generation and LuminOrigin v1.4 for synthesis.*

**Note:** These demos are optimized for both Google Colab and local environments. For deep technical insights into the v1.4 C kernel, refer to the core documentation in the root folder.
