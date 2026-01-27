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
* **Local Script:**
  ```bash
  python demo_001_stress_test_50D.py
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
  python demo_002_non_linear_stress_test_20D.py
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
  python demo_003_evolutionary_pipeline_10D.py
  ```
---

## 🚀 Demo 004: Multidimensional Origin Engine (High-Fidelity)

This laboratory marks the transition of the **Lumin-Origin** engine to industrial-grade standards, focusing on mathematical honesty, multidimensional integrity, and scale agnosticism.

### 🛠️ Architectural Breakthroughs

#### 1. Hybrid Epsilon Logic (Dual-Thresholding)
The engine now supports two distinct ways of detecting "reality fractures," critical for handling complex data topologies:
* **Absolute Mode ('abs'):** Constant error corridor. Prevents artificial sector fragmentation in data crossing the zero-origin (polar data).
* **Relative Mode ('rel'):** Threshold proportional to the signal's magnitude. Optimal for exponential trends where precision must scale with the data.

#### 2. The Symmetric Normalization Shield (Max-Abs Mapping)
To protect FPU stability and maximize matrix performance, we implement **Symmetric Max-Absolute Scaling**:
* **Stability:** Prevents floating-point errors and matrix ill-conditioning in high-dimensional spaces by mapping data to a $[-1, 1]$ range.
* **Integrity:** Unlike standard Min-Max, this method preserves the absolute zero ($0.0$) at the center, maintaining the geometric relationship between positive and negative vectors.

#### 3. Complete Binary "DNA" (Triple-Key Metadata)
The output `.npy` file is fully self-contained. The first row (Row 0) acts as a persistent contract between Synthesis and Resolution:
* **Key [0,0]:** `max_abs_y` factor for real-world scale recovery.
* **Key [0,1]:** `Epsilon Type` flag (1 for Absolute / 0 for Relative).
* **Key [0,2]:** `Epsilon Value` (The exact threshold used during synthesis).
* **Utility:** This allows **Demo 005** to reconstruct reality with 100% deterministic precision without external configuration.

#### 4. Multidimensional Hyperplane Synthesis
The engine uses **Least Squares (LSTSQ)** to solve the optimal law for every sector, considering all dimensions ($X_0$ to $X_n$) simultaneously. It includes **Auto-Sorting** to ensure geometric continuity and jurisdiction alignment.

### 📊 Performance & Stability (v1.4 High-Fidelity)
| Metric | Specification |
| :--- | :--- |
| **Synthesis Logic** | nD Least Squares (Hyperplane) |
| **Normalization** | Symmetric Max-Abs ([-1, 1] Range) |
| **I/O Format** | Binary NumPy (.npy) with Triple-Key DNA |
| **Structure Modes** | Mode 1 (Diversity) / Mode 2 (Purity) |

### 🧠 Research Conclusion
By decoupling the **Real-World Scale** from the **Geometric Core** through the DNA header, SLRM-nD achieves true "Scale Agnosticism." This ensures the synthesized map remains a mathematically perfect representation of the underlying law, optimized for ultra-fast vectorized inference.

### 🛠️ How to run
* **Local Script:**
  ```bash
  python demo_004_lumin_origin.py
  ```
---

## 🚀 Demo 005: F1-Vectorized Resolution (Massive Throughput)

The ultimate "Pilot" for the SLRM-nD architecture. Demo 005 introduces a **Zero-Loop Inference** engine designed to handle millions of points with near-zero latency by leveraging modern CPU SIMD capabilities.

### 🛠️ Key Architectural Innovations

#### 1. Zero-Loop Vectorization (The NumPy Engine)
Unlike traditional iterative resolvers, Demo 005 uses advanced linear algebra operations to process data batches:
* **Broadcasting:** Identifies point-to-sector ownership across the entire multidimensional map simultaneously, creating a boolean jurisdiction matrix.
* **Einstein Summation (`np.einsum`):** Executes the linear laws ($Y = XW + B$) for thousands of points in a single matrix operation, bypassing the overhead of Python `for` loops.

#### 2. DNA-Synchronized Reconstruction (The 3-Point Check)
By reading the **Metadata Header** generated by Origin 004, the Resolution engine is no longer a "blind" executor. It automatically configures itself based on:
* **Scale Recovery:** Applies the `max_abs_y` factor only at the final stage to maintain maximum floating-point precision during calculation.
* **Logic Alignment:** Identifies whether the synthesis used Absolute or Relative Epsilon, ensuring that any future integrity checks remain consistent with the Origin's "truth."

#### 3. Deterministic "Void" Detection
The engine maintains strict geometric honesty. If a point falls outside all synthesized sector jurisdictions (Bounding Boxes), the engine returns `NaN` (The Void) instead of hallucinating a value, effectively solving the reliability problem of traditional black-box models.

### 📊 Benchmark Capabilities (Vectorized v1.4)
| Metric | Specification |
| :--- | :--- |
| **Inference Mode** | 100% Vectorized (Matrix-Based) |
| **Throughput** | Millions of points per second (Single-CPU) |
| **DNA Synchronization** | Triple-Key (Scale, Mode, Epsilon) |
| **Precision** | Deterministic (Zero Hallucination) |
| **Memory Map** | Binary `.npy` load (Instantaneous) |

### 🧠 Why this matters
This engine proves that **SLRM-nD** is not just a theoretical model but a production-ready solution for **Edge Computing** and **Real-Time Digital Twins**. It solves the "Inference Bottleneck" common in high-dimensional AI by replacing complex neural weights with direct, deterministic geometric laws that run at the speed of hardware.

### 🛠️ How to run
* **Prerequisite:** Ensure you have a map file generated by `004_lumin_origin.py`.
* **Local Script:**
  ```bash
  python demo_005_lumin_resolution.py
  ```
---

## 🚀 Demo 006: Lumin Fusion (Integrated Engine)

The most advanced iteration of the **Lumin Core**. Demo 006 unifies the Synthesis (Origin) and Inference (Resolution) engines into a single, seamless pipeline designed for real-world benchmarking and high-performance experimentation.

### 🛠️ Strategic Enhancements

#### 1. Unified Fusion Pipeline
Demo 006 eliminates the gap between learning and resolving. It allows for immediate verification of the synthesized map's fidelity by running a **Stress Test** instantly after the ingestion phase, ensuring the structural laws are perfectly captured.

#### 2. Persistence & Cache Management
To optimize developer workflow, the engine introduces a **Session Persistence Layer**:
* **Memory Cache:** Re-use massive datasets across different engine configurations (changing Epsilon or Normalization) without the overhead of re-loading or re-generating data.
* **Session Token:** Every execution is tagged with a unique hex-token for precise tracking of reports and binary exports.

#### 3. Real-Time Ignition Reports
The engine provides a high-fidelity dashboard after each "Ignition" (Synthesis process), delivering critical metrics:
* **Structural Synthesis:** Direct visualization of the compression ratio and sector count.
* **Mathematical Fidelity:** Real-time calculation of MAE (Mean Absolute Error) and Global Fidelity percentage.
* **Hardware Performance:** Throughput metrics (points/sec) and operational latency.

#### 4. Dual-Normalization Shield
Supports both **Symmetric [-1, 1]** and **Direct [0, 1]** mapping, allowing the user to adapt the engine's geometric sensitivity based on the data's topology.

### 📊 Performance Metrics (Fusion v006)
| Metric | Specification |
| :--- | :--- |
| **Engine State** | Fully Integrated (Origin + Resolution) |
| **Learning Speed** | Optimized single-pass ingestion |
| **Resolution** | 100% Vectorized Matrix Operations |
| **Data Integrity** | DNA-Synchronized with persistent cache |
| **Export Format** | Version-stamped Binary Map (.npy) |

### 🧠 Why this matters
**Lumin Fusion 006** is the definitive tool for researchers. It allows for the rapid "tuning" of the Epsilon-to-Synthesis ratio, making it easy to find the perfect balance between model simplicity and mathematical precision. It is the core engine used for the official **SRLM-nD Benchmarks**.

### 🛠️ How to run
* **Local Script:**
  ```bash
  python demo_006_lumin_fusion.py
  ```
---
**Note:** All tests were performed in a Google Colab environment using vectorized NumPy operations for data generation and Lumin-Origin v1.4 for synthesis.
