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

# 🚀 Demo 004: Industrial Scale & Range Resilience (1M Points)

This laboratory focuses on the structural maturation of the **LuminOrigin** engine, moving from basic prototyping to industrial-grade data synthesis. Demo 4 solves the challenges of **Massive Throughput** and **Geometric Standardization**.

### 🛠️ Key Architectural Innovations

#### 1. The Normalization Shield ([-1, 1] Mapping)
Unlike previous versions, Demo 4 implements a mandatory internal mapping to a unit hyper-cube. 
* **Stability:** Prevents floating-point errors and matrix ill-conditioning in high-dimensional spaces (50D+).
* **Relative Epsilon Consistency:** By normalizing to a fixed range, the `Epsilon` parameter (e.g., 0.05) consistently represents a **2.5% structural tolerance** regardless of the raw data's original magnitude (e.g., micro-volts vs. mega-watts).

#### 2. Dual-Mode Structural Logic
The engine now features a toggleable philosophical approach to reality fractures:
* **MODE 1: DIVERSITY (The Sprinter):** Maintains point continuity between sectors. Optimized for high-speed fluid data streams (~400,000 ops/sec).
* **MODE 2: PURITY (The Philosopher):** Resets the sector buffer upon fracture detection. Optimized for high-integrity legal isolation and clean mathematical boundaries.

### 📊 Benchmark Results (1,000,000 Points | 50D)

| Metric | Mode 1 (Diversity) | Mode 2 (Purity) |
| :--- | :--- | :--- |
| **Throughput** | ~395,000 pts/sec | ~16,000 pts/sec |
| **Compression** | ~15-20% (Noise) | ~50%+ (Fractures) |
| **Stability** | High | Ultra-High |
| **RAM Usage** | Constant (Chunked) | Constant (Chunked) |

### 🧠 Research Conclusion
By decoupling the **Real-World Scale** from the **Geometric Core**, SLRM-nD achieves "Scale Agnosticism." This ensures that the synthesized knowledge map (`MASTER_MAP_D4.csv`) remains a mathematically perfect representation of the underlying law, ready for ultra-fast vectorized inference in Demo 5.

---
*Note: All tests were performed in a Google Colab environment using vectorized NumPy operations for data generation and LuminOrigin v1.4 for synthesis.*

**Note:** These demos are optimized for both Google Colab and local environments. For deep technical insights into the v1.4 C kernel, refer to the core documentation in the root folder.
