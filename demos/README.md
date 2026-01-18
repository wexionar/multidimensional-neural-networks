# 📂 SLRM-nD Demos & Benchmarks

Welcome to the demonstration laboratory of the **SLRM-nD (Lumin Core)** project. These scripts are designed to showcase the power of Hierarchical Axis-Pivot Compilation in high-dimensional environments.

This folder contains stress tests and practical demonstrations of the SLRM-nD engine's capabilities in high-dimensional environments.

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
Traditional neural networks and clustering algorithms often struggle with the "Curse of Dimensionality," leading to data fragmentation. **Lumin-Synthesis v1.4 B** uses deductive geometry to find the optimal axis pivot, allowing it to "see" the underlying law through the noise and unify the entire dataset without losing precision.

### 🛠️ How to run
1. **Interactive Notebook:** Open the `001_stress_test_50D.ipynb` file in this folder. You can run it directly in your browser using the "Open in Colab" button for a zero-installation experience.
2. **Local Script:** Ensure you have `numpy` and `pandas` installed, then run:
   ```bash
   python 001_stress_test_50D.py
   ```
3. The system will generate a synthetic 50D dataset, compile it, and validate the compression success.

---
**Note:** These demos are optimized for both Google Colab and local environments. The architecture ensures that **Inference (Resolution)** remains near-instantaneous once the synthesis is complete.

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
| **Inference Speed** | **< 0.03 ms / point** |
| **Void Detection** | **Success (Zero Hallucination)** |

### 🧠 Why this matters
Real-world data is rarely perfectly linear. **Lumin-Synthesis** demonstrates how it can "carve" a multidimensional curve into a set of efficient linear laws. Furthermore, the **Resolution** engine proves it can detect when it's being asked about unknown territory, providing `NaN` (or `None`) instead of guessing (hallucinating).

### 🛠️ How to run
```bash
python 002_non_linear_stress_test_20D.py
```bash
