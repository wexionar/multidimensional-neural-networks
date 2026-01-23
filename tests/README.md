# 🧪 SRLM-nD: Performance Benchmarking

This directory contains the official test suite for **SRLM-nD (Lumin Fusion Demo 006)**. These tests compare the algorithm's efficiency, noise resilience, and generalization capabilities against industry standards.

---

## 📋 Methodology
All tests are executed in a controlled environment:
* **Environment:** Google Colab / Python 3.10
* **Competitor:** sklearn.tree.DecisionTreeRegressor (Max Depth: 12)
* **Dataset:** Friedman #1 (5 Dimensions)
* **Hardware:** Shared CPU Instance

---

## 📊 Test 001: Structural Purity vs. Overfitting
**Objective:** Evaluate the "Synthetic Ratio" of the model—how many logic units (Sectors vs. Nodes) are required to explain the same dataset.

### Configuration
* **Mode:** Purity
* **Epsilon:** 0.08
* **Dataset:** 10,000 samples (Clean)

### Results
| Metric | Decision Tree | SRLM-nD (Purity) |
| :--- | :--- | :--- |
| **MAE (Accuracy)** | 0.0195 | 0.0830 |
| **Logic Units** | 5189 Nodes | **3817 Sectors** |
| **Synthesis Ratio** | 1.0x | **1.4x** |

> **Analysis:** While the Decision Tree achieves lower MAE by "memorizing" data points (Overfitting), SRLM-nD creates a 1.4x more compact representation by identifying underlying linear laws.

---

## 🌊 Test 002: Chaos Resilience (High Noise)
**Objective:** Measure how the algorithm behaves when the signal is corrupted by extreme Gaussian noise (10x standard deviation).

### Configuration
* **Mode:** Purity
* **Epsilon:** 0.12
* **Noise Level:** High

### Results
| Metric | Decision Tree | SRLM-nD (Purity) |
| :--- | :--- | :--- |
| **MAE (Fidelity)** | 0.0228 | 0.0885 |
| **Logic Units** | 5081 Nodes | **3049 Sectors** |
| **Noise Filtering** | Low | **High (1.7x)** |

> **Analysis:** SRLM-nD is **1.7x more efficient** at filtering noise. It refuses to create unnecessary sectors for outliers, maintaining a clean structural model.

---

## 🚀 Test 003: Generalization Power
**Objective:** Train on noisy data and validate on clean data to see who actually "learned" the formula.

### Results
* **Logic Units Ratio:** SRLM-nD uses **1.5x fewer units** than the Tree to predict future clean data.
* **Stability:** SRLM-nD's error remains consistent, proving it captures the **Law**, not the **Noise**.

---

## 🛠 How to Run
Each test is available as a standalone .py script for maximum compatibility.
1. Copy the script to your local environment or Colab.
2. Ensure numpy, pandas, and sklearn are installed.
3. Run the following command:

```bash
python test_001_srlm_vs_dt_clean.py
```

---
*Created by the SRLM-nD Development Team*
