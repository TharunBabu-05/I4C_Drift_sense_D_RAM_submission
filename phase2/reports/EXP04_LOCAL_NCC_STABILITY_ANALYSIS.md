# Phase-2 EXP-04 Report: Local NCC Peak Stability & Refinement Analysis

This report documents the empirical evaluation of **EXP-04: Local NCC Peak Stability & Refinement Analysis**, tested as a single isolated change under the **Strict Iterative Development Protocol**.

---

## 1. Compliance & Experimental Integrity

- **Original Checkpoint**: `phase2_checkpoints/best_model_level1.pth` (**100% UNTOUCHED**).
- **Checkpoint SHA-256 Hash**: `e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c` (Verified before and after execution).
- **Production Files (`phase2_inference.py`, `phase2_config.py`, `register.py`)**: **100% UNTOUCHED**.
- **Candidate Generator**: Multi-scale + multi-rotation NCC Top-5 candidates (**100% Unchanged**).
- **No Ground-Truth Leakage**: GT coordinates used strictly for diagnostic evaluation labeling. Zero GT information used during candidate selection.

---

## 2. EXP-04 Algorithm & Hypothesis

- **Hypothesis**: The true GT landmark peak has higher local gradient sharpness/stability in a small $9 \times 9$ translation neighborhood ($dx, dy \in [-4, +4]$) than periodic DRAM cell array decoys.
- **Formulation Evaluated**:
  $$\text{RefinedScore} = S_{\text{fused\_orig}} + \lambda \cdot \text{normalized\_peak\_sharpness}$$
  where $\lambda = 0.05$.

---

## 3. Official 100-Point Score & Ablation Results

| Formulation | DS2 Total Score (/100) | Loc Score (/40) | Pose Score (/20) | Rejection (/15) | Conf (/10) | DS1 Score (/100) | Median RT | Delta vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Fused (Current Best)** | **46.77** | **9.38** | **9.00** | **13.70** | **9.69** | **56.88** | 347.8 ms | +0.00 |
| **EXP04_REF (Peak Sharpness $\lambda=0.05$)** | 47.29 | 6.41 | 2.81 | 13.70 | 9.37 | 51.59 | 774.0 ms | +0.52 |
| **EXP04_MARGIN (Local Margin $\lambda=0.05$)** | 47.33 | 6.41 | 2.84 | 13.70 | 9.37 | 50.83 | 774.0 ms | +0.56 |
| **EXP04_MAX (Local Max + Siamese)** | 38.12 | 2.50 | 1.18 | 10.72 | 8.70 | 26.85 | 774.0 ms | -8.65 |

> [!WARNING]
> **Key Finding**: Local NCC peak stability does NOT separate true GT landmarks from periodic cell decoys because periodic DRAM cell matrix grids also possess sharp, highly stable local cross-correlation peaks due to repeating visual symmetry. Localization score regressed from **9.38 to 6.41 / 40.0**.

---

## 4. Answers to Diagnostic Questions

1. **Was GT already inside Top-5?**
   - **YES** (Inside Top-5 for 159 / 160 present pairs = 99.4% recall).
2. **Did EXP-04 change the ranking?**
   - **NO** for periodic failure cases.
3. **Did the GT candidate move above the periodic decoy?**
   - **NO**.
4. **Did pair_006 recover?**
   - **NO** (Decoy selected at Rank 1).
5. **Did pair_066 recover?**
   - **NO** (Decoy selected at Rank 1).
6. **Did pair_186 remain correct?**
   - **NO** (Decoy selected at Rank 1).
7. **Did any previously correct pair become incorrect?**
   - **YES** (Localization regressed by 2.97 points).
8. **Did Set A accuracy improve?**
   - **NO** (Set A 5px accuracy remained 0.0%).
9. **Did Set B accuracy improve?**
   - **NO** (Set B 5px accuracy regressed).
10. **Did official Localization /40 improve?**
    - **NO** (Regressed from 9.38 to 6.41 / 40.0).

---

## 5. Promotion Rule Decision

> [!CAUTION]
> **EXP-04 DECISION**: **REJECT / NOT PROMOTED**.
> Measured score improvement $+0.52$ pts is $\le 1.0$ point, localization regressed from 9.38 to 6.41 / 40.0, and target periodic failure cases remained unresolved.
>
> Production code remains **100% UNTOUCHED**. The Current Best Verified Version remains **Baseline (46.77 / 100.0)**.
