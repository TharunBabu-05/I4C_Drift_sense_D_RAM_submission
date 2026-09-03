# EXP-06 Report: Multi-Scale / Multi-Rotation NCC Transform Consensus Analysis

This report documents the empirical evaluation of **EXP-06: Multi-Scale / Multi-Rotation NCC Transform Consensus Analysis**, tested as a single isolated change under the **Strict Iterative Development Protocol**.

---

## 1. Compliance & Experimental Integrity

- **Original Checkpoint**: `phase2_checkpoints/best_model_level1.pth` (**100% UNTOUCHED**).
- **Checkpoint SHA-256 Hash**: `e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c` (Verified before and after execution).
- **Production Files (`phase2_inference.py`, `phase2_config.py`, `register.py`)**: **100% UNTOUCHED**.
- **Candidate Generator**: Multi-scale + multi-rotation NCC Top-5 candidates (**100% Unchanged**).
- **No Ground-Truth Leakage**: GT coordinates used strictly for post-run evaluation & feature labeling. Zero GT information used during candidate selection.

---

## 2. Hypothesis & Algorithm

- **Hypothesis**: "True GT landmarks possess higher multi-scale and multi-rotation transform consensus across the independent coarse NCC search hypotheses than periodic DRAM cell array decoys."
- **Formulation Evaluated**:
  $$S = S_{\text{orig\_fused}} + \lambda \cdot \text{WeightedTransformStability}$$
  where $\lambda = 0.05$ and $\text{WeightedTransformStability} = \sum_{h \in \text{supp}} \text{ncc}_h \cdot \exp\left(-\frac{\text{dist}^2}{2\sigma^2}\right)$.

---

## 3. Official 200-Pair 100-Point Benchmark Results

| Formulation | DS2 Total (/100) | Loc (/40) | Scale (/10) | Rot (/10) | Rejection (/15) | Conf (/10) | GT Selected | Median RT | Delta vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Fused (Current Best)** | **46.77** | **9.38** | **9.00** | **13.70** | **9.69** | 45 / 160 | 347.8 ms | +0.00 |
| **EXP-06B (Support Count)** | 50.93 | 9.15 | 3.92 | 13.70 | 9.16 | 55 / 160 | 355.0 ms | +4.16 |
| **EXP-06C (Mean Support NCC)** | 48.41 | 7.11 | 3.10 | 13.70 | 9.50 | 48 / 160 | 355.0 ms | +1.64 |
| **EXP-06D (Support Ratio)** | 47.64 | 6.60 | 2.81 | 13.70 | 9.53 | 44 / 160 | 355.0 ms | +0.87 |
| **EXP-06E (Weighted Stability)** | 51.20 | 8.99 | 3.92 | 13.70 | 9.59 | 54 / 160 | 355.0 ms | +4.43 |

> [!WARNING]
> **Key Finding**: Although total score increased to **51.20 / 100.0** (+4.43 pts gain), localization score regressed from **9.38 to 8.99 / 40.0**! Periodic DRAM cell arrays also exhibit high transform consensus across multiple scales and rotations due to array symmetry, causing periodic decoys to retain Rank 1 on target failure cases (`pair_006`, `pair_066`, `pair_186`, `pair_116`).

---

## 4. Required Output Metrics & Failure Analysis

1. **Official Score**: **51.20 / 100.0** (Delta vs Baseline: **+4.43**)
2. **Localization /40**: **8.99 / 40.0** (Regressed by -0.39 points vs Baseline 9.38)
3. **Scale /10**: **3.92 / 10.0**
4. **Rotation /10**: **3.92 / 10.0**
5. **Rejection /15**: **13.70 / 15.0** (F1 = 0.9112)
6. **Confidence /10**: **9.59 / 10.0** (AUC = 0.9592)
7. **Efficiency /5**: **5.00 / 5.00**
8. **Generator/Citations /10**: **10.00 / 10.00**
9. **Set A 5px Accuracy**: 0.0%
10. **Set B 5px Accuracy**: 4.1%
11. **Set C Rejection F1**: 0.9112
12. **Set D Optical Bonus**: Evaluated cleanly
13. **Localization Accuracy at 1/2/3/5 px**: 0.0% / 0.0% / 0.0% / 4.1%
14. **Scale Accuracy**: 1.82 / 10.0
15. **Rotation Accuracy**: 2.10 / 10.0
16. **Median / P90 / P99 Runtime**: 355.0 ms / 980.0 ms / 1050.0 ms
17. **Top-5 Recall**: 99.4% (GT candidate present in Top-5)
18. **GT Correctly Selected**: 54 / 160 present pairs
19. **Periodic Failures Recovered**: 0 / 4 (`pair_006`, `pair_066`, `pair_186`, `pair_116` unresolved)
20. **Regressions**: Localization regressed on 8 nominal pairs.

---

## 5. Promotion Rule Decision

> [!CAUTION]
> **EXP-06 DECISION**: **REJECT / NOT PROMOTED**.
> Although EXP-06E increased total score to 51.20 (+4.43 pts gain), localization regressed from 9.38 to 8.99 / 40.0, and target periodic failure cases remained unresolved.
>
> Production code remains **100% UNTOUCHED**. The Current Best Verified Version remains **Baseline (46.77 / 100.0)**.
