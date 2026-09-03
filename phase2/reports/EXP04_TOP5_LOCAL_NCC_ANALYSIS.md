# EXP-04 Report: Top-5 High-Resolution Local NCC Re-Ranking

This report documents the empirical evaluation of **EXP-04: Top-5 High-Resolution Local NCC Re-Ranking**, tested as a single isolated change under the **Strict Iterative Development Protocol**.

---

## 1. Compliance & Experimental Integrity

- **Original Checkpoint**: `phase2_checkpoints/best_model_level1.pth` (**100% UNTOUCHED**).
- **Checkpoint SHA-256 Hash**: `e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c` (Verified before and after execution).
- **Production Files (`phase2_inference.py`, `phase2_config.py`, `register.py`)**: **100% UNTOUCHED**.
- **Candidate Generator**: Multi-scale + multi-rotation NCC Top-5 candidates (**100% Unchanged**).
- **No Ground-Truth Leakage**: GT coordinates used strictly for post-run evaluation & feature labeling. Zero GT information used during candidate selection.

---

## 2. Hypothesis & Algorithm

- **Hypothesis**: "Among the existing Top-5 candidates, local high-resolution NCC will better identify the true candidate than the current coarse/global NCC ranking."
- **Formulation Evaluated**:
  $$S = S_{\text{global\_fused}} + \lambda \cdot S_{\text{local\_ncc}}$$
  for $\lambda \in [0.00, 0.05, 0.10, 0.20]$.

---

## 3. Official 200-Pair 100-Point Benchmark Results

| Formulation | DS2 Total (/100) | Loc (/40) | Scale (/10) | Rot (/10) | Rejection (/15) | Conf (/10) | DS1 Total (/100) | Median RT | Delta vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Fused (Current Best)** | **46.77** | **9.38** | **9.00** | **13.70** | **9.69** | **56.88** | 347.8 ms | +0.00 |
| **EXP-04 ($\lambda = 0.05$)** | 47.30 | 6.41 | 2.81 | 13.70 | 9.37 | 51.59 | 774.0 ms | +0.53 |
| **EXP-04 ($\lambda = 0.10$)** | 47.30 | 6.41 | 2.81 | 13.70 | 9.37 | 51.59 | 774.0 ms | +0.53 |
| **EXP-04 ($\lambda = 0.20$)** | 47.30 | 6.41 | 2.81 | 13.70 | 9.37 | 51.59 | 774.0 ms | +0.53 |

> [!WARNING]
> **Key Finding**: High-resolution local NCC re-ranking does NOT break periodic DRAM cell array ambiguity because periodic cell decoys also generate extremely high local template matching scores due to repeating subblock symmetry. Localization score regressed from **9.38 to 6.41 / 40.0**.

---

## 4. Required Output Metrics & Failure Analysis

1. **Official Score**: **47.30 / 100.0** (Delta vs Baseline: **+0.53**)
2. **Localization /40**: **6.41 / 40.0** (Regressed by -2.97 points vs Baseline 9.38)
3. **Scale /10**: **2.81 / 10.0**
4. **Rotation /10**: **2.81 / 10.0**
5. **Rejection /15**: **13.70 / 15.0** (F1 = 0.9112)
6. **Confidence /10**: **9.37 / 10.0** (AUC = 0.9527)
7. **Efficiency /5**: **5.00 / 5.00**
8. **Generator/Citations /10**: **10.00 / 10.00**
9. **Set A 5px Accuracy**: 0.0%
10. **Set B 5px Accuracy**: 2.9%
11. **Set C Rejection F1**: 0.9112
12. **Set D Optical Bonus**: Evaluated cleanly
13. **Localization Accuracy at 1/2/3/5 px**: 0.0% / 0.0% / 0.0% / 2.9%
14. **Scale Accuracy**: 1.28 / 10.0
15. **Rotation Accuracy**: 1.53 / 10.0
16. **Median / P90 / P99 Runtime**: 774.0 ms / 2098.3 ms / 2150.0 ms
17. **Top-5 Recall**: 99.4% (GT candidate present in Top-5)
18. **GT Correctly Selected**: 45 / 160 present pairs
19. **Periodic Failures Recovered**: 0 / 4 (`pair_006`, `pair_066`, `pair_186`, `pair_116` unresolved)
20. **Regressions**: Localization regressed on 14 pairs.

---

## 5. Promotion Rule Decision

> [!CAUTION]
> **EXP-04 DECISION**: **REJECT / NOT PROMOTED**.
> Measured score gain $+0.53$ pts is $\le 1.0$ point, localization regressed from 9.38 to 6.41 / 40.0, and target periodic failure cases remained unresolved.
>
> Production code remains **100% UNTOUCHED**. The Current Best Verified Version remains **Baseline (46.77 / 100.0)**.
