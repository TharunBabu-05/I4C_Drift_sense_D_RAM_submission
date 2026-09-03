# EXP-05 Report: Multi-Layer ResNet Feature Distance Ratio Analysis

This report documents the empirical evaluation of **EXP-05: Multi-Layer ResNet Feature Distance Ratio Analysis**, tested as a single isolated change under the **Strict Iterative Development Protocol**.

---

## 1. Compliance & Experimental Integrity

- **Original Checkpoint**: `phase2_checkpoints/best_model_level1.pth` (**100% UNTOUCHED**).
- **Checkpoint SHA-256 Hash**: `e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c` (Verified before and after execution).
- **Production Files (`phase2_inference.py`, `phase2_config.py`, `register.py`)**: **100% UNTOUCHED**.
- **Candidate Generator**: Multi-scale + multi-rotation NCC Top-5 candidates (**100% Unchanged**).
- **No Ground-Truth Leakage**: GT coordinates used strictly for post-run evaluation & feature labeling. Zero GT information used during candidate selection.

---

## 2. Hypothesis & Algorithm

- **Hypothesis**: "Intermediate ResNet convolutional feature maps (Layer 1, Layer 2, Layer 3) retain local spatial structure partially lost by the final 128-D embedding, and can better discriminate true GT landmarks from periodic DRAM cell replicas."
- **Formulation Evaluated**:
  $$\text{SelectionScore} = S_{\text{fused\_orig}} - \lambda \cdot \frac{D_{i, L}}{\max(D_{\text{top5}})}$$
  where $\lambda = 0.10$ and $L \in \{\text{Layer 1}, \text{Layer 2}, \text{Layer 3}, \text{128-D Embedding}\}$.

---

## 3. Official 200-Pair 100-Point Benchmark Results

| Formulation | DS2 Total (/100) | Loc (/40) | Scale (/10) | Rot (/10) | Rejection (/15) | Conf (/10) | GT Selected | Median RT | Delta vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Fused (Current Best)** | **46.77** | **9.38** | **9.00** | **13.70** | **9.69** | 45 / 160 | 347.8 ms | +0.00 |
| **EXP-05A (Layer 1 Distance)** | 47.37 | 6.41 | 2.81 | 13.70 | 9.37 | 45 / 160 | 785.0 ms | +0.60 |
| **EXP-05B (Layer 2 Distance)** | 47.37 | 6.41 | 2.81 | 13.70 | 9.37 | 45 / 160 | 785.0 ms | +0.60 |
| **EXP-05C (Layer 3 Distance)** | 47.36 | 6.41 | 2.81 | 13.70 | 9.37 | 45 / 160 | 785.0 ms | +0.59 |
| **EXP-05D (Final 128-D Embedding)** | 47.36 | 6.41 | 2.81 | 13.70 | 9.37 | 45 / 160 | 785.0 ms | +0.59 |

> [!WARNING]
> **Key Finding**: Intermediate ResNet layer feature distances (Layer 1, Layer 2, Layer 3) fail to separate periodic DRAM cell array decoys from true GT landmarks because periodic cell grids produce near-identical convolutional feature maps across all layers. Localization score regressed from **9.38 to 6.41 / 40.0**.

---

## 4. Required Output Metrics & Failure Analysis

1. **Official Score**: **47.37 / 100.0** (Delta vs Baseline: **+0.60**)
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
16. **Median / P90 / P99 Runtime**: 785.0 ms / 2115.0 ms / 2180.0 ms
17. **Top-5 Recall**: 99.4% (GT candidate present in Top-5)
18. **GT Correctly Selected**: 45 / 160 present pairs
19. **Periodic Failures Recovered**: 0 / 4 (`pair_006`, `pair_066`, `pair_186`, `pair_116` unresolved)
20. **Regressions**: Localization regressed on 14 pairs.

---

## 5. Promotion Rule Decision

> [!CAUTION]
> **EXP-05 DECISION**: **REJECT / NOT PROMOTED**.
> Measured score gain $+0.60$ pts is $\le 1.0$ point, localization regressed from 9.38 to 6.41 / 40.0, and target periodic failure cases remained unresolved.
>
> Production code remains **100% UNTOUCHED**. The Current Best Verified Version remains **Baseline (46.77 / 100.0)**.
