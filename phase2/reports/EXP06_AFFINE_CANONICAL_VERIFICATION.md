# EXP-06 Report: Post-Top-5 Affine-Canonical Candidate Verification Analysis

This report documents the empirical evaluation of **EXP-06: Post-Top-5 Affine-Canonical Candidate Verification Analysis**, tested as a single isolated change under the **Strict Iterative Development Protocol**.

---

## 1. Compliance & Experimental Integrity

- **Original Checkpoint**: `phase2_checkpoints/best_model_level1.pth` (**100% UNTOUCHED**).
- **Checkpoint SHA-256 Hash**: `e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c` (Verified before and after execution).
- **Production Files (`phase2_inference.py`, `phase2_config.py`, `register.py`)**: **100% UNTOUCHED**.
- **Candidate Generator**: Multi-scale + multi-rotation NCC Top-5 candidates (**100% Unchanged**).
- **No Ground-Truth Leakage**: GT coordinates used strictly for post-run evaluation & feature labeling. Zero GT information used during candidate selection.

---

## 2. Hypothesis & Algorithm

- **Hypothesis**: "For each candidate in the existing Top-5 pool, candidate-local affine canonicalization to remove scale/rotation pose differences followed by fine local scale/rotation pose verification ($\text{scale} \pm 0.005$, $\text{rotation} \pm 0.25^\circ$) will better discriminate true GT landmarks from periodic DRAM cell replicas."
- **Formulation Evaluated**:
  $$S = (1 - w_{\text{verif}}) \cdot S_{\text{fused\_orig}} + w_{\text{verif}} \cdot S_{\text{canonical\_local\_ncc}}$$
  where $w_{\text{verif}} \in [0.10, 0.20, 0.30, 1.00]$.

---

## 3. Official 200-Pair 100-Point Benchmark Results

| Formulation | DS2 Total (/100) | Loc (/40) | Scale (/10) | Rot (/10) | Rejection (/15) | Conf (/10) | GT Selected | Median RT | Delta vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Fused (Current Best)** | **46.77** | **9.38** | **9.00** | **13.70** | **9.69** | 45 / 160 | 347.8 ms | +0.00 |
| **Canonical Local NCC Only** | 51.64 | 8.05 | 1.94 | 2.09 | 14.67 | 9.90 | 52 / 160 | 516.4 ms | +4.87 |
| **Fused 90% / Canonical 10%** | 47.36 | 6.41 | 1.28 | 1.59 | 13.70 | 9.37 | 45 / 160 | 516.4 ms | +0.59 |
| **Fused 80% / Canonical 20%** | 47.36 | 6.41 | 1.28 | 1.59 | 13.70 | 9.37 | 45 / 160 | 516.4 ms | +0.59 |
| **Fused 70% / Canonical 30%** | 47.36 | 6.41 | 1.28 | 1.59 | 13.70 | 9.37 | 45 / 160 | 516.4 ms | +0.59 |

> [!WARNING]
> **Key Finding**: Canonical local NCC pose verification fails to break periodic DRAM cell array ambiguity because inverse affine unwarping back to reference space yields identical or higher local template correlation on periodic cell matrices. Localization score regressed from **9.38 to 8.05 / 40.0** (and 6.41 / 40.0 in fused combinations). Target failure cases (`pair_006`, `pair_066`, `pair_186`, `pair_116`) remained unresolved.

---

## 4. Required Output Metrics & Failure Analysis

1. **Official Score**: **51.64 / 100.0** (Canonical Local NCC Only, Delta vs Baseline: **+4.87**)
2. **Localization /40**: **8.05 / 40.0** (Regressed by -1.33 points vs Baseline 9.38)
3. **Scale /10**: **1.94 / 10.0**
4. **Rotation /10**: **2.09 / 10.0**
5. **Rejection /15**: **14.67 / 15.0** (F1 = 0.978)
6. **Confidence /10**: **9.90 / 10.0** (AUC = 0.990)
7. **Efficiency /5**: **5.00 / 5.00**
8. **Generator/Citations /10**: **10.00 / 10.00**
9. **Set A 5px Accuracy**: 1.4%
10. **Set B 5px Accuracy**: 4.3%
11. **Set C Rejection F1**: 0.978
12. **Set D Optical Bonus**: Evaluated cleanly
13. **Localization Accuracy at 1/2/3/5 px**: 0.0% / 0.0% / 0.0% / 4.3%
14. **Scale Accuracy**: 1.94 / 10.0
15. **Rotation Accuracy**: 2.09 / 10.0
16. **Median / P90 / P99 Runtime**: 516.4 ms / 571.9 ms / 617.4 ms
17. **Top-5 Recall**: 99.4% (GT candidate present in Top-5)
18. **GT Correctly Selected**: 52 / 160 present pairs
19. **Periodic Failures Recovered**: 0 / 4 (`pair_006`, `pair_066`, `pair_186`, `pair_116` unresolved)
20. **Regressions**: Localization regressed on 10 nominal pairs.

---

## 5. Promotion Rule Decision

> [!CAUTION]
> **EXP-06 DECISION**: **REJECT / NOT PROMOTED**.
> Although Canonical Local NCC Only increased total score to 51.64 (+4.87 pts gain), localization regressed from 9.38 to 8.05 / 40.0, and target periodic failure cases remained unresolved.
>
> Production code remains **100% UNTOUCHED**. The Current Best Verified Version remains **Baseline (46.77 / 100.0)**.
