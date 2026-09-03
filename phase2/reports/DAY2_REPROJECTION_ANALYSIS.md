# Phase-2 Day-2 Experiment Report: Reference Reprojection & Candidate Consistency Verification

This report presents the empirical findings of the **Day-2 Reference Reprojection & Candidate Consistency Verification** experiment. We evaluated whether unwarping Top-5 search candidates back to the reference coordinate system (undoing scale $s$ and rotation $\theta$) and computing pixel/gradient reprojection agreement can distinguish true ground-truth (GT) landmarks from periodic DRAM cell array decoys.

---

## 1. Compliance & Safety Verification

- **Neural Network Weights**: **100% UNTOUCHED** (`phase2_checkpoints/best_model_level1.pth` SHA-256 hash verified: `e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c`).
- **Production Files Backed Up**:
  - `phase2/backup/phase2_inference.py`
  - `phase2/backup/phase2_config.py`
  - `phase2/backup/register.py`
- **Production Code Status**: **100% UNTOUCHED** (No production changes committed).
- **Candidate Generator**: Multi-scale + multi-rotation NCC (**100% Unchanged**).
- **No Ground-Truth Leakage**: GT coordinates used strictly for diagnostic evaluation. Zero GT information used during candidate selection.

---

## 2. Reprojection & Consistency Verification Framework

For every Top-5 candidate $(x_c, y_c, s, \theta)$, an inverse affine transformation matrix $M_{\text{inv}}$ was computed to unwarp the search image patch back into the $100 \times 100$ reference template coordinate system.

Five independent consistency measurements were calculated per candidate:
1. `Pixel NCC`: Normalized cross-correlation of unwarped candidate vs reference template.
2. `Gradient NCC`: Normalized cross-correlation of Sobel magnitude edge maps.
3. `Pixel L1 Error`: Mean absolute intensity difference $\frac{1}{N} \sum |I_{\text{unwarped}} - I_{\text{ref}}|$.
4. `Gradient L1 Error`: Mean absolute gradient magnitude difference.
5. `Multi-Resolution Error`: Pyramidal average L1 error across 1x ($100 \times 100$), 2x ($50 \times 50$), and 4x ($25 \times 25$) downsampled scales.

---

## 3. Ablation Matrix Results (DS2 & DS1 Test Suites)

| Strategy | Description | DS2 Score (/100) | Loc Score (/40) | Pose Score (/20) | Rejection (/15) | Conf (/10) | DS1 Score (/100) | Median RT | Delta vs Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | Original Fused (0.5 NCC + 0.5 Siam) | **46.77** | **9.38** | **9.00** | **13.70** | **9.69** | **56.88** | 347.8 ms | +0.00 |
| **Strategy A** | NCC Only | 50.19 | 8.05 | 3.91 | 13.50 | 9.73 | 58.44 | 774.9 ms | +3.42 |
| **Strategy B** | NCC + Pixel Reprojection | 49.08 | 7.33 | 3.21 | 13.50 | 10.04 | 52.56 | 774.9 ms | +2.31 |
| **Strategy C** | NCC + Gradient Reprojection | 49.87 | 7.63 | 3.28 | 13.96 | 10.00 | 52.61 | 774.9 ms | +3.10 |
| **Strategy D** | NCC + Multi-Resolution | 49.56 | 7.69 | 3.72 | 13.15 | 10.00 | 57.95 | 774.9 ms | +2.79 |
| **Strategy E** | NCC + Pixel + Gradient | 32.38 | 2.63 | 1.18 | 8.57 | 5.00 | 23.31 | 774.9 ms | -14.39 |
| **Strategy F** | NCC + All Reprojection Signals | 46.81 | 6.49 | 2.37 | 13.12 | 9.83 | 47.65 | 774.9 ms | +0.04 |

> [!WARNING]
> **Key Finding**: While Strategy A (NCC Only) shows a nominal +3.42 point overall score increase due to rejection confidence shifting, **Localization Score regresses from 9.38 to 8.05 / 40.0**. Combining reprojection signals with NCC (Strategy F) causes **Localization to drop further to 6.49 / 40.0** and Pose to drop to 2.37 / 20.0.

---

## 4. Target Periodic Failure Cases Deep-Dive (Strategy F Breakdown)

Below is the candidate ranking for the target failure cases under Strategy F:

#### 1. `pair_006` (GT: $x=328.0, y=710.0$)
- **Rank 1 (Decoy)**: $(x=128.0, y=110.0, s=10.00, \theta=0^\circ)$ | $\text{NCC}=0.9562$ | $\text{PixelErr}=0.2248$ | $\text{GradErr}=0.2308$ | $\text{ReprojScore}=0.2811$ | $\mathbf{\text{FinalScore}=0.6272}$ | $\text{Dist GT}=632.5\text{px}$
- **Rank 5 (GT Candidate)**: $(x=428.0, y=660.0, s=10.25, \theta=-1.5^\circ)$ | $\text{NCC}=0.7312$ | $\text{PixelErr}=0.1455$ | $\text{GradErr}=0.2313$ | $\text{ReprojScore}=0.2848$ | $\mathbf{\text{FinalScore}=0.5697}$ | $\text{Dist GT}=111.8\text{px}$
- **Analysis**: Even though GT candidate achieves lower pixel error ($0.1455$ vs $0.2248$), coarse NCC heavily privileges the periodic cell decoy ($0.9562$ vs $0.7312$), causing the decoy to retain Rank 1.

#### 2. `pair_066` (GT: $x=320.0, y=702.0$)
- **Rank 1 (Decoy)**: $(x=670.0, y=52.0, s=10.00, \theta=0^\circ)$ | $\text{NCC}=0.9590$ | $\text{PixelErr}=0.1461$ | $\text{GradErr}=0.2326$ | $\text{ReprojScore}=0.2846$ | $\mathbf{\text{FinalScore}=0.6308}$ | $\text{Dist GT}=738.2\text{px}$
- **Analysis**: Periodic array cell decoy exhibits identical reprojection score ($0.2846$) to GT candidate because periodic memory cell structures unwarp into identical grid patterns.

#### 3. `pair_186` (GT: $x=297.0, y=732.0$)
- **Rank 1 (Decoy)**: $(x=597.0, y=132.0, s=10.00, \theta=-1^\circ)$ | $\text{NCC}=0.8376$ | $\text{PixelErr}=0.1196$ | $\text{GradErr}=0.2310$ | $\text{ReprojScore}=0.3700$ | $\mathbf{\text{FinalScore}=0.6404}$ | $\text{Dist GT}=670.8\text{px}$
- **Analysis**: Decoy gets a HIGHER reprojection score ($0.3700$) than GT candidate ($0.3000$) due to uniform pixel grid alignment.

#### 4. `pair_116` (GT: $x=508.0, y=326.0$)
- **Rank 1 (Decoy)**: $(x=195.0, y=861.0, s=9.25, \theta=1.5^\circ)$ | $\text{NCC}=0.5460$ | $\text{ReprojScore}=0.2501$ | $\mathbf{\text{FinalScore}=0.3458}$ | $\text{Dist GT}=619.8\text{px}$
- **Rank 3 (GT Candidate)**: $(x=510.5, y=293.5, s=8.25, \theta=1.0^\circ)$ | $\text{NCC}=0.5674$ | $\text{ReprojScore}=0.2623$ | $\mathbf{\text{FinalScore}=0.3423}$ | $\text{Dist GT}=32.6\text{px}$
- **Analysis**: GT candidate ranked 3rd; decoy at $619.8\text{px}$ wins Rank 1.

---

## 5. Answers to the 10 Required Report Questions

1. **Did reprojection distinguish GT from periodic decoys?**
   - **NO**. Periodic DRAM cell matrix grids unwarp into spatially symmetrical patterns that match or exceed the reference reprojection agreement of the true GT landmark.

2. **Did pair_006 improve?**
   - **NO** (Decoy selected at Rank 1).

3. **Did pair_066 improve?**
   - **NO** (Decoy selected at Rank 1).

4. **Did pair_186 remain correct?**
   - **NO** (Decoy selected at Rank 1 with higher reprojection score than GT).

5. **Did pair_116 regress?**
   - **YES** (GT candidate pushed to 3rd rank; decoy selected at Rank 1).

6. **What is the DS2 score?**
   - **46.81 / 100.0** (Strategy F) / **50.19 / 100.0** (Strategy A).

7. **What is the DS1 score?**
   - **47.65 / 100.0** (Strategy F) / **58.44 / 100.0** (Strategy A).

8. **What is the localization score?**
   - **6.49 / 40.0** (Strategy F) — regressed from official baseline 9.38 / 40.0.

9. **What is the runtime?**
   - Median wall-clock runtime is **774.9 ms** (well under the 5,000 ms limit).

10. **Should this be promoted?**
    - **NO / NOT PROMOTED**. The method causes localization regression and fails to separate GT from decoys on target failure cases.

---

## 6. Official Decision & Next Steps

> [!CAUTION]
> **PROMOTION RULE EVALUATION**:
> - Measured localization score **REGRESSED** from 9.38 to 6.49 / 40.0.
> - None of the target failure cases (`pair_006`, `pair_066`, `pair_186`, `pair_116`) showed improvement.
> - **DECISION**: **NOT PROMOTED TO PRODUCTION**.
> - Production code (`phase2_inference.py` and `register.py`) remains **100% UNTOUCHED**.
