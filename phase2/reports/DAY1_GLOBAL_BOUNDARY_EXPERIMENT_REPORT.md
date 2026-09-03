# Phase-2 Day-1 Diagnostic Experiment Report: Global Landmark Boundary & Asymmetric Context Alignment

This report documents the diagnostic experiment evaluating whether lightweight global landmark boundary and asymmetric context alignment features can distinguish true ground-truth (GT) landmarks from periodic DRAM cell array decoys among Top-5 candidate matches.

---

## 1. Compliance & Experimental Integrity

- **Original Checkpoint**: `phase2_checkpoints/best_model_level1.pth` (**100% UNTOUCHED**)
- **Original Checkpoint SHA-256 Hash**: `e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c` (**Match Verified**)
- **Production Code (`phase2_inference.py` / `register.py`)**: **100% UNMODIFIED**
- **Candidate Generator**: Multi-scale + multi-rotation NCC (**100% Unchanged**)
- **No Ground-Truth Leakage**: GT coordinates used strictly for diagnostic labeling and post-run score metrics calculation. Zero GT information used during candidate selection.

---

## 2. Feature Extraction Framework

For every candidate in the NCC Top-5 candidates, five features were recorded:
1. `NCC Score`: Coarse normalized cross-correlation score.
2. `Boundary Distance Score`: L2 distance transform score relative to major macro-cell boundary edge contours in a 300x300 context window.
3. `Asymmetric Context Score`: Standard deviation of edge gradient orientation magnitudes across 4 quadrants in a 300x300 context window.
4. `Local Structural Score`: Ratio of inner core (50px) vs outer border (150px) edge contrast variance.
5. `Combined Score`: $S_{\text{fused}} + 0.30 \cdot S_{\text{boundary}} + 0.20 \cdot S_{\text{asymmetric}} + 0.10 \cdot S_{\text{structural}}$.

---

## 3. Diagnostic Breakdown on Periodic Failure Cases

Below is the empirical feature diagnostic on the four target failure cases (`pair_006`, `pair_066`, `pair_186`, `pair_116`):

#### 1. `pair_006` (GT: $x=428.0, y=660.0$)
- **Rank 1 (Selected Decoy)**: $(x=128.0, y=110.0)$ | $\text{NCC}=0.9562$ | $\text{Siam}=0.9903$ | $\text{Boundary}=1.0000$ | $\text{Asym}=0.0121$ | $\text{Struct}=0.9958$ | $\mathbf{\text{COMBINED}=1.3752}$ | $\text{Dist GT}=632.5\text{px}$
- **Rank 5 (True GT Candidate)**: $(x=428.0, y=660.0)$ | $\text{NCC}=0.7312$ | $\text{Siam}=0.9780$ | $\text{Boundary}=0.9615$ | $\text{Asym}=0.0558$ | $\text{Struct}=1.0490$ | $\mathbf{\text{COMBINED}=1.2591}$ | $\text{Dist GT}=111.8\text{px}$
- **Diagnostic Result**: **FAILED TO SEPARATE**. The periodic decoy receives a higher boundary score ($1.0000$ vs $0.9615$) and a higher combined score ($\mathbf{1.3752}$ vs $\mathbf{1.2591}$).

#### 2. `pair_066` (GT: $x=320.0, y=702.0$)
- **Rank 1 (Selected Decoy)**: $(x=670.0, y=52.0)$ | $\text{NCC}=0.9590$ | $\text{Siam}=0.9949$ | $\text{Boundary}=0.9259$ | $\text{Asym}=0.0130$ | $\text{Struct}=0.9132$ | $\mathbf{\text{COMBINED}=1.3487}$ | $\text{Dist GT}=738.2\text{px}$
- **Diagnostic Result**: **FAILED TO SEPARATE**. Decoy retains top rank.

#### 3. `pair_186` (GT: $x=297.0, y=732.0$)
- **Rank 1 (Selected Decoy)**: $(x=597.0, y=132.0)$ | $\text{NCC}=0.9545$ | $\text{Siam}=0.9842$ | $\text{Boundary}=0.9615$ | $\text{Asym}=0.0078$ | $\text{Struct}=1.0037$ | $\mathbf{\text{COMBINED}=1.3597}$ | $\text{Dist GT}=670.8\text{px}$
- **Diagnostic Result**: **FAILED TO SEPARATE**. Decoy retains top rank.

#### 4. `pair_116` (GT: $x=508.0, y=326.0$)
- **Rank 1 (Selected Decoy)**: $(x=194.0, y=861.0)$ | $\text{NCC}=0.5480$ | $\text{Siam}=0.3451$ | $\text{Boundary}=0.9804$ | $\text{Asym}=0.0315$ | $\text{Struct}=1.3398$ | $\mathbf{\text{COMBINED}=0.8810}$ | $\text{Dist GT}=620.3\text{px}$
- **Rank 4 (True GT Candidate)**: $(x=510.5, y=293.5)$ | $\text{NCC}=0.5676$ | $\text{Siam}=0.2773$ | $\text{Boundary}=0.9804$ | $\text{Asym}=0.0294$ | $\text{Struct}=1.1763$ | $\mathbf{\text{COMBINED}=0.8401}$ | $\text{Dist GT}=32.6\text{px}$
- **Diagnostic Result**: **FAILED TO SEPARATE**.

---

## 4. Official 100-Point Score Comparison

| Metric Category | Max Points | Official Baseline Score | Day-1 Method Score | Score Delta |
| :--- | :---: | :---: | :---: | :---: |
| **1. Localization** | 40.0 | **9.38** | **9.38** | +0.00 |
| **2. Pose Scale Recovery** | 10.0 | **4.12** | **4.12** | +0.00 |
| **3. Pose Rotation Recovery** | 10.0 | **4.88** | **4.88** | +0.00 |
| **4. Rejection F1** | 15.0 | **13.70** | **13.70** | +0.00 |
| **5. Confidence Calibration** | 10.0 | **9.69** | **9.69** | +0.00 |
| **6. CPU Efficiency** | 5.0 | **5.00** | **5.00** | +0.00 |
| **7. Generator / Citations** | 10.0 | **10.00** | **10.00** | +0.00 |
| **TOTAL SCORE** | **100.0** | **46.77** | **46.77** | **+0.00** |

- **Target Periodic Failures Status**:
  - `pair_006`: **UNRESOLVED** (Decoy selected)
  - `pair_066`: **UNRESOLVED** (Decoy selected)
  - `pair_186`: **UNRESOLVED** (Decoy selected)
  - `pair_116`: **UNRESOLVED** (Decoy selected)
- **Median Wall-Clock Runtime**: **352.4 ms** (Stable).

---

## 5. Mandatory Decision & Promotion Rule Evaluation

> [!CAUTION]
> **MANDATORY DECISION RULE TRIGGERED**:
> The proposed Day-1 Global Boundary & Asymmetric Context method **FAILED** to separate true GT landmarks from periodic cell array decoys on the target periodic failure cases (`pair_006`, `pair_066`, `pair_186`, `pair_116`).
>
> Total Score improvement is **+0.00 points** (46.77 -> 46.77 / 100.0).

### Official Decision:
- **DO NOT PROMOTE TO PRODUCTION**: `phase2_inference.py` and `register.py` will remain **100% UNTOUCHED**.
- **STOP TUNING THIS FEATURE**: As instructed by the user, we STOP further tuning of 2D non-neural boundary/context heuristics because periodic cell matrices maintain edge symmetry across boundaries.
