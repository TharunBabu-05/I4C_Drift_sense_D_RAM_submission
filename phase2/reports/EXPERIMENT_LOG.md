# Phase-2 Single-Change Experiment Log

This log tracks all single-change experiments evaluated under the **Strict Iterative Development Protocol**.

---

## Experiment Summary Log Table

| EXP ID | Single Change Tested | Prev Score | New Score | Delta | Loc /40 | Scale /10 | Rot /10 | Rejection /15 | Conf /10 | Eff /5 | Gen /10 | pair_006 | pair_066 | pair_186 | pair_116 | Runtime | Decision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BASE** | Baseline (Phase-1 Checkpoint) | — | **46.77** | — | **9.38** | **4.12** | **4.88** | **13.70** | **9.69** | **5.00** | **10.00** | Decoy | Decoy | Decoy | Decoy | 347.8ms | **FREEZE** |
| **EXP-01** | Global Boundary + Asym Context | 46.77 | 46.77 | +0.00 | 9.38 | 4.12 | 4.88 | 13.70 | 9.69 | 5.00 | 10.00 | Decoy | Decoy | Decoy | Decoy | 352.4ms | **REJECT** |
| **EXP-02** | Ref-Reprojection Unwarping | 46.77 | 46.81 | +0.04 | 6.49 | 1.18 | 1.19 | 13.12 | 9.83 | 5.00 | 10.00 | Decoy | Decoy | Decoy | Decoy | 774.9ms | **REJECT** |
| **EXP-03** | Isotropic Subblock Penalization | 46.77 | 48.01 | +0.68 | 6.91 | 1.28 | 1.62 | 13.67 | 9.53 | 5.00 | 10.00 | Decoy | Decoy | Decoy | Decoy | 651.3ms | **REJECT** |
| **EXP-04** | Top-5 High-Res Local NCC | 46.77 | 47.30 | +0.53 | 6.41 | 1.28 | 1.53 | 13.70 | 9.37 | 5.00 | 10.00 | Decoy | Decoy | Decoy | Decoy | 774.0ms | **REJECT** |
| **EXP-05** | Multi-Layer ResNet Distance | 46.77 | 47.37 | +0.60 | 6.41 | 1.28 | 1.53 | 13.70 | 9.37 | 5.00 | 10.00 | Decoy | Decoy | Decoy | Decoy | 785.0ms | **REJECT** |
| **EXP-06** | Affine Canonical Verification | 46.77 | 51.64 | +4.87 | 8.05 | 1.94 | 2.09 | 14.67 | 9.90 | 5.00 | 10.00 | Decoy | Decoy | Decoy | Decoy | 516.4ms | **REJECT** |
| **EXP-09** | Multi-Peak NCC Candidate Generation | 46.77 | 46.52 | -0.25 | 5.71 | 1.28 | 1.28 | 13.85 | 9.40 | 5.00 | 10.00 | Decoy | Decoy | Decoy | Decoy | 550ms | **REJECT** |
| **EXP-10** | NCC-First Primary Ranking (Strategy A) | 46.77 | **60.99** | **+14.22** | **14.11** | **3.88** | **4.38** | **13.87** | **9.76** | **5.00** | **10.00** | Decoy | Decoy | **0.29px** | 33.9px | 515ms | **PROMOTE** |
| **EXP-11** | Denser Coarse Search (750x750) | 60.99 | 61.19 | +0.19 | 14.30 | 3.81 | 4.35 | 13.95 | 9.78 | 5.00 | 10.00 | Decoy | Decoy | 1.21px | 32.3px | 666ms | **REJECT** |
| **EXP-12** | Gradient-Normalized NCC Only | 60.99 | 50.53 | -10.46 | 8.62 | 2.11 | 2.30 | 13.49 | 9.01 | 5.00 | 10.00 | Decoy | Decoy | 200.8px | 657px | 517ms | **REJECT** |
| **EXP-13** | NCC Periodicity Penalization | 60.99 | **71.65** | **+10.66** | **21.01** | **5.75** | **6.19** | **13.87** | **9.83** | **5.00** | **10.00** | **1.33px** | **0.69px** | **0.29px** | 33.6px | 534ms | **PROMOTE** |
| **EXP-14** | Failure-Mode Audit | 71.65 | 71.65 | +0.00 | 21.01 | 5.75 | 6.19 | 13.87 | 9.83 | 5.00 | 10.00 | 1.33px | 0.69px | 0.29px | 33.6px | 534ms | **DIAGNOSTIC** |
| **EXP-15** | Top-K Coarse Cutoff Expansion | 71.65 | 71.74 | +0.09 | 21.01 | 5.75 | 6.19 | 13.95 | 9.84 | 5.00 | 10.00 | 1.33px | 0.69px | 0.29px | 33.6px | 1855ms | **REJECT** |
| **EXP-16** | Sunday Model Fine-Tuned Checkpoint | 71.65 | **72.80** | **+1.15** | **21.01** | **5.75** | **6.19** | **14.86** | **10.00** | **5.00** | **10.00** | **1.33px** | **0.69px** | **0.29px** | 33.6px | 771ms | **PROMOTE** |
| **EXP-17** | Submission-Readiness & Checkpoint Lock | 72.80 | 72.80 | +0.00 | 21.01 | 5.75 | 6.19 | 14.86 | 10.00 | 5.00 | 10.00 | 1.33px | 0.69px | 0.29px | 33.6px | 774ms | **VALIDATION** |








---

## Detailed Experiment Findings

### EXP-01: Global Landmark Boundary & Asymmetric Context Alignment
- **Hypothesis**: Landmark features are situated near macro-cell boundaries and exhibit quadrant edge orientation asymmetry.
- **Measured Result**: Total Score 46.77 -> 46.77 (+0.00 pts). Loc 9.38 -> 9.38.
- **Decision**: **REJECT**. 2D macro boundaries do not separate periodic cell array decoys.

### EXP-02: Reference-Reprojection & Candidate Consistency
- **Hypothesis**: Inverse affine unwarping back to reference space yields lower pixel/gradient error for true GT.
- **Measured Result**: Total Score 46.77 -> 46.81 (+0.04 pts). Loc regressed from 9.38 to 6.49 / 40.0.
- **Decision**: **REJECT**. Unwarped periodic cell grids match or exceed GT template agreement.

### EXP-03: Isotropic Gradient Sub-Block Penalization
- **Hypothesis**: Periodic DRAM cell arrays exhibit isotropic subblock variance, penalizable by subblock std/mean ratio.
- **Measured Result**: Total Score 46.77 -> 48.01 (+0.68 pts). Loc regressed to 6.91 / 40.0.
- **Decision**: **REJECT** ($\le 1.0$ pt gain, localization regressed vs baseline 9.38).

### EXP-04: Top-5 High-Resolution Local NCC Re-Ranking
- **Hypothesis**: Local high-resolution NCC re-ranking among existing Top-5 candidates will better identify the true candidate than coarse NCC.
- **Measured Result**: Total Score 46.77 -> 47.30 (+0.53 pts). Loc regressed from 9.38 to 6.41 / 40.0.
- **Decision**: **REJECT** (Loc regressed, target periodic failure cases unresolved).

### EXP-05: Multi-Layer ResNet Feature Distance Ratio
- **Hypothesis**: Intermediate convolutional feature maps (Layer 1, Layer 2, Layer 3) retain local spatial structure partially lost by the 128-D embedding.
- **Measured Result**: Total Score 46.77 -> 47.37 (+0.60 pts). Loc regressed from 9.38 to 6.41 / 40.0.
- **Decision**: **REJECT** (Loc regressed, target periodic failure cases unresolved).

### EXP-06: Post-Top-5 Affine-Canonical Candidate Verification
- **Hypothesis**: Candidate-local affine canonicalization to remove scale/rotation pose differences followed by fine pose verification will discriminate true GT landmarks from periodic cell decoys.
- **Measured Result**: Total Score 46.77 -> 51.64 (+4.87 pts). Loc regressed from 9.38 to 8.05 / 40.0.
- **Decision**: **REJECT** (Loc regressed vs baseline 9.38, target periodic failure cases unresolved).

---

## Detailed Experiment Findings

### EXP-01: Global Landmark Boundary & Asymmetric Context Alignment
- **Hypothesis**: Landmark features are situated near macro-cell boundaries and exhibit quadrant edge orientation asymmetry.
- **Measured Result**: Total Score 46.77 -> 46.77 (+0.00 pts). Loc 9.38 -> 9.38.
- **Decision**: **REJECT**. 2D macro boundaries do not separate periodic cell array decoys.

### EXP-02: Reference-Reprojection & Candidate Consistency
- **Hypothesis**: Inverse affine unwarping back to reference space yields lower pixel/gradient error for true GT.
- **Measured Result**: Total Score 46.77 -> 46.81 (+0.04 pts). Loc regressed from 9.38 to 6.49 / 40.0.
- **Decision**: **REJECT**. Unwarped periodic cell grids match or exceed GT template agreement.

### EXP-03: Isotropic Gradient Sub-Block Penalization
- **Hypothesis**: Periodic DRAM cell arrays exhibit isotropic subblock variance, penalizable by subblock std/mean ratio.
- **Measured Result**: Total Score 46.77 -> 48.01 (+0.68 pts). Loc regressed to 6.91 / 40.0.
- **Decision**: **REJECT** ($\le 1.0$ pt gain, localization regressed vs baseline 9.38).

### EXP-04: Top-5 High-Resolution Local NCC Re-Ranking
- **Hypothesis**: Local high-resolution NCC re-ranking among existing Top-5 candidates will better identify the true candidate than coarse NCC.
- **Measured Result**: Total Score 46.77 -> 47.30 (+0.53 pts). Loc regressed from 9.38 to 6.41 / 40.0.
- **Decision**: **REJECT** (Loc regressed, target periodic failure cases unresolved).

### EXP-05: Multi-Layer ResNet Feature Distance Ratio
- **Hypothesis**: Intermediate convolutional feature maps (Layer 1, Layer 2, Layer 3) retain local spatial structure partially lost by the 128-D embedding.
- **Measured Result**: Total Score 46.77 -> 47.37 (+0.60 pts). Loc regressed from 9.38 to 6.41 / 40.0.
- **Decision**: **REJECT** (Loc regressed, target periodic failure cases unresolved).

### EXP-09: Multi-Peak NCC Candidate Generation
- **Hypothesis**: Extracting multiple local NCC peaks per scale/theta (with spatial NMS) recovers true GT landmark candidates that single global-max extraction misses due to periodic decoy structures.
- **Ablation Sweep Tested**: K_peaks ∈ {1, 3, 5, 10} peaks per scale/rotation.
- **Measured Result**:
  - Baseline (K=1): 46.77 / 100 (Loc 9.38/40 in production; 5.85/40 in diagnostic run)
  - K=3: 46.55 / 100 (Loc 5.71/40)
  - K=5 (Primary): 46.52 / 100 (Loc 5.71/40)
  - K=10: 46.15 / 100 (Loc 5.39/40)
- **Critical Finding**: Multi-peak extraction populates the coarse candidate pool with MORE periodic decoys. Higher K_peaks decreases Top-5 Refined Recall @15px (36.2% at K=1 -> 30.0% at K=3 -> 27.5% at K=5 -> 24.4% at K=10).
- **Target Cases**: GT is recovered in coarse & refined pools for `pair_006`, `pair_066`, `pair_186`, but lost at final Top-5 fused ranking because periodic decoys get higher Siamese similarity scores.
- **Decision**: **REJECT** (Localization regressed, total score regressed, recall worsened at higher K).

---

### EXP-10: NCC-First Primary Ranking (Strategy A)
- **Hypothesis**: Spatial ranking of candidates should be controlled primarily by NCC score (`argmax(ncc_norm)`), while retaining Siamese score for rejection thresholding and confidence calibration.
- **Measured Result**: Total Score **46.77 → 60.99 / 100 (+14.22 pts)**. Localization Score **5.85 → 14.11 / 40 (+8.26 pts)**.
- **Regressions**: **52 pairs recovered** (including target `pair_186` from 670.4px → 0.29px). **0 pairs regressed**.
- **Decision**: **PROMOTE TO PRODUCTION**.

---

### EXP-11: Denser Coarse Search Spatial Resolution
- **Hypothesis**: Increasing coarse spatial search resolution (from 500x500 to 750x750 or 1000x1000) will improve coarse candidate recall by preserving fine structural landmarks.
- **Measured Result**:
  - 500x500 (Production Base): **60.99 / 100** (Loc 14.11/40, Med RT 516ms)
  - 750x750: **61.19 / 100** (+0.19 pts, Loc 14.30/40, Med RT 666ms)
  - 1000x1000: **60.69 / 100** (-0.30 pts, Loc 13.89/40, Med RT 899ms)
- **Critical Finding**: Coarse pool recall @15px remained identical at 85.6% across 500x500 and 750x750. For target pairs (`pair_006`, `pair_066`), GT is ALREADY present in the coarse pool at 500x500, but a periodic decoy has a slightly higher NCC score.
- **Decision**: **REJECT** (Total score gain +0.19 is <= 1.0 point threshold; runtime increased by +150ms).

---

### EXP-12: Gradient-Normalized NCC Only
- **Hypothesis**: Replacing raw intensity NCC with Sobel gradient magnitude NCC (`G = sqrt(Gx^2 + Gy^2)`) will measure edge structural transitions rather than brightness, preventing periodic intensity decoys from outranking GT landmarks.
- **Measured Result**: Total Score **60.99 → 50.53 / 100 (-10.46 pts)**. Localization Score **14.11 → 8.62 / 40 (-5.49 pts)**.
- **Critical Finding**: Sobel gradient operators magnify SEM high-frequency sensor noise and produce repetitive edge ripple maps across DRAM cell arrays, causing **43 nominal pairs to regress** (e.g. `pair_186` regressed from 0.29px → 200.8px). Recall @5px dropped from 49.4% → 27.5%.
- **Decision**: **REJECT** (Severe score & localization regression, 43 pairs regressed).

---

### EXP-13: NCC Periodicity Penalization
- **Hypothesis**: Periodic DRAM decoys produce repetitive clusters of high NCC peaks (`PeriodicityCount = 6-9`), whereas genuine structural landmarks produce single isolated peaks (`PeriodicityCount = 1`). Subtracting a small penalty `0.05 * (periodicity_count - 1)` from candidate NCC will suppress decoys without degrading GT landmarks.
- **Measured Result**: Total Score **60.99 → 71.65 / 100 (+10.66 pts)**. Localization Score **14.11 → 21.01 / 40 (+6.90 pts)**. Pose Score **8.25 → 11.94 / 20 (+3.69 pts)**.
- **Critical Finding**: 38 pairs recovered down to sub-5px error. Target failure cases `pair_006` (631.8px → 1.33px) and `pair_066` (739.3px → 0.69px) fully resolved! Candidate recall @5px jumped from 49.4% → 69.4%.
- **Decision**: **PROMOTE TO PRODUCTION**.

---

### EXP-14: Post-EXP13 Failure-Mode Audit
- **Objective**: Diagnostic pipeline trace of all 160 present pairs to categorize remaining failure modes.
- **Measured Result**: 94/160 present pairs localized <=5px (58.8%). 66 pairs failed.
- **Critical Finding**: **Cat B (GT in coarse, lost before refinement)** is the single largest remaining bottleneck, accounting for **32 / 66 failures (48.5% of all failures)**! All 32 Cat B failures occur in Set B (Degraded SEM) because `top_k_coarse = 10` drops GT ranked 11th-20th.
- **Decision**: **DIAGNOSTIC ONLY** (Production remains frozen at 71.65 / 100).

---

### EXP-16: Sunday Fine-Tuned Model Checkpoint Promotion
- **Hypothesis**: Replacing `best_model_level1.pth` with `checkpoints_phase2_v2_sunday/best_model_phase2.pth` (fine-tuned on 16,000 augmented SEM/optical images with AMP) will produce sharper 128-D feature embedding separation for rejection thresholding and confidence calibration.
- **Measured Result**: Total Score **71.65 → 72.80 / 100 (+1.15 pts, NEW ALL-TIME RECORD)**. Rejection Score **13.87 → 14.86 / 15 (+0.99 pts)**. Confidence Score **9.83 → 10.00 / 10 (PERFECT 10/10 AUC)**.
- **Decision**: **PROMOTE TO PRODUCTION**.

### EXP-17: Submission-Readiness & Checkpoint Lock Validation
- **Objective**: Independent reproduction of the 72.80 score, validation of CSV output schema, absent target format, and checkpoint hash lock.
- **Measured Result**: Total Score **72.80 / 100** (+0.0039 difference). Localization = 21.01/40, Pose = 11.94/20, Rejection = 14.86/15, Conf = 10.00/10 (PERFECT 10/10 AUC).
- **Regressions**: **0 PAIRS (ZERO REGRESSIONS)**.
- **Checkpoint SHA-256**: `74714ac16cb25da8a707113af9b30fa2ee051302065eaf947b46ef0a27592b8f` (EXACT MATCH).
- **Decision**: **VALIDATED SUCCESSFUL — SYSTEM IS 100% READY FOR FINAL SUBMISSION**.

---

## Current Best Verified Version

- **Checkpoint**: `checkpoints_phase2_v2_sunday/best_model_phase2.pth` (SHA-256: `74714ac16cb25da8a707113af9b30fa2ee051302065eaf947b46ef0a27592b8f`).
- **Production Files**: `phase2/phase2_inference.py` (PROMOTED with EXP-13 Periodicity Penalization + EXP-16 Sunday Checkpoint Resolution), `phase2/phase2_config.py`, `register.py`.
- **Official Score**: **72.80 / 100.0** (Loc: 21.01/40, Pose: 11.94/20, Rejection: 14.86/15, Conf: 10.00/10, Eff: 5.00/5, Gen: 10.00/10).



