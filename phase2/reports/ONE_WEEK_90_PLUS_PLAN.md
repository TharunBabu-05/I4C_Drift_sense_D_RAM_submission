# One-Week Strategic Engineering Plan to Achieve >= 90/100 Phase-2 Score

This document outlines a targeted, step-by-step 7-day engineering plan to elevate the Phase-2 competition score from **46.77 / 100.0** to **>= 90.00 / 100.0**.

---

## Executive Summary & Core Engineering Strategy

The official 100-point audit proved that **Candidate Generation is NOT the bottleneck**:
- **Candidate Generation Recall**: **99.4%** (GT candidate inside NCC Top-5 for 159 out of 160 present pairs).
- **Maximum Theoretical Score**: **97.70 / 100.0** if candidate selection is perfect.

Therefore, achieving **>= 90/100** requires **ZERO changes** to:
- Candidate generation (Multi-scale NCC)
- ResNet architecture & weights
- CAD 60-generators & search images

All gains will be achieved by introducing a **High-Precision Algorithmic Candidate Selection Layer** that operates strictly after Top-5 candidate generation.

---

## 7-Day Implementation Roadmap

```
Day 1: Architectural Alignment & Landmark Boundary Verification
      │
Day 2: Structural Keypoint & Non-Periodic Feature Verification
      │
Day 3: Hierarchical Multi-Scale Edge-Vector Verification
      │
Day 4: Adaptive Rejection Thresholding & Re-calibration
      │
Day 5: HighRes Sub-Pixel Parabola Refinement Integration
      │
Day 6: Complete End-to-End Test Suite Validation (DS1 + DS2)
      │
Day 7: Code Freeze, Verification & Submission Build
```

---

### Day 1: Global Landmark Boundary & Context Alignment
- **Goal**: Resolve periodic cell array symmetry by measuring spatial alignment relative to chip macro-cell boundaries.
- **Action**: Compute distance ratios from candidate crop center to nearest high-contrast macro-boundary edges.
- **Expected Score Gain**: **+25.0 Points** (Loc: 9.38 -> 24.0, Pose: 9.0 -> 14.5).

### Day 2: Structural Corner & Asymmetric Keypoint Matching
- **Goal**: Distinguish true landmark features from uniform cell matrix grids.
- **Action**: Extract Harris corner density and FAST keypoint spatial asymmetry within candidate crops.
- **Expected Score Gain**: **+15.0 Points** (Loc: 24.0 -> 32.0, Pose: 14.5 -> 17.5).

### Day 3: Multi-Scale Gradient Orientation Matching
- **Goal**: Eliminate high-frequency periodic decoy preference.
- **Action**: Calculate Sobel gradient orientation histograms across inner (50px) vs outer (100px) candidate rings.
- **Expected Score Gain**: **+8.0 Points** (Loc: 32.0 -> 36.0, Pose: 17.5 -> 18.5).

### Day 4: Adaptive Rejection Thresholding & Calibration
- **Goal**: Maximize Rejection F1 score to 14.8+ / 15.0.
- **Action**: Implement dynamic rejection gating $\tau_{\text{adaptive}} = f(\text{Top-1 Score} - \text{Top-2 Score})$.
- **Expected Score Gain**: **+1.2 Points** (Rejection: 13.70 -> 14.90).

### Day 5: Sub-Pixel Local Crop Parabola Refinement
- **Goal**: Convert 2px - 5px localized candidates into <= 1px precision (1.00 credit).
- **Action**: Apply 2D 3x3 parabolic peak interpolation around selected candidate.
- **Expected Score Gain**: **+3.0 Points** (Loc: 36.0 -> 38.5).

### Day 6: Benchmark Suite & Stress Testing
- **Goal**: Verify zero-regression across all 400 test pairs (DS1 + DS2).
- **Action**: Run full 100-point benchmark evaluation scripts `evaluate_phase2_inference.py`.

### Day 7: Final Verification & Code Freeze
- **Goal**: Finalize submission code.
- **Verification Criteria**:
  1. Official Score $\ge 90.00 / 100.0$
  2. Median CPU Runtime $\le 500\text{ms}$
  3. SHA-256 Checkpoint Verification
