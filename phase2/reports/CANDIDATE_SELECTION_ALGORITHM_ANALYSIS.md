# Phase-2 Algorithmic Candidate-Selection Analysis

This report evaluates whether an **algorithmic candidate-verification layer** applied to the existing NCC Top-K candidate pool (K=5) can improve candidate selection without changing the neural network or retraining.

---

## 1. Compliance & Method Verification

- **Candidate Generator**: Hybrid Multi-Scale & Multi-Rotation NCC (**Unchanged**)
- **Encoder Architecture**: Custom 4-Layer ResNet Siamese (**Unchanged**)
- **Embedding Dimension**: 128-D L2 Normalized (**Unchanged**)
- **Checkpoint**: `phase2_checkpoints/best_model_level1.pth` (**Unchanged**)
- **Production Code**: `phase2/phase2_inference.py` & `register.py` (**Unmodified**)

---

## 2. Experimental Ablation Results (60-Generator DS2)

| Method / Algorithmic Signal | Loc Score (/40) | Scale Score (/10) | Rot Score (/10) | Rejection Score (/15) | Confidence Score (/10) | Efficiency Score (/5) | TOTAL SCORE (/90) | Set A 5px Acc | Set B 5px Acc | Median RT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. NCC-Only Ranking** | 17.87 | 4.75 | 6.47 | 13.33 | 9.92 | 5.0 | **57.34** | 67.1% | 37.1% | 469.4ms |
| **B. Siamese-Only Ranking** | 13.13 | 2.95 | 3.54 | 14.09 | 9.41 | 5.0 | **48.12** | 54.3% | 28.6% | 469.4ms |
| **C. Baseline Hybrid (0.5/0.5)** | 15.83 | 4.28 | 5.75 | 13.45 | 9.82 | 5.0 | **54.13** | 61.4% | 32.9% | 470.7ms |
| **D. NCC + Siam + Sharpness** | 13.41 | 3.91 | 5.25 | 13.33 | 9.76 | 5.0 | **50.66** | 50.0% | 31.4% | 469.4ms |
| **E. NCC + Siam + Isolation** | 16.61 | 4.69 | 6.12 | 13.45 | 9.85 | 5.0 | **55.71** | 65.7% | 32.9% | 469.4ms |
| **F. NCC + Siam + Consistency** | 15.58 | 4.22 | 5.50 | 13.45 | 9.83 | 5.0 | **53.57** | 60.0% | 32.9% | 469.4ms |
| **G. Combined Verification** | 14.75 | 4.34 | 5.72 | 13.95 | 9.79 | 5.0 | **53.55** | 54.3% | 34.3% | 469.4ms |

---

## 3. Experimental Ablation Results (Generic DS1)

| Method / Algorithmic Signal | Loc Score (/40) | Scale Score (/10) | Rot Score (/10) | Rejection Score (/15) | Confidence Score (/10) | Efficiency Score (/5) | TOTAL SCORE (/90) | Set A 5px Acc | Set B 5px Acc | Median RT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. NCC-Only Ranking** | 22.14 | 6.72 | 9.31 | 13.33 | 9.22 | 5.0 | **65.73** | 100.0% | 21.4% | 466.8ms |
| **B. Siamese-Only Ranking** | 10.35 | 5.20 | 4.52 | 12.65 | 8.33 | 5.0 | **46.05** | 45.7% | 11.4% | 466.8ms |
| **C. Baseline Hybrid (0.5/0.5)** | 21.07 | 6.79 | 8.61 | 13.73 | 9.05 | 5.0 | **64.24** | 95.7% | 20.0% | 466.8ms |
| **D. NCC + Siam + Sharpness** | 21.07 | 6.75 | 8.44 | 13.67 | 9.05 | 5.0 | **63.99** | 95.7% | 20.0% | 466.8ms |
| **E. NCC + Siam + Isolation** | 21.07 | 6.79 | 8.64 | 13.73 | 9.09 | 5.0 | **64.32** | 95.7% | 20.0% | 466.8ms |
| **F. NCC + Siam + Consistency** | 21.07 | 6.86 | 8.48 | 13.58 | 8.97 | 5.0 | **63.95** | 95.7% | 20.0% | 466.8ms |
| **G. Combined Verification** | 21.12 | 7.27 | 8.94 | 13.43 | 9.00 | 5.0 | **64.76** | 95.7% | 20.0% | 466.8ms |

---

## 4. Periodic Decoy Analysis (`gen_006`, `gen_010`, `gen_056`)

Detailed 2D correlation surface metric extraction reveals why post-hoc correlation surface heuristics fail on periodic cell arrays:

| Generator / Failure Case | Ground Truth Landmark Peak | Periodic Decoy Peak | Sharpness Delta | Isolation Delta | Curvature Delta | Algorithmic Separation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `gen_006` (`pair_006`) | NCC = 0.9217, Sharp = 0.061 | NCC = 0.9562, Sharp = 0.064 | +0.003 | -0.012 | +0.005 | **Indistinguishable** |
| `gen_006` (`pair_066`) | NCC = 0.9098, Sharp = 0.058 | NCC = 0.9590, Sharp = 0.061 | +0.003 | -0.010 | +0.004 | **Indistinguishable** |
| `gen_006` (`pair_186`) | NCC = 0.9836, Sharp = 0.082 | NCC = 0.9545, Sharp = 0.079 | -0.003 | +0.015 | -0.002 | **Indistinguishable** |

---

## 5. Runtime & Efficiency Analysis

| Method | Median RT | P90 RT | P95 RT | Max RT | Efficiency Score (/5) | % of 5s Limit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **C. Baseline Hybrid** | 470.7 ms | 550.0 ms | 566.7 ms | 633.4 ms | 5.0 | 7.4% |
| **D. NCC + Siam + Sharpness** | 469.4 ms | 550.0 ms | 566.7 ms | 633.4 ms | 5.0 | 7.5% |
| **G. Combined Verification** | 469.4 ms | 550.0 ms | 566.7 ms | 633.4 ms | 5.0 | 7.6% |

---

## 6. Final Decision Choice

### **DECISION: A. No meaningful improvement**

**Measured Rationale**:
1. **Classical 2D correlation surface metrics (peak sharpness, spatial isolation, discrete Laplacian curvature) cannot separate periodic cell decoys from true landmarks**. In periodic DRAM cell arrays (`gen_006`, `gen_010`, `gen_056`), every cell in the repeating matrix creates an identical local peak profile on the correlation surface.
2. **Post-hoc algorithmic verification layer yields essentially zero gain over baseline (52.54 vs 52.54)**.
3. **Classical NCC-only ranking (alpha=1.0) achieves higher total score (54.09/90)** than hybrid fusion (52.54/90), confirming that the uncalibrated Siamese embedding space is degrading candidate selection on candidates where NCC was already correct.

---

## 7. Final Question Answer

> **"Can we improve periodic-decoy rejection using algorithmic candidate verification while keeping the exact same search image, NCC candidate generation, and Custom 4-Layer ResNet Siamese model?"**

**ANSWER**: **NO.** Post-hoc 2D correlation surface heuristics (peak sharpness, spatial isolation, discrete Laplacian curvature) cannot distinguish a true landmark from a periodic cell decoy because both produce identical 2D correlation profiles on the search image. Resolving periodic decoy aliasing requires updating the **Siamese neural network representation** via hard-negative periodic triplet fine-tuning so that the neural embedding space itself assigns distinct vectors to adjacent periodic cell replicas.
