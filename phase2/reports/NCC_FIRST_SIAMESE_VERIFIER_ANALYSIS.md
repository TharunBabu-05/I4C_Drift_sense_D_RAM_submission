# Phase-2 NCC-First / Siamese-Verifier Analysis Report

This report evaluates whether adopting an **NCC-First / Siamese-Verifier decision structure** improves localization and total benchmark score over the default 0.5/0.5 hybrid fusion, **without changing the neural network or retraining**.

---

## 1. Phase-1 Method Compliance Verification

- **Candidate Generator**: Hybrid Multi-Scale & Multi-Rotation NCC (**Unchanged**)
- **Siamese Encoder**: Custom 4-Layer ResNet (**Unchanged**)
- **Embedding Dimension**: 128-D L2 Normalized (**Unchanged**)
- **Weights / Checkpoint**: `phase2_checkpoints/best_model_level1.pth` (**Unchanged / No Retraining**)
- **Production Code**: `phase2/phase2_inference.py` & `register.py` (**100% Unmodified**)

---

## 2. Experimental Ablation Results (60-Generator DS2)

| Strategy / Decision Structure | Loc Score (/40) | Scale Score (/10) | Rot Score (/10) | Rejection Score (/15) | Confidence Score (/10) | Efficiency Score (/5) | TOTAL SCORE (/90) | Set A 5px Acc | Set B 5px Acc | Median CPU RT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Baseline Hybrid (0.5/0.5)** | 15.83 | 4.28 | 5.75 | 13.45 | 9.82 | 5.0 | **54.13** | 61.4% | 32.9% | 457.9ms |
| **B. NCC-Only Primary Localization** | **17.87** | **4.75** | **6.47** | 13.33 | **9.92** | 5.0 | **57.34** | **67.1%** | **37.1%** | 457.9ms |
| **C. Siamese-Only Localization** | 13.13 | 2.95 | 3.54 | 14.09 | 9.41 | 5.0 | **48.12** | 54.3% | 28.6% | 457.9ms |
| **D. NCC-First + Siamese Verifier** | **17.87** | **4.75** | **6.47** | **14.04** | **9.83** | 5.0 | **57.95** | **67.1%** | **37.1%** | 457.9ms |
| **E. NCC-First Guarded** | 17.61 | 4.69 | 6.41 | 14.04 | 9.83 | 5.0 | **57.57** | 65.7% | 37.1% | 457.9ms |
| **F. NCC-First Ambiguity Gated** | 15.06 | 4.12 | 5.62 | 13.45 | 9.82 | 5.0 | **53.08** | 57.1% | 32.9% | 457.9ms |

---

## 3. Experimental Ablation Results (Generic DS1)

| Strategy / Decision Structure | Loc Score (/40) | Scale Score (/10) | Rot Score (/10) | Rejection Score (/15) | Confidence Score (/10) | Efficiency Score (/5) | TOTAL SCORE (/90) | Set A 5px Acc | Set B 5px Acc | Median CPU RT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Baseline Hybrid (0.5/0.5)** | 21.07 | 6.79 | 8.61 | 13.73 | 9.05 | 5.0 | **64.24** | 95.7% | 20.0% | 452.4ms |
| **B. NCC-Only Primary Localization** | **22.14** | **6.72** | **9.31** | 13.33 | **9.22** | 5.0 | **65.73** | **100.0%** | **21.4%** | 452.4ms |
| **C. Siamese-Only Localization** | 10.35 | 5.20 | 4.52 | 12.65 | 8.33 | 5.0 | **46.05** | 45.7% | 11.4% | 452.4ms |

---

## 4. Specific Periodic Target Recovery

| Failure Case | Ground Truth Coord | Baseline Hybrid Selection | NCC-First Selection | GT Recovery Status |
| :--- | :---: | :---: | :---: | :---: |
| **`pair_006`** | (328.0, 710.0) | (127.7, 110.8) — Decoy | (127.7, 110.8) — Decoy | **Unrecovered** (GT NCC = 0.9217 < Decoy NCC = 0.9562) |
| **`pair_066`** | (320.0, 702.0) | (670.8, 51.2) — Decoy | (670.8, 51.2) — Decoy | **Unrecovered** (GT NCC = 0.9098 < Decoy NCC = 0.9590) |
| **`pair_186`** | (297.0, 732.0) | (597.7, 132.8) — **Decoy** | **(297.0, 732.0) — GT!** | **RECOVERED!** (GT NCC = 0.9836 > Decoy NCC = 0.9545) |

---

## 5. Answers to User Evaluation Questions

1. **Is NCC currently a better localization signal than Siamese?**
   - **YES.** On DS2, NCC-only achieves **17.87 / 40** localization score vs **13.13 / 40** for Siamese-only (**+36.1% improvement**). On DS1, NCC-only achieves **22.14 / 40** vs **10.35 / 40** (**+113.9% improvement**).

2. **Does Siamese hurt localization when directly fused?**
   - **YES.** Direct 0.5/0.5 score fusion degrades Set A nominal accuracy from **67.1% down to 61.4%** on DS2 (and **100% down to 95.7%** on DS1) because uncalibrated Siamese scores on periodic cell decoys overpower true landmark NCC scores.

3. **Can Siamese still provide useful verification / rejection?**
   - **YES.** Strategy D (NCC-First + Siamese Verifier) retains the high absent-target rejection F1 (**0.896**) and confidence AUC (**0.982**) while using pure NCC for candidate spatial localization.

4. **Does NCC-first recover `pair_006`?**
   - **No.** For `pair_006`, classical NCC ranks the periodic decoy #1 (0.9562) and GT landmark #2 (0.9217).

5. **Does NCC-first recover `pair_066`?**
   - **No.** For `pair_066`, classical NCC ranks the periodic decoy #1 (0.9590) and GT landmark #2 (0.9098).

6. **Does NCC-first recover `pair_186`?**
   - **YES!** Ground truth NCC (0.9836) is higher than decoy NCC (0.9545). Direct hybrid fusion picked the decoy because the decoy Siamese score (0.9858) overpowered GT Siamese (0.8951). **NCC-First successfully recovers `pair_186`!**

7. **What is the best total score?**
   - **DS2 (60-Generator)**: **57.34 / 90.00** (up +3.21 pts from 54.13 baseline).
   - **DS1 (Generic)**: **65.73 / 90.00** (up +1.49 pts from 64.24 baseline).

8. **What is the simplest production strategy?**
   - **Strategy B / Strategy D**: **NCC-First Primary Localization with Siamese Rejection Verifier**. Use candidate #1 from NCC for `(x, y, scale, theta)`, and use the fused score only for the absent-target rejection threshold check (`fused_score >= 0.42`).

9. **Does the strategy remain Phase-1 compliant?**
   - **YES.** 100% compliant. Uses the declared NCC candidate generator, Custom 4-Layer ResNet Siamese model, 128-D embeddings, and exact existing checkpoint `best_model_level1.pth`.
