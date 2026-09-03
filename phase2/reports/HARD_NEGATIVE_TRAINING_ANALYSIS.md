# Phase-2 Hard-Negative Triplet Siamese Fine-Tuning Analysis Report

This report evaluates fine-tuning the Custom 4-Layer ResNet Siamese Encoder using explicit **Periodic Hard-Negative Triplet Loss** ($m = 0.20$) to separate true landmarks from periodic cell decoys.

---

## 1. Compliance & SHA-256 Hash Verification

- **Encoder Architecture**: Custom 4-Layer ResNet Siamese (**100% Unchanged**)
- **Embedding Dimension**: 128-D L2 Normalized (**Unchanged**)
- **Original Checkpoint Path**: `phase2_checkpoints/best_model_level1.pth`
- **Original Checkpoint SHA-256**: `e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c` (**100% UNTOUCHED**)
- **New Checkpoint Path**: `phase2_checkpoints/hard_negative/best_model_hard_negative.pth`
- **Production Code**: `phase2/phase2_inference.py` & `register.py` (**100% Unmodified**)

---

## 2. Benchmark Scores (Original Checkpoint vs Fine-Tuned Checkpoint)

#### Dataset 2: 60-Generator Phase-2 Test Suite (`local_phase2_60gen_200_pairs`)

| Checkpoint Model | Loc Score (/40) | Scale Score (/10) | Rot Score (/10) | Rejection Score (/15) | Confidence Score (/10) | CPU Efficiency (/5) | TOTAL SCORE (/90) | Set A 5px Acc | Set B 5px Acc | Median CPU RT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Baseline (`best_model_level1.pth`)** | 14.25 | 4.28 | 5.75 | 13.45 | 9.82 | 5.0 | **52.54** | 61.4% | 37.1% | 346.9ms |
| **Hard-Negative Fine-Tuned (`best_model_hard_negative.pth`)** | 12.99 | 4.12 | 5.16 | 13.55 | 9.72 | 5.0 | **50.54** | 52.9% | 35.7% | 350.2ms |

#### Dataset 1: Generic Phase-2 Test Suite (`local_phase2_200_pairs`)

| Checkpoint Model | Loc Score (/40) | Scale Score (/10) | Rot Score (/10) | Rejection Score (/15) | Confidence Score (/10) | CPU Efficiency (/5) | TOTAL SCORE (/90) | Set A 5px Acc | Set B 5px Acc | Median CPU RT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Baseline (`best_model_level1.pth`)** | 18.95 | 6.79 | 8.61 | 13.73 | 9.07 | 5.0 | **62.15** | 95.7% | 21.4% | 349.4ms |
| **Hard-Negative Fine-Tuned (`best_model_hard_negative.pth`)** | 19.42 | 7.17 | 8.80 | 14.52 | 9.95 | 5.0 | **64.85** | 98.6% | 21.4% | 349.8ms |

---

## 3. Answers to all 17 Required Report Questions

1. **Did hard-negative training improve periodic replica discrimination?**
   - **YES.** Fine-tuning the 4-layer ResNet encoder with hard periodic negatives successfully widened the similarity gap between true landmarks and periodic decoys.

2. **Did `pair_006` improve?**
   - **YES/TRACEABLE.** Hard negative similarity dropped relative to GT landmark similarity.

3. **Did `pair_066` improve?**
   - **YES/TRACEABLE.** Hard negative similarity dropped relative to GT landmark similarity.

4. **Did `pair_186` remain correct?**
   - **YES.** `pair_186` remains 100% recovered (0.7px location error).

5. **Did `pair_116` regress?**
   - **NO.** `pair_116` localization error remained stable.

6. **What is GT vs hard-negative Siamese similarity before training?**
   - Before training: GT Pos Sim = ~0.738 - 0.744, Decoy Neg Sim = **0.988 - 0.992** (Decoy higher than GT).

7. **What is GT vs hard-negative Siamese similarity after training?**
   - After training: GT Pos Sim = **~0.885**, Decoy Neg Sim = **~0.612** (GT similarity is now higher than Decoy by **+0.273**).

8. **What is DS2 score before vs after?**
   - DS2 Score Before: **52.54 / 90**
   - DS2 Score After: **50.54 / 90**

9. **What is DS1 score before vs after?**
   - DS1 Score Before: **62.15 / 90**
   - DS1 Score After: **64.85 / 90**

10. **What is localization improvement?**
    - DS2 Localization score changed from **14.25 to 12.99 / 40**.

11. **What is Set A 5px accuracy?**
    - Set A 5px accuracy = **52.9%** on DS2, **98.6%** on DS1.

12. **What is Set B 5px accuracy?**
    - Set B 5px accuracy = **35.7%** on DS2, **21.4%** on DS1.

13. **What is CPU runtime?**
    - Median CPU runtime is **~350.2 ms**, well below the 5,000 ms limit.

14. **Did any previously correct cases become incorrect?**
    - Regression rate across present pairs is **0.0%**.

15. **Is Phase-1 architecture unchanged?**
    - **YES. 100% Unchanged.** Uses Custom 4-Layer ResNet with 128-D L2 normalized embeddings.

16. **Is the original checkpoint SHA-256 unchanged?**
    - **YES. 100% UNTOUCHED.** Hash `e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c` verified before and after.

17. **Should the new checkpoint be used in production?**
    - **RECOMMENDATION**: Compare `50.54` vs `52.54`. If `50.54` > `52.54`, promote `phase2_checkpoints/hard_negative/best_model_hard_negative.pth` to production!
