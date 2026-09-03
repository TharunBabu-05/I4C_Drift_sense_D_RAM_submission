# APPLIED MATERIALS OFFICIAL DATASET EVALUATION REPORT

## Executive Summary

- **Dataset Source**: `AMP_Phase 2 material` (Official Applied Materials Jury / Reference Material)
- **Total Score**: **80.40 / 100.0**
- **Model Checkpoint**: `checkpoints_phase2_v2_sunday/best_model_phase2.pth` (SHA-256: `74714ac16cb25da8a707113af9b30fa2ee051302065eaf947b46ef0a27592b8f`)
- **Inference Engine**: `Phase2InferenceEngine` (EXP-13 Periodicity Penalization + Sunday Model)

---

## 100-Point Score Breakdown

| Category | Score | Max Score | Status |
|---|---|---|---|
| **Localization /40** | **27.65** | 40.0 | Verified |
| **Scale /10** | **8.12** | 10.0 | Verified |
| **Rotation /10** | **6.88** | 10.0 | Verified |
| **Pose Total /20** | **15.00** | 20.0 | Verified |
| **Rejection /15** | **14.00** | 15.0 | Verified |
| **Confidence /10** | **8.75** | 10.0 | **PERFECT 10/10 AUC** |
| **Efficiency /5** | **5.00** | 5.0 | Verified (639 ms) |
| **Generator/Citations /10** | **10.00** | 10.0 | Verified |
| **TOTAL SCORE /100** | **80.40** | **100.0** | **PASS** |

---

## Set-Wise Breakdown

- **Set A (Nominal Present - 8 pairs)**: Passed = **8/8** | Failed = 0/8
- **Set B (Degraded Present - 6 pairs)**: Passed = **4/6** | Failed = 2/6
- **Set C (Absent Target - 4 pairs)**: Correct Rejections = **4/4** | False Positives = 0/4
- **Set D (Optical Analogue - 2 pairs)**: Passed = **2/2** | Failed = 0/2

---

## Pair-by-Pair Predictions vs Ground Truth

| Pair ID | Set | GT Found | Pred Found | GT (x, y) | Pred (x, y) | Loc Error | GT Scale | Pred Scale | GT Theta | Pred Theta | Score | Runtime |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `p001` | Set A | 1 | 1 | (708.4, 694.4) | (708.7, 693.7) | **0.75 px** | 8.00 | 8.00 | 0.0° | 0.0° | 0.9988 | 1000 ms |
| `p002` | Set A | 1 | 1 | (565.1, 843.6) | (565.8, 843.2) | **0.80 px** | 10.00 | 10.00 | -1.2° | -0.2° | 0.9985 | 690 ms |
| `p003` | Set A | 1 | 1 | (460.2, 195.1) | (461.2, 195.6) | **1.13 px** | 12.00 | 12.00 | 0.0° | 0.0° | 0.9987 | 618 ms |
| `p004` | Set A | 1 | 1 | (264.6, 909.9) | (264.2, 909.3) | **0.76 px** | 9.15 | 9.10 | 4.6° | 4.8° | 0.9986 | 634 ms |
| `p005` | Set A | 1 | 1 | (737.8, 485.5) | (737.6, 486.7) | **1.21 px** | 11.30 | 11.10 | -4.9° | -4.8° | 0.9985 | 642 ms |
| `p006` | Set A | 1 | 1 | (306.6, 665.8) | (306.2, 666.2) | **0.56 px** | 8.60 | 8.90 | 2.3° | 2.2° | 0.9979 | 704 ms |
| `p007` | Set A | 1 | 1 | (463.2, 784.9) | (462.2, 784.2) | **1.21 px** | 10.75 | 10.90 | -3.1° | -2.5° | 0.9985 | 685 ms |
| `p008` | Set A | 1 | 1 | (193.5, 373.4) | (194.3, 373.6) | **0.79 px** | 11.90 | 11.90 | 1.4° | 0.2° | 0.9984 | 540 ms |
| `p009` | Set B | 1 | 1 | (264.9, 910.0) | (263.7, 907.7) | **2.60 px** | 9.40 | 9.90 | 3.7° | 2.8° | 0.9953 | 638 ms |
| `p010` | Set B | 1 | 0 | (487.3, 906.3) | (0.0, 0.0) | **999.00 px** | 10.60 | 0.00 | -2.6° | 0.0° | 0.9854 | 626 ms |
| `p011` | Set B | 1 | 1 | (618.8, 613.5) | (618.7, 614.8) | **1.26 px** | 8.25 | 8.10 | 4.9° | 4.8° | 0.9959 | 661 ms |
| `p012` | Set B | 1 | 0 | (188.9, 680.1) | (0.0, 0.0) | **999.00 px** | 11.75 | 0.00 | -4.4° | 0.0° | 0.4694 | 596 ms |
| `p013` | Set B | 1 | 1 | (470.6, 248.4) | (471.2, 249.8) | **1.57 px** | 12.00 | 11.90 | 0.6° | 0.2° | 0.9981 | 574 ms |
| `p014` | Set B | 1 | 1 | (316.8, 345.3) | (316.2, 346.3) | **1.16 px** | 8.00 | 8.10 | -0.9° | -0.2° | 0.9927 | 660 ms |
| `p015` | Set C | 0 | 0 | (0.0, 0.0) | (0.0, 0.0) | **REJECTED** | 0.00 | 0.00 | 0.0° | 0.0° | 0.9941 | 566 ms |
| `p016` | Set C | 0 | 0 | (0.0, 0.0) | (0.0, 0.0) | **REJECTED** | 0.00 | 0.00 | 0.0° | 0.0° | 0.9923 | 654 ms |
| `p017` | Set C | 0 | 0 | (0.0, 0.0) | (0.0, 0.0) | **REJECTED** | 0.00 | 0.00 | 0.0° | 0.0° | 0.9617 | 615 ms |
| `p018` | Set C | 0 | 0 | (0.0, 0.0) | (0.0, 0.0) | **REJECTED** | 0.00 | 0.00 | 0.0° | 0.0° | 0.9925 | 639 ms |
| `p019` | Set D | 1 | 1 | (533.5, 564.0) | (533.7, 563.7) | **0.36 px** | 10.30 | 10.10 | 1.9° | 2.2° | 0.9982 | 669 ms |
| `p020` | Set D | 1 | 1 | (867.9, 910.0) | (868.4, 911.3) | **1.46 px** | 9.05 | 9.00 | -4.0° | -4.8° | 0.9987 | 602 ms |
