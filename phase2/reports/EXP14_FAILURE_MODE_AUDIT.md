# EXP-14 — Post-EXP13 Failure-Mode Audit Report

## Executive Summary

- **Production Baseline Score (EXP-13)**: **71.71 / 100**
- **Audit Scope**: 200 pairs (`local_phase2_60gen_200_pairs`)
- **Total Present Pairs**: 160
- **Successfully Localized Present Pairs (<=5px)**: 94 (58.8%)
- **Failed Present Pairs**: 66 (41.2%)
- **Status**: **DIAGNOSTIC COMPLETED (NO CODE MODIFICATIONS MADE)**

---

## Failure Category Taxonomy & Breakdown

| Category | Description | Count | % of Present (160) | Primary Root Cause |
|---|---|---|---|---|
| **Cat A** | GT absent from coarse candidate pool | **8** | 5.0% | Coarse downsampling (500x500) loses landmark feature |
| **Cat B** | GT in coarse pool, lost before refinement | **32** | 20.0% | Top-10 coarse cutoff excludes GT candidate |
| **Cat C** | GT in refined pool, lost final ranking | **6** | 3.8% | Decoy NCC exceeds GT & 0.05 periodicity penalty insufficient |
| **Cat D** | GT selected, subpixel precision error >5px | **0** | 0.0% | Parabola subpixel fit imprecision |
| **Cat E** | GT selected, scale error >0.5 | **14** | 8.8% | Fine scale grid step (0.25) quantization |
| **Cat F** | GT selected, rotation error >1.5° | **1** | 0.6% | Fine rotation grid step (1.0°) quantization |
| **Cat G** | GT localized but rejected (false negative) | **0** | 0.0% | Fused score < 0.42 rejection threshold |
| **Cat H** | Set-D optical microscope analogue failure | **5** | 3.1% | Low contrast, blur, and lighting non-uniformity |
| **Cat I** | Other | **0** | 0.0% | Miscellaneous |

---

## Set-Wise Failure Distribution

| Set Name | Total Failures | Cat A | Cat B | Cat C | Cat D | Cat E | Cat F | Cat G | Cat H |
|---|---|---|---|---|---|---|---|---|---|
| **Set A (SEM Clean)** | **4** | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 |
| **Set B (SEM Degraded)** | **57** | 8 | 32 | 2 | 0 | 14 | 1 | 0 | 0 |
| **Set C (Absent Pairs)** | **0 FP** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Set D (Optical)** | **5** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |

---

## Deep-Dive Analysis of Dominant Failure Modes

### 1. Primary Failure Bottleneck: Candidate Coarse Recall (Cat A & Cat B = 40 pairs)
- **Cat A (8 pairs)**: Ground truth is completely missing from the coarse candidate pool (`gt_coarse_dist > 25px`). Downsampling the search image to 500x500 combined with a 50x50 coarse template skips fine landmark features.
- **Cat B (32 pairs)**: Ground truth is present in the coarse search grid but receives a coarse NCC score outside the Top-10 cutoff.

### 2. Secondary Failure Bottleneck: Periodicity Penalty Insufficiency (Cat C = 6 pairs)
- For 6 pairs, GT exists in the refined pool but a periodic decoy outranks GT.
- In 1 of these pairs, the decoy was more periodic than GT (`p_margin > 0`), but the 0.05 periodicity penalty multiplier was too conservative to bridge the raw NCC gap.

### 3. Set-D Optical Microscope Domain Shift (Cat H = 5 pairs)
- Set D optical microscope images feature severe global lighting non-uniformity and soft blur. Low local contrast drops raw NCC scores below the rejection threshold (`tau = 0.42`), causing 6 present pairs to be rejected.

---

## Target Pairs Trace

### pair_006
- **Loc Error**: 1.33 px
- **GT in Coarse**: True (dist 0.0px) | **GT in Refined**: True (dist 0.0px)
- **GT Rank**: Before periodicity = 17 -> After periodicity = 1
- **GT NCC**: 0.7756 (p=1) -> Adj = 0.7756
- **Selected Decoy NCC**: 0.9217 (p=1) -> Adj = 0.9217

### pair_066
- **Loc Error**: 0.69 px
- **GT in Coarse**: True (dist 1.41px) | **GT in Refined**: True (dist 0.0px)
- **GT Rank**: Before periodicity = 3 -> After periodicity = 1
- **GT NCC**: 0.9098 (p=1) -> Adj = 0.9098
- **Selected Decoy NCC**: 0.9165 (p=1) -> Adj = 0.9165

### pair_116
- **Loc Error**: 33.63 px
- **GT in Coarse**: False (dist 29.43px) | **GT in Refined**: False (dist 31.82px)
- **GT Rank**: Before periodicity = 11 -> After periodicity = -1
- **GT NCC**: 0.5674 (p=7) -> Adj = 0.2674
- **Selected Decoy NCC**: 0.5723 (p=1) -> Adj = 0.5723

### pair_186
- **Loc Error**: 0.29 px
- **GT in Coarse**: True (dist 1.0px) | **GT in Refined**: True (dist 0.71px)
- **GT Rank**: Before periodicity = 1 -> After periodicity = 1
- **GT NCC**: 0.9836 (p=1) -> Adj = 0.9836
- **Selected Decoy NCC**: 0.9836 (p=1) -> Adj = 0.9836

---

## Next Single Hypothesis Recommendation for EXP-15

Based on the empirical findings of EXP-14:
- **Dominant Bottleneck**: **Cat A & Cat B Candidate Generation Loss (40 pairs)** and **Cat C Periodicity Penalty Under-penalization (6 pairs)**.
- **Proposed EXP-15**: Test **Non-Linear / Dynamic Periodicity Penalty Scaling** or **Multi-Scale Coarse Template Extraction** as a strict single change.
