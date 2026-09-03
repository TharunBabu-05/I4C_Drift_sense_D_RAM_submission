# EXP-11 — DENSER COARSE SEARCH ANALYSIS REPORT

## Executive Summary

- **Production Baseline Total (500x500)**: 61.00 / 100
- **750x750 Coarse Search Total**: 61.19 / 100 (Delta: +0.19)
- **1000x1000 Coarse Search Total**: 60.69 / 100 (Delta: -0.30)
- **Production Baseline Localization**: 14.06 / 40
- **750x750 Localization**: 14.30 / 40
- **1000x1000 Localization**: 13.89 / 40
- **Decision**: **REJECT**

---

## 100-Point Score Breakdown

| Metric | 500x500 (Production Base) | 750x750 Coarse | 1000x1000 Coarse |
|---|---|---|---|
| **Localization /40** | 14.06 | 14.30 | 13.89 |
| **Scale /10** | 3.88 | 3.81 | 3.81 |
| **Rotation /10** | 4.34 | 4.34 | 4.28 |
| **Pose Total /20** | 8.22 | 8.16 | 8.09 |
| **Rejection /15** | 13.95 | 13.95 | 13.95 |
| **Confidence /10** | 9.77 | 9.78 | 9.75 |
| **Efficiency /5** | 5.00 | 5.00 | 5.00 |
| **Generator/Citations /10** | 10.00 | 10.00 | 10.00 |
| **TOTAL SCORE /100** | **61.00** | **61.19** | **60.69** |

---

## Candidate Recall Audit

| Stage / Threshold | 500x500 (Base) @15px | 750x750 @15px | 1000x1000 @15px |
|---|---|---|---|
| **Coarse Pool Recall** | 85.62% | 85.62% | 84.38% |
| **Refined Pool Recall** | 83.12% | 83.12% | 83.12% |
| **Final Selected Recall @5px** | 48.75% | 48.75% | 48.12% |

---

## Target Pairs Analysis

### pair_006
- **GT Location**: (328.0, 710.0)
  - **500x500**: Coarse GT = YES, Refined GT = YES, LocErr = 631.84px, NCC = 0.9562
  - **750x750**: Coarse GT = YES, Refined GT = YES, LocErr = 363.46px, NCC = 0.9562
  - **1000x1000**: Coarse GT = YES, Refined GT = YES, LocErr = 538.07px, NCC = 0.9562

### pair_066
- **GT Location**: (320.0, 702.0)
  - **500x500**: Coarse GT = YES, Refined GT = YES, LocErr = 739.29px, NCC = 0.9590
  - **750x750**: Coarse GT = YES, Refined GT = YES, LocErr = 584.24px, NCC = 0.9590
  - **1000x1000**: Coarse GT = YES, Refined GT = YES, LocErr = 658.28px, NCC = 0.9590

### pair_116
- **GT Location**: (508.0, 326.0)
  - **500x500**: Coarse GT = NO (28.64px), Refined GT = NO (31.82px), LocErr = 33.92px, NCC = 0.5767
  - **750x750**: Coarse GT = NO (29.16px), Refined GT = NO (31.82px), LocErr = 32.34px, NCC = 0.5767
  - **1000x1000**: Coarse GT = NO (32.5px), Refined GT = NO (31.82px), LocErr = 32.34px, NCC = 0.5767

### pair_186
- **GT Location**: (297.0, 732.0)
  - **500x500**: Coarse GT = YES, Refined GT = YES, LocErr = 1.63px, NCC = 0.9836
  - **750x750**: Coarse GT = YES, Refined GT = YES, LocErr = 1.21px, NCC = 0.9836
  - **1000x1000**: Coarse GT = YES, Refined GT = YES, LocErr = 1.63px, NCC = 0.9836

---

## Regression Analysis (750x750 vs 500x500 Production Base)

- **Recovered Pairs**: 1
  - `pair_103`: 500x500 error 6.43px -> 750x750 error 4.88px
- **Regressed Pairs**: 1
  - `pair_195`: 500x500 error 0.30px -> 750x750 error 212.90px
- **Unchanged Pairs**: 198

---

## Runtime Performance

| Resolution | Median Runtime | P90 Runtime | P99 Runtime |
|---|---|---|---|
| **500x500** | 516 ms | 582 ms | 615 ms |
| **750x750** | 666 ms | 733 ms | 789 ms |
| **1000x1000** | 899 ms | 959 ms | 996 ms |

---

## Final Decision: REJECT
