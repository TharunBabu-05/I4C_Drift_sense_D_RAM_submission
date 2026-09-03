# EXP-15 TOP-K COARSE EXPANSION REPORT

## Executive Summary

- **Baseline Score (EXP-13, K=10)**: **71.65 / 100**
- **EXP-15 Score (K=20)**: **71.74 / 100**
- **Delta Total Score**: **+0.09**
- **Baseline Localization**: 21.01 / 40
- **EXP-15 Localization**: **21.01 / 40** (+0.00)
- **Decision**: **REJECT**

---

## 100-Point Score Breakdown

| Category | Baseline (K=10) | EXP-15 (K=20) | Delta |
|---|---|---|---|
| **Localization /40** | 21.01 | **21.01** | **+0.00** |
| **Scale /10** | 5.75 | **5.75** | **+0.00** |
| **Rotation /10** | 6.19 | **6.19** | **+0.00** |
| **Pose Total /20** | 11.94 | **11.94** | **+0.00** |
| **Rejection /15** | 13.87 | **13.95** | **0.00** |
| **Confidence /10** | 9.83 | **9.84** | **+0.01** |
| **Efficiency /5** | 5.00 | **5.00** | **0.00** |
| **Generator/Citations /10** | 10.00 | **10.00** | **0.00** |
| **TOTAL SCORE /100** | **71.65** | **71.74** | **+0.09** |

---

## Cat-B Recovery Audit (32 Total EXP-14 Baseline Cat-B Failures)

- **Cat-B Coarse Rank was 11–20**: 2 / 32
- **Cat-B Coarse Rank was > 20**: 1 / 32
- **Cat-B Pairs RECOVERED (<=5px)**: **0 / 32**
- **Cat-B Pairs STILL FAILED (>5px)**: 32 / 32

---

## Candidate Recall Audit

| Threshold | Baseline (K=10) | EXP-15 (K=20) | Delta |
|---|---|---|---|
| **Recall @1px** | 26.88% | **26.88%** | +0.00% |
| **Recall @5px** | 69.38% | **69.38%** | +0.00% |
| **Recall @15px** | 74.38% | **74.38%** | +0.00% |
| **Recall @50px** | 88.75% | **88.75%** | +0.00% |

---

## Target Pairs Comparison

### pair_006
- **Baseline (K=10)**: LocErr = 1.33px, Found = 1, Score = 0.9928
- **EXP-15 (K=20)**  : LocErr = 1.33px, Found = 1, Score = 0.9928

### pair_066
- **Baseline (K=10)**: LocErr = 0.69px, Found = 1, Score = 0.9925
- **EXP-15 (K=20)**  : LocErr = 0.69px, Found = 1, Score = 0.9925

### pair_116
- **Baseline (K=10)**: LocErr = 33.63px, Found = 1, Score = 0.9857
- **EXP-15 (K=20)**  : LocErr = 33.63px, Found = 1, Score = 0.9857

### pair_186
- **Baseline (K=10)**: LocErr = 0.29px, Found = 1, Score = 0.998
- **EXP-15 (K=20)**  : LocErr = 0.29px, Found = 1, Score = 0.998

---

## Regression Analysis (K=20 vs Baseline K=10)

- **Recovered Pairs (2)**:
  - `pair_151`: Baseline error 999.00px -> EXP-15 error 0.00px
  - `pair_152`: Baseline error 999.00px -> EXP-15 error 0.00px
- **Regressed Pairs (0)**:
- **Unchanged Pairs**: 198

---

## Runtime Performance

- **Baseline Median Runtime**: 1211 ms
- **EXP-15 Median Runtime**: **1855 ms** (Delta: +644 ms)
- **EXP-15 P90 Runtime**: 2050 ms
- **EXP-15 P99 Runtime**: 2179 ms

---

## Technical Conclusion & Decision: REJECT
