# EXP-10 — NCC-FIRST + SIAMESE VERIFIER ANALYSIS REPORT

## Executive Summary

- **Baseline Total**: 46.77 / 100
- **Strategy A Total (Pure NCC)**: 57.93 / 100
- **Strategy B Total (NCC + Guard)**: 57.64 / 100
- **Delta Total (Strategy A)**: +11.16
- **Baseline Localization**: 5.85 / 40
- **Strategy A Localization**: 13.09 / 40
- **Delta Localization**: +7.23
- **Decision**: **PROMOTE**

---

## 100-Point Score Breakdown

| Metric | Baseline (Production) | Strategy A (Pure NCC) | Strategy B (NCC + Guard) |
|---|---|---|---|
| **Localization /40** | 5.85 | 13.09 | 12.83 |
| **Scale /10** | 1.28 | 3.41 | 3.34 |
| **Rotation /10** | 1.56 | 3.72 | 3.66 |
| **Pose Total /20** | 2.84 | 7.12 | 7.00 |
| **Rejection /15** | 13.70 | 13.61 | 13.71 |
| **Confidence /10** | 9.38 | 9.11 | 9.10 |
| **Efficiency /5** | 5.00 | 5.00 | 5.00 |
| **Generator/Citations /10** | 10.00 | 10.00 | 10.00 |
| **TOTAL SCORE /100** | **46.77** | **57.93** | **57.64** |

---

## Candidate Diagnostics (Present N=160)

| Category | Production Baseline | Strategy A (Pure NCC) |
|---|---|---|
| **GT lost because not generated (>15px)** | 27 | 27 |
| **GT lost because NCC rank > 1** | 102 | 102 |
| **GT lost because Siamese rank > 1** | 126 | 126 |
| **GT lost because Fused rank > 1** | 120 | 120 |
| **GT selected successfully (<= 5.0px)** | 32 | 67 |

---

## Target Case Analysis

### pair_006
- **GT Location**: (328.0, 710.0)
- **GT in Candidate Pool**: YES
- **GT Ranks**: NCC Rank = 17, Siamese Rank = 66, Fused Rank = 52
- **Production Baseline Selected**: (127.7, 110.75) — Error: 631.84px
- **Strategy A Selected**: (128.0, 110.0) — Error: 632.46px

### pair_066
- **GT Location**: (320.0, 702.0)
- **GT in Candidate Pool**: YES
- **GT Ranks**: NCC Rank = 3, Siamese Rank = 43, Fused Rank = 27
- **Production Baseline Selected**: (670.8, 51.24) — Error: 739.29px
- **Strategy A Selected**: (670.0, 52.0) — Error: 738.24px

### pair_116
- **GT Location**: (508.0, 326.0)
- **GT in Candidate Pool**: NO (31.82px)
- **GT Ranks**: NCC Rank = 99, Siamese Rank = 99, Fused Rank = 99
- **Production Baseline Selected**: (292.2, 257.2) — Error: 226.5px
- **Strategy A Selected**: (0.0, 0.0) — Error: 999.0px

### pair_186
- **GT Location**: (297.0, 732.0)
- **GT in Candidate Pool**: YES
- **GT Ranks**: NCC Rank = 1, Siamese Rank = 31, Fused Rank = 20
- **Production Baseline Selected**: (597.68, 132.8) — Error: 670.41px
- **Strategy A Selected**: (296.5, 732.5) — Error: 0.71px

---

## Regression Analysis (Strategy A vs Production Baseline)

- **Recovered Pairs**: 52
  - `pair_003`: Baseline error 210.56px -> Strategy A error 0.71px
  - `pair_004`: Baseline error 69.64px -> Strategy A error 0.71px
  - `pair_017`: Baseline error 95.38px -> Strategy A error 0.71px
  - `pair_018`: Baseline error 502.36px -> Strategy A error 0.71px
  - `pair_022`: Baseline error 447.44px -> Strategy A error 1.00px
  - `pair_026`: Baseline error 5.23px -> Strategy A error 0.71px
  - `pair_028`: Baseline error 443.95px -> Strategy A error 0.00px
  - `pair_036`: Baseline error 132.58px -> Strategy A error 0.00px
  - `pair_037`: Baseline error 594.80px -> Strategy A error 0.71px
  - `pair_038`: Baseline error 292.80px -> Strategy A error 0.71px
  - `pair_039`: Baseline error 475.37px -> Strategy A error 0.71px
  - `pair_040`: Baseline error 519.97px -> Strategy A error 1.00px
  - `pair_044`: Baseline error 215.20px -> Strategy A error 0.71px
  - `pair_046`: Baseline error 432.37px -> Strategy A error 0.71px
  - `pair_048`: Baseline error 440.80px -> Strategy A error 0.00px
  - `pair_059`: Baseline error 384.80px -> Strategy A error 0.71px
  - `pair_060`: Baseline error 416.12px -> Strategy A error 0.00px
  - `pair_068`: Baseline error 140.65px -> Strategy A error 0.71px
  - `pair_070`: Baseline error 282.95px -> Strategy A error 0.71px
  - `pair_078`: Baseline error 8.29px -> Strategy A error 3.81px
  - `pair_087`: Baseline error 469.21px -> Strategy A error 0.71px
  - `pair_089`: Baseline error 588.62px -> Strategy A error 2.12px
  - `pair_099`: Baseline error 8.21px -> Strategy A error 3.54px
  - `pair_110`: Baseline error 216.88px -> Strategy A error 1.41px
  - `pair_117`: Baseline error 235.71px -> Strategy A error 3.54px
  - `pair_119`: Baseline error 429.68px -> Strategy A error 0.71px
  - `pair_123`: Baseline error 420.80px -> Strategy A error 0.71px
  - `pair_141`: Baseline error 999.00px -> Strategy A error 0.00px
  - `pair_142`: Baseline error 999.00px -> Strategy A error 0.00px
  - `pair_147`: Baseline error 999.00px -> Strategy A error 0.00px
  - `pair_148`: Baseline error 999.00px -> Strategy A error 0.00px
  - `pair_151`: Baseline error 999.00px -> Strategy A error 0.00px
  - `pair_152`: Baseline error 999.00px -> Strategy A error 0.00px
  - `pair_153`: Baseline error 999.00px -> Strategy A error 0.00px
  - `pair_154`: Baseline error 999.00px -> Strategy A error 0.00px
  - `pair_163`: Baseline error 999.00px -> Strategy A error 0.00px
  - `pair_164`: Baseline error 999.00px -> Strategy A error 0.00px
  - `pair_171`: Baseline error 999.00px -> Strategy A error 0.00px
  - `pair_172`: Baseline error 999.00px -> Strategy A error 0.00px
  - `pair_175`: Baseline error 999.00px -> Strategy A error 0.00px
  - `pair_176`: Baseline error 999.00px -> Strategy A error 0.00px
  - `pair_178`: Baseline error 999.00px -> Strategy A error 0.00px
  - `pair_179`: Baseline error 999.00px -> Strategy A error 0.00px
  - `pair_180`: Baseline error 999.00px -> Strategy A error 0.00px
  - `pair_184`: Baseline error 350.23px -> Strategy A error 0.71px
  - `pair_186`: Baseline error 670.41px -> Strategy A error 0.71px
  - `pair_187`: Baseline error 349.20px -> Strategy A error 0.71px
  - `pair_188`: Baseline error 70.69px -> Strategy A error 0.00px
  - `pair_190`: Baseline error 164.75px -> Strategy A error 0.00px
  - `pair_195`: Baseline error 342.56px -> Strategy A error 0.71px
  - `pair_197`: Baseline error 433.95px -> Strategy A error 0.00px
  - `pair_198`: Baseline error 272.48px -> Strategy A error 0.00px
- **Regressed Pairs**: 0
- **Unchanged Pairs**: 148

---

## Runtime Performance

- **Median Runtime**: 526 ms
- **P90 Runtime**: 583 ms
- **P99 Runtime**: 612 ms

---

## Final Decision: PROMOTE
