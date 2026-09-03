# EXP-12 — GRADIENT-NORMALIZED NCC ANALYSIS REPORT

## Executive Summary

- **Baseline Total Score (Raw NCC)**: 60.99 / 100
- **EXP-12 Total Score (Gradient NCC)**: 50.53 / 100
- **Delta Total Score**: -10.46
- **Baseline Localization Score**: 14.11 / 40
- **EXP-12 Localization Score**: 8.62 / 40
- **Delta Localization Score**: -5.49
- **Decision**: **REJECT**

---

## 100-Point Score Breakdown

| Metric | Baseline (Raw Intensity NCC) | EXP-12 (Gradient-Normalized NCC) |
|---|---|---|
| **Localization /40** | 14.11 | 8.62 |
| **Scale /10** | 3.88 | 1.97 |
| **Rotation /10** | 4.38 | 2.44 |
| **Pose Total /20** | 8.25 | 4.41 |
| **Rejection /15** | 13.87 | 13.49 |
| **Confidence /10** | 9.76 | 9.01 |
| **Efficiency /5** | 5.00 | 5.00 |
| **Generator/Citations /10** | 10.00 | 10.00 |
| **TOTAL SCORE /100** | **60.99** | **50.53** |

---

## Candidate Recall Audit

| Metric / Threshold | Baseline (Raw NCC) | EXP-12 (Gradient NCC) |
|---|---|---|
| **Final Selected Recall @1px** | 17.5% | 11.88% |
| **Final Selected Recall @5px** | 49.38% | 27.5% |
| **Final Selected Recall @15px** | 53.12% | 30.0% |
| **Final Selected Recall @50px** | 65.0% | 36.88% |

---

## Target Pairs Forensic Breakdown

### pair_006
- **GT Location**: (328.0, 710.0)
- **Baseline (Raw NCC)**: LocErr = 631.84px, GT in Refined = True, Raw NCC = 0.9562
- **EXP-12 (Grad NCC)**: LocErr = 213.21px, GT in Refined = True, GT Rank = 2, GT GradNCC = 0.8522 vs Decoy GradNCC = 0.9624

### pair_066
- **GT Location**: (320.0, 702.0)
- **Baseline (Raw NCC)**: LocErr = 739.29px, GT in Refined = True, Raw NCC = 0.9590
- **EXP-12 (Grad NCC)**: LocErr = 390.41px, GT in Refined = True, GT Rank = 2, GT GradNCC = 0.8448 vs Decoy GradNCC = 0.9583

### pair_116
- **GT Location**: (508.0, 326.0)
- **Baseline (Raw NCC)**: LocErr = 33.92px, GT in Refined = False, Raw NCC = 0.5767
- **EXP-12 (Grad NCC)**: LocErr = 657.02px, GT in Refined = False, GT Rank = 99, GT GradNCC = 0.0000 vs Decoy GradNCC = 0.5559

### pair_186
- **GT Location**: (297.0, 732.0)
- **Baseline (Raw NCC)**: LocErr = 0.29px, GT in Refined = True, Raw NCC = 0.9836
- **EXP-12 (Grad NCC)**: LocErr = 200.8px, GT in Refined = True, GT Rank = 2, GT GradNCC = 0.9327 vs Decoy GradNCC = 0.9479

---

## Regression Analysis (EXP-12 vs Baseline 60.99)

- **Recovered Pairs**: 14
  - `pair_093`: Baseline error 7.65px -> EXP-12 error 4.94px
  - `pair_107`: Baseline error 240.48px -> EXP-12 error 2.95px
  - `pair_143`: Baseline error 999.00px -> EXP-12 error 0.00px
  - `pair_144`: Baseline error 999.00px -> EXP-12 error 0.00px
  - `pair_153`: Baseline error 999.00px -> EXP-12 error 0.00px
  - `pair_154`: Baseline error 999.00px -> EXP-12 error 0.00px
  - `pair_161`: Baseline error 999.00px -> EXP-12 error 0.00px
  - `pair_162`: Baseline error 999.00px -> EXP-12 error 0.00px
  - `pair_165`: Baseline error 999.00px -> EXP-12 error 0.00px
  - `pair_166`: Baseline error 999.00px -> EXP-12 error 0.00px
  - `pair_169`: Baseline error 999.00px -> EXP-12 error 0.00px
  - `pair_170`: Baseline error 999.00px -> EXP-12 error 0.00px
  - `pair_171`: Baseline error 999.00px -> EXP-12 error 0.00px
  - `pair_178`: Baseline error 999.00px -> EXP-12 error 0.00px
- **Regressed Pairs**: 43
  - `pair_007`: Baseline error 0.69px -> EXP-12 error 350.39px
  - `pair_008`: Baseline error 1.10px -> EXP-12 error 97.99px
  - `pair_018`: Baseline error 0.42px -> EXP-12 error 550.90px
  - `pair_020`: Baseline error 0.30px -> EXP-12 error 307.94px
  - `pair_021`: Baseline error 0.96px -> EXP-12 error 187.85px
  - `pair_022`: Baseline error 1.53px -> EXP-12 error 494.01px
  - `pair_023`: Baseline error 0.48px -> EXP-12 error 127.68px
  - `pair_036`: Baseline error 1.13px -> EXP-12 error 107.21px
  - `pair_038`: Baseline error 1.33px -> EXP-12 error 289.83px
  - `pair_039`: Baseline error 1.84px -> EXP-12 error 548.42px
  - `pair_040`: Baseline error 1.97px -> EXP-12 error 252.80px
  - `pair_045`: Baseline error 1.13px -> EXP-12 error 162.80px
  - `pair_048`: Baseline error 1.11px -> EXP-12 error 167.20px
  - `pair_052`: Baseline error 0.45px -> EXP-12 error 192.80px
  - `pair_063`: Baseline error 0.85px -> EXP-12 error 70.01px
  - `pair_067`: Baseline error 0.85px -> EXP-12 error 210.18px
  - `pair_068`: Baseline error 1.84px -> EXP-12 error 349.54px
  - `pair_070`: Baseline error 0.42px -> EXP-12 error 344.16px
  - `pair_071`: Baseline error 2.35px -> EXP-12 error 330.48px
  - `pair_077`: Baseline error 2.34px -> EXP-12 error 10.09px
  - `pair_083`: Baseline error 4.17px -> EXP-12 error 8.98px
  - `pair_103`: Baseline error 4.88px -> EXP-12 error 6.26px
  - `pair_105`: Baseline error 4.02px -> EXP-12 error 999.00px
  - `pair_110`: Baseline error 2.55px -> EXP-12 error 308.85px
  - `pair_111`: Baseline error 2.64px -> EXP-12 error 412.75px
  - `pair_113`: Baseline error 2.81px -> EXP-12 error 216.30px
  - `pair_119`: Baseline error 1.33px -> EXP-12 error 446.74px
  - `pair_123`: Baseline error 0.95px -> EXP-12 error 420.02px
  - `pair_127`: Baseline error 4.72px -> EXP-12 error 999.00px
  - `pair_130`: Baseline error 1.13px -> EXP-12 error 999.00px
  - `pair_167`: Baseline error 0.00px -> EXP-12 error 999.00px
  - `pair_168`: Baseline error 0.00px -> EXP-12 error 999.00px
  - `pair_173`: Baseline error 0.00px -> EXP-12 error 999.00px
  - `pair_174`: Baseline error 0.00px -> EXP-12 error 999.00px
  - `pair_179`: Baseline error 0.00px -> EXP-12 error 999.00px
  - `pair_180`: Baseline error 0.00px -> EXP-12 error 999.00px
  - `pair_186`: Baseline error 0.29px -> EXP-12 error 200.80px
  - `pair_187`: Baseline error 0.66px -> EXP-12 error 210.32px
  - `pair_188`: Baseline error 0.16px -> EXP-12 error 279.57px
  - `pair_190`: Baseline error 0.46px -> EXP-12 error 252.47px
  - `pair_195`: Baseline error 0.42px -> EXP-12 error 95.80px
  - `pair_198`: Baseline error 0.98px -> EXP-12 error 247.82px
  - `pair_200`: Baseline error 1.69px -> EXP-12 error 489.46px
- **Unchanged Pairs**: 143

---

## Runtime Performance

| Metric | Baseline (Raw NCC) | EXP-12 (Gradient NCC) |
|---|---|---|
| **Median Runtime** | 533 ms | 517 ms |
| **P90 Runtime** | 583 ms | 588 ms |
| **P99 Runtime** | 621 ms | 626 ms |

---

## Final Decision: REJECT
