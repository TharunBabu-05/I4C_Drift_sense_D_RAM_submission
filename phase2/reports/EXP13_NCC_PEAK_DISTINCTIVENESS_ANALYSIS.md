# EXP-13 — NCC PEAK DISTINCTIVENESS / PERIODICITY PENALTY REPORT

## Executive Summary

- **Production Baseline Total (Strategy A)**: 60.99 / 100
- **Best EXP-13 Variant Total (Periodicity Penalized (ncc - 0.05*period_count))**: **71.65 / 100**
- **Delta Total Score**: **+10.65**
- **Production Baseline Localization**: 14.11 / 40
- **Best EXP-13 Localization**: **21.01 / 40**
- **Delta Localization Score**: **+6.90**
- **Decision**: **PROMOTE**

---

## 100-Point Score Breakdown across Variants

| Category | Baseline (Strategy A) | Prominence Mult | Prominence Add | Periodicity Pen |
|---|---|---|---|---|
| **Localization /40** | 14.11 | 12.09 | 12.86 | 21.01 |
| **Scale /10** | 3.88 | 3.38 | 3.59 | 5.75 |
| **Rotation /10** | 4.38 | 3.78 | 4.00 | 6.19 |
| **Pose Total /20** | 8.25 | 7.16 | 7.59 | 11.94 |
| **Rejection /15** | 13.87 | 13.87 | 13.87 | 13.87 |
| **Confidence /10** | 9.76 | 9.73 | 9.75 | 9.83 |
| **Efficiency /5** | 5.00 | 5.00 | 5.00 | 5.00 |
| **Generator/Citations /10** | 10.00 | 10.00 | 10.00 | 10.00 |
| **TOTAL SCORE /100** | **60.99** | **57.84** | **59.08** | **71.65** |

---

## Target Pairs Analysis

### pair_006
- **GT Location**: (656.0, 366.0)
  - **base**: LocErr = 631.84px, Found = 1, NCC = 0.9562, Prominence = 0.8777
  - **prominence_mult**: LocErr = 631.84px, Found = 1, NCC = 0.9562, Prominence = 0.8777
  - **prominence_add**: LocErr = 631.84px, Found = 1, NCC = 0.9562, Prominence = 0.8777
  - **periodicity_pen**: LocErr = 1.33px, Found = 1, NCC = 0.9217, Prominence = 0.6801

### pair_066
- **GT Location**: (656.0, 366.0)
  - **base**: LocErr = 739.29px, Found = 1, NCC = 0.9590, Prominence = 0.8730
  - **prominence_mult**: LocErr = 739.29px, Found = 1, NCC = 0.9590, Prominence = 0.8730
  - **prominence_add**: LocErr = 739.29px, Found = 1, NCC = 0.9590, Prominence = 0.8730
  - **periodicity_pen**: LocErr = 0.69px, Found = 1, NCC = 0.9165, Prominence = 0.6927

### pair_116
- **GT Location**: (656.0, 366.0)
  - **base**: LocErr = 33.92px, Found = 1, NCC = 0.5767, Prominence = 0.0847
  - **prominence_mult**: LocErr = 33.92px, Found = 1, NCC = 0.5767, Prominence = 0.0847
  - **prominence_add**: LocErr = 33.92px, Found = 1, NCC = 0.5767, Prominence = 0.0847
  - **periodicity_pen**: LocErr = 33.63px, Found = 1, NCC = 0.5723, Prominence = 0.0720

### pair_186
- **GT Location**: (656.0, 366.0)
  - **base**: LocErr = 0.29px, Found = 1, NCC = 0.9836, Prominence = 0.8420
  - **prominence_mult**: LocErr = 0.29px, Found = 1, NCC = 0.9836, Prominence = 0.8420
  - **prominence_add**: LocErr = 0.29px, Found = 1, NCC = 0.9836, Prominence = 0.8420
  - **periodicity_pen**: LocErr = 0.29px, Found = 1, NCC = 0.9836, Prominence = 0.8420

---

## Regression Analysis (Periodicity Penalized (ncc - 0.05*period_count) vs Production Baseline)

- **Recovered Pairs**: 38
  - `pair_005`: Baseline error 180.43px -> Best EXP-13 error 0.69px
  - `pair_006`: Baseline error 631.84px -> Best EXP-13 error 1.33px
  - `pair_009`: Baseline error 323.28px -> Best EXP-13 error 1.84px
  - `pair_011`: Baseline error 233.03px -> Best EXP-13 error 0.98px
  - `pair_012`: Baseline error 466.92px -> Best EXP-13 error 1.13px
  - `pair_027`: Baseline error 545.18px -> Best EXP-13 error 1.81px
  - `pair_029`: Baseline error 147.33px -> Best EXP-13 error 0.42px
  - `pair_030`: Baseline error 182.57px -> Best EXP-13 error 0.92px
  - `pair_035`: Baseline error 313.67px -> Best EXP-13 error 0.64px
  - `pair_041`: Baseline error 359.54px -> Best EXP-13 error 1.33px
  - `pair_042`: Baseline error 144.59px -> Best EXP-13 error 1.13px
  - `pair_043`: Baseline error 159.92px -> Best EXP-13 error 0.42px
  - `pair_047`: Baseline error 576.87px -> Best EXP-13 error 1.33px
  - `pair_049`: Baseline error 206.64px -> Best EXP-13 error 0.42px
  - `pair_050`: Baseline error 578.34px -> Best EXP-13 error 0.85px
  - `pair_051`: Baseline error 432.96px -> Best EXP-13 error 0.95px
  - `pair_053`: Baseline error 387.17px -> Best EXP-13 error 1.13px
  - `pair_054`: Baseline error 429.72px -> Best EXP-13 error 0.30px
  - `pair_055`: Baseline error 587.97px -> Best EXP-13 error 1.76px
  - `pair_056`: Baseline error 347.35px -> Best EXP-13 error 0.42px
  - `pair_057`: Baseline error 383.20px -> Best EXP-13 error 1.29px
  - `pair_058`: Baseline error 192.04px -> Best EXP-13 error 1.33px
  - `pair_064`: Baseline error 349.65px -> Best EXP-13 error 1.59px
  - `pair_065`: Baseline error 353.68px -> Best EXP-13 error 1.06px
  - `pair_066`: Baseline error 739.29px -> Best EXP-13 error 0.69px
  - `pair_069`: Baseline error 511.13px -> Best EXP-13 error 0.84px
  - `pair_107`: Baseline error 240.48px -> Best EXP-13 error 3.96px
  - `pair_115`: Baseline error 473.85px -> Best EXP-13 error 1.13px
  - `pair_125`: Baseline error 539.72px -> Best EXP-13 error 1.15px
  - `pair_131`: Baseline error 329.26px -> Best EXP-13 error 1.33px
  - `pair_143`: Baseline error 999.00px -> Best EXP-13 error 0.00px
  - `pair_144`: Baseline error 999.00px -> Best EXP-13 error 0.00px
  - `pair_153`: Baseline error 999.00px -> Best EXP-13 error 0.00px
  - `pair_154`: Baseline error 999.00px -> Best EXP-13 error 0.00px
  - `pair_169`: Baseline error 999.00px -> Best EXP-13 error 0.00px
  - `pair_170`: Baseline error 999.00px -> Best EXP-13 error 0.00px
  - `pair_185`: Baseline error 224.68px -> Best EXP-13 error 0.81px
  - `pair_189`: Baseline error 203.18px -> Best EXP-13 error 0.93px
- **Regressed Pairs**: 6
  - `pair_151`: Baseline error 0.00px -> Best EXP-13 error 999.00px
  - `pair_152`: Baseline error 0.00px -> Best EXP-13 error 999.00px
  - `pair_167`: Baseline error 0.00px -> Best EXP-13 error 999.00px
  - `pair_168`: Baseline error 0.00px -> Best EXP-13 error 999.00px
  - `pair_173`: Baseline error 0.00px -> Best EXP-13 error 999.00px
  - `pair_174`: Baseline error 0.00px -> Best EXP-13 error 999.00px
- **Unchanged Pairs**: 156

---

## Runtime Performance

- **Median Runtime**: 534 ms
- **P90 Runtime**: 603 ms
- **P99 Runtime**: 654 ms

---

## Final Decision: PROMOTE
