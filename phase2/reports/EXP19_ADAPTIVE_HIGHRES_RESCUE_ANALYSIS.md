# EXP-19 ADAPTIVE HIGH-RESOLUTION RESCUE REPORT

## Executive Summary

- **Production Baseline Score**: **72.80 / 100**
- **EXP-19 Score**: **72.76 / 100**
- **Delta Total Score**: **-0.04**
- **Production Baseline Localization**: 21.01 / 40
- **EXP-19 Localization**: **21.06 / 40** (+0.05)
- **Pairs Triggering Rescue**: 115 / 200
- **Decision**: **REJECT**

---

## 100-Point Score Breakdown

| Category | Production Baseline (72.80) | EXP-19 | Delta |
|---|---|---|---|
| **Localization /40** | 21.01 | **21.06** | **+0.05** |
| **Scale /10** | 5.75 | **5.69** | **-0.06** |
| **Rotation /10** | 6.19 | **6.16** | **-0.03** |
| **Pose Total /20** | 11.94 | **11.84** | **-0.09** |
| **Rejection /15** | 14.86 | **14.86** | **+0.00** |
| **Confidence /10** | 10.00 | **10.00** | **+0.00** |
| **Efficiency /5** | 5.00 | **5.00** | **0.00** |
| **Generator/Citations /10** | 10.00 | **10.00** | **0.00** |
| **TOTAL SCORE /100** | **72.80** | **72.76** | **-0.04** |

---

## Set-B & Cat-A Candidate Recovery Audit

- **Pairs Triggering High-Res Rescue**: 115 / 200
- **Cat-A Baseline Coarse-Missing Pairs Recovered**: 0 / 8
- **Set-B Degraded SEM Baseline Pairs Recovered**: 0 / 40
- **Total Recovered Pairs**: 0
- **Total Regressed Pairs**: 1

---

## Target Pairs Comparison

### pair_006
- **Production Baseline**: LocErr = 1.33px, Found = 1, Score = 0.9978
- **EXP-19**             : LocErr = 1.33px, Found = 1, Score = 0.9978

### pair_066
- **Production Baseline**: LocErr = 0.69px, Found = 1, Score = 0.9978
- **EXP-19**             : LocErr = 0.69px, Found = 1, Score = 0.9978

### pair_116
- **Production Baseline**: LocErr = 33.63px, Found = 1, Score = 0.9868
- **EXP-19**             : LocErr = 33.63px, Found = 1, Score = 0.9868

### pair_160
- **Production Baseline**: LocErr = 0.0px, Found = 0, Score = 0.2174
- **EXP-19**             : LocErr = 0.0px, Found = 0, Score = 0.2454

### pair_186
- **Production Baseline**: LocErr = 0.29px, Found = 1, Score = 0.9988
- **EXP-19**             : LocErr = 0.29px, Found = 1, Score = 0.9988

---

## Runtime Performance

- **Production Baseline Median Runtime**: 535 ms
- **EXP-19 Median Runtime**: **467 ms** (Delta: -68 ms)
- **EXP-19 P90 Runtime**: 582 ms
- **EXP-19 P99 Runtime**: 614 ms

---

## Decision & Technical Conclusion: REJECT
