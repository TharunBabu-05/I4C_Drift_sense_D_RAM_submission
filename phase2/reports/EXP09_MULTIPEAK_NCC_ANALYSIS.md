# EXP-09 MULTI-PEAK NCC REPORT

## Summary

- **Baseline Total**: 46.77 / 100
- **EXP-09 Total (K=5)**: 46.52 / 100
- **Delta Total**: -0.25
- **Baseline Loc**: 9.38 / 40
- **EXP-09 Loc (K=5)**: 5.71 / 40
- **Delta Loc**: -3.67
- **Decision**: **REJECT**

---

## Candidate Recall (K_peaks = 5)

| Stage | @1px | @5px | @15px | @50px |
|---|---|---|---|---|
| **All Coarse** | 62.5% | 77.5% | 90.0% | 100.0% |
| **Top-K Coarse** | 55.0% | 72.5% | 79.38% | 91.25% |
| **All Refined** | 63.12% | 73.75% | 80.62% | 92.5% |
| **Final Top-5** | 13.75% | 21.25% | 27.5% | 36.25% |

---

## K_peaks Ablation Table

| K_peaks | Total Score | Loc /40 | Pose /20 | Rej /15 | Conf /10 | Eff /5 | Top-5 Rec@5px | Top-5 Rec@15px | Med Runtime |
|---|---|---|---|---|---|---|---|---|---|
| **BASE (1)** | **46.77** | **9.38** | **9.00** | **13.70** | **9.69** | **5.0** | **28.7%** | **36.2%** | **347ms** |
| 1 | 46.77 | 5.85 | 2.84 | 13.70 | 9.38 | 5.0 | 28.8% | 36.2% | 533ms |
| 3 | 46.55 | 5.71 | 2.56 | 13.85 | 9.43 | 5.0 | 23.1% | 30.0% | 564ms |
| 5 | 46.52 | 5.71 | 2.56 | 13.85 | 9.40 | 5.0 | 21.2% | 27.5% | 550ms |
| 10 | 46.15 | 5.39 | 2.50 | 13.89 | 9.36 | 5.0 | 19.4% | 24.4% | 566ms |

---

## Target Failure Cases (K_peaks = 5)

### pair_006
- **GT**: (328.0, 710.0)
- **GT Recovered Coarse**: YES (dist=0.0px)
- **GT Survived Top-K**: YES (dist=1.41px)
- **GT Survived Refined**: YES (dist=0.0px)
- **GT Survived Top-5**: NO (dist=618.47px)
- **Final Selected**: DECOY (loc_err=619.29px)

### pair_066
- **GT**: (320.0, 702.0)
- **GT Recovered Coarse**: YES (dist=1.41px)
- **GT Survived Top-K**: YES (dist=1.41px)
- **GT Survived Refined**: YES (dist=0.0px)
- **GT Survived Top-5**: NO (dist=650.0px)
- **Final Selected**: DECOY (loc_err=668.0px)

### pair_116
- **GT**: (508.0, 326.0)
- **GT Recovered Coarse**: NO (dist=15.81px)
- **GT Survived Top-K**: NO (dist=32.56px)
- **GT Survived Refined**: NO (dist=31.82px)
- **GT Survived Top-5**: NO (dist=32.6px)
- **Final Selected**: DECOY (loc_err=619.67px)

### pair_186
- **GT**: (297.0, 732.0)
- **GT Recovered Coarse**: YES (dist=1.0px)
- **GT Survived Top-K**: YES (dist=1.0px)
- **GT Survived Refined**: YES (dist=0.71px)
- **GT Survived Top-5**: NO (dist=206.16px)
- **Final Selected**: DECOY (loc_err=640.34px)

---

## Failure Stage Diagnostics (K_peaks = 5, Present N=160)

- GT absent from coarse search (>15px): **16**
- GT lost at coarse Top-K selection (>15px): **17**
- GT lost during fine grid refinement (>15px): **0**
- GT lost at final Top-5 fused ranking: **84**
- GT successfully selected (Top-1 <= 5px): **29**
- GT in Top-5 but wrong candidate selected: **14**

---

## Runtime Performance (K_peaks = 5)

- **Median**: 550 ms
- **P90**: 617 ms
- **P99**: 664 ms

---

## Final Decision: REJECT
