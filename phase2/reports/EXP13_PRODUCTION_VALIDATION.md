# EXP-13 Production Validation Report

## Executive Summary

- **Pre-EXP-13 Baseline Score**: 60.99 / 100
- **Promoted EXP-13 Production Score**: **71.65 / 100**
- **Score Improvement**: **++10.66 points**
- **Pre-EXP-13 Localization**: 14.11 / 40
- **Promoted EXP-13 Localization**: **21.01 / 40**
- **Checkpoint SHA-256**: `e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c` (100% UNTOUCHED)
- **Status**: **PROMOTED AND VALIDATED SUCCESSFUL**

---

## Official 100-Point Score Breakdown

| Category | Pre-EXP-13 Baseline | Promoted Production | Delta |
|---|---|---|---|
| **Localization /40** | 14.11 | **21.01** | **++6.90** |
| **Scale /10** | 3.88 | **5.75** | **++1.87** |
| **Rotation /10** | 4.38 | **6.19** | **++1.81** |
| **Pose Total /20** | 8.25 | **11.94** | **++3.69** |
| **Rejection /15** | 13.87 | **13.87** | **0.00** |
| **Confidence /10** | 9.76 | **9.83** | **++0.07** |
| **Efficiency /5** | 5.00 | **5.00** | **0.00** |
| **Generator/Citations /10** | 10.00 | **10.00** | **0.00** |
| **TOTAL SCORE /100** | **60.99** | **71.65** | **++10.66** |

---

## Candidate Recall Audit

| Threshold | Pre-EXP-13 Baseline | Promoted EXP-13 Production |
|---|---|---|
| **Recall @1px** | 21.2% | **26.88%** |
| **Recall @5px** | 49.4% | **69.38%** |
| **Recall @15px** | 53.1% | **74.38%** |
| **Recall @50px** | 60.6% | **88.75%** |

---

## Target Pairs Verification

### pair_006
- **Prediction**: (x=327.7, y=711.3) scale=9.0 theta=1.0
- **Found**: 1 (score=0.9928)
- **Raw NCC**: 0.9217 | **Raw Siamese**: 0.7383
- **Localization Error**: **1.33 px**

### pair_066
- **Prediction**: (x=319.32, y=702.09) scale=8.75 theta=-3.5
- **Found**: 1 (score=0.9925)
- **Raw NCC**: 0.9165 | **Raw Siamese**: 0.7371
- **Localization Error**: **0.69 px**

### pair_116
- **Prediction**: (x=512.7, y=292.7) scale=8.0 theta=-1.5
- **Found**: 1 (score=0.9857)
- **Raw NCC**: 0.5723 | **Raw Siamese**: 0.9732
- **Localization Error**: **33.63 px**

### pair_186
- **Prediction**: (x=297.02, y=732.29) scale=11.75 theta=-5.0
- **Found**: 1 (score=0.998)
- **Raw NCC**: 0.9836 | **Raw Siamese**: 0.8951
- **Localization Error**: **0.29 px**

---

## Set-D Regression Audit (Optical Microscope Analogue)

The 6 Set-D pairs that regressed due to low contrast / diffuse responses under periodicity penalization:
- `pair_151`: Found=1 (GT=0), Score=0.5204, Error=999.0px
- `pair_152`: Found=1 (GT=0), Score=0.5157, Error=999.0px
- `pair_167`: Found=1 (GT=0), Score=0.6023, Error=999.0px
- `pair_168`: Found=1 (GT=0), Score=0.6023, Error=999.0px
- `pair_173`: Found=1 (GT=0), Score=0.5464, Error=999.0px
- `pair_174`: Found=1 (GT=0), Score=0.5693, Error=999.0px
---

## Runtime Performance

- **Median Runtime**: 533 ms
- **P90 Runtime**: 587 ms
- **P99 Runtime**: 617 ms

---

## Backup Information

Pre-EXP-13 promotion codebase archives:
- `phase2/backup/pre_exp13_promotion/phase2_inference.py`
- `phase2/backup/pre_exp13_promotion/phase2_config.py`
- `phase2/backup/pre_exp13_promotion/register.py`
