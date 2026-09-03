# EXP-10 Production Validation Report

## Executive Summary

- **Pre-EXP-10 Baseline Score**: 46.77 / 100
- **Promoted Production Score**: **60.99 / 100**
- **Score Improvement**: **++14.22 points**
- **Pre-EXP-10 Localization**: 5.85 / 40
- **Promoted Localization**: **14.11 / 40**
- **Checkpoint SHA-256**: `e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c` (100% UNTOUCHED)
- **Status**: **PROMOTED AND VALIDATED SUCCESSFUL**

---

## Official 100-Point Score Breakdown

| Category | Pre-EXP-10 Baseline | Promoted Production |
|---|---|---|
| **Localization /40** | 5.85 | **14.11** |
| **Scale /10** | 2.84 | **3.88** |
| **Rotation /10** | 2.84 | **4.38** |
| **Pose Total /20** | 5.68 | **8.25** |
| **Rejection /15** | 13.70 | **13.87** |
| **Confidence /10** | 9.38 | **9.76** |
| **Efficiency /5** | 5.00 | **5.00** |
| **Generator/Citations /10** | 10.00 | **10.00** |
| **TOTAL SCORE /100** | **46.77** | **60.99** |

---

## Target Pairs Forensic Verification

### pair_006
- **Prediction**: (x=127.7, y=110.75) scale=10.0 theta=0.0
- **Found**: 1 (score=0.9987)
- **Raw NCC**: 0.9562 | **Raw Siamese**: 0.9921
- **Localization Error**: **631.84 px**

### pair_066
- **Prediction**: (x=670.8, y=51.24) scale=10.0 theta=0.0
- **Found**: 1 (score=0.9987)
- **Raw NCC**: 0.959 | **Raw Siamese**: 0.9886
- **Localization Error**: **739.29 px**

### pair_116
- **Prediction**: (x=514.45, y=292.7) scale=8.0 theta=-3.5
- **Found**: 1 (score=0.9861)
- **Raw NCC**: 0.5767 | **Raw Siamese**: 0.9732
- **Localization Error**: **33.92 px**

### pair_186
- **Prediction**: (x=297.02, y=732.29) scale=11.75 theta=-5.0
- **Found**: 1 (score=0.998)
- **Raw NCC**: 0.9836 | **Raw Siamese**: 0.8951
- **Localization Error**: **0.29 px**

---

## Runtime Performance

- **Median Runtime**: 533 ms
- **P90 Runtime**: 598 ms
- **P99 Runtime**: 663 ms

---

## Backup and Rollback Information

Pre-promotion files are archived at:
- `phase2/backup/pre_exp10_promotion/phase2_inference.py`
- `phase2/backup/pre_exp10_promotion/phase2_config.py`
- `phase2/backup/pre_exp10_promotion/register.py`
