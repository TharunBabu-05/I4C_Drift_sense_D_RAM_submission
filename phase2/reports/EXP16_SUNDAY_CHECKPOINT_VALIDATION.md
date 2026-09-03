# EXP-16 SUNDAY CHECKPOINT VALIDATION REPORT

OLD EXP-13 BASELINE:

    71.65 /100

SUNDAY CHECKPOINT:

    72.80 /100

DELTA:

    +1.16

Localization:

    21.01 /40

Scale:

    5.75 /10

Rotation:

    6.19 /10

Pose:

    11.94 /20

Rejection:

    14.86 /15

Confidence:

    10.00 /10

Efficiency:

    5.00 /5

Generator/Citations:

    10.00 /10

Runtime:

    Median = 781 ms
    P90    = 1387 ms
    P99    = 1610 ms

Recovered:

    23 pairs

Regressed:

    0 pairs

Unchanged:

    177 pairs

---

## Set-Wise Breakdown

### Set A (SEM Clean - 70 pairs)
- Old Baseline: Localized = 66, Failed = 4
- Sunday Model: Localized = 66, Failed = 4

### Set B (SEM Degraded - 70 pairs)
- Old Baseline: Localized = 30, Failed = 40
- Sunday Model: Localized = 30, Failed = 40

### Set C (Absent Pairs - 40 pairs)
- Old Baseline: Correct Rejections (TN) = 14/40, False Positives (FP) = 26/40
- Sunday Model: Correct Rejections (TN) = 37/40, False Positives (FP) = 3/40

### Set D (Optical Analogue - 20 pairs)
- Old Baseline: Localized = 15, Failed = 5
- Sunday Model: Localized = 15, Failed = 5

---

## Target Pairs Trace

### pair_006
- **Old Baseline**: LocErr = 1.33px, Found = 1, Score = 0.9928
- **Sunday Model**: LocErr = 1.33px, Found = 1, Score = 0.9978

### pair_066
- **Old Baseline**: LocErr = 0.69px, Found = 1, Score = 0.9925
- **Sunday Model**: LocErr = 0.69px, Found = 1, Score = 0.9978

### pair_116
- **Old Baseline**: LocErr = 33.63px, Found = 1, Score = 0.9857
- **Sunday Model**: LocErr = 33.63px, Found = 1, Score = 0.9868

### pair_160
- **Old Baseline**: LocErr = 999.0px, Found = 1, Score = 0.7354
- **Sunday Model**: LocErr = 0.0px, Found = 0, Score = 0.2174

### pair_186
- **Old Baseline**: LocErr = 0.29px, Found = 1, Score = 0.998
- **Sunday Model**: LocErr = 0.29px, Found = 1, Score = 0.9988

---

## Checkpoint Metadata & Integrity

- **Checkpoint File**: `checkpoints_phase2_v2_sunday/best_model_phase2.pth`
- **File Size**: 1380297 bytes
- **SHA-256 Hash**: `74714ac16cb25da8a707113af9b30fa2ee051302065eaf947b46ef0a27592b8f`

---

## DECISION

**PROMOTE CHECKPOINT ONLY**
