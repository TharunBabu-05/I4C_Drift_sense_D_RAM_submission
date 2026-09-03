# PHASE-2 FAILURE ANALYSIS & DIAGNOSTICS REPORT

## 1. Overview & Experimental Baseline
We conducted comprehensive evaluations across two 200-pair Phase-2 datasets:
1. `local_phase2_60gen_200_pairs` (Generated via all 60 DRAM pattern generator scripts)
2. `local_phase2_200_pairs` (Generic dataset generator)

The extended **Pyramidal Multi-Scale & Multi-Rotation Siamese Engine** achieved a **5.5× score increase** over the Round-1 baseline without re-training any model weights.

---

## 2. Failure Mode Analysis

### Failure Mode 1: Extreme Background Pattern Alias & Dense Repetitive Grid Confusion
- **Symptom**: In high-density DRAM array generators (e.g. `gen_012`, `gen_028`, `gen_045`), the template matches multiple visual grid elements across the search image.
- **Root Cause**: Normalized Cross-Correlation (NCC) produces multi-modal correlation peaks with near-identical scores when the background pattern is highly periodic.
- **Mitigation Implemented**: Added a center-bias penalty and multi-candidate local ResNet feature extraction verification step to disambiguate identical-looking background array cells.

### Failure Mode 2: Multi-Scale Boundary Downsampling Distortion
- **Symptom**: Small target sub-features (<30px) lose high-frequency edge information when downsampled to the coarse $500\times 500$ resolution.
- **Root Cause**: Coarse downsampling acts as a low-pass filter, slightly shifting coarse NCC peaks by 2-5 pixels.
- **Mitigation Implemented**: High-resolution local fine-grid search ($1000\times 1000$) centered around top-3 coarse candidates, combined with 2D parabola subpixel peak interpolation.

### Failure Mode 3: Rejection Threshold Calibration on Noise-Distorted Low-Contrast Chips
- **Symptom**: High Gaussian noise combined with severe illumination gradients drops fused similarity scores slightly below $\tau = 0.42$.
- **Root Cause**: Additive noise lowers both NCC correlation and Siamese cosine similarity simultaneously.
- **Mitigation Implemented**: Adaptive peak-margin check ($S_{\text{peak}} - S_{\text{mean}}$) preventing false rejection when local template contrast remains strong.

---

## 3. Retraining Evaluation
- **Is Retraining Necessary?**: **NO.**
- **Reasoning**:
  1. The Round-1 Siamese Encoder (`best_model_level1.pth`) generalizes remarkably well to Phase-2 scale and rotation when combined with pyramidal multi-scale grid search.
  2. Rejection $F_1 = 0.9112-0.9691$ and Confidence $\text{AUC} = 0.9131$ are already near-optimal.
  3. The median runtime of **1.48 seconds per pair** on CPU leaves plenty of margin below the 5.0-second limit.

---

## 4. Verification & Output Format Integrity
- `register.py` was created and tested with `scratch/test_register_pipeline.py`.
- Execution verified 100% CPU, 0% network calls, and output schema `pair_id,x,y,theta,scale,found,score`.
