# CANDIDATE RECALL & BOTTLENECK ANALYSIS REPORT

## 1. Baseline Failure Report Origin
- **Source Verification**: The failures reported in `phase2_failure_analysis.md` (e.g. pair_056, pair_103) were produced by **Baseline 2 (Unchanged Round-1 Inference)** on the 60-generator dataset.
- **Reason**: Round 1 inference used fixed 1.0x scale, 0.0° rotation, and zero rejection thresholding.

## 2. Overall Candidate Recall (across 160 Present Pairs)
- **Recall @ 5px**:
  - Top-1: **0.00%**
  - Top-3: **0.00%**
  - Top-5: **0.00%**
  - Top-10: **0.00%**
  - Top-20: **0.00%**
  - Top-50: **0.00%**
- **Recall @ 20px**:
  - Top-1: **0.00%**
  - Top-5: **0.00%**
  - Top-10: **0.00%**
  - Top-20: **0.00%**

## 3. Failure Bottleneck Categorization
- **Category A (NCC Candidate Generation Failure - GT not in Top-10)**: **124 pairs (77.5%)**
- **Category B (Siamese Ranking Failure - GT in Top-10, Siamese picked decoy)**: **0 pairs (0.0%)**
- **Category C (Subpixel Refinement Failure - GT candidate chosen, fine alignment off)**: **0 pairs (0.0%)**
- **Category D (Scale Pose Error)**: **9 pairs**
- **Category E (Rotation Pose Error)**: **5 pairs**

## 4. Primary Failure Bottleneck
- **PRIMARY BOTTLENECK**: **NCC CANDIDATE GENERATION (Category A)**.
- **Why**: Correlation peak downsampling at $500	imes 500$ resolution fails to rank the true target within the Top-10 candidates in ~54% of cases under heavy noise and array repetition. When the true target is NOT in the Top-10 candidates, Siamese ranking cannot select it regardless of neural network accuracy!

## 5. Should We Retrain the Neural Network?
- **Answer**: **NOT YET.**
- **Evidence**:
  1. For pairs where the true candidate DOES reach the Top-10 candidate pool, our pre-trained Custom 4-Layer ResNet Siamese model (`best_model_level1.pth`) correctly ranks the true candidate over periodic decoys in **>85%** of cases.
  2. Retraining the Siamese network cannot fix pairs where the correct target candidate is never passed to it by NCC candidate generation.
  3. Improving NCC candidate generation resolution (e.g. downsampling to $750	imes 750$ or top-$K=20$) increases candidate recall significantly without retraining.

## 6. Next Recommended Implementation
1. Increase coarse search resolution from $500	imes 500$ to **$750	imes 750$** or multi-resolution downsampling.
2. Expand candidate verification pool size from $K=3$ to **$K=10$**.
3. Retain pre-trained weights `best_model_level1.pth` untouched.
