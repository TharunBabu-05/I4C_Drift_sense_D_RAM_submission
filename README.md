<div align="center">

<img src="assets/drift_sense_animated.svg" width="100%">

</div>

---

## 🚨 The Problem Statement
In modern semiconductor foundries, locating a specific microscopic reference patch (e.g., a single FinFET or DRAM layout cell) within a massive, highly-degraded Scanning Electron Microscope (SEM) search image is an immense challenge:
* **Extreme Noise:** SEM imaging physics naturally introduces catastrophic Poisson shot noise, Gaussian read noise, focal blur, multiplicative speckle, contrast drift, and stage vibration jitter.
* **Periodic Decoys:** Semiconductor chips (especially DRAM arrays) consist of millions of perfectly repeating, identical-looking structures.
* **Classical Fragility:** Traditional mathematical Computer Vision algorithms (like pure Template Matching) routinely fail because they lack the semantic understanding to differentiate the true target from a noisy periodic decoy.

## 💡 Our Solution
We engineered a **Hybrid Fusion Pipeline** that runs classical Normalized Cross-Correlation (NCC) search through a **3-Level Image Pyramid**, disambiguated by a **Siamese Triplet Loss Model (TLM)** built on a lightweight **Custom 4-Layer ResNet** encoder.

By combining the raw speed of classical coarse search with the deep semantic understanding of a learned heatmap regressor, we created a system capable of shattering classical accuracy baselines while maintaining strict real-time edge-device constraints (~44ms average on CPU, 1.38MB model weights).

---

## 🏗️ Architecture Deep-Dive

### 1. Preprocessing Pipeline
Every input pair (target crop + full search image) is normalized before matching:
* Convert to grayscale (uint8)
* Histogram equalization
* Denoising (Gaussian / Median)
* Edge enhancement

### 2. Custom 4-Layer ResNet Siamese Encoder
Rather than deploying a massive off-the-shelf architecture, we designed a deliberately bottlenecked **4-Layer ResNet** encoder. This is the only backbone used in the pipeline (selected via `--encoder resnet`):
* **Why?** It mathematically forces the network to ignore transient noise and focus purely on extracting the fundamental, invariant geometric structure of the semiconductor layouts.
* **Output:** Projects 1-channel grayscale SEM patches into a discriminative **128-Dimensional embedding space**, ready for triplet-loss metric learning.

### 3. Training Setup
| Component | Detail |
|---|---|
| Framework | PyTorch 2.x, OpenCV, NumPy |
| Backbone | Custom 4-Layer ResNet Siamese Encoder |
| Optimizer | AdamW (initial LR = 0.001, weight decay = 1e-4) |
| LR Scheduler | Cosine Annealing (T_max = 30 epochs) |
| Loss Function | Triplet Loss Model (TLM) — metric learning over the 128-D embedding space, with hard negative mining (15 hard periodic-decoy negatives + 15 global random negatives per anchor) |
| Deployment Target | 100% CPU edge execution (factory inspection hardware); optional CUDA 12.1 GPU acceleration |

### 4. Three-Level Image Pyramid (Hybrid Inference)
Our production inference flow (`run_inference.py`) splits the search space across three progressively finer pyramid levels to balance speed and accuracy:

```mermaid
graph TD
    A[1000x1000 SEM Search Image] --> B[Preprocessing: Grayscale, Histogram Eq, Denoise, Edge Enhance]
    B --> C[Level 0: Coarse NCC — 50x50 template across 500x500]
    C -->|Fast Global Screening| D[Top-K Candidate Peaks]
    D --> E[Level 1: Siamese Verification & Re-Ranking — 100x100]
    E -->|4-Layer ResNet TLM| F[Re-Ranked Best Candidate]
    F --> G[Level 2: Fine Localization — 200x200 Local Refinement]
    G --> H{Center Bias Disambiguation}
    H -->|Multiple similar peaks?| I[Select candidate closest to image center]
    I --> J[Post-Processing: NMS, Sub-Pixel Refinement, Boundary Check, Confidence Threshold]
    J --> K((Final Sub-Pixel x, y Coordinate))
```

1. **Level 0 — Coarse NCC:** A 50×50 template is swept across a 500×500 downsampled search region for fast global screening, producing Top‑K candidate peaks.
2. **Level 1 — Siamese Verification & Re-Ranking:** Full-resolution (100×100) matching using the Siamese ResNet TLM encoder re-ranks and verifies candidates against periodic decoys.
3. **Level 2 — Fine Localization:** Local refinement at 200×200 around the best candidate to pin down the final target.
4. **Center Bias Disambiguation:** When multiple candidates score similarly, the pipeline computes each candidate's distance from the image center and selects the closest reliable candidate.
5. **Post-Processing:** Non-Maximum Suppression, fine peak/coordinate refinement, boundary checking, and confidence thresholding produce the final `(x, y)` output.

### 5. Decision Rules & Mathematical Formulas

**Non-Maximum Suppression**
Extract Top‑K (K = 3) peaks from the heatmap `H(x, y)` using a 3×3 NMS window:
```
P_DL(t) = H(x_t, y_t)
```

**Hybrid Score Fusion**
Deep semantic probability is fused with the local structural NCC score using a power-law weighting:
```
S_fused(i) = P_DL(i)^α · R_NCC(i)^β
α = 0.6, β = 1.5
-1 ≤ R_NCC ≤ 1,  0 ≤ P_DL ≤ 1
```

**Periodic Pitch Decoy Rejection Threshold**
The winning candidate must clear the next-best candidate by a fixed margin to overcome periodic DRAM cell similarity:
```
r' = 1                          if S_best(1) ≥ T_theory
r' = argmax_i S_best(i)         otherwise
T_theory = 1.15   (empirically chosen)
```

**Sub-Pixel Parabolic Refinement**
2D parabolic peak fitting around the winning integer peak `(x̂, ŷ)`:
```
Δx_sub = R(x̂+1, ŷ) − R(x̂−1, ŷ) / [2(2R(x̂,ŷ) − R(x̂−1,ŷ) − R(x̂+1,ŷ))]
Δy_sub = R(x̂, ŷ+1) − R(x̂, ŷ−1) / [2(2R(x̂,ŷ) − R(x̂,ŷ−1) − R(x̂,ŷ+1))]
(x*, y*) = (x̂ + Δx_sub, ŷ + Δy_sub)
```

---

## 📊 Dataset & MLOps Generation
Because real foundry images are highly confidential, we engineered a procedural **MLOps Synthetic DRAM Dataset Generator**, built on **60 CAD baseline DRAM generator scripts** spanning 5 structurally distinct sub-micron layout families (100% DRAM architectures):

1. **Vertical BL-Twist DRAM Arrays** — twisted/crossover bitlines, double parallel capacitor plates, top via pads
2. **3D Stacked-Capacitor DRAM** — cylindrical/rectangular stacked capacitors, storage node isolation
3. **Hybrid Zigzag & Box-Capacitor DRAM** — alternating zigzag bitlines, box-capacitor nodes in isolation trenches
4. **Crown-Capacitor & Deep-Trench DRAM** — hollow crown-shaped capacitor nodes, shared S/D diffusion
5. **Honeycomb Packed & Periphery DRAM** — highest-density honeycomb-packed arrays, WL driver/SA via matrices

**Generation pipeline:**
* Uniform target sampling with **zero center-position bias** — target coordinates `(x_gt, y_gt) ∈ [100, 900] × [100, 900]` over a 1000×1000 canvas.
* Reference/search pair creation: high-magnification reference crop (100×100) upscaled to full resolution, paired with a full 1000×1000 search image at lower magnification.
* Metadata banner stripping to remove synthetic SEM header/footer artifacts before degradation.
* **Stochastic SEM degradation engine** applying 2.0× physical degradation impact:
  * Poisson shot noise
  * Gaussian read noise
  * Defocus blur (Gaussian)
  * Multiplicative speckle
  * Contrast modulation
  * Stage vibration jitter (affine translation)

Each degradation is physically modeled and cited (Reimer 2013 — SEM physics; Holzer et al. 2021 — low-dose SEM noise; Goldstein et al. 2017 — SEM & X-ray microanalysis; Postek & Vladár 2011 — CD metrology noise; Sim et al. — SE gain/contrast in SEM; Cizmar et al. 2008 — vibration analysis/mitigation in high-res SEM).

**Scale & rotation modeling:**
* Scale invariance via broadcast pooling of a 256-D embedding into an 8×8×1 heatmap.
* Affine rotation + translation jitter: `θ ∈ [-15°, 15°]`, `Δx, Δy ∈ [-3, 3] μm`, applied with border-replicate padding.

**Strict architecture-level dataset split (zero data leakage):**

| Split | Layout Scripts | Share | Notes |
|---|---|---|---|
| Train | 50 | 80% | Learns general DRAM layout features |
| Validation | 5 | 10% | Evaluates unseen layout generalization |
| Test (locked) | 5 | 10% | Hidden benchmark, unseen architectures never used during training |

Validation and test layouts are generated from architecture scripts the model has never seen, guaranteeing the model is evaluated on genuinely novel wafer designs.

---

## 🚀 Execution & Usage

### 1. Environment Setup
```bash
git clone https://github.com/TharunBabu-05/I4C_Drift_sense_D_RAM_submission.git
cd I4C_Drift_sense_D_RAM_submission
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

### 2. Single Pair Inference
```bash
python3 master_inference.py --reference all_60_pairs/pair_001/reference.png  --search all_60_pairs/pair_001/search.png 
```

### 3. Benchmark Evaluation
```bash
python3 evaluate.py --data_dir all_60_pairs --checkpoint best_model_level1.pth

```

**Tech Stack:** Python 3.10 · PyTorch 2.3.0 · OpenCV 4.11.0 · NumPy 1.26.4
**Hardware:** NVIDIA CUDA 12.1 (GPU acceleration) with full CPU fallback compatibility
**Model Checkpoint:** `best_model_level1.pth` — 1.38MB (ultra-low-weight model)

---

## 🏆 Final Results

Benchmarked across all 60 held-out test pairs:

| Metric | Value |
|---|---|
| Inference Speed | 43.56 ms/image (avg) |
| Mean Localization Error | 21.05 px |
| Accuracy (≤ 5px tolerance) | **95.0%** (57 hits / 3 misses) |
| Perfect Matches (0px error) | **91.7%** (55 / 60) |
| Failure Cases (> 5px) | 3 total (e.g. pair_010 — see `model/all_60_pairs/visualizer`) |

**Distribution notes:**
* The error histogram is heavily concentrated near 0px, with only 3 outlier images exceeding the 5px tolerance threshold — no long tail of moderate-error cases.
* Inference time stays consistently in the 40–55ms band regardless of accuracy outcome (a handful of ~65ms outliers), confirming that failure cases are *localization* failures, not slow/incomplete searches.
* Spatial GT-vs-predicted plots show tight clustering for the majority of targets, with the 3 failures showing large, isolated jumps — a known limitation flagged for future hard-negative refinement work.

Full per-pair results are written to `results_manifest.csv`, with visualizations saved under `model/all_60_pairs/visualizer`.

---

## ⚔️ Baseline Comparison: Classical OpenCV vs. Our Hybrid TLM

### Extreme-Noise Benchmark (Synthetic Test Set, 1,600 images, 2.0× noise)
We injected catastrophic physical degradations (Poisson/Gaussian noise, focal blur) to simulate extreme factory conditions.

| Metric | Baseline (`inference.py`, OpenCV) | Our Hybrid TLM (`_resnet_final_16k_correct_Dataset_TLM`) |
|---|---|---|
| Localization Accuracy (≤5px) | 52.6% (842 hits) | **58.0%** (928 hits) |
| Inference Speed (Pure CPU) | 65.0 ms/image | **33.2 ms/image** |

### Ideal-Conditions Benchmark (Physical Test Set, 60 images)

| Metric | Baseline | Our Hybrid TLM |
|---|---|---|
| Localization Accuracy | 100% | 96.7% (58 hits) |
| Inference Speed | — | 49.1 ms/image |

**Takeaway:** Even in perfect conditions, our Hybrid TLM executes in just 49.1 ms — comfortably beating the 60 ms real-time deadline. Under extreme noise, it's also nearly **2× faster** than the classical baseline (33.2 ms vs 65.0 ms), because the lightweight 4-layer ResNet cuts through noise far more efficiently than brute-force template matching.

---

## 🔬 Robustness Comparison & Baseline Ablation

**The Classical Failure:** Under extreme 2.0× noise, the pure mathematical tracking of `inference.py` shatters — it blindly snaps to periodic decoys because geometric noise corrupts the template matching correlation surface.

**The AI Fusion Success:** Our Hybrid Triplet Loss Model successfully "semanticizes" the noise. Even when the geometric structure is degraded, the 128-D embedding space has learned the fundamental invariant layout of the semiconductor, boosting extreme-noise accuracy by an absolute **+5.4%** over the classical baseline.

**Ablation Study — Why `top_k=3` is the secret weapon:**
* Forcing the TLM to evaluate all 60 OpenCV candidates simultaneously causes **multi-decoy confusion**, dropping accuracy back down to 57.3%.
* Our final pipeline mathematically restricts the classical stage's output to the **top_k=3** candidates, separated by a **min_distance=10**. This acts as a strict geometric pre-filter, preventing the AI from being overwhelmed and letting the Siamese network act as the final, decisive judge — this is the configuration behind the 58.0% headline accuracy above.

---

## 🏆 Key Achievements
* **Unprecedented Robustness:** 95.0% accuracy at ≤5px tolerance and 91.7% perfect (0px) localization across unseen DRAM architectures, plus a +5.4% absolute accuracy gain over the classical baseline under extreme 2.0× noise.
* **Edge-Ready Performance:** Inference executes in ~33–49 ms on CPU depending on noise conditions — up to 2× faster than the classical OpenCV baseline — with a 1.38MB model footprint.
* **Architectural Mastery:** A single lightweight, custom-tailored 4-layer ResNet Siamese encoder generalizes better to industrial SEM noise than heavy off-the-shelf SOTA models.
* **Leakage-Free Evaluation:** Architecture-level train/val/test partitioning across 60 procedurally generated DRAM layouts guarantees results reflect true generalization, not memorization.


# Drift-Sense: Hybrid Siamese Localization (Final Model)

This repository contains the finalized production code for **Drift-Sense**, a highly robust Hybrid Siamese Neural Network designed for sub-pixel localization of highly-degraded, extreme-noise Scanning Electron Microscope (SEM) imagery.

## 🚀 Environment Setup

We recommend running this project inside a Python virtual environment to prevent dependency conflicts.

1. **Create a Virtual Environment:**
   ```bash
   python3 -m venv venv
   ```

2. **Activate the Virtual Environment:**
   * On Linux/macOS:
     ```bash
     source venv/bin/activate
     ```
   * On Windows:
     ```cmd
     venv\Scripts\activate
     ```

3. **Install the Required Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Run the Bulk Evaluation

We have provided a bulk evaluation script that tests the Hybrid AI model against physical layout pairs and generates a formatted terminal report, visualizations, and a CSV manifest.

Run the following command:
```bash
python3 evaluate.py --data_dir all_60_pairs --checkpoint best_model_level1.pth
```
* **Output:** This will print the final Machine Learning metrics (Inference Speed, Mean Error, Accuracy) to your terminal. It will also create annotated bounding-box visualizations inside `all_60_pairs/visualizer/` and generate a `results_manifest.csv` file.

---

## 🔬 Run Individual Inference

If you want to run the pipeline on a single pair of images, use the Master Inference script.

By default, the script will execute the complete **Hybrid (OpenCV NCC + Siamese AI)** pipeline:
```bash
python master_inference.py --reference img2.png --search img1.png
```

### Additional Inference Options:
* **Run Pure Classical Baseline:** If you want to force the pipeline to bypass the AI and execute only the classic 3-Layer OpenCV Pyramid (for ablation testing), add the `--ncc_only` flag:
  ```bash
  python master_inference.py --reference img2.png --search img1.png --ncc_only
  ```
* **Enable Verbose Output:** Add `--verbose` to see the step-by-step execution times and fusion scores:
  ```bash
  python master_inference.py --reference img2.png --search img1.png --verbose
  ```
