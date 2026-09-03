# Phase-1 Method Baseline

## Declared Phase-1 Method (verbatim from README.md)

> "We engineered a **Hybrid Fusion Pipeline** that runs classical Normalized Cross-Correlation (NCC) search through a **3-Level Image Pyramid**, disambiguated by a **Siamese Triplet Loss Model (TLM)** built on a lightweight **Custom 4-Layer ResNet** encoder."

**Short form declaration:**

> **PHASE-1 DECLARED METHOD: "Hybrid NCC + Custom 4-Layer ResNet Siamese (Triplet Loss Model)"**

---

## 1. NCC Implementation

**File:** `master_inference.py` — `ncc_search()` (lines 154–180)

```python
def ncc_search(search_image, template, top_k=20, min_score=-1.0):
    res = cv2.matchTemplate(search_f32, template_f32, cv2.TM_CCOEFF_NORMED)
    # → returns Top-K peaks by NMS
```

- Algorithm: `cv2.TM_CCOEFF_NORMED` (standard Normalized Cross-Correlation)
- Used at 3 pyramid levels: L0 (50×50 template, 500×500 search), L1 (100×100), L2 (200×200)
- Top-K extraction by iterative NMS with peak suppression

---

## 2. Custom 4-Layer ResNet Implementation

**File:** `models/siamese_encoder.py` — `SiameseEncoder`

```
Layer Structure:
  conv1:   Conv2d(1, 16, 3×3)  + BN + ReLU + MaxPool2d(2×2)
  layer1:  ResidualBlock(16→32,  stride=2)
  layer2:  ResidualBlock(32→64,  stride=2)
  layer3:  ResidualBlock(64→128, stride=2)
  avgpool: AdaptiveAvgPool2d(1,1)
  fc:      Linear(128 → embedding_dim)
  L2 normalize → 128-D unit sphere embedding
```

Each `ResidualBlock` = Conv3×3 + BN + ReLU + Conv3×3 + BN + skip connection

**This is the "Custom 4-Layer ResNet" declared in Phase 1.**

**Activated via:** `encoder_type='resnet'` in `PyramidSiameseNetwork`

---

## 3. Siamese Architecture

**File:** `models/pyramid_siamese.py` — `PyramidSiameseNetwork`

- Weight-shared encoder (same `SiameseEncoder` instance for both branches)
- `model.encoder(ref)` → ref_emb; `model.encoder(candidate)` → cand_emb
- Similarity: cosine similarity (dot product of L2-normalized embeddings)

---

## 4. Embedding Dimension

**128-D** — declared in README and confirmed in `siamese_encoder.py`:
```python
self.fc = nn.Linear(128, embedding_dim)  # embedding_dim defaults to 128
```

---

## 5. Loss Function

**InfoNCE** (at Level 1) with 30 negatives per anchor:
- 15 local hard negatives: periodic DRAM decoy crops shifted ±10–15 px
- 15 global random negatives

**File:** `training/losses.py` — `InfoNCELoss`

README also mentions "Triplet Loss Model (TLM)" — this was the original framing; the code uses InfoNCE which is a generalized form of triplet loss.

---

## 6. Candidate Generation

**Phase 1 flow:**
```
L0: 50×50 template NCC across 500×500 search → Top-20 coarse candidates
L1: 100×100 template NCC across full 1000×1000 → Top-30, refined
Fusion: score = 0.35 * NCC_L0 + 0.65 * NCC_L1
```

**Hybrid disambiguation:**
```python
# master_inference.py lines 398–437
ncc_result = cv2.matchTemplate(search_eq, ref_eq, cv2.TM_CCOEFF_NORMED)
top_peaks = non_max_suppression_peaks(ncc_result, min_distance=10, top_k=3)
# → batch Siamese embedding comparison → fusion_score = 0.3*NCC + 0.7*Siamese
```

---

## 7. Score Fusion Formula

```
fusion_score(i) = 0.3 × NCC_score(i) + 0.7 × Siamese_cosine_sim(i)
```
*(master_inference.py line 451)*

---

## 8. Localization / Refinement

- Parabolic sub-pixel refinement around best integer peak
- Level-2 fine NCC on 2× upscaled window (200×200 template)
- Center-bias disambiguation when tied

---

## 9. Model Checkpoint

**File:** `best_model_level1.pth`
- **Size:** 1,379,880 bytes (1.38 MB)
- **SHA-256:** `267A7AC9B6F2A077F18E9BE0274E604C5D1268F6564874603BD122AEC5F97178`
- **Encoder type:** `resnet` (SiameseEncoder with 4-layer residual backbone)

---

## 10. Training Method

**File:** `training/train_siamese_v2.py`
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Scheduler: Cosine annealing with 3-epoch warmup
- Augmentation: SEM-style shot noise + Gaussian + blur + brightness jitter
- Best checkpoint saved on highest validation accuracy

---

## 11. Inference Entry Point

**Phase-1 production entry point:**
```bash
python master_inference.py --reference ref.png --search search.png
# → default: Hybrid NCC + Siamese pipeline
# → fallback: Pure NCC pyramid if checkpoint unavailable
```

**Phase-1 Round-1 batch evaluation:**
```bash
python evaluate.py --data_dir all_60_pairs --checkpoint best_model_level1.pth
```

---

## 12. Phase-1 Performance (from README)

| Metric | Value |
|--------|-------|
| Inference speed | 43.56 ms/image avg |
| Mean localization error | 21.05 px |
| Accuracy ≤5px | 95.0% (57/60) |
| Perfect 0px matches | 91.7% (55/60) |
| Model size | 1.38 MB |
