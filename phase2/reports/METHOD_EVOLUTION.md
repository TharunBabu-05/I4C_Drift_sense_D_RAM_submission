# Method Evolution: Phase 1 → Phase 2

## Phase 1 Pipeline

```
Reference (1000×1000 upscaled)    Search (1000×1000)
           │                              │
           └──────────── PREPROCESSING ───┘
                         (histogram eq, denoise)
                                │
                    ┌───────────▼───────────┐
                    │  LEVEL 0: Coarse NCC  │
                    │  50×50 template       │
                    │  500×500 search       │
                    │  cv2.TM_CCOEFF_NORMED │
                    └───────────┬───────────┘
                                │ Top-20 candidates
                    ┌───────────▼───────────┐
                    │  LEVEL 1: NCC + SIAM  │
                    │  100×100 template     │
                    │  Full 1000×1000       │
                    │  cv2.TM_CCOEFF_NORMED │
                    └───────────┬───────────┘
                                │ Top-K fused candidates
                    ┌───────────▼──────────────────────┐
                    │  Custom 4-Layer ResNet Siamese   │
                    │  SiameseEncoder: 128-D embedding │
                    │  fusion = 0.3×NCC + 0.7×Siamese  │
                    └───────────┬──────────────────────┘
                                │ Best candidate
                    ┌───────────▼───────────┐
                    │  LEVEL 2: Fine NCC    │
                    │  200×200 local refine │
                    │  Parabolic sub-pixel  │
                    └───────────┬───────────┘
                                │
                           (x, y) output
                     scale fixed at 10×, theta ≈ 0°
```

---

## Extension Arrow

```
                    ↓ PHASE-2 EXTENSION ↓

  New requirements disclosed by organizers:
    • scale unknown ∈ [8×, 12×]
    • theta unknown ∈ [−5°, +5°]
    • ~20% absent targets (rejection required)
    • Phase-2 degradation conditions
    • Output: x, y, theta, scale, found, score
```

---

## Phase 2 Pipeline

```
Reference (100×100)             Search (1000×1000)
           │                              │
           └─────── PREPROCESSING ────────┘
                    (grayscale convert)
                           │
           ┌───────────────▼──────────────────┐
           │  MULTI-SCALE COARSE NCC SEARCH   │  ← EXTENSION of Phase-1 L0
           │  For scale ∈ {8, 8.5, …, 12}:   │
           │    patch_size = 1000 / scale      │
           │    Resize ref to patch_size       │
           │    For theta ∈ {−5°, …, +5°}:    │
           │      Rotate ref by theta          │
           │      cv2.TM_CCOEFF_NORMED on      │
           │      500×500 coarse search        │
           └───────────────┬──────────────────┘
                           │ Top-K coarse candidates (scale, theta per cand)
           ┌───────────────▼──────────────────┐
           │  FINE SCALE+ROTATION NCC GRID    │  ← EXTENSION of Phase-1 L1
           │  Around each coarse candidate:   │
           │  fine_scales = [s−0.25, s, s+0.25]│
           │  fine_thetas = [θ−1°, θ, θ+1°]  │
           │  Local window NCC search         │
           │  cv2.TM_CCOEFF_NORMED            │
           └───────────────┬──────────────────┘
                           │ Candidate patches (100×100 crops)
           ┌───────────────▼──────────────────────────┐
           │  SAME Custom 4-Layer ResNet Siamese      │  ← IDENTICAL to Phase-1
           │  SiameseEncoder (unchanged architecture) │
           │  same best_model_level1.pth weights      │
           │  Batch 128-D embedding extraction        │
           │  Cosine similarity to reference          │
           │  fusion = α×NCC_norm + (1−α)×Siamese_sim │
           └───────────────┬──────────────────────────┘
                           │ Best fused candidate
           ┌───────────────▼──────────────────┐
           │  POSE ESTIMATION                  │  ← NEW supporting logic
           │  Best candidate carries:          │
           │    pred_scale = best_scale        │
           │    pred_theta = best_theta        │
           │  Parabolic sub-pixel refinement   │
           └───────────────┬──────────────────┘
                           │
           ┌───────────────▼──────────────────┐
           │  REJECTION / CONFIDENCE           │  ← NEW supporting logic
           │  if fused_score ≥ threshold:      │
           │    found = 1 (target present)     │
           │  else:                            │
           │    found = 0 (target absent)      │
           │  conf = sigmoid(score − tau)      │
           └───────────────┬──────────────────┘
                           │
                 (x, y, theta, scale, found, score)
```

---

## Component-by-Component Comparison

| Component | Phase 1 | Phase 2 | Status |
|-----------|---------|---------|--------|
| **NCC algorithm** | `cv2.TM_CCOEFF_NORMED` | Same `cv2.TM_CCOEFF_NORMED` | **A. SAME** |
| **NCC scale** | Fixed 10× | Swept [8×, 12×] | **B. EXTENSION** |
| **NCC rotation** | ~0° only | Swept [−5°, +5°] | **B. EXTENSION** |
| **NCC pyramid** | 3 levels (L0/L1/L2) | Coarse→Fine grid | **B. EXTENSION** |
| **ResNet architecture** | 4-layer SiameseEncoder | Same 4-layer SiameseEncoder | **A. SAME** |
| **Siamese weight sharing** | Yes | Yes | **A. SAME** |
| **Embedding dimension** | 128-D | 128-D | **A. SAME** |
| **Similarity metric** | Cosine (dot of L2-norm) | Same cosine similarity | **A. SAME** |
| **Score fusion** | 0.3×NCC + 0.7×Siamese | α×NCC + (1−α)×Siamese | **A. SAME** |
| **Sub-pixel refinement** | Parabolic | Same parabolic | **A. SAME** |
| **Model checkpoint** | best_model_level1.pth | Same (until retrained) | **A. SAME** |
| **Scale output** | Not required (10×) | pred_scale estimated | **C. NEW SUPPORTING** |
| **Rotation output** | Not required (~0°) | pred_theta estimated | **C. NEW SUPPORTING** |
| **Rejection** | Not required | Sigmoid threshold on score | **C. NEW SUPPORTING** |
| **Confidence score** | Not required | Sigmoid calibrated | **C. NEW SUPPORTING** |
| **Output format** | (x, y) | (x, y, theta, scale, found, score) | **C. NEW SUPPORTING** |

---

## Key Architectural Truth

The **central algorithm** did not change:

> **Phase 1 core: NCC candidate generation → Siamese verification → best match**
>
> **Phase 2 core: NCC candidate generation → Siamese verification → best match**
>
> The only addition is that Phase 2 sweeps NCC over a 2D grid of (scale, rotation)
> values instead of a single fixed (10×, 0°) point — and adds pose/rejection output.
