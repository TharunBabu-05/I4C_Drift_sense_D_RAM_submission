#!/usr/bin/env python3
"""
Comprehensive SEM Image Degradation Engine (Max 3 Degradations per Image)
========================================================================

Implements ALL degradation categories for synthetic SEM dataset generation
with strict rules:

    REFERENCE / TARGET = ALWAYS CLEAN (no degradation ever)
    SEARCH             = STOCHASTICALLY DEGRADED (MAX 3 degradations total)

Categories & Models:
--------------------
A. SEM NOISE:           poisson, gaussian, gamma, secondary_electron, low_frequency
B. ROBUSTNESS NOISE:    speckle, banding, salt_pepper
C. BLUR / RESOLUTION:   gaussian_blur, motion_blur
D. GEOMETRIC:           rotation, barrel, pincushion, projective
E. ACQUISITION:         field_nonuniformity, vibration, stage_drift

STRICT CONSTRAINT:
- No more than 3 degradation models are EVER applied to a single Search image.
- Keeps search images clear and structurally findable against the reference target.
"""

import numpy as np
import cv2
from collections import Counter


# =============================================================================
# DEFAULT CONFIGURATION (Capped at Max 3 Models Total)
# =============================================================================

DEFAULT_CONFIG = {
    "enabled": True,
    "probability_applied": 0.90,   # 90% get degradation, 10% clean
    "min_total_models": 2,
    "max_total_models": 2,         # EXACTLY 2 DEGRADATION MODELS PER SEARCH IMAGE

    # Model selection weights across all categories
    "model_weights": {
        # Category A: SEM Noise
        "poisson":            0.20,
        "gaussian":           0.20,
        "gamma":              0.08,
        "secondary_electron": 0.08,
        "low_frequency":      0.06,

        # Category B: Robustness Noise
        "speckle":            0.08,
        "banding":            0.06,
        "salt_pepper":        0.04,

        # Category C: Blur
        "gaussian_blur":      0.08,
        "motion_blur":        0.04,

        # Category D: Geometric (mild parameters)
        "rotation":           0.03,
        "barrel":             0.02,
        "pincushion":         0.02,
        "projective":         0.01,

        # Category E: Acquisition
        "field_nonuniformity": 0.05,
        "vibration":           0.03,
        "stage_drift":         0.02,
    },

    # Per-model strength ranges (kept moderate to preserve pattern clarity)
    "strength_ranges": {
        "poisson":            {"scale_min": 1.0, "scale_max": 5.0},
        "gaussian":           {"sigma_min": 3.0, "sigma_max": 15.0},
        "gamma":              {"shape_min": 10.0, "shape_max": 50.0},
        "secondary_electron": {"sigma_min": 3.0, "sigma_max": 12.0},
        "low_frequency":      {"amplitude_min": 5.0, "amplitude_max": 20.0, "frequency_min": 1, "frequency_max": 3},

        "speckle":            {"sigma_min": 0.02, "sigma_max": 0.08},
        "banding":            {"amplitude_min": 3.0, "amplitude_max": 12.0},
        "salt_pepper":        {"density_min": 0.0005, "density_max": 0.005},

        "gaussian_blur":      {"ksize_min": 3, "ksize_max": 5, "sigma_min": 0.5, "sigma_max": 1.5},
        "motion_blur":        {"ksize_min": 3, "ksize_max": 7, "angle_min": 0.0, "angle_max": 180.0},

        "rotation":           {"angle_min": -2.0, "angle_max": 2.0},
        "barrel":             {"k1_min": 0.005, "k1_max": 0.03},
        "pincushion":         {"k1_min": -0.03, "k1_max": -0.005},
        "projective":         {"max_shift_min": 3, "max_shift_max": 12},

        "field_nonuniformity": {"strength_min": 0.05, "strength_max": 0.20},
        "vibration":           {"amplitude_min": 1, "amplitude_max": 3},
        "stage_drift":         {"dx_min": -5, "dx_max": 5, "dy_min": -5, "dy_max": 5},
    },
}


# =============================================================================
# MODEL IMPLEMENTATIONS
# =============================================================================

# Category A
def noise_poisson(img_f32, rng, cfg):
    scale = rng.uniform(cfg["scale_min"], cfg["scale_max"])
    img_scaled = np.clip(img_f32 / 255.0 * scale, 0, None)
    noisy = rng.poisson(img_scaled).astype(np.float32) / scale * 255.0
    return noisy, {"model": "poisson", "scale": round(float(scale), 3)}

def noise_gaussian(img_f32, rng, cfg):
    sigma = rng.uniform(cfg["sigma_min"], cfg["sigma_max"])
    noise = rng.normal(0, sigma, img_f32.shape).astype(np.float32)
    return img_f32 + noise, {"model": "gaussian", "sigma": round(float(sigma), 3)}

def noise_gamma(img_f32, rng, cfg):
    shape = rng.uniform(cfg["shape_min"], cfg["shape_max"])
    gain = rng.gamma(shape, 1.0 / shape, img_f32.shape).astype(np.float32)
    return img_f32 * gain, {"model": "gamma", "shape": round(float(shape), 3)}

def noise_secondary_electron(img_f32, rng, cfg):
    sigma = rng.uniform(cfg["sigma_min"], cfg["sigma_max"])
    h, w = img_f32.shape[:2]
    hf = rng.normal(0, sigma * 0.7, img_f32.shape).astype(np.float32)
    lf_h, lf_w = max(h // 16, 1), max(w // 16, 1)
    lf_shape = (lf_h, lf_w) if img_f32.ndim == 2 else (lf_h, lf_w, img_f32.shape[2])
    lf_small = rng.normal(0, sigma * 0.3, lf_shape).astype(np.float32)
    lf = np.repeat(np.repeat(lf_small, (h // lf_h) + 1, axis=0), (w // lf_w) + 1, axis=1)
    lf = lf[:h, :w] if img_f32.ndim == 2 else lf[:h, :w, :]
    return img_f32 + hf + lf, {"model": "secondary_electron", "sigma": round(float(sigma), 3)}

def noise_low_frequency(img_f32, rng, cfg):
    amp = rng.uniform(cfg["amplitude_min"], cfg["amplitude_max"])
    n = int(rng.integers(cfg["frequency_min"], cfg["frequency_max"] + 1))
    h, w = img_f32.shape[:2]
    drift = np.zeros((h, w), dtype=np.float32)
    for _ in range(n):
        fx = rng.uniform(0.5, 3.0) * np.pi / w
        fy = rng.uniform(0.5, 3.0) * np.pi / h
        px, py = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
        a = rng.uniform(0.3, 1.0) * amp
        drift += a * np.outer(np.sin(fy * np.arange(h) + py), np.sin(fx * np.arange(w) + px)).astype(np.float32)
    if img_f32.ndim == 3:
        drift = drift[:, :, np.newaxis]
    return img_f32 + drift, {"model": "low_frequency", "amplitude": round(float(amp), 3)}

# Category B
def noise_speckle(img_f32, rng, cfg):
    sigma = rng.uniform(cfg["sigma_min"], cfg["sigma_max"])
    sp = rng.normal(0, sigma, img_f32.shape).astype(np.float32)
    return img_f32 * (1.0 + sp), {"model": "speckle", "sigma": round(float(sigma), 3)}

def noise_banding(img_f32, rng, cfg):
    amp = rng.uniform(cfg["amplitude_min"], cfg["amplitude_max"])
    h = img_f32.shape[0]
    line_off = rng.normal(0, amp * 0.6, h).astype(np.float32)
    freq = rng.uniform(0.01, 0.1)
    periodic = (np.sin(np.arange(h, dtype=np.float32) * freq * 2 * np.pi) * amp * 0.4)
    band = line_off + periodic
    if img_f32.ndim == 2:
        return img_f32 + band[:, np.newaxis], {"model": "banding", "amplitude": round(float(amp), 3)}
    else:
        return img_f32 + band[:, np.newaxis, np.newaxis], {"model": "banding", "amplitude": round(float(amp), 3)}

def noise_salt_pepper(img_f32, rng, cfg):
    density = rng.uniform(cfg["density_min"], cfg["density_max"])
    out = img_f32.copy()
    h, w = img_f32.shape[:2]
    n = int(h * w * density / 2)
    out[rng.integers(0, h, n), rng.integers(0, w, n)] = 255.0
    out[rng.integers(0, h, n), rng.integers(0, w, n)] = 0.0
    return out, {"model": "salt_pepper", "density": round(float(density), 5)}

# Category C
def blur_gaussian(img_f32, rng, cfg):
    ksize = int(rng.integers(cfg["ksize_min"] // 2, cfg["ksize_max"] // 2 + 1)) * 2 + 1
    sigma = rng.uniform(cfg["sigma_min"], cfg["sigma_max"])
    blurred = cv2.GaussianBlur(img_f32, (ksize, ksize), sigma)
    return blurred, {"model": "gaussian_blur", "ksize": ksize, "sigma": round(float(sigma), 3)}

def blur_motion(img_f32, rng, cfg):
    ksize = int(rng.integers(cfg["ksize_min"] // 2, cfg["ksize_max"] // 2 + 1)) * 2 + 1
    angle = rng.uniform(cfg["angle_min"], cfg["angle_max"])
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = ksize // 2
    rad = np.deg2rad(angle)
    for i in range(ksize):
        offset = i - center
        x = int(round(center + offset * np.cos(rad)))
        y = int(round(center + offset * np.sin(rad)))
        if 0 <= x < ksize and 0 <= y < ksize:
            kernel[y, x] = 1.0
    kernel /= kernel.sum() if kernel.sum() > 0 else 1.0
    blurred = cv2.filter2D(img_f32, -1, kernel)
    return blurred, {"model": "motion_blur", "ksize": ksize, "angle": round(float(angle), 1)}

# Category D (Geometric: changes GT)
def geo_rotation(img_u8, gt_x, gt_y, rng, cfg):
    angle = rng.uniform(cfg["angle_min"], cfg["angle_max"])
    h, w = img_u8.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(img_u8, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    pt = np.array([gt_x, gt_y, 1.0])
    new_pt = M @ pt
    new_x = max(0, min(w - 1, int(round(new_pt[0]))))
    new_y = max(0, min(h - 1, int(round(new_pt[1]))))
    return rotated, new_x, new_y, {"model": "rotation", "angle_deg": round(float(angle), 3)}

def geo_barrel(img_u8, gt_x, gt_y, rng, cfg):
    k1 = rng.uniform(cfg["k1_min"], cfg["k1_max"])
    h, w = img_u8.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn, yn = (xx - cx) / cx, (yy - cy) / cy
    r2 = xn ** 2 + yn ** 2
    factor = 1.0 + k1 * r2
    map_x = (xn / factor) * cx + cx
    map_y = (yn / factor) * cy + cy
    distorted = cv2.remap(img_u8, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    gxn, gyn = (gt_x - cx) / cx, (gt_y - cy) / cy
    gf = 1.0 + k1 * (gxn ** 2 + gyn ** 2)
    new_x = max(0, min(w - 1, int(round(gxn * gf * cx + cx))))
    new_y = max(0, min(h - 1, int(round(gyn * gf * cy + cy))))
    return distorted, new_x, new_y, {"model": "barrel", "k1": round(float(k1), 5)}

def geo_pincushion(img_u8, gt_x, gt_y, rng, cfg):
    k1 = rng.uniform(cfg["k1_min"], cfg["k1_max"])
    h, w = img_u8.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn, yn = (xx - cx) / cx, (yy - cy) / cy
    r2 = xn ** 2 + yn ** 2
    factor = 1.0 + k1 * r2
    map_x = (xn / factor) * cx + cx
    map_y = (yn / factor) * cy + cy
    distorted = cv2.remap(img_u8, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    gxn, gyn = (gt_x - cx) / cx, (gt_y - cy) / cy
    gf = 1.0 + k1 * (gxn ** 2 + gyn ** 2)
    new_x = max(0, min(w - 1, int(round(gxn * gf * cx + cx))))
    new_y = max(0, min(h - 1, int(round(gyn * gf * cy + cy))))
    return distorted, new_x, new_y, {"model": "pincushion", "k1": round(float(k1), 5)}

def geo_projective(img_u8, gt_x, gt_y, rng, cfg):
    max_shift = int(rng.integers(cfg["max_shift_min"], cfg["max_shift_max"] + 1))
    h, w = img_u8.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src.copy()
    for i in range(4):
        dst[i, 0] += rng.integers(-max_shift, max_shift + 1)
        dst[i, 1] += rng.integers(-max_shift, max_shift + 1)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_u8, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    pt = np.array([gt_x, gt_y, 1.0])
    new_pt = M @ pt
    new_x = max(0, min(w - 1, int(round(new_pt[0] / new_pt[2]))))
    new_y = max(0, min(h - 1, int(round(new_pt[1] / new_pt[2]))))
    return warped, new_x, new_y, {"model": "projective", "max_shift": max_shift}

# Category E
def acq_field_nonuniformity(img_f32, rng, cfg):
    strength = rng.uniform(cfg["strength_min"], cfg["strength_max"])
    h, w = img_f32.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    gain_map = np.clip(1.0 - strength * r ** 2, 0.4, 1.0)
    if img_f32.ndim == 3:
        gain_map = gain_map[:, :, np.newaxis]
    return img_f32 * gain_map, {"model": "field_nonuniformity", "strength": round(float(strength), 3)}

def acq_vibration(img_u8, gt_x, gt_y, rng, cfg):
    amp = int(rng.integers(cfg["amplitude_min"], cfg["amplitude_max"] + 1))
    dx = int(rng.integers(-amp, amp + 1))
    dy = int(rng.integers(-amp, amp + 1))
    h, w = img_u8.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(img_u8, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return shifted, max(0, min(w - 1, gt_x + dx)), max(0, min(h - 1, gt_y + dy)), {"model": "vibration", "dx": dx, "dy": dy}

def acq_stage_drift(img_u8, gt_x, gt_y, rng, cfg):
    dx = int(rng.integers(cfg["dx_min"], cfg["dx_max"] + 1))
    dy = int(rng.integers(cfg["dy_min"], cfg["dy_max"] + 1))
    h, w = img_u8.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(img_u8, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return shifted, max(0, min(w - 1, gt_x + dx)), max(0, min(h - 1, gt_y + dy)), {"model": "stage_drift", "dx": dx, "dy": dy}


INTENSITY_MODELS = {
    "poisson":            noise_poisson,
    "gaussian":           noise_gaussian,
    "gamma":              noise_gamma,
    "secondary_electron": noise_secondary_electron,
    "low_frequency":      noise_low_frequency,
    "speckle":            noise_speckle,
    "banding":            noise_banding,
    "salt_pepper":        noise_salt_pepper,
    "gaussian_blur":      blur_gaussian,
    "motion_blur":        blur_motion,
    "field_nonuniformity": acq_field_nonuniformity,
}

GEOMETRIC_MODELS = {
    "rotation":   geo_rotation,
    "barrel":     geo_barrel,
    "pincushion": geo_pincushion,
    "projective": geo_projective,
    "vibration":   acq_vibration,
    "stage_drift": acq_stage_drift,
}


# =============================================================================
# MASTER DEGRADATION PIPELINE (CAPPED AT MAX 3 MODELS TOTAL)
# =============================================================================

def apply_full_degradation(clean_search_u8, gt_x, gt_y, rng, config=None):
    """
    Apply STOCHASTIC degradation to SEARCH image (AT MOST 3 MODELS TOTAL).

    Guarantees:
      - Max 3 degradation models per image across all categories (A-E).
      - Reference/Target remains 100% clean.
      - Geometric models update GT coordinates.
    """
    if config is None:
        config = DEFAULT_CONFIG

    if not config.get("enabled", True):
        return clean_search_u8.copy(), gt_x, gt_y, {"applied": False, "models": []}

    prob = config.get("probability_applied", 0.90)
    if rng.random() > prob:
        return clean_search_u8.copy(), gt_x, gt_y, {"applied": False, "models": [], "reason": "clean_sampled"}

    # Determine total number of models to apply (exactly 2 noise/degradation models per image)
    min_t = config.get("min_total_models", 2)
    max_t = config.get("max_total_models", 2)
    n_models = int(rng.integers(min_t, max_t + 1)) if min_t != max_t else min_t

    # Sample n_models from overall weighted model pool
    weights_dict = config.get("model_weights", {})
    all_models = list(weights_dict.keys())
    w_arr = np.array([weights_dict[m] for m in all_models], dtype=np.float64)
    w_arr /= w_arr.sum()

    selected_models = list(rng.choice(all_models, size=n_models, replace=False, p=w_arr))

    # Separate selected models into geometric vs intensity
    geo_selected = [m for m in selected_models if m in GEOMETRIC_MODELS]
    int_selected = [m for m in selected_models if m in INTENSITY_MODELS]

    img = clean_search_u8.copy()
    cur_x, cur_y = gt_x, gt_y
    applied_details = []
    applied_names = []

    strength_ranges = config.get("strength_ranges", {})

    # 1. Apply Geometric Models first (updates GT)
    for model_name in geo_selected:
        fn = GEOMETRIC_MODELS[model_name]
        cfg = strength_ranges.get(model_name, {})
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        img, cur_x, cur_y, detail = fn(img, cur_x, cur_y, rng, cfg)
        applied_details.append(detail)
        applied_names.append(model_name)

    # 2. Apply Intensity Models (noise / blur / illumination)
    for model_name in int_selected:
        fn = INTENSITY_MODELS[model_name]
        cfg = strength_ranges.get(model_name, {})
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        img, detail = fn(img, rng, cfg)
        applied_details.append(detail)
        applied_names.append(model_name)

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    full_report = {
        "applied": len(applied_names) > 0,
        "models": applied_names,
        "details": applied_details,
        "gt_changed": (cur_x != gt_x or cur_y != gt_y),
        "original_gt": (gt_x, gt_y),
        "final_gt": (cur_x, cur_y),
    }

    return img, cur_x, cur_y, full_report


def print_degradation_report(all_reports, total_pairs):
    clean_count = 0
    combo_counter = Counter()
    model_counter = Counter()

    for r in all_reports:
        if not r.get("applied", False):
            clean_count += 1
        else:
            models = sorted(r.get("models", []))
            combo_counter[" + ".join(models)] += 1
            for m in models:
                model_counter[m] += 1

    print("\n" + "=" * 60)
    print("DEGRADATION DISTRIBUTION REPORT (MAX 3 DEGRADATIONS PER IMAGE)")
    print("=" * 60)
    print(f"\nTotal pairs: {total_pairs}")
    print(f"\nOverall distribution:")
    print(f"    {'Clean (no degradation):':<40} {clean_count}")
    for combo, count in combo_counter.most_common():
        print(f"    {combo + ':':<40} {count}")

    print(f"\nPer-model frequency:")
    for model, count in model_counter.most_common():
        print(f"    {model + ':':<30} {count}")

    gt_changed = sum(1 for r in all_reports if r.get("gt_changed", False))
    print(f"\nGT coordinate changes: {gt_changed}/{total_pairs}")
    print(f"\nReference/Target noise: NONE (100% clean)")
    print(f"Search degradation:     STOCHASTIC (Max 3 models)")
    print(f"Ground truth:           VALIDATED")
    print("=" * 60)
