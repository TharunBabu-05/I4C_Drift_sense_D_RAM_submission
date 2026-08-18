#!/usr/bin/env python3
"""
Stochastic SEM Noise Engine
============================
Implements a configurable, hierarchical stochastic noise pipeline for synthetic
SEM dataset generation.

Design Principles (MANDATORY):
------------------------------
- Reference/Target images NEVER receive sensor noise. They stay clean.
- Search images receive STOCHASTIC noise drawn from a configurable pool.
- Noise selection is hierarchical and independently randomized per image:
    1. Whether noise is applied at all
    2. How many noise models are selected
    3. Which noise models are selected
    4. The strength of each selected noise model
    5. The spatial/random realization of each selected noise model

Supported Noise Models:
-----------------------
- poisson:           Photon shot noise (Poisson statistics)
- gaussian:          Thermal / readout Gaussian additive noise
- gamma:             Multiplicative gamma-distributed detector gain noise
- secondary_electron: Secondary electron yield fluctuation noise
- low_frequency:     Low-frequency spatial intensity drift / banding
- speckle:           Multiplicative speckle (surface roughness)
- banding:           Horizontal line-correlated banding artifacts
- salt_pepper:       Salt-and-pepper impulse noise (dead/hot pixels)
"""

import numpy as np
from collections import Counter


# =============================================================================
# Default Noise Configuration
# =============================================================================

DEFAULT_NOISE_CONFIG = {
    "enabled": True,

    # Probability that ANY noise is applied to a given search image.
    # 1.0 - probability_noise_applied fraction of images remain clean.
    "probability_noise_applied": 0.90,

    # Random number of noise types selected per image
    "minimum_noise_models": 1,
    "maximum_noise_models": 3,

    # Selection weights (unnormalized; will be normalized internally)
    "model_selection_weights": {
        "poisson":            0.30,
        "gaussian":           0.30,
        "gamma":              0.10,
        "secondary_electron": 0.10,
        "low_frequency":      0.08,
        "speckle":            0.05,
        "banding":            0.04,
        "salt_pepper":        0.03,
    },

    # Per-model strength ranges (sampled uniformly per image)
    "strength_ranges": {
        "poisson": {
            "scale_min": 0.5,    # Lower = more noise (fewer effective photons)
            "scale_max": 5.0,
        },
        "gaussian": {
            "sigma_min": 3.0,
            "sigma_max": 25.0,
        },
        "gamma": {
            "shape_min": 5.0,    # Lower shape = more noise
            "shape_max": 50.0,
        },
        "secondary_electron": {
            "sigma_min": 5.0,
            "sigma_max": 20.0,
        },
        "low_frequency": {
            "amplitude_min": 5.0,
            "amplitude_max": 30.0,
            "frequency_min": 1,
            "frequency_max": 5,
        },
        "speckle": {
            "sigma_min": 0.02,
            "sigma_max": 0.15,
        },
        "banding": {
            "amplitude_min": 3.0,
            "amplitude_max": 20.0,
        },
        "salt_pepper": {
            "density_min": 0.001,
            "density_max": 0.01,
        },
    },
}


# =============================================================================
# Individual Noise Model Implementations
# =============================================================================

def apply_poisson_noise(img_f32, rng, config):
    """
    Poisson shot noise.

    Models the discrete photon counting statistics of the electron detector.
    The number of detected electrons at each pixel follows a Poisson distribution
    whose mean is proportional to the true signal intensity.

    A lower scale value simulates fewer effective photons → more noise.
    """
    scale = rng.uniform(config["scale_min"], config["scale_max"])
    # Normalize to [0, scale], apply Poisson, then rescale
    img_scaled = np.clip(img_f32 / 255.0 * scale, 0, None)
    noisy = rng.poisson(img_scaled).astype(np.float32) / scale * 255.0
    return noisy, {"model": "poisson", "scale": round(float(scale), 3)}


def apply_gaussian_noise(img_f32, rng, config):
    """
    Additive Gaussian noise.

    Models thermal noise from the detector electronics and readout amplifier
    chain. Pixel intensity deviations are i.i.d. Gaussian with zero mean
    and configurable standard deviation sigma.
    """
    sigma = rng.uniform(config["sigma_min"], config["sigma_max"])
    noise = rng.normal(0, sigma, img_f32.shape).astype(np.float32)
    noisy = img_f32 + noise
    return noisy, {"model": "gaussian", "sigma": round(float(sigma), 3)}


def apply_gamma_noise(img_f32, rng, config):
    """
    Multiplicative gamma noise.

    Models gain variations in the electron multiplier / scintillator detector.
    Each pixel's intensity is multiplied by a gamma-distributed random variable
    with mean 1.0 and shape parameter controlling the spread.
    Lower shape → more variable gain → more noise.
    """
    shape = rng.uniform(config["shape_min"], config["shape_max"])
    gain = rng.gamma(shape, 1.0 / shape, img_f32.shape).astype(np.float32)
    noisy = img_f32 * gain
    return noisy, {"model": "gamma", "shape": round(float(shape), 3)}


def apply_secondary_electron_noise(img_f32, rng, config):
    """
    Secondary electron yield fluctuation noise.

    Models the stochastic variation in secondary electron yield from the
    sample surface. Combines a spatially-correlated component (low-frequency
    yield variation across the sample) with a high-frequency component
    (per-pixel yield fluctuation).
    """
    sigma = rng.uniform(config["sigma_min"], config["sigma_max"])
    h, w = img_f32.shape[:2]

    # High-frequency per-pixel noise
    hf_noise = rng.normal(0, sigma * 0.7, img_f32.shape).astype(np.float32)

    # Low-frequency spatially-correlated component
    lf_h, lf_w = max(h // 16, 1), max(w // 16, 1)
    lf_shape = (lf_h, lf_w) if img_f32.ndim == 2 else (lf_h, lf_w, img_f32.shape[2])
    lf_small = rng.normal(0, sigma * 0.3, lf_shape).astype(np.float32)

    # Use simple repeat-based upscaling (no cv2 dependency here)
    if img_f32.ndim == 2:
        lf_noise = np.repeat(np.repeat(lf_small, h // lf_h + 1, axis=0), w // lf_w + 1, axis=1)[:h, :w]
    else:
        lf_noise = np.repeat(np.repeat(lf_small, h // lf_h + 1, axis=0), w // lf_w + 1, axis=1)[:h, :w, :]

    noisy = img_f32 + hf_noise + lf_noise
    return noisy, {"model": "secondary_electron", "sigma": round(float(sigma), 3)}


def apply_low_frequency_noise(img_f32, rng, config):
    """
    Low-frequency spatial intensity drift.

    Models slow spatial variations in beam current, detector gain, or sample
    charging that create smooth intensity gradients across the field of view.
    Implemented as a sum of low-frequency 2D sinusoidal components.
    """
    amplitude = rng.uniform(config["amplitude_min"], config["amplitude_max"])
    n_freq = rng.integers(config["frequency_min"], config["frequency_max"] + 1)
    h, w = img_f32.shape[:2]

    drift = np.zeros((h, w), dtype=np.float32)
    for _ in range(n_freq):
        fx = rng.uniform(0.5, 3.0) * np.pi / w
        fy = rng.uniform(0.5, 3.0) * np.pi / h
        phase_x = rng.uniform(0, 2 * np.pi)
        phase_y = rng.uniform(0, 2 * np.pi)
        amp_i = rng.uniform(0.3, 1.0) * amplitude

        xx = np.arange(w, dtype=np.float32)
        yy = np.arange(h, dtype=np.float32)
        drift += amp_i * np.outer(np.sin(fy * yy + phase_y), np.sin(fx * xx + phase_x))

    if img_f32.ndim == 3:
        drift = drift[:, :, np.newaxis]

    noisy = img_f32 + drift
    return noisy, {"model": "low_frequency", "amplitude": round(float(amplitude), 3),
                   "n_components": int(n_freq)}


def apply_speckle_noise(img_f32, rng, config):
    """
    Multiplicative speckle noise.

    Models granular intensity fluctuations caused by surface roughness,
    grain boundaries, and coherent electron scattering. Each pixel is
    multiplied by (1 + sigma * N(0,1)), producing intensity-dependent noise.
    """
    sigma = rng.uniform(config["sigma_min"], config["sigma_max"])
    speckle = rng.normal(0, sigma, img_f32.shape).astype(np.float32)
    noisy = img_f32 * (1.0 + speckle)
    return noisy, {"model": "speckle", "sigma": round(float(sigma), 3)}


def apply_banding_noise(img_f32, rng, config):
    """
    Horizontal banding / scanline artifacts.

    Models periodic and random horizontal intensity variations caused by
    beam current instability, ground loops, or scan electronics interference
    during raster scanning.
    """
    amplitude = rng.uniform(config["amplitude_min"], config["amplitude_max"])
    h = img_f32.shape[0]

    # Random per-scanline offset
    line_offsets = rng.normal(0, amplitude * 0.6, h).astype(np.float32)
    # Periodic component
    freq = rng.uniform(0.01, 0.1)
    periodic = (np.sin(np.arange(h, dtype=np.float32) * freq * 2 * np.pi) * amplitude * 0.4)
    band = line_offsets + periodic

    if img_f32.ndim == 2:
        noisy = img_f32 + band[:, np.newaxis]
    else:
        noisy = img_f32 + band[:, np.newaxis, np.newaxis]

    return noisy, {"model": "banding", "amplitude": round(float(amplitude), 3)}


def apply_salt_pepper_noise(img_f32, rng, config):
    """
    Salt-and-pepper impulse noise.

    Models dead pixels (always dark) and hot pixels (always bright) in the
    detector array. Randomly selected pixels are set to either 0 (pepper)
    or 255 (salt) with configurable density.
    """
    density = rng.uniform(config["density_min"], config["density_max"])
    noisy = img_f32.copy()
    h, w = img_f32.shape[:2]

    total_pixels = h * w
    n_salt = int(total_pixels * density / 2)
    n_pepper = int(total_pixels * density / 2)

    # Salt (hot pixels)
    salt_y = rng.integers(0, h, n_salt)
    salt_x = rng.integers(0, w, n_salt)
    noisy[salt_y, salt_x] = 255.0

    # Pepper (dead pixels)
    pepper_y = rng.integers(0, h, n_pepper)
    pepper_x = rng.integers(0, w, n_pepper)
    noisy[pepper_y, pepper_x] = 0.0

    return noisy, {"model": "salt_pepper", "density": round(float(density), 5)}


# Map model name → function
NOISE_MODEL_REGISTRY = {
    "poisson":            apply_poisson_noise,
    "gaussian":           apply_gaussian_noise,
    "gamma":              apply_gamma_noise,
    "secondary_electron": apply_secondary_electron_noise,
    "low_frequency":      apply_low_frequency_noise,
    "speckle":            apply_speckle_noise,
    "banding":            apply_banding_noise,
    "salt_pepper":        apply_salt_pepper_noise,
}


# =============================================================================
# Hierarchical Stochastic Noise Application
# =============================================================================

def apply_stochastic_noise(clean_image_uint8, rng, noise_config=None):
    """
    Apply stochastic noise to a SEARCH image following the hierarchical pipeline.

    Steps:
        1. Decide whether noise is applied at all (probability_noise_applied).
        2. Randomly select how many noise models (min..max).
        3. Randomly select which models from the pool (weighted sampling).
        4. For each selected model, sample a random strength.
        5. Generate an independent noise realization and apply.

    Parameters:
    -----------
    clean_image_uint8 : np.ndarray
        Clean search image (uint8, grayscale or RGB).
    rng : np.random.Generator
        Independent random number generator for this image.
    noise_config : dict, optional
        Noise configuration dict. Uses DEFAULT_NOISE_CONFIG if None.

    Returns:
    --------
    noisy_image_uint8 : np.ndarray
        Degraded search image (uint8).
    noise_report : dict
        Report of what noise was applied (models, strengths, etc.).
    """
    if noise_config is None:
        noise_config = DEFAULT_NOISE_CONFIG

    if not noise_config.get("enabled", True):
        return clean_image_uint8.copy(), {"applied": False, "reason": "noise_disabled"}

    # Step 1: Should noise be applied to this image?
    prob = noise_config.get("probability_noise_applied", 0.90)
    if rng.random() > prob:
        return clean_image_uint8.copy(), {"applied": False, "reason": "probability_check_failed", "models": []}

    # Step 2: How many noise models?
    min_models = noise_config.get("minimum_noise_models", 1)
    max_models = noise_config.get("maximum_noise_models", 3)
    available_models = list(noise_config.get("model_selection_weights", {}).keys())
    max_models = min(max_models, len(available_models))
    min_models = min(min_models, max_models)
    n_models = int(rng.integers(min_models, max_models + 1))

    # Step 3: Which noise models? (weighted sampling without replacement)
    weights = noise_config.get("model_selection_weights", {})
    model_names = list(weights.keys())
    model_weights = np.array([weights[m] for m in model_names], dtype=np.float64)
    model_weights /= model_weights.sum()  # Normalize

    # Sample without replacement
    selected_indices = rng.choice(len(model_names), size=n_models, replace=False, p=model_weights)
    selected_models = [model_names[i] for i in selected_indices]

    # Step 4 & 5: Apply each selected model with random strength
    img_f32 = clean_image_uint8.astype(np.float32)
    applied_details = []

    strength_ranges = noise_config.get("strength_ranges", {})

    for model_name in selected_models:
        if model_name not in NOISE_MODEL_REGISTRY:
            continue

        model_fn = NOISE_MODEL_REGISTRY[model_name]
        model_config = strength_ranges.get(model_name, {})

        img_f32, detail = model_fn(img_f32, rng, model_config)
        applied_details.append(detail)

    # Clamp and convert back to uint8
    noisy_uint8 = np.clip(img_f32, 0, 255).astype(np.uint8)

    noise_report = {
        "applied": True,
        "num_models": n_models,
        "models": selected_models,
        "details": applied_details,
    }

    return noisy_uint8, noise_report


# =============================================================================
# Noise Distribution Summary Reporter
# =============================================================================

def print_noise_distribution_report(all_noise_reports, total_pairs):
    """
    Print the noise distribution summary to the console after dataset generation.

    Counts how many images received each noise combination, how many were clean, etc.
    """
    clean_count = 0
    combo_counter = Counter()

    for report in all_noise_reports:
        if not report.get("applied", False):
            clean_count += 1
        else:
            models = sorted(report.get("models", []))
            combo_key = " + ".join(models) if models else "Clean"
            combo_counter[combo_key] += 1

    print("\n" + "=" * 60)
    print("NOISE DISTRIBUTION REPORT")
    print("=" * 60)
    print(f"\nTotal pairs: {total_pairs}")
    print(f"\nNoise distribution:")
    print(f"    {'Clean (no noise):':<35} {clean_count}")
    for combo, count in combo_counter.most_common():
        label = f"    {combo + ':':<35}"
        print(f"{label} {count}")

    print(f"\nReference noise:")
    print(f"    NONE")
    print(f"\nSearch noise:")
    print(f"    STOCHASTIC")
    print(f"\nGround truth:")
    print(f"    VALIDATED")
    print("=" * 60)
