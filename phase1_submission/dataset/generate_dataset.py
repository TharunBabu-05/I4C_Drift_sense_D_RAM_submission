#!/usr/bin/env python3
"""
Drift-Sense Siamese Training Dataset Generator
Generates realistic synthetic DRAM SEM image pairs for Siamese training.
Outputs train/val/test splits with metadata.csv for DataLoader.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

# =============================================================================
# DRAM Layout & Noise Parameters (Inherited from v2.5 Baseline)
# =============================================================================
DRAM_PARAMS = {
    "cell_pitch_x_range": (42, 65),
    "cell_pitch_y_range": (42, 65),
    "cell_fill_fraction": (0.55, 0.75),
    "corner_radius_range": (3, 8),
    "body_intensity_range": (15, 55),
    "wall_intensity_range": (170, 230),
    "intersection_boost": (15, 40),
    "pitch_jitter_std": 0.8,
    "ler_amplitude": (1.0, 3.0),
    "ler_correlation_length": (8, 25),
    "cd_gradient_strength": (0.0, 0.04),
    "block_period_x": (1800, 2400),
    "block_period_y": (1800, 2400),
    "block_dimming": (0.04, 0.10),
    "defect_density": (0.0, 2.0),
    "defect_types": ["missing_contact", "particle", "line_bridge", "line_break"],
    "particle_radius_range": (3, 9),
    "bridge_length_range": (10, 35),
    "break_length_range": (8, 25),
}

NOISE_PARAMS = {
    "ref_poisson_scale": (15.0, 25.0),
    "ref_gaussian_std": (0.5, 2.0),
    "search_poisson_scale": (8.0, 15.0),
    "search_gaussian_std": (1.0, 2.5),
    "edge_brightness_ref": (0.10, 0.22),
    "edge_brightness_search": (0.06, 0.15),
    "blur_sigma_ref": (0.4, 1.0),
    "blur_sigma_search": (0.6, 1.5),
    "rotation_range_deg": (-0.5, 0.5),
    "vignette_strength": (0.08, 0.20),
    "gain_variation": (0.92, 1.08),
    "offset_variation": (-6, 6),
    "beam_drift_amplitude": (0.0, 0.04),
    "beam_drift_period": (80, 300),
}

# =============================================================================
# Core Generator Functions (Condensed from v2.5)
# =============================================================================
def generate_ler_profile(length, amplitude, correlation_length, rng):
    white_noise = rng.normal(0, 1, size=length)
    kernel_size = min(length, int(correlation_length * 6) | 1)
    k = np.arange(kernel_size)
    kernel = np.exp(-k / correlation_length)
    kernel = kernel / kernel.sum()
    profile = ndimage.convolve1d(white_noise, kernel, mode='wrap')
    if profile.std() > 0:
        profile = profile / profile.std() * amplitude
    return profile

def generate_dram_layout(width, height, params, rng):
    pitch_x = rng.integers(*params["cell_pitch_x_range"])
    pitch_y = rng.integers(*params["cell_pitch_y_range"])
    fill_frac = rng.uniform(*params["cell_fill_fraction"])
    corner_r = rng.integers(*params["corner_radius_range"])
    body_int = rng.uniform(*params["body_intensity_range"])
    wall_int = rng.uniform(*params["wall_intensity_range"])
    inter_boost = rng.uniform(*params["intersection_boost"])
    ler_amp = rng.uniform(*params["ler_amplitude"])
    cd_grad = rng.uniform(*params["cd_gradient_strength"])
    pitch_jitter = params["pitch_jitter_std"]

    cell_w = int(pitch_x * fill_frac)
    cell_h = int(pitch_y * fill_frac)
    wall_w = pitch_x - cell_w
    wall_h = pitch_y - cell_h

    layout = np.full((height, width), wall_int, dtype=np.float64)
    phase_x = rng.integers(0, pitch_x)
    phase_y = rng.integers(0, pitch_y)

    cell_centers_x = []
    x = phase_x + pitch_x // 2
    while x < width + pitch_x:
        cell_centers_x.append(x)
        x += pitch_x + rng.normal(0, pitch_jitter)

    cell_centers_y = []
    y = phase_y + pitch_y // 2
    while y < height + pitch_y:
        cell_centers_y.append(y)
        y += pitch_y + rng.normal(0, pitch_jitter)

    for cy in cell_centers_y:
        for cx in cell_centers_x:
            cy_int, cx_int = int(round(cy)), int(round(cx))
            local_cd_factor = 1.0 + cd_grad * (cx / width - 0.5) * 2
            lw = max(4, int(cell_w * local_cd_factor))
            lh = max(4, int(cell_h * local_cd_factor))

            top_ler = int(round(rng.normal(0, ler_amp * 0.5)))
            bot_ler = int(round(rng.normal(0, ler_amp * 0.5)))
            lft_ler = int(round(rng.normal(0, ler_amp * 0.5)))
            rgt_ler = int(round(rng.normal(0, ler_amp * 0.5)))

            y0 = max(0, min(height - 1, cy_int - lh // 2 + top_ler))
            y1 = max(0, min(height - 1, cy_int + lh // 2 + bot_ler))
            x0 = max(0, min(width - 1, cx_int - lw // 2 + lft_ler))
            x1 = max(0, min(width - 1, cx_int + lw // 2 + rgt_ler))

            if y1 > y0 and x1 > x0:
                local_body = body_int + rng.normal(0, 4)
                layout[y0:y1, x0:x1] = np.clip(local_body, 0, 120)

    body_mask = (layout < (body_int + wall_int) / 2).astype(np.float64)
    body_mask_blurred = ndimage.gaussian_filter(body_mask, sigma=corner_r * 0.4)
    layout = layout * (1 - body_mask_blurred) + body_int * body_mask_blurred

    # Block banding
    block_px = rng.integers(*params["block_period_x"])
    block_py = rng.integers(*params["block_period_y"])
    block_dim = rng.uniform(*params["block_dimming"])
    block_width = max(3, pitch_x)

    y_bound = rng.integers(0, block_py)
    while y_bound < height:
        layout[max(0, y_bound - block_width // 2):min(height, y_bound + block_width // 2), :] *= (1.0 - block_dim)
        y_bound += block_py

    x_bound = rng.integers(0, block_px)
    while x_bound < width:
        layout[:, max(0, x_bound - block_width // 2):min(width, x_bound + block_width // 2)] *= (1.0 - block_dim)
        x_bound += block_px

    defect_log = []
    # simplified defect injection for speed in training data generation
    num_defects = rng.poisson(rng.uniform(*params["defect_density"]))
    for _ in range(num_defects):
        dtype = rng.choice(params["defect_types"])
        dy = rng.integers(50, height - 50)
        dx = rng.integers(50, width - 50)
        if dtype == "missing_contact":
            r = rng.integers(4, 10)
            layout[max(0, dy-r):min(height, dy+r), max(0, dx-r):min(width, dx+r)] = body_int * 0.7
            defect_log.append({"type": dtype, "x": int(dx), "y": int(dy)})
        elif dtype == "particle":
            pr = rng.integers(*params["particle_radius_range"])
            yy, xx = np.ogrid[:height, :width]
            mask = ((xx - dx) / pr) ** 2 + ((yy - dy) / pr) ** 2 <= 1
            layout[mask] = np.clip(wall_int + 40, 0, 255)
            defect_log.append({"type": dtype, "x": int(dx), "y": int(dy)})

    return np.clip(layout, 0, 255), defect_log

def apply_full_sem_noise(image, noise_cfg, rng):
    blur_sigma = rng.uniform(*noise_cfg["blur_range"])
    image = ndimage.gaussian_filter(image, sigma=blur_sigma)
    
    edge_str = rng.uniform(*noise_cfg["edge_range"])
    edges = np.sqrt(ndimage.sobel(image, axis=1)**2 + ndimage.sobel(image, axis=0)**2)
    if edges.max() > 0:
        edges = edges / edges.max() * 255.0
    image = np.clip(image + edge_str * edges, 0, 255)

    drift_amp = rng.uniform(*noise_cfg["beam_drift_range"])
    drift_period = rng.uniform(*noise_cfg["beam_drift_period_range"])
    if drift_amp > 0:
        t = np.arange(image.shape[0])
        drift = 1.0 + drift_amp * np.sin(2 * np.pi * t / drift_period + rng.uniform(0, 2*np.pi))
        image = np.clip(image * drift[:, np.newaxis], 0, 255)

    poisson_scale = rng.uniform(*noise_cfg["poisson_range"])
    lam = np.clip(np.clip(image, 0.001, 255) * poisson_scale, 0.001, 1e7)
    image = np.clip(rng.poisson(lam).astype(np.float64) / poisson_scale, 0, 255)

    gauss_std = rng.uniform(*noise_cfg["gaussian_range"])
    image = np.clip(image + rng.normal(0, gauss_std, size=image.shape), 0, 255)

    rot = rng.uniform(*noise_cfg["rotation_range"])
    if abs(rot) > 0.01:
        image = ndimage.rotate(image, rot, reshape=False, order=1, mode='reflect')

    gain = rng.uniform(*noise_cfg["gain_range"])
    offset = rng.uniform(*noise_cfg["offset_range"])
    image = np.clip(image * gain + offset, 0, 255)
    
    return image

def generate_pair(master_size=10000, rng=None):
    master_layout, defects = generate_dram_layout(master_size, master_size, DRAM_PARAMS, rng)
    
    ref_size = 1000
    center_y, center_x = master_size // 2, master_size // 2
    drift_y, drift_x = rng.integers(-22, 22) * 10, rng.integers(-22, 22) * 10
    ref_y = max(0, min(master_size - ref_size, center_y + drift_y - (ref_size // 2)))
    ref_x = max(0, min(master_size - ref_size, center_x + drift_x - (ref_size // 2)))

    reference_clean = master_layout[ref_y:ref_y+ref_size, ref_x:ref_x+ref_size].copy()
    
    search_size = 1000
    master_pil = Image.fromarray(master_layout.astype(np.uint8), mode='L')
    search_pil = master_pil.resize((search_size, search_size), Image.LANCZOS)
    search_clean = np.array(search_pil, dtype=np.float64)

    gt_center_x = max(0, min(search_size - 1, int(round((ref_x + ref_size / 2) / 10.0))))
    gt_center_y = max(0, min(search_size - 1, int(round((ref_y + ref_size / 2) / 10.0))))

    ref_noise = {
        "blur_range": NOISE_PARAMS["blur_sigma_ref"],
        "edge_range": NOISE_PARAMS["edge_brightness_ref"],
        "poisson_range": NOISE_PARAMS["ref_poisson_scale"],
        "gaussian_range": NOISE_PARAMS["ref_gaussian_std"],
        "beam_drift_range": (0.0, NOISE_PARAMS["beam_drift_amplitude"][1] * 0.3),
        "beam_drift_period_range": NOISE_PARAMS["beam_drift_period"],
        "rotation_range": (0.0, 0.0),
        "gain_range": (0.95, 1.05),
        "offset_range": (-4, 4),
    }

    search_noise = {
        "blur_range": NOISE_PARAMS["blur_sigma_search"],
        "edge_range": NOISE_PARAMS["edge_brightness_search"],
        "poisson_range": NOISE_PARAMS["search_poisson_scale"],
        "gaussian_range": NOISE_PARAMS["ref_gaussian_std"],
        "beam_drift_range": NOISE_PARAMS["beam_drift_amplitude"],
        "beam_drift_period_range": NOISE_PARAMS["beam_drift_period"],
        "rotation_range": NOISE_PARAMS["rotation_range_deg"],
        "gain_range": NOISE_PARAMS["gain_variation"],
        "offset_range": NOISE_PARAMS["offset_variation"],
    }

    ref_out = apply_full_sem_noise(reference_clean, ref_noise, rng)
    search_out = apply_full_sem_noise(search_clean, search_noise, rng)

    return (
        np.clip(ref_out, 0, 255).astype(np.uint8), 
        np.clip(search_out, 0, 255).astype(np.uint8), 
        gt_center_x, gt_center_y
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train", type=int, default=500, help="Train pairs")
    parser.add_argument("--num_val", type=int, default=100, help="Val pairs")
    parser.add_argument("--num_test", type=int, default=100, help="Test pairs")
    parser.add_argument("--output_dir", type=str, default="../data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    base_dir = Path(args.output_dir)

    splits = {
        "train": args.num_train,
        "val": args.num_val,
        "test": args.num_test
    }

    for split, count in splits.items():
        if count == 0:
            continue
        print(f"Generating {split} split ({count} pairs)...")
        split_dir = base_dir / split
        (split_dir / "references").mkdir(parents=True, exist_ok=True)
        (split_dir / "searches").mkdir(parents=True, exist_ok=True)

        meta_path = split_dir / "metadata.csv"
        with open(meta_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["pair_id", "reference_path", "search_path", "target_x", "target_y"])

            for i in range(count):
                ref_img, search_img, tx, ty = generate_pair(master_size=10000, rng=rng)
                
                ref_path = f"references/ref_{i:04d}.png"
                search_path = f"searches/search_{i:04d}.png"
                
                Image.fromarray(ref_img).save(split_dir / ref_path)
                Image.fromarray(search_img).save(split_dir / search_path)
                
                writer.writerow([f"{i:04d}", ref_path, search_path, tx, ty])
                if (i+1) % 10 == 0:
                    print(f"  {i+1}/{count} done.")

if __name__ == "__main__":
    main()
