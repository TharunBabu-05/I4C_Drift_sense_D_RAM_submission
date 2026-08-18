#!/usr/bin/env python3
"""
Generate Pair 1 for generated_new (High-Density SEM Transistor Array - Ultra-Heavy Multi-Noise + Raster Banding)
==============================================================================================================
Synthesizes a 1000x1000 px uint8 grayscale search image and 1000x1000 px grayscale target reference image
PURELY PROGRAMMATICALLY IN PYTHON CODE (matching the user's reference SEM memory cell array image).

Design Features:
  - Dense SEM DRAM/FeRAM memory cell array (pitch = 40px, 600+ cells).
  - Transistor bodies, top pads, stems, and side gates remain 100% separate across all cells.
  - UNIQUE TARGET LANDMARK: At Ground Truth (460, 460), ONLY THE BOTTOM TWO PARALLEL CAPACITOR LINES
    ARE CONTINUOUSLY JOINED ACROSS TWO ADJACENT CELLS!
  - ULTRA-HEAVY Independent Noise applied to BOTH search.png and target.png
    (Speckle + Vignette + SEM Raster Line Banding + Heavy Shot/Read Noise + Defocus Blur).

Files created in generated_new/pair1/:
  - groundtruth.json
  - search.png    (1000x1000 px grayscale with ultra-heavy noise)
  - target.png    (1000x1000 px grayscale 100x high-mag reference target with independent ultra-heavy noise)
  - reference.png (1000x1000 px grayscale, copy of target.png)
"""

import json
import os
import shutil
import cv2
import numpy as np


def render_dense_sem_memory_array(h=1000, w=1000, gt_x=460, gt_y=460, seed=1101):
    """
    Render 1000x1000 uint8 grayscale SEM memory cell scene pure programmatically:
    - High-density cell array matching user's SEM image (pitch = 40px)
    - All transistors remain separate
    - Ground Truth target at (gt_x, gt_y) features JOINED BOTTOM TWO PARALLEL CAPACITOR LINES
    """
    np.random.seed(seed)
    # Dark SEM substrate background
    img = np.full((h, w), 25, dtype=np.float32)

    pitch = 40

    # 1. Render vertical & horizontal bus grid lines across field
    for bx in range(20, w, pitch):
        cv2.line(img, (bx, 0), (bx, h), 160, 2)
    for by in range(20, h, pitch):
        cv2.line(img, (0, by), (w, by), 160, 2)

    # 2. Render 600+ memory cell units across matrix
    for cy in range(20, h, pitch):
        for cx in range(20, w, pitch):
            if 0 <= cx < w and 0 <= cy < h:
                # 1. Top via contact pad (separate for all cells)
                cv2.rectangle(img, (cx - 3, cy - 15), (cx + 3, cy - 9), 240, -1)

                # 2. Vertical stem line (separate for all cells)
                cv2.line(img, (cx, cy - 9), (cx, cy + 8), 190, 1)

                # 3. Transistor access gate contact pad (separate for all cells)
                cv2.rectangle(img, (cx + 3, cy - 4), (cx + 7, cy), 220, -1)
                cv2.line(img, (cx, cy - 2), (cx + 3, cy - 2), 220, 1)

                # 4. Outer transistor boundary frame (separate for all cells)
                cv2.rectangle(img, (cx - 7, cy - 8), (cx + 7, cy + 3), 180, 1)

                # 5. Bottom double parallel capacitor plate lines
                is_gt_left = (cx == gt_x and cy == gt_y)
                is_gt_right = (cx == gt_x + pitch and cy == gt_y)

                if is_gt_left:
                    # UNIQUE TARGET LANDMARK: Join ONLY the bottom two parallel capacitor lines across adjacent cells!
                    cx2 = gt_x + pitch
                    cv2.line(img, (cx - 9, cy + 8), (cx2 + 9, cy + 8), 255, 2)
                    cv2.line(img, (cx - 9, cy + 12), (cx2 + 9, cy + 12), 255, 2)
                elif is_gt_right:
                    # Bottom two lines already joined continuously from the left cell above
                    pass
                else:
                    # Regular separate bottom capacitor lines
                    cv2.line(img, (cx - 9, cy + 8), (cx + 9, cy + 8), 235, 2)
                    cv2.line(img, (cx - 9, cy + 12), (cx + 9, cy + 12), 235, 2)

    return np.clip(img, 0, 255).astype(np.uint8)


def apply_vignette_noise(image_float, strength=0.50):
    """Apply optical/SEM lens vignette corner shading noise."""
    h, w = image_float.shape
    v1 = (1.0 - strength * (np.linspace(-1.0, 1.0, w, dtype=np.float32)**2)).astype(np.float32)
    v2 = (1.0 - strength * (np.linspace(-1.0, 1.0, h, dtype=np.float32)**2)).astype(np.float32)
    vignette = np.outer(v2, v1)
    return image_float * vignette


def apply_speckle_noise(image_float, sigma=0.28):
    """Apply multiplicative SEM electron speckle noise."""
    noise = np.random.normal(0, sigma, image_float.shape).astype(np.float32)
    return image_float * (1.0 + noise)


def apply_sem_scan_line_noise(image_float, sigma=15.0):
    """Apply horizontal SEM electron beam scanning raster line noise."""
    h, w = image_float.shape
    line_noise = np.random.normal(0, sigma, (h, 1)).astype(np.float32)
    return image_float + line_noise


# [STRIPPED] def add_heavy_multi_sem_noise(image_uint8, seed=1101, is_target=False):
# [STRIPPED]     """
# [STRIPPED]     Apply ULTRA-HEAVY multi-type imaging noise pipeline:
# [STRIPPED]     (Vignette + Speckle + SEM Scan Line Banding + Shot/Read Noise + Blur):
# [STRIPPED]     - Uses DIFFERENT random seeds for search image vs target image!
# [STRIPPED]     """
# [STRIPPED]     np.random.seed(seed if not is_target else seed + 9999)
# [STRIPPED]     img_float = image_uint8.astype(np.float32)
# [STRIPPED] 
# [STRIPPED]     # 1. Vignette corner shading
# [STRIPPED]     vignetted = apply_vignette_noise(img_float, strength=0.50 if not is_target else 0.30)
# [STRIPPED] 
# [STRIPPED]     # 2. Multiplicative Speckle noise
# [STRIPPED]     speckled = apply_speckle_noise(vignetted, sigma=0.28 if not is_target else 0.18)
# [STRIPPED] 
# [STRIPPED]     # 3. SEM Scanning Raster Line noise
# [STRIPPED]     scanned = apply_sem_scan_line_noise(speckled, sigma=15.0 if not is_target else 8.0)
# [STRIPPED] 
# [STRIPPED]     # 4. Heavy Poisson shot & Gaussian detector read noise
# [STRIPPED]     shot_noise = np.random.normal(0, 32.0 if not is_target else 20.0, img_float.shape).astype(np.float32)
# [STRIPPED]     read_noise = np.random.normal(0, 24.0 if not is_target else 14.0, img_float.shape).astype(np.float32)
# [STRIPPED]     noisy_img = np.clip(scanned + shot_noise + read_noise, 0, 255).astype(np.uint8)
# [STRIPPED] 
# [STRIPPED]     # 5. Defocus blur
# [STRIPPED]     final_img = cv2.GaussianBlur(noisy_img, (3, 3), 1.0 if not is_target else 0.6)
# [STRIPPED] 
# [STRIPPED]     return final_img
# [STRIPPED] 
# [STRIPPED] 
def generate_new_pair1(output_dir="generated_new/pair1", gt_x=460, gt_y=460, seed=1101):
    """Generate dense SEM pair1 with ultra-heavy independent noise applied to BOTH search and target images."""
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    # Render clean dense SEM scene programmatically
    search_clean = render_dense_sem_memory_array(h=h, w=w, gt_x=gt_x, gt_y=gt_y, seed=seed)

    # Extract 100x100 crop around GT center (460, 460) and scale 10x to 1000x1000 px for target.png
    crop_size = 100
    x0 = max(0, min(w - crop_size, gt_x - crop_size // 2))
    y0 = max(0, min(h - crop_size, gt_y - crop_size // 2))
    x1, y1 = x0 + crop_size, y0 + crop_size

    crop_patch = search_clean[y0:y1, x0:x1]
    
    # 1000x1000 px clean target reference image (100x magnification view)
    target_clean = cv2.resize(crop_patch, (1000, 1000), interpolation=cv2.INTER_CUBIC)

    # Apply ULTRA-HEAVY INDEPENDENT noise to search image
    search_noisy = search_clean  # noise stripped by v2 pipeline

    # Apply ULTRA-HEAVY INDEPENDENT noise to target image (different noise instance!)
    target_noisy = target_clean  # noise stripped by v2 pipeline

    gt_info = {
        "center_x": gt_x,
        "center_y": gt_y,
        "target_name": "Dense SEM Memory Array with Ultra-Heavy Multi-Noise & Banding (Target & Search) 1",
        "pair_id": "pair1",
        "scale_factor": 10.0
    }

    # Save output grayscale files
    gt_path = os.path.join(output_dir, "groundtruth.json")
    search_path = os.path.join(output_dir, "search.png")
    target_path = os.path.join(output_dir, "target.png")
    ref_path = os.path.join(output_dir, "reference.png")

    with open(gt_path, "w") as f:
        json.dump(gt_info, f, indent=2)

    cv2.imwrite(search_path, search_noisy)
    cv2.imwrite(target_path, target_noisy)
    shutil.copyfile(target_path, ref_path)

    print(f"[generate_new_pair1] Successfully generated ultra-heavy noisy target & search pair {output_dir} | GT: ({gt_x}, {gt_y})")
    return search_path, target_path, gt_info


if __name__ == "__main__":
    generate_new_pair1()
