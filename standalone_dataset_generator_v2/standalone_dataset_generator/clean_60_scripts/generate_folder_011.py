#!/usr/bin/env python3
"""
Generate Pair 1 for generated_new (Silicon Wafer Multi-Chip Array - Thin Line Trace Joined Boxes - Center 500,500)
===================================================================================================================
Synthesizes a 1000x1000 px uint8 grayscale search image and 1000x1000 px target reference image
PURELY PROGRAMMATICALLY IN PYTHON CODE (matching the user's reference SEM microchip image).

Design Features:
  - Full Silicon Wafer disk containing a 5x5 array of repeated semiconductor microchips.
  - Top-Left section of every chip features a 4x4 matrix of 16 separate square boxes / via pads.
  - NO ARTIFICIAL MARKINGS & NO SOLID FILLED BLOCKS!
  - UNIQUE TARGET LANDMARK (Pair 1 - Center Chip 500,500): Boxes remain separate individual squares,
    joined by a THIN METALLIC INTERCONNECT TRACE LINE running through the box row!
  - Independent SEM noise applied to search.png and target.png.
  - Razor-sharp edges with nearest-neighbor scaling for target.png.

Files created in generated_new/pair1/ (and copied to generated_wafer/pair1/):
  - groundtruth.json
  - search.png    (1000x1000 px grayscale wafer image with sharp thin trace joined boxes & noise)
  - target.png    (1000x1000 px grayscale 100x high-mag target image centered at 500, 500)
  - reference.png (1000x1000 px grayscale, copy of target.png)
"""

import json
import os
import shutil
import cv2
import numpy as np


def render_semiconductor_wafer_scene(h=1000, w=1000, gt_x=500, gt_y=500, seed=2001):
    """
    Render 1000x1000 uint8 grayscale Semiconductor Wafer multi-chip layout for Pair 1:
    - 5x5 array of identical semiconductor microchips on a circular wafer disk
    - Regular chips: Top-left 4x4 box matrix of 16 separate square via pads
    - Target chip at Center (gt_x, gt_y) = (500, 500): Boxes remain separate individual squares, joined by a THIN METALLIC TRACE LINE
    """
    np.random.seed(seed)
    # Dark SEM substrate background
    img = np.full((h, w), 20, dtype=np.float32)

    # 1. Circular Silicon Wafer Disk
    center_wafer = (w // 2, h // 2)
    radius_wafer = 470
    cv2.circle(img, center_wafer, radius_wafer, 40, -1)

    # Metallic radial shading on silicon wafer surface
    y_indices, x_indices = np.indices((h, w))
    dist_from_center = np.sqrt((x_indices - center_wafer[0])**2 + (y_indices - center_wafer[1])**2)
    wafer_mask = dist_from_center <= radius_wafer
    img[wafer_mask] += 15.0 * (1.0 - dist_from_center[wafer_mask] / radius_wafer)

    # Wafer alignment notch at top
    cv2.rectangle(img, (w // 2 - 15, h // 2 - radius_wafer - 5), (w // 2 + 15, h // 2 - radius_wafer + 15), 15, -1)

    # 2. Render 5x5 Microchip Die Array across wafer
    chip_w, chip_h = 140, 140
    pitch_x, pitch_y = 165, 165
    
    # 5x5 die centers centered on wafer
    cols = [500 + i * pitch_x for i in range(-2, 3)] # [170, 335, 500, 665, 830]
    rows = [500 + j * pitch_y for j in range(-2, 3)] # [170, 335, 500, 665, 830]

    for cy in rows:
        for cx in cols:
            # Render chip if it fits inside silicon wafer radius
            if np.sqrt((cx - center_wafer[0])**2 + (cy - center_wafer[1])**2) < radius_wafer - 55:
                x1, y1 = cx - chip_w // 2, cy - chip_h // 2
                x2, y2 = cx + chip_w // 2, cy + chip_h // 2

                # Crisp Scribe line outer die frame
                cv2.rectangle(img, (x1, y1), (x2, y2), 220, 2)

                # Inner die substrate
                cv2.rectangle(img, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), 60, -1)

                # Top-Right Section: Word Lines (horizontal parallel stripes)
                for wy in range(y1 + 10, y1 + 55, 4):
                    cv2.line(img, (cx - 10, wy), (x2 - 10, wy), 235, 1)

                # Middle Section: Bit Lines & Memory Cell Matrix (crosshatch grid)
                for bx in range(x1 + 12, x2 - 12, 6):
                    cv2.line(img, (bx, y1 + 60), (bx, y2 - 38), 190, 1)
                for my in range(y1 + 64, y2 - 40, 10):
                    cv2.line(img, (x1 + 12, my), (x2 - 12, my), 150, 1)

                # Bottom Section: Sense Amplifier & I/O Circuitry (dense blocks & via pads)
                cv2.rectangle(img, (x1 + 12, y2 - 34), (x2 - 12, y2 - 8), 120, -1)
                for px in range(x1 + 16, x2 - 16, 12):
                    cv2.rectangle(img, (px, y2 - 30), (px + 6, y2 - 12), 245, -1)

                # Top-Left Section: 4x4 Box Matrix (16 separate individual square boxes)
                box_pitch = 12
                for r_idx in range(4):
                    for c_idx in range(4):
                        box_x = cx - 52 + c_idx * box_pitch
                        box_y = cy - 52 + r_idx * box_pitch
                        cv2.rectangle(img, (box_x - 3, box_y - 3), (box_x + 3, box_y + 3), 210, -1)
                        cv2.rectangle(img, (box_x - 4, box_y - 4), (box_x + 4, box_y + 4), 150, 1)

                # PAIR 1 UNIQUE TARGET LANDMARK: Thin metallic line joining boxes (boxes remain 100% separate individual squares!)
                is_gt_chip = (cx == gt_x and cy == gt_y)
                if is_gt_chip:
                    b0_x = gt_x - 52
                    b0_y = gt_y - 52
                    b2_x = b0_x + 2 * box_pitch

                    # Thin metallic wire trace line joining Box 0 <-> Box 1 <-> Box 2 in Row 0
                    cv2.line(img, (b0_x, b0_y), (b2_x, b0_y), 255, 3)
                    # Extended thin bus stub line
                    cv2.line(img, (b2_x, b0_y), (b2_x + 8, b0_y), 255, 3)

    return np.clip(img, 0, 255).astype(np.uint8)


def apply_vignette_noise(image_float, strength=0.25):
    """Apply optical/SEM lens vignette corner shading noise."""
    h, w = image_float.shape
    v1 = (1.0 - strength * (np.linspace(-1.0, 1.0, w, dtype=np.float32)**2)).astype(np.float32)
    v2 = (1.0 - strength * (np.linspace(-1.0, 1.0, h, dtype=np.float32)**2)).astype(np.float32)
    vignette = np.outer(v2, v1)
    return image_float * vignette


def apply_speckle_noise(image_float, sigma=0.08):
    """Apply multiplicative SEM electron speckle noise."""
    noise = np.random.normal(0, sigma, image_float.shape).astype(np.float32)
    return image_float * (1.0 + noise)


# [STRIPPED] def add_heavy_multi_sem_noise(image_uint8, seed=2001, is_target=False):
# [STRIPPED]     """
# [STRIPPED]     Apply crisp SEM noise pipeline (Vignette + Speckle + Shot/Read Noise):
# [STRIPPED]     - NO DEFOCUS BLUR to preserve 100% razor-sharp circuit edges!
# [STRIPPED]     """
# [STRIPPED]     np.random.seed(seed if not is_target else seed + 9999)
# [STRIPPED]     img_float = image_uint8.astype(np.float32)
# [STRIPPED] 
# [STRIPPED]     # 1. Vignette corner shading
# [STRIPPED]     vignetted = apply_vignette_noise(img_float, strength=0.25 if not is_target else 0.15)
# [STRIPPED] 
# [STRIPPED]     # 2. Speckle noise
# [STRIPPED]     speckled = apply_speckle_noise(vignetted, sigma=0.08 if not is_target else 0.04)
# [STRIPPED] 
# [STRIPPED]     # 3. Poisson shot & Gaussian detector read noise
# [STRIPPED]     shot_noise = np.random.normal(0, 10.0 if not is_target else 6.0, img_float.shape).astype(np.float32)
# [STRIPPED]     read_noise = np.random.normal(0, 6.0 if not is_target else 4.0, img_float.shape).astype(np.float32)
# [STRIPPED] 
# [STRIPPED]     # Sharp output without defocus blur
# [STRIPPED]     noisy_img = np.clip(speckled + shot_noise + read_noise, 0, 255).astype(np.uint8)
# [STRIPPED]     return noisy_img
# [STRIPPED] 
# [STRIPPED] 
def generate_new_pair1(output_dir="generated_new/pair1", gt_x=500, gt_y=500, seed=2001):
    """Generate pair1 with Target Chip at Center (500, 500)."""
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    # Render clean wafer scene programmatically
    search_clean = render_semiconductor_wafer_scene(h=h, w=w, gt_x=gt_x, gt_y=gt_y, seed=seed)

    # Extract 100x100 crop around GT center (500, 500) and scale 10x to 1000x1000 px for target.png
    crop_size = 100
    x0 = max(0, min(w - crop_size, gt_x - crop_size // 2))
    y0 = max(0, min(h - crop_size, gt_y - crop_size // 2))
    x1, y1 = x0 + crop_size, y0 + crop_size

    crop_patch = search_clean[y0:y1, x0:x1]
    
    # 1000x1000 px sharp target reference image (nearest-neighbor scaling)
    target_clean = cv2.resize(crop_patch, (1000, 1000), interpolation=cv2.INTER_NEAREST)

    # Apply INDEPENDENT noise to search image and target image (preserving sharp edges)
    search_noisy = search_clean  # noise stripped by v2 pipeline
    target_noisy = target_clean  # noise stripped by v2 pipeline

    gt_info = {
        "center_x": gt_x,
        "center_y": gt_y,
        "target_name": "Silicon Wafer Multi-Chip Array (Thin Trace Line Joined Boxes - Center Chip GT 500,500) 1",
        "pair_id": "pair1",
        "scale_factor": 10.0
    }

    gt_path = os.path.join(output_dir, "groundtruth.json")
    search_path = os.path.join(output_dir, "search.png")
    target_path = os.path.join(output_dir, "target.png")
    ref_path = os.path.join(output_dir, "reference.png")

    with open(gt_path, "w") as f:
        json.dump(gt_info, f, indent=2)

    cv2.imwrite(search_path, search_noisy)
    cv2.imwrite(target_path, target_noisy)
    shutil.copyfile(target_path, ref_path)

    # Also sync to generated_wafer/pair1 for seamless compatibility
    wafer_dir = os.path.join("generated_wafer", "pair1")
    os.makedirs(wafer_dir, exist_ok=True)
    with open(os.path.join(wafer_dir, "groundtruth.json"), "w") as f:
        json.dump(gt_info, f, indent=2)
    cv2.imwrite(os.path.join(wafer_dir, "search.png"), search_noisy)
    cv2.imwrite(os.path.join(wafer_dir, "target.png"), target_noisy)
    shutil.copyfile(os.path.join(wafer_dir, "target.png"), os.path.join(wafer_dir, "reference.png"))

    print(f"[generate_new_pair1] Successfully generated pair 1 {output_dir} | GT: ({gt_x}, {gt_y})")
    return search_path, target_path, gt_info


if __name__ == "__main__":
    generate_new_pair1()
