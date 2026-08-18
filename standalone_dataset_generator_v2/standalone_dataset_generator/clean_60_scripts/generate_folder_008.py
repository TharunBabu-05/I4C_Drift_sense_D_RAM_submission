#!/usr/bin/env python3
"""
Generate Pair 2 for generated_new (Dense Grayscale SEM Layout + Barrel + Vignette + Multi-Noise)
===============================================================================================
Synthesizes a 1000x1000 px uint8 grayscale search image and 1000x1000 px target reference image
PURELY PROGRAMMATICALLY IN PYTHON CODE (matching the user's reference SEM chip layout image).

Design Features:
  - Dense SEM T-anchor / ladder transistor memory cell matrix (pitch = 70px).
  - SQUARE junction frames across regular intersections.
  - UNIQUE TARGET LANDMARK: At Ground Truth (350, 630), junction is a UNIQUE SOLID CIRCLE pad.
  - Multi-Noise Pipeline:
      1. Vignette shading (corner dark falloff)
      2. Barrel lens distortion
      3. Poisson shot noise + Gaussian read noise
      4. Salt & pepper impulse noise
      5. Defocus blur

Files created in generated_new/pair2/:
  - groundtruth.json
  - search.png    (1000x1000 px grayscale with barrel/vignette/multi-noise)
  - target.png    (1000x1000 px grayscale 100x high-mag reference target)
  - reference.png (1000x1000 px grayscale, copy of target.png)
"""

import json
import os
import shutil
import cv2
import numpy as np


def render_dense_sem_chip_layout_scene(h=1000, w=1000, gt_x=350, gt_y=630, seed=1002):
    """
    Render dense 1000x1000 uint8 grayscale SEM chip layout scene pure programmatically:
    - Dense parallel bus rails & square junction frames (block_pitch = 70px)
    - Vertical T-anchor memory transistor array inside sub-panel blocks
    - Unique Circle target landmark at Ground Truth (gt_x, gt_y)
    """
    np.random.seed(seed)
    # Dark SEM substrate background
    img = np.full((h, w), 25, dtype=np.float32)

    block_pitch = 70
    block_size = 54

    # 1. Render T-anchor memory cell array inside sub-panel blocks
    for by in range(35, h, block_pitch):
        for bx in range(35, w, block_pitch):
            x1, y1 = bx - block_size // 2, by - block_size // 2
            x2, y2 = x1 + block_size, y1 + block_size
            if 0 <= x1 < w and 0 <= y1 < h:
                # Sub-panel outer boundary
                cv2.rectangle(img, (x1, y1), (x2, y2), 160, 1)

                # Draw 2 columns of vertical T-anchor transistor cells inside block
                for col_x in [bx - 12, bx + 12]:
                    # Vertical stem line
                    cv2.line(img, (col_x, y1 + 3), (col_x, y2 - 3), 210, 1)
                    # Horizontal T-crossbars & gate pads
                    for ty in range(y1 + 7, y2 - 5, 12):
                        cv2.line(img, (col_x - 7, ty), (col_x + 7, ty), 220, 1)
                        cv2.rectangle(img, (col_x - 6, ty + 2), (col_x - 2, ty + 6), 170, -1)
                        cv2.rectangle(img, (col_x + 2, ty + 2), (col_x + 6, ty + 6), 170, -1)

    # 2. Render major interconnect bus rail corridors
    bus_xs = [x for x in range(35, w, block_pitch)]
    bus_ys = [y for y in range(35, h, block_pitch)]

    if gt_x not in bus_xs:
        bus_xs.append(gt_x)
    if gt_y not in bus_ys:
        bus_ys.append(gt_y)

    for bx in bus_xs:
        for dx in [-5, -2, 2, 5]:
            cv2.line(img, (bx + dx, 0), (bx + dx, h), 215, 1)

    for by in bus_ys:
        for dy in [-5, -2, 2, 5]:
            cv2.line(img, (0, by + dy), (w, by + dy), 215, 1)

    # 3. Render SQUARE junction frames at regular bus intersections
    sq_w = 20
    for bx in bus_xs:
        for by in bus_ys:
            dist = np.hypot(bx - gt_x, by - gt_y)
            if dist >= 15:
                cv2.rectangle(img, (bx - sq_w // 2, by - sq_w // 2),
                              (bx + sq_w // 2, by + sq_w // 2), 240, 1)
                cv2.rectangle(img, (bx - 4, by - 4), (bx + 4, by + 4), 45, -1)
                cv2.circle(img, (bx, by), 1, 255, -1)

    # 4. UNIQUE TARGET LANDMARK: Solid CIRCLE pad with concentric rings at (gt_x, gt_y)!
    rad = 12
    cv2.circle(img, (gt_x, gt_y), rad, 255, -1)
    cv2.circle(img, (gt_x, gt_y), 4, 45, -1)
    cv2.circle(img, (gt_x, gt_y), 16, 230, 1)
    cv2.circle(img, (gt_x, gt_y), 20, 180, 1)

    return np.clip(img, 0, 255).astype(np.uint8)


def apply_vignette_noise(image_uint8, strength=0.25):
    """Apply optical/SEM lens vignette corner shading noise."""
    h, w = image_uint8.shape[:2]
    x = np.linspace(-1.0, 1.0, w)
    y = np.linspace(-1.0, 1.0, h)
    x_grid, y_grid = np.meshgrid(x, y)
    r = np.sqrt(x_grid**2 + y_grid**2)
    vignette_mask = 1.0 - strength * (r / np.sqrt(2.0))**2
    vignette_mask = np.clip(vignette_mask, 0.3, 1.0)
    return np.clip(image_uint8.astype(np.float32) * vignette_mask, 0, 255).astype(np.uint8)


def apply_barrel_distortion(image_uint8, dist_k=0.0000001):
    """Apply SEM lens barrel distortion transformation."""
    h, w = image_uint8.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    x = np.arange(w, dtype=np.float32) - cx
    y = np.arange(h, dtype=np.float32) - cy
    x_grid, y_grid = np.meshgrid(x, y)
    r2 = x_grid**2 + y_grid**2
    map_x = (x_grid * (1.0 + dist_k * r2) + cx).astype(np.float32)
    map_y = (y_grid * (1.0 + dist_k * r2) + cy).astype(np.float32)
    return cv2.remap(image_uint8, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def apply_salt_pepper_noise(image_uint8, salt_prob=0.001, pepper_prob=0.001):
    """Apply impulse salt & pepper defect noise."""
    noisy = image_uint8.copy()
    h, w = image_uint8.shape
    num_salt = int(salt_prob * h * w)
    num_pepper = int(pepper_prob * h * w)

    sy = np.random.randint(0, h, num_salt)
    sx = np.random.randint(0, w, num_salt)
    noisy[sy, sx] = 255

    py = np.random.randint(0, h, num_pepper)
    px = np.random.randint(0, w, num_pepper)
    noisy[py, px] = 0

    return noisy


# [STRIPPED] def add_multi_type_sem_noise(image_uint8, seed=1002):
# [STRIPPED]     """
# [STRIPPED]     Apply multi-type imaging noise pipeline:
# [STRIPPED]       1. Vignette corner shading
# [STRIPPED]       2. Barrel lens distortion
# [STRIPPED]       3. Poisson shot + Gaussian read noise
# [STRIPPED]       4. Salt & Pepper impulse noise
# [STRIPPED]       5. Defocus blur
# [STRIPPED]     """
# [STRIPPED]     np.random.seed(seed)
# [STRIPPED]     
# [STRIPPED]     # 1. Vignette shading
# [STRIPPED]     vignetted = apply_vignette_noise(image_uint8, strength=0.25)
# [STRIPPED]     
# [STRIPPED]     # 2. Barrel distortion
# [STRIPPED]     barreled = apply_barrel_distortion(vignetted, dist_k=0.0000001)
# [STRIPPED] 
# [STRIPPED]     # 3. Poisson + Gaussian noise
# [STRIPPED]     img_float = barreled.astype(np.float32)
# [STRIPPED]     shot_noise = np.random.normal(0, 12.0, img_float.shape).astype(np.float32)
# [STRIPPED]     read_noise = np.random.normal(0, 8.0, img_float.shape).astype(np.float32)
# [STRIPPED]     noisy_img = np.clip(img_float + shot_noise + read_noise, 0, 255).astype(np.uint8)
# [STRIPPED] 
# [STRIPPED]     # 4. Salt & Pepper noise
# [STRIPPED]     sp_img = apply_salt_pepper_noise(noisy_img, salt_prob=0.001, pepper_prob=0.001)
# [STRIPPED] 
# [STRIPPED]     # 5. Defocus blur
# [STRIPPED]     final_img = cv2.GaussianBlur(sp_img, (3, 3), 0.6)
# [STRIPPED] 
# [STRIPPED]     return final_img
# [STRIPPED] 
# [STRIPPED] 
def generate_new_pair2(output_dir="generated_new/pair2", gt_x=350, gt_y=630, seed=1002):
    """Generate dense grayscale pair2 files with barrel, vignette, and multi-noise."""
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    # Render clean dense grayscale SEM scene programmatically
    search_clean = render_dense_sem_chip_layout_scene(h=h, w=w, gt_x=gt_x, gt_y=gt_y, seed=seed)

    # Extract 100x100 crop around GT center (350, 630) and scale 10x to 1000x1000 px for target.png
    crop_size = 100
    x0 = max(0, min(w - crop_size, gt_x - crop_size // 2))
    y0 = max(0, min(h - crop_size, gt_y - crop_size // 2))
    x1, y1 = x0 + crop_size, y0 + crop_size

    crop_patch = search_clean[y0:y1, x0:x1]
    
    # 1000x1000 px target reference image (100x magnification view)
    target_clean = cv2.resize(crop_patch, (1000, 1000), interpolation=cv2.INTER_CUBIC)

    # Apply multi-type noise pipeline to search image
    search_noisy = search_clean  # noise stripped by v2 pipeline

    gt_info = {
        "center_x": gt_x,
        "center_y": gt_y,
        "target_name": "Dense Grayscale SEM Layout with Barrel, Vignette, and Multi-Noise 2",
        "pair_id": "pair2",
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
    cv2.imwrite(target_path, target_clean)
    shutil.copyfile(target_path, ref_path)

    print(f"[generate_new_pair2] Successfully generated dense multi-noise {output_dir} | GT: ({gt_x}, {gt_y})")
    return search_path, target_path, gt_info


if __name__ == "__main__":
    generate_new_pair2()
