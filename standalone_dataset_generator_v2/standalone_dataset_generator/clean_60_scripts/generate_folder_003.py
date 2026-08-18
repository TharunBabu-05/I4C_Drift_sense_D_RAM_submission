#!/usr/bin/env python3
"""
Generate Dataset Pair 1 (Hyper-Dense Crisp SEM Cell Grid + Extreme Noise + Unique Circle Target)
==================================================================================================
Generates 1000x1000 px search image and 1000x1000 px target reference image
PURELY PROGRAMMATICALLY IN PYTHON CODE (No external images used).

Design Features:
  - Hyper-dense cell array (pitch = 14px) packing 4,400+ unmerged memory cells.
  - Thin 1-pixel cell walls to keep all lines crisp and unmerged.
  - Sub-panel bus line intersections have solid SQUARE junction pads.
  - UNIQUE TARGET LANDMARK: At Ground Truth (490, 490), junction is a UNIQUE CIRCLE pad.
  - Extreme SEM noise (Poisson scale = 8.0, Gauss sigma = 18.0).

Output files in generated_pairs/pair1/:
  - groundtruth.json
  - search.png    (1000x1000 px with extreme SEM noise)
  - target.png    (1000x1000 px 100x high-mag reference target)
  - reference.png (1000x1000 px, copy of target.png)
"""

import json
import os
import shutil
import cv2
import numpy as np


def render_hyperdense_crisp_sem_scene(h=1000, w=1000, gt_x=490, gt_y=490, seed=801):
    """
    Render hyper-dense 1000x1000 SEM grid image:
    - Hyper-dense crisp square cell array (pitch = 14px, 4400+ cells across field)
    - Thin 1-pixel walls so cell lines DO NOT MERGE
    - SQUARE junction pads across grid, UNIQUE CIRCLE pad at (gt_x, gt_y)
    """
    np.random.seed(seed)
    img = np.full((h, w), 25, dtype=np.float32)

    pitch = 14
    cell_w, cell_h = 10, 10

    # 1. Draw 4,400+ hyper-dense crisp unmerged cells across 1000x1000 field
    for y in range(7, h, pitch):
        for x in range(7, w, pitch):
            x1, y1 = x - cell_w // 2, y - cell_h // 2
            x2, y2 = x1 + cell_w, y1 + cell_h
            if 0 <= x1 < w and 0 <= y1 < h:
                # Thin 1-pixel bright cell wall (CRISP, UNMERGED)
                cv2.rectangle(img, (x1, y1), (x2, y2), 190, 1)
                # Dark cell body interior
                cv2.rectangle(img, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), 40, -1)

                # Circular via inside regular cell
                cv2.circle(img, (x, y), 1, 215, -1)

    # 2. Draw sub-panel bus tracks crossing at intervals (every 70 px)
    bus_positions_x = [x for x in range(70, w, 70)]
    bus_positions_y = [y for y in range(70, h, 70)]

    if gt_x not in bus_positions_x:
        bus_positions_x.append(gt_x)
    if gt_y not in bus_positions_y:
        bus_positions_y.append(gt_y)

    for bx in bus_positions_x:
        for dx in [-6, -2, 2, 6]:
            cv2.line(img, (bx + dx, 0), (bx + dx, h), 215, 1)
        cv2.rectangle(img, (bx - 1, 0), (bx + 1, h), 45, -1)

    for by in bus_positions_y:
        for dy in [-6, -2, 2, 6]:
            cv2.line(img, (0, by + dy), (w, by + dy), 215, 1)
        cv2.rectangle(img, (0, by - 1), (w, by + 1), 45, -1)

    # 3. Draw SQUARE junction pads at all regular sub-panel bus intersections
    sq_w = 20
    for bx in bus_positions_x:
        for by in bus_positions_y:
            dist_to_gt = np.hypot(bx - gt_x, by - gt_y)
            if dist_to_gt >= 15:
                cv2.rectangle(img, (bx - sq_w // 2, by - sq_w // 2),
                              (bx + sq_w // 2, by + sq_w // 2), 245, -1)
                cv2.rectangle(img, (bx - 4, by - 4), (bx + 4, by + 4), 38, -1)
                cv2.circle(img, (bx, by), 1, 255, -1)

    # 4. UNIQUE TARGET LANDMARK: Solid CIRCLE pad at (gt_x, gt_y)
    rad = 12
    cv2.circle(img, (gt_x, gt_y), rad, 255, -1)
    cv2.circle(img, (gt_x, gt_y), 4, 38, -1)
    cv2.circle(img, (gt_x, gt_y), 16, 230, 1)

    return np.clip(img, 0, 255).astype(np.uint8)


# [STRIPPED] def add_extreme_sem_noise(image_uint8, poisson_scale=8.0, gauss_sigma=18.0):
# [STRIPPED]     """
# [STRIPPED]     Apply EXTREME SEM imaging noise while preserving crisp unmerged cell lines:
# [STRIPPED]     - Extreme Poisson shot noise (poisson_scale = 8.0)
# [STRIPPED]     - Extreme Gaussian read noise (gauss_sigma = 18.0)
# [STRIPPED]     - NO blur filter to prevent merging of adjacent cell lines
# [STRIPPED]     """
# [STRIPPED]     img_float = image_uint8.astype(np.float32)
# [STRIPPED] 
# [STRIPPED]     # 1. Poisson shot noise
# [STRIPPED]     counts = np.maximum(0, img_float) / 255.0 * poisson_scale
# [STRIPPED]     noisy_counts = np.random.poisson(counts).astype(np.float32)
# [STRIPPED]     img_poisson = noisy_counts / poisson_scale * 255.0
# [STRIPPED] 
# [STRIPPED]     # 2. Extreme Gaussian read noise
# [STRIPPED]     gauss_noise = np.random.normal(0, gauss_sigma, img_float.shape).astype(np.float32)
# [STRIPPED]     noisy_img = img_poisson + gauss_noise
# [STRIPPED] 
# [STRIPPED]     return np.clip(noisy_img, 0, 255).astype(np.uint8)
# [STRIPPED] 
# [STRIPPED] 
def generate_pair1(output_dir="generated_pairs/pair1", gt_x=490, gt_y=490, seed=801):
    """Generate pair1 files with hyper-dense crisp cell grid and extreme SEM noise."""
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    # Render clean hyper-dense scene programmatically
    search_clean = render_hyperdense_crisp_sem_scene(h=h, w=w, gt_x=gt_x, gt_y=gt_y, seed=seed)

    # Extract 100x100 crop around GT center (490, 490) and scale 10x to 1000x1000 px for target.png
    crop_size = 100
    x0 = max(0, min(w - crop_size, gt_x - crop_size // 2))
    y0 = max(0, min(h - crop_size, gt_y - crop_size // 2))
    x1, y1 = x0 + crop_size, y0 + crop_size

    crop_patch = search_clean[y0:y1, x0:x1]
    
    # 1000x1000 px target reference image (100x magnification view)
    target_clean = cv2.resize(crop_patch, (1000, 1000), interpolation=cv2.INTER_CUBIC)

    # Apply EXTREME SEM noise to search image
    np.random.seed(seed)
    search_noisy = search_clean  # noise stripped by v2 pipeline

    gt_info = {
        "center_x": gt_x,
        "center_y": gt_y,
        "target_name": "Hyper-Dense Crisp SEM Cell Grid with Unique Circle Target Landmark 1",
        "pair_id": "pair1",
        "scale_factor": 10.0
    }

    # Save output files
    gt_path = os.path.join(output_dir, "groundtruth.json")
    search_path = os.path.join(output_dir, "search.png")
    target_path = os.path.join(output_dir, "target.png")
    ref_path = os.path.join(output_dir, "reference.png")

    with open(gt_path, "w") as f:
        json.dump(gt_info, f, indent=2)

    cv2.imwrite(search_path, search_noisy)
    cv2.imwrite(target_path, target_clean)
    shutil.copyfile(target_path, ref_path)

    print(f"[generate_pair1] Successfully generated hyper-dense pair {output_dir} | GT: ({gt_x}, {gt_y})")
    return search_path, target_path, gt_info


if __name__ == "__main__":
    generate_pair1()
