#!/usr/bin/env python3
"""
Generate Pair 2 for generated_rgb (Extremely Dense Vibrant RGB IC Layout Grid - Unique Circle Target)
===================================================================================================
Synthesizes a 1000x1000 px 3-channel RGB search image and 1000x1000 px RGB target reference image
PURELY PROGRAMMATICALLY IN PYTHON CODE (matching the user's colorful IC layout reference image).

Design Features:
  - Extremely dense multi-colored IC layout grid (pitch = 50px) packing 400+ colorful blocks.
  - Thin 1-pixel crisp line boundaries to prevent colors/lines from merging.
  - Regular IC sub-panel blocks contain central SQUARE features.
  - UNIQUE TARGET LANDMARK: At Ground Truth (300, 700), the central square is replaced with
    a UNIQUE VIBRANT CONCENTRIC CIRCLE landmark!
  - Heavy imaging noise applied across RGB channels.

Files created in generated_rgb/pair2/:
  - groundtruth.json
  - search.png    (1000x1000 px RGB with heavy noise)
  - target.png    (1000x1000 px RGB 100x high-mag reference target)
  - reference.png (1000x1000 px RGB, copy of target.png)
"""

import json
import os
import shutil
import cv2
import numpy as np


def render_extremedense_rgb_ic_layout_scene(h=1000, w=1000, gt_x=300, gt_y=700, seed=902):
    """
    Render extremely dense 1000x1000 3-channel RGB IC layout grid image pure programmatically:
    - Extremely high-density colorful interconnect tracks (magenta, cyan, yellow, green, orange)
    - Concentric square frames on regular blocks (block_pitch = 50px, 400+ blocks across field)
    - Unique Circle target feature at Ground Truth (gt_x, gt_y)
    """
    np.random.seed(seed)
    img = np.full((h, w, 3), 12, dtype=np.float32)

    # Color Palette (BGR format for OpenCV)
    C_MAGENTA = (200, 0, 220)
    C_CYAN    = (230, 220, 0)
    C_YELLOW  = (0, 220, 240)
    C_GREEN   = (0, 210, 80)
    C_ORANGE  = (0, 140, 240)
    C_PURPLE  = (220, 50, 140)
    C_BRIGHT  = (255, 255, 255)

    block_pitch = 50
    block_size = 38

    # 1. Render 400+ extremely dense IC sub-panel layout blocks
    for by in range(25, h, block_pitch):
        for bx in range(25, w, block_pitch):
            x1, y1 = bx - block_size // 2, by - block_size // 2
            x2, y2 = x1 + block_size, y1 + block_size
            if 0 <= x1 < w and 0 <= y1 < h:
                # Outer block frame (Purple / Cyan)
                cv2.rectangle(img, (x1, y1), (x2, y2), C_PURPLE, 1)
                cv2.rectangle(img, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), C_CYAN, 1)

                # Four corner square pads
                pad_s = 4
                corners = [(x1 + 2, y1 + 2), (x2 - 2 - pad_s, y1 + 2),
                           (x1 + 2, y2 - 2 - pad_s), (x2 - 2 - pad_s, y2 - 2 - pad_s)]
                for cx, cy in corners:
                    cv2.rectangle(img, (cx, cy), (cx + pad_s, cy + pad_s), C_YELLOW, 1)

                # Internal parallel rail tracks
                cv2.line(img, (x1 + 3, by - 6), (x2 - 3, by - 6), C_YELLOW, 1)
                cv2.line(img, (x1 + 3, by + 6), (x2 - 3, by + 6), C_YELLOW, 1)
                cv2.line(img, (bx - 6, y1 + 3), (bx - 6, y2 - 3), C_GREEN, 1)
                cv2.line(img, (bx + 6, y1 + 3), (bx + 6, y2 - 3), C_GREEN, 1)

                # Check if this block is the Ground Truth target block
                dist = np.hypot(bx - gt_x, by - gt_y)
                if dist >= block_pitch * 0.4:
                    # Regular block: Concentric SQUARE feature at center
                    cv2.rectangle(img, (bx - 6, by - 6), (bx + 6, by + 6), C_MAGENTA, 1)
                    cv2.rectangle(img, (bx - 3, by - 3), (bx + 3, by + 3), C_CYAN, -1)

    # 2. Render global interconnect bus corridors
    for bx in range(25, w, block_pitch):
        cv2.line(img, (bx, 0), (bx, h), C_GREEN, 1)
    for by in range(25, h, block_pitch):
        cv2.line(img, (0, by), (w, by), C_YELLOW, 1)

    # 3. UNIQUE TARGET LANDMARK: Concentric CIRCLE feature at (gt_x, gt_y)!
    cv2.circle(img, (gt_x, gt_y), 14, C_MAGENTA, 2)
    cv2.circle(img, (gt_x, gt_y), 10, C_CYAN, 2)
    cv2.circle(img, (gt_x, gt_y), 6, C_YELLOW, 1)
    cv2.circle(img, (gt_x, gt_y), 3, C_ORANGE, -1)

    # Crosshair alignment ticks on target circle
    cv2.line(img, (gt_x - 18, gt_y), (gt_x - 15, gt_y), C_BRIGHT, 1)
    cv2.line(img, (gt_x + 15, gt_y), (gt_x + 18, gt_y), C_BRIGHT, 1)
    cv2.line(img, (gt_x, gt_y - 18), (gt_x, gt_y - 15), C_BRIGHT, 1)
    cv2.line(img, (gt_x, gt_y + 15), (gt_x, gt_y + 18), C_BRIGHT, 1)

    return np.clip(img, 0, 255).astype(np.uint8)


# [STRIPPED] def add_heavy_rgb_noise(img_rgb, poisson_scale=15.0, gauss_sigma=12.0):
# [STRIPPED]     """Apply heavy imaging noise across RGB channels."""
# [STRIPPED]     img_float = img_rgb.astype(np.float32)
# [STRIPPED] 
# [STRIPPED]     # 1. Poisson shot noise per channel
# [STRIPPED]     counts = np.maximum(0, img_float) / 255.0 * poisson_scale
# [STRIPPED]     noisy_counts = np.random.poisson(counts).astype(np.float32)
# [STRIPPED]     img_poisson = noisy_counts / poisson_scale * 255.0
# [STRIPPED] 
# [STRIPPED]     # 2. Heavy Gaussian read noise
# [STRIPPED]     gauss_noise = np.random.normal(0, gauss_sigma, img_float.shape).astype(np.float32)
# [STRIPPED]     noisy_img = img_poisson + gauss_noise
# [STRIPPED] 
# [STRIPPED]     return np.clip(noisy_img, 0, 255).astype(np.uint8)
# [STRIPPED] 
# [STRIPPED] 
def generate_rgb_pair2(output_dir="generated_rgb/pair2", gt_x=300, gt_y=700, seed=902):
    """Generate extremely dense RGB pair2 files with UNIQUE CIRCLE target landmark."""
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    # Render clean extremely dense RGB IC layout scene programmatically
    search_clean = render_extremedense_rgb_ic_layout_scene(h=h, w=w, gt_x=gt_x, gt_y=gt_y, seed=seed)

    # Extract 100x100 crop around GT center (300, 700) and scale 10x to 1000x1000 px for target.png
    crop_size = 100
    x0 = max(0, min(w - crop_size, gt_x - crop_size // 2))
    y0 = max(0, min(h - crop_size, gt_y - crop_size // 2))
    x1, y1 = x0 + crop_size, y0 + crop_size

    crop_patch = search_clean[y0:y1, x0:x1]
    
    # 1000x1000 px RGB target reference image (100x magnification view)
    target_clean = cv2.resize(crop_patch, (1000, 1000), interpolation=cv2.INTER_CUBIC)

    # Apply heavy RGB noise to search image
    np.random.seed(seed)
    search_noisy = search_clean  # noise stripped by v2 pipeline

    gt_info = {
        "center_x": gt_x,
        "center_y": gt_y,
        "target_name": "Extremely Dense RGB IC Layout Grid with Unique Circle Target Landmark 2",
        "pair_id": "pair2",
        "scale_factor": 10.0
    }

    # Save output RGB files
    gt_path = os.path.join(output_dir, "groundtruth.json")
    search_path = os.path.join(output_dir, "search.png")
    target_path = os.path.join(output_dir, "target.png")
    ref_path = os.path.join(output_dir, "reference.png")

    with open(gt_path, "w") as f:
        json.dump(gt_info, f, indent=2)

    cv2.imwrite(search_path, search_noisy)
    cv2.imwrite(target_path, target_clean)
    shutil.copyfile(target_path, ref_path)

    print(f"[generate_rgb_pair2] Successfully generated extremely dense RGB {output_dir} | GT: ({gt_x}, {gt_y})")
    return search_path, target_path, gt_info


if __name__ == "__main__":
    generate_rgb_pair2()
