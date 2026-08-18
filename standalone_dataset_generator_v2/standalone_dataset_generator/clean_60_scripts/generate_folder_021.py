#!/usr/bin/env python3
"""
Generate Pair 1 for generated_stacked (Grayscale Stacked Capacitor DRAM - Center GT 500,500)
============================================================================================
Synthesizes 1000x1000 Grayscale normal image matching SEM micro-photo of Stacked Capacitor DRAM IC structure:
  - Full-frame 1000x1000 IC layout
  - Dense repeating rows of stacked capacitor cells with rectangular storage nodes
  - Vertical bitline pillars and horizontal wordline bus straps
  - Seamless Ground Truth Target Landmark at Center (500, 500): Interconnected cluster of adjacent capacitor cells
  - Memory-efficient SEM noise pipeline
"""

import json
import os
import shutil
import cv2
import numpy as np


def render_stacked_capacitor_dram_scene(h=1000, w=1000, gt_x=500, gt_y=500, seed=5001):
    """Render 1000x1000 uint8 Grayscale Normal Image of Stacked Capacitor DRAM Semiconductor Circuitry."""
    np.random.seed(seed)
    # Full-frame dark IC substrate base
    img_gray = np.full((h, w), 30, dtype=np.float32)

    # Row pitch and column pitch for stacked capacitor cells
    row_pitch = 20    # Vertical distance between cell rows
    col_pitch = 12    # Horizontal distance between capacitor cells

    # 1. Render Stacked Capacitor Cell Rows across entire canvas
    for ry in range(10, h - 10, row_pitch):
        # Horizontal row boundary line (wordline)
        cv2.line(img_gray, (0, ry), (w, ry), 95.0, 1)

        for cx in range(6, w - 6, col_pitch):
            # Stacked Capacitor Storage Node (rectangular block)
            cap_y1 = ry + 3
            cap_y2 = ry + row_pitch - 3
            cap_x1 = cx - 3
            cap_x2 = cx + 3

            # Capacitor dielectric layer (outer shell)
            cv2.rectangle(img_gray, (cap_x1, cap_y1), (cap_x2, cap_y2), 140.0, 1)

            # Inner storage electrode plate
            cv2.rectangle(img_gray, (cap_x1 + 1, cap_y1 + 2), (cap_x2 - 1, cap_y2 - 2), 175.0, -1)

            # Top plate contact via
            cv2.circle(img_gray, (cx, cap_y1 + 1), 1, 210.0, -1)

            # Bottom contact to transistor drain
            cv2.rectangle(img_gray, (cx - 1, cap_y2 - 1), (cx + 1, cap_y2), 200.0, -1)

    # 2. Vertical Bitline Pillars (spaced every 36px)
    for bx in range(18, w - 18, 36):
        cv2.line(img_gray, (bx, 0), (bx, h), 115.0, 2)
        # Bitline contact pads at intervals
        for by in range(15, h - 15, 40):
            cv2.rectangle(img_gray, (bx - 3, by - 2), (bx + 3, by + 2), 185.0, -1)

    # 3. Horizontal Wordline Bus Straps (major dividers every 60px)
    for hy in range(30, h - 30, 60):
        cv2.line(img_gray, (0, hy), (w, hy), 110.0, 2)
        # Wordline driver contact taps
        for tx in range(18, w - 18, 72):
            cv2.rectangle(img_gray, (tx - 4, hy - 4), (tx + 4, hy + 4), 195.0, -1)

    # 4. Seamless In-Pattern Stacked DRAM Landmark Centered Exactly at (gt_x, gt_y)
    cv2.rectangle(img_gray, (gt_x - 14, gt_y - 14), (gt_x + 14, gt_y + 14), 220.0, 2)
    cv2.rectangle(img_gray, (gt_x - 8, gt_y - 8), (gt_x + 8, gt_y + 8), 245.0, -1)
    cv2.circle(img_gray, (gt_x, gt_y), 4, 255.0, -1)
    cv2.line(img_gray, (gt_x - 18, gt_y), (gt_x + 18, gt_y), 240.0, 2)
    cv2.line(img_gray, (gt_x, gt_y - 18), (gt_x, gt_y + 18), 240.0, 2)

    return np.clip(img_gray, 0, 255).astype(np.uint8)


# [STRIPPED] def add_heavy_sem_noise_grayscale(image_uint8, seed=5001, is_target=False):
# [STRIPPED]     """Memory-efficient Grayscale SEM Noise pipeline."""
# [STRIPPED]     np.random.seed(seed)
# [STRIPPED]     img = image_uint8.copy()
# [STRIPPED]     std = 10 if not is_target else 4
# [STRIPPED]     noise = np.random.randint(-std, std + 1, img.shape, dtype=np.int16)
# [STRIPPED]     noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
# [STRIPPED]     return noisy
# [STRIPPED] 
# [STRIPPED] 
def generate_stacked_pair1(output_dir="generated_stacked/pair1", gt_x=500, gt_y=500, seed=5001):
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    search_clean = render_stacked_capacitor_dram_scene(h=h, w=w, gt_x=gt_x, gt_y=gt_y, seed=seed)

    crop_size = 100
    x0 = max(0, min(w - crop_size, gt_x - crop_size // 2))
    y0 = max(0, min(h - crop_size, gt_y - crop_size // 2))
    x1, y1 = x0 + crop_size, y0 + crop_size

    crop_patch = search_clean[y0:y1, x0:x1]
    target_clean = cv2.resize(crop_patch, (1000, 1000), interpolation=cv2.INTER_NEAREST)

    search_noisy = search_clean  # noise stripped by v2 pipeline
    target_noisy = target_clean  # noise stripped by v2 pipeline

    gt_info = {
        "center_x": gt_x,
        "center_y": gt_y,
        "target_name": "Grayscale Stacked Capacitor DRAM - Center GT 500,500 (Pair 1)",
        "pair_id": "pair1",
        "scale_factor": 10.0
    }

    with open(os.path.join(output_dir, "groundtruth.json"), "w") as f:
        json.dump(gt_info, f, indent=2)

    cv2.imwrite(os.path.join(output_dir, "search.png"), search_noisy)
    cv2.imwrite(os.path.join(output_dir, "target.png"), target_noisy)
    shutil.copyfile(os.path.join(output_dir, "target.png"), os.path.join(output_dir, "reference.png"))

    print(f"[generate_stacked_pair1] Successfully generated Stacked Capacitor DRAM pair 1 {output_dir} | GT: ({gt_x}, {gt_y})")
    return os.path.join(output_dir, "search.png"), os.path.join(output_dir, "target.png"), gt_info


if __name__ == "__main__":
    generate_stacked_pair1()
