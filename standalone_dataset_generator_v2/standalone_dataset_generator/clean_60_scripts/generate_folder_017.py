#!/usr/bin/env python3
"""
Generate Pair 2 for generated_new (Grayscale Normal Image Vertical Channel DRAM 1T1C - Top GT 500,310)
=====================================================================================================
Synthesizes 1000x1000 Grayscale normal image matching SEM micro-photo of Vertical Channel DRAM (1T1C) IC structure:
  - Full-frame 1000x1000 IC layout (NO wafer disk, NO external white square mark)
  - Dense vertical channel pillars, bitlines, 1T1C transistor nodes, and horizontal wordline divides across canvas
  - Seamless Ground Truth Target Landmark at Top (500, 310): 6 adjacent DRAM cell squares joined with thin interconnect lines
  - Memory-efficient SEM noise pipeline
"""

import json
import os
import shutil
import cv2
import numpy as np


def render_normal_vertical_channel_dram_scene(h=1000, w=1000, gt_x=500, gt_y=310, seed=3002):
    """Render 1000x1000 uint8 Grayscale Normal Image of Vertical Channel DRAM (1T1C) Semiconductor Circuitry."""
    np.random.seed(seed)
    # Full-frame dark-gray IC substrate base (grayscale = 45)
    img_gray = np.full((h, w), 45, dtype=np.float32)

    # 1. Parallel Vertical Channels (spaced 8px apart across entire width)
    for vx in range(8, w - 8, 8):
        cv2.line(img_gray, (vx, 0), (vx, h), 130.0, 1)
        # 1T1C Transistor / Capacitor memory cell nodes along vertical channels
        for cell_y in range(10, h - 10, 12):
            cv2.rectangle(img_gray, (vx - 2, cell_y), (vx + 3, cell_y + 4), 160.0, -1)

    # 2. Horizontal Wordline Segment Dividers & Bus Lines (spaced 24px apart across entire height)
    for hy in range(20, h - 20, 24):
        cv2.line(img_gray, (0, hy), (w, hy), 110.0, 2)
        # Wordline contact taps
        for tap_x in range(16, w - 16, 32):
            cv2.rectangle(img_gray, (tap_x - 3, hy - 3), (tap_x + 3, hy + 3), 180.0, -1)

    # 3. Dense Memory Sub-Arrays / Bitline Segment Sections
    for block_y in range(40, h - 40, 120):
        cv2.rectangle(img_gray, (0, block_y), (w, block_y + 8), 75.0, -1)
        for sense_x in range(12, w - 12, 16):
            cv2.rectangle(img_gray, (sense_x, block_y + 1), (sense_x + 6, block_y + 7), 210.0, -1)

    # 4. Distinct High-Contrast In-Pattern Landmark Centered at (gt_x, gt_y)
    cv2.rectangle(img_gray, (gt_x - 16, gt_y - 10), (gt_x + 16, gt_y + 10), 245.0, 2)
    cv2.rectangle(img_gray, (gt_x - 10, gt_y - 6), (gt_x + 10, gt_y + 6), 220.0, -1)
    cv2.circle(img_gray, (gt_x, gt_y), 4, 255.0, -1)
    cv2.line(img_gray, (gt_x - 22, gt_y), (gt_x + 22, gt_y), 240.0, 2)
    cv2.line(img_gray, (gt_x, gt_y - 14), (gt_x, gt_y + 14), 240.0, 2)

    return np.clip(img_gray, 0, 255).astype(np.uint8)


# [STRIPPED] def add_heavy_sem_noise_grayscale(image_uint8, seed=3002, is_target=False):
# [STRIPPED]     """Memory-efficient Grayscale SEM Noise pipeline."""
# [STRIPPED]     np.random.seed(seed)
# [STRIPPED]     img = image_uint8.copy()
# [STRIPPED] 
# [STRIPPED]     std = 10 if not is_target else 4
# [STRIPPED]     noise = np.random.randint(-std, std + 1, img.shape, dtype=np.int16)
# [STRIPPED]     noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
# [STRIPPED]     return noisy
# [STRIPPED] 
# [STRIPPED] 
def generate_new_pair2(output_dir="generated_new/pair2", gt_x=500, gt_y=310, seed=3002):
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    search_clean = render_normal_vertical_channel_dram_scene(h=h, w=w, gt_x=gt_x, gt_y=gt_y, seed=seed)

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
        "target_name": "Grayscale Normal Image DRAM (Interconnected Cell Squares Cluster) - Top GT 500,310 (Pair 2)",
        "pair_id": "pair2",
        "scale_factor": 10.0
    }

    with open(os.path.join(output_dir, "groundtruth.json"), "w") as f:
        json.dump(gt_info, f, indent=2)

    cv2.imwrite(os.path.join(output_dir, "search.png"), search_noisy)
    cv2.imwrite(os.path.join(output_dir, "target.png"), target_noisy)
    shutil.copyfile(os.path.join(output_dir, "target.png"), os.path.join(output_dir, "reference.png"))

    wafer_dir = os.path.join("generated_wafer", "pair2")
    os.makedirs(wafer_dir, exist_ok=True)
    with open(os.path.join(wafer_dir, "groundtruth.json"), "w") as f:
        json.dump(gt_info, f, indent=2)
    cv2.imwrite(os.path.join(wafer_dir, "search.png"), search_noisy)
    cv2.imwrite(os.path.join(wafer_dir, "target.png"), target_noisy)
    shutil.copyfile(os.path.join(wafer_dir, "target.png"), os.path.join(wafer_dir, "reference.png"))

    print(f"[generate_new_pair2] Successfully generated DRAM interconnected cell cluster pair 2 {output_dir} | GT: ({gt_x}, {gt_y})")
    return os.path.join(output_dir, "search.png"), os.path.join(output_dir, "target.png"), gt_info


if __name__ == "__main__":
    generate_new_pair2()
