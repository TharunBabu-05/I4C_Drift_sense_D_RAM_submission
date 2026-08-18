#!/usr/bin/env python3
"""
Generate Pair 1 for Grayscale Single-Sided Stitched Word-Line DRAM (Center GT 500,500)
=====================================================================================
Synthesizes 1000x1000 Grayscale SEM image matching Single-Sided Stitched Word-Line DRAM:
  - Full-frame 1000x1000 IC layout
  - Long continuous Wordlines (WL) with periodic single-sided metal stitching straps (top side taps)
  - Asymmetric WL driver pickup blocks and cell landing pads
  - High-density cell grid (pitch 18px x 18px)
  - Embedded Double-Stitched Landmark at Center (500, 500):
    Two adjacent stitched WL tap blocks joined together directly in the grid (no big white mark!)
  - Memory-optimized uint8 pipeline with full SEM noise
"""

import json
import os
import shutil
import cv2
import numpy as np


def render_stitchedwl_dram_scene(h=1000, w=1000, gt_x=500, gt_y=500, seed=17001):
    """Render 1000x1000 uint8 Grayscale image of Single-Sided Stitched Word-Line DRAM Circuitry."""
    np.random.seed(seed)

    col_pitch = 18  # Column pitch
    row_pitch = 18  # Wordline pitch

    # Dark insulating substrate base
    img_gray = np.full((h, w), 30, dtype=np.uint8)

    # ─── 1. Vertical Bitlines (BL) ────────────────────────────────────
    for cx in range(col_pitch // 2, w, col_pitch):
        cv2.line(img_gray, (cx, 0), (cx, h), 110, 2)
        cv2.line(img_gray, (cx - 1, 0), (cx - 1, h), 75, 1)

    # ─── 2. Long Continuous Stitched Wordlines (WL) ────────────────────
    for wy in range(row_pitch // 2, h, row_pitch):
        # Continuous horizontal Wordline trace
        cv2.line(img_gray, (0, wy), (w, wy), 170, 3)
        cv2.line(img_gray, (0, wy - 1), (w, wy - 1), 195, 1)

        # Single-Sided Stitching Straps (asymmetric top-side taps every 4 columns)
        for col_idx, cx in enumerate(range(col_pitch // 2, w, col_pitch)):
            if col_idx % 4 == 0:
                # Top-side metal stitch tap block
                cv2.rectangle(img_gray, (cx - 3, wy - 6), (cx + 3, wy - 1), 220, -1)
                cv2.rectangle(img_gray, (cx - 3, wy - 6), (cx + 3, wy - 1), 245, 1)
                cv2.circle(img_gray, (cx, wy - 3), 1, 255, -1)
            else:
                # Asymmetric bottom-side cell landing pad
                cv2.rectangle(img_gray, (cx - 2, wy + 1), (cx + 2, wy + 4), 185, -1)

    # ─── 3. Asymmetric Single-Sided WL Driver & Logic Bands ────────────
    sense_interval = row_pitch * 8  # 144px
    for sy in range(sense_interval, h - 10, sense_interval):
        # Asymmetric driver band offset to one side
        cv2.rectangle(img_gray, (0, sy - 5), (w, sy + 2), 60, -1)
        cv2.line(img_gray, (0, sy - 6), (w, sy - 6), 145, 1)
        cv2.line(img_gray, (0, sy + 3), (w, sy + 3), 145, 1)

    # ─── 4. Embedded Double-Stitched Landmark Centered at (gt_x, gt_y) ─
    cv2.rectangle(img_gray, (gt_x - 18, gt_y - 6), (gt_x + 18, gt_y + 6), 20, -1)
    cv2.line(img_gray, (gt_x - 17, gt_y), (gt_x + 17, gt_y), 180, 3)
    cv2.rectangle(img_gray, (gt_x - 15, gt_y - 4), (gt_x + 15, gt_y + 4), 225, -1)
    cv2.rectangle(img_gray, (gt_x - 15, gt_y - 4), (gt_x + 15, gt_y + 4), 245, 1)
    cv2.circle(img_gray, (gt_x - 9, gt_y), 2, 255, -1)
    cv2.circle(img_gray, (gt_x + 9, gt_y), 2, 255, -1)

    return img_gray


# [STRIPPED] def add_sem_noise_grayscale(image_uint8, seed=17001, is_target=False):
# [STRIPPED]     np.random.seed(seed)
# [STRIPPED]     img = image_uint8.astype(np.float32)
# [STRIPPED] 
# [STRIPPED]     if not is_target:
# [STRIPPED]         poisson = np.random.poisson(img / 255.0 * 12.0) / 12.0 * 255.0
# [STRIPPED]         gauss = np.random.normal(0, 18.0, img.shape)
# [STRIPPED]         speckle = np.random.randn(*img.shape) * 0.15
# [STRIPPED]         noisy = (poisson + gauss) * (1.0 + speckle)
# [STRIPPED]     else:
# [STRIPPED]         gauss = np.random.normal(0, 6.0, img.shape)
# [STRIPPED]         noisy = img + gauss
# [STRIPPED] 
# [STRIPPED]     return np.clip(noisy, 0, 255).astype(np.uint8)
# [STRIPPED] 
# [STRIPPED] 
def generate_stitchedwl_pair1(output_dir="generated_stitchedwl/pair1", gt_x=500, gt_y=500, seed=17001):
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    search_clean = render_stitchedwl_dram_scene(h=h, w=w, gt_x=gt_x, gt_y=gt_y, seed=seed)

    crop_size = 100
    x0 = max(0, min(w - crop_size, gt_x - crop_size // 2))
    y0 = max(0, min(h - crop_size, gt_y - crop_size // 2))
    x1, y1 = x0 + crop_size, y0 + crop_size

    crop_patch = search_clean[y0:y1, x0:x1]
    target_clean = cv2.resize(crop_patch, (1000, 1000), interpolation=cv2.INTER_NEAREST)

    search_out = search_clean  # noise stripped by v2 pipeline
    target_out = target_clean  # noise stripped by v2 pipeline

    gt_info = {
        "center_x": gt_x,
        "center_y": gt_y,
        "target_name": "Grayscale Single-Sided Stitched WL DRAM - Center GT 500,500 (Pair 1)",
        "pair_id": "pair1",
        "scale_factor": 10.0
    }

    with open(os.path.join(output_dir, "groundtruth.json"), "w") as f:
        json.dump(gt_info, f, indent=2)

    cv2.imwrite(os.path.join(output_dir, "search.png"), search_out)
    cv2.imwrite(os.path.join(output_dir, "target.png"), target_out)
    shutil.copyfile(os.path.join(output_dir, "target.png"), os.path.join(output_dir, "reference.png"))

    print(f"[generate_stitchedwl_pair1] Successfully generated Stitched WL DRAM pair 1 {output_dir} | GT: ({gt_x}, {gt_y})")
    return os.path.join(output_dir, "search.png"), os.path.join(output_dir, "target.png"), gt_info


if __name__ == "__main__":
    generate_stitchedwl_pair1()
