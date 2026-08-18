#!/usr/bin/env python3
"""
Generate Pair 2 for Grayscale 3D Capacitorless DRAM (Top GT 500,310)
====================================================================
Synthesizes 1000x1000 Grayscale SEM image matching 3D Capacitorless DRAM architecture:
  - Full-frame 1000x1000 IC layout with HIGH-DENSITY 3D vertical floating body pillars (pitch 18px x 18px)
  - Gate-All-Around (GAA) wordline sheets & 3D vertical channel cores
  - Embedded Double-Pillar Landmark at Top (500, 310):
    Two adjacent 3D channel pillars joined together directly in the grid (no big white mark!)
  - Memory-optimized uint8 pipeline with full SEM noise
"""

import json
import os
import shutil
import cv2
import numpy as np


def render_3dcdram_scene(h=1000, w=1000, gt_x=500, gt_y=310, seed=16002):
    """Render 1000x1000 uint8 Grayscale image of 3D Capacitorless DRAM Circuitry."""
    np.random.seed(seed)

    col_pitch = 18  # 3D pillar column pitch
    row_pitch = 18  # 3D pillar row pitch

    # Dark silicon substrate base
    img_gray = np.full((h, w), 30, dtype=np.uint8)

    # ─── 1. Vertical Bitline Bus Tracks ───────────────────────────────
    for bx in range(col_pitch // 2, w, col_pitch):
        cv2.line(img_gray, (bx, 0), (bx, h), 105, 1)

    # ─── 2. Horizontal GAA Wordline Plane Sheets ───────────────────────
    for ry in range(row_pitch // 2, h, row_pitch):
        cv2.line(img_gray, (0, ry), (w, ry), 155, 2)
        cv2.line(img_gray, (0, ry - 1), (w, ry - 1), 185, 1)

    # ─── 3. Dense 3D Vertical Floating Body Pillars ───────────────────
    for cx in range(col_pitch // 2, w, col_pitch):
        for ry in range(row_pitch // 2, h, row_pitch):
            cv2.circle(img_gray, (cx, ry), 6, 175, -1)  # Outer GAA wordline gate ring
            cv2.circle(img_gray, (cx, ry), 4, 115, -1)  # Gate oxide dielectric interface
            cv2.circle(img_gray, (cx, ry), 2, 225, -1)  # Vertical silicon floating body channel core
            cv2.circle(img_gray, (cx, ry), 1, 255, -1)  # Center bitline via dot

            # Inter-pillar S/D contact pads
            sd_y = ry + row_pitch // 2
            if sd_y + 1 < h:
                cv2.rectangle(img_gray, (cx - 2, sd_y - 1), (cx + 2, sd_y + 1), 200, -1)

    # ─── 4. Sub-Array Logic Bands ──────────────────────────────────────
    sense_interval = row_pitch * 8  # 144px
    for sy in range(sense_interval, h - 10, sense_interval):
        cv2.rectangle(img_gray, (0, sy - 3), (w, sy + 3), 55, -1)
        cv2.line(img_gray, (0, sy - 4), (w, sy - 4), 140, 1)
        cv2.line(img_gray, (0, sy + 4), (w, sy + 4), 140, 1)

    # ─── 5. Embedded Double-Pillar Landmark Centered at (gt_x, gt_y) ──
    cv2.rectangle(img_gray, (gt_x - 18, gt_y - 6), (gt_x + 18, gt_y + 6), 20, -1)
    cv2.line(img_gray, (gt_x - 17, gt_y), (gt_x + 17, gt_y), 180, 3)
    cv2.rectangle(img_gray, (gt_x - 15, gt_y - 4), (gt_x + 15, gt_y + 4), 225, -1)
    cv2.rectangle(img_gray, (gt_x - 15, gt_y - 4), (gt_x + 15, gt_y + 4), 245, 1)
    cv2.circle(img_gray, (gt_x - 9, gt_y), 2, 255, -1)
    cv2.circle(img_gray, (gt_x + 9, gt_y), 2, 255, -1)

    return img_gray


# [STRIPPED] def add_sem_noise_grayscale(image_uint8, seed=16002, is_target=False):
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
def generate_3dcdram_pair2(output_dir="generated_3dcdram/pair2", gt_x=500, gt_y=310, seed=16002):
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    search_clean = render_3dcdram_scene(h=h, w=w, gt_x=gt_x, gt_y=gt_y, seed=seed)

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
        "target_name": "Grayscale 3D Capacitorless DRAM - Top GT 500,310 (Pair 2)",
        "pair_id": "pair2",
        "scale_factor": 10.0
    }

    with open(os.path.join(output_dir, "groundtruth.json"), "w") as f:
        json.dump(gt_info, f, indent=2)

    cv2.imwrite(os.path.join(output_dir, "search.png"), search_out)
    cv2.imwrite(os.path.join(output_dir, "target.png"), target_out)
    shutil.copyfile(os.path.join(output_dir, "target.png"), os.path.join(output_dir, "reference.png"))

    print(f"[generate_3dcdram_pair2] Successfully generated 3D Capacitorless DRAM pair 2 {output_dir} | GT: ({gt_x}, {gt_y})")
    return os.path.join(output_dir, "search.png"), os.path.join(output_dir, "target.png"), gt_info


if __name__ == "__main__":
    generate_3dcdram_pair2()
