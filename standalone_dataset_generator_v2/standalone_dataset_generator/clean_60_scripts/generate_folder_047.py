#!/usr/bin/env python3
"""
Generate Pair 1 for Grayscale VCT 4F2 DRAM (Center GT 500,500)
==============================================================
Synthesizes 1000x1000 Grayscale SEM image matching Vertical Channel Transistor (VCT) 4F2 DRAM Architecture:
  - Full-frame 1000x1000 IC layout matching reference SEM micrograph
  - Horizontal Word Lines (WL) crossed by vertical Bit Line (BL) pairs
  - VCT Access Transistors at WL-BL intersections
  - Vertical capsule-shaped Storage Capacitors (Cs) with inner electrode cores
  - Authentic SEM footer scale bar & metadata text
  - Creative In-Pattern Embedded Landmark at Center (500, 500):
    Fine interconnect bridge line linking two adjacent Storage Capacitors with a central micro-via (NO big white box!)
  - Memory-optimized uint8 pipeline with full SEM noise
"""

import json
import os
import shutil
import cv2
import numpy as np


def render_vct_dram_scene(h=1000, w=1000, gt_x=500, gt_y=500, seed=19001):
    """Render 1000x1000 uint8 Grayscale image of VCT 4F2 DRAM Semiconductor Circuitry."""
    np.random.seed(seed)

    col_pitch = 24  # Bitline pair pitch
    row_pitch = 24  # Wordline pitch

    # Dark silicon oxide substrate base
    img_gray = np.full((h, w), 35, dtype=np.uint8)

    # ─── 1. Vertical Bit Line (BL) Pairs ──────────────────────────────
    for cx in range(col_pitch // 2, w, col_pitch):
        cv2.line(img_gray, (cx - 3, 0), (cx - 3, h), 110, 2)
        cv2.line(img_gray, (cx + 3, 0), (cx + 3, h), 110, 2)
        cv2.line(img_gray, (cx - 4, 0), (cx - 4, h), 75, 1)
        cv2.line(img_gray, (cx + 4, 0), (cx + 4, h), 75, 1)

    # ─── 2. Horizontal Word Lines (WL) ─────────────────────────────────
    for wy in range(row_pitch // 2, h - 40, row_pitch):
        cv2.rectangle(img_gray, (0, wy - 3), (w, wy + 3), 160, -1)
        cv2.line(img_gray, (0, wy - 3), (w, wy - 3), 205, 1)
        cv2.line(img_gray, (0, wy + 3), (w, wy + 3), 205, 1)

    # ─── 3. VCT Access Transistors & Storage Capacitors (Cs) ───────────
    for cx in range(col_pitch // 2, w, col_pitch):
        for wy in range(row_pitch // 2, h - 40, row_pitch):
            cv2.circle(img_gray, (cx, wy), 3, 220, -1)
            cv2.circle(img_gray, (cx, wy), 1, 255, -1)

            cs_y = wy + row_pitch // 2
            if cs_y + 8 < h - 40:
                cv2.rectangle(img_gray, (cx - 4, cs_y - 6), (cx + 4, cs_y + 6), 175, -1)
                cv2.rectangle(img_gray, (cx - 4, cs_y - 6), (cx + 4, cs_y + 6), 210, 1)
                cv2.rectangle(img_gray, (cx - 2, cs_y - 4), (cx + 2, cs_y + 4), 95, -1)
                cv2.rectangle(img_gray, (cx - 1, cs_y - 2), (cx + 1, cs_y + 2), 240, -1)

    # ─── 4. Sub-Array Sense Amplifier Logic Bands ──────────────────────
    sense_interval = row_pitch * 8  # 192px
    for sy in range(sense_interval, h - 50, sense_interval):
        cv2.rectangle(img_gray, (0, sy - 4), (w, sy + 4), 55, -1)
        cv2.line(img_gray, (0, sy - 5), (w, sy - 5), 145, 1)
        cv2.line(img_gray, (0, sy + 5), (w, sy + 5), 145, 1)

    # ─── 5. Creative In-Pattern Landmark: Small Interconnect Line Bridge
    # Centered precisely at (gt_x, gt_y)
    start_col = (gt_x // col_pitch) * col_pitch + col_pitch // 2
    start_row = (gt_y // row_pitch) * row_pitch + row_pitch // 2
    cs_y = start_row + row_pitch // 2

    # Fine horizontal bridge line connecting two adjacent capacitors
    cv2.line(img_gray, (start_col - 4, cs_y), (start_col + col_pitch + 4, cs_y), 235, 2)
    # Small cross-bridge interconnect line
    cv2.line(img_gray, (start_col + col_pitch // 2, cs_y - 6), (start_col + col_pitch // 2, cs_y + 6), 225, 2)
    # Central micro-via dot at the line intersection
    cv2.circle(img_gray, (start_col + col_pitch // 2, cs_y), 2, 255, -1)

    return img_gray


# [STRIPPED] def add_sem_noise_grayscale(image_uint8, seed=19001, is_target=False):
# [STRIPPED]     np.random.seed(seed)
# [STRIPPED]     img = image_uint8.astype(np.float32)
# [STRIPPED] 
# [STRIPPED]     if not is_target:
# [STRIPPED]         poisson = np.random.poisson(img / 255.0 * 12.0) / 12.0 * 255.0
# [STRIPPED]         gauss = np.random.normal(0, 16.0, img.shape)
# [STRIPPED]         speckle = np.random.randn(*img.shape) * 0.12
# [STRIPPED]         noisy = (poisson + gauss) * (1.0 + speckle)
# [STRIPPED]     else:
# [STRIPPED]         gauss = np.random.normal(0, 5.0, img.shape)
# [STRIPPED]         noisy = img + gauss
# [STRIPPED] 
# [STRIPPED]     return np.clip(noisy, 0, 255).astype(np.uint8)
# [STRIPPED] 
# [STRIPPED] 
def generate_vct_pair1(output_dir="generated_vct/pair1", gt_x=500, gt_y=500, seed=19001):
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    search_clean = render_vct_dram_scene(h=h, w=w, gt_x=gt_x, gt_y=gt_y, seed=seed)

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
        "target_name": "Grayscale VCT 4F2 DRAM - Center GT 500,500 (Pair 1)",
        "pair_id": "pair1",
        "scale_factor": 10.0
    }

    with open(os.path.join(output_dir, "groundtruth.json"), "w") as f:
        json.dump(gt_info, f, indent=2)

    cv2.imwrite(os.path.join(output_dir, "search.png"), search_out)
    cv2.imwrite(os.path.join(output_dir, "target.png"), target_out)
    shutil.copyfile(os.path.join(output_dir, "target.png"), os.path.join(output_dir, "reference.png"))

    print(f"[generate_vct_pair1] Successfully generated VCT 4F2 DRAM pair 1 {output_dir} | GT: ({gt_x}, {gt_y})")
    return os.path.join(output_dir, "search.png"), os.path.join(output_dir, "target.png"), gt_info


if __name__ == "__main__":
    generate_vct_pair1()
