#!/usr/bin/env python3
"""
Generate Pair 1 for Grayscale Saddle-Fin DRAM (Center GT 500,500)
=================================================================
Synthesizes 1000x1000 Grayscale SEM image matching Saddle-Fin DRAM architecture:
  - Full-frame 1000x1000 IC layout
  - Hourglass bow-tie shaped Saddle-Fin channel geometry (pinched center under gate, flared at S/D)
  - Horizontal Wordlines (WL) & Vertical Bitlines (BL)
  - Embedded Double-Cell Box Landmark at Center (500, 500):
    Two adjacent saddle-fin cell boxes joined together directly in the grid (no big white mark!)
  - Memory-optimized uint8 pipeline with full SEM noise
"""

import json
import os
import shutil
import cv2
import numpy as np


def render_saddlefin_dram_scene(h=1000, w=1000, gt_x=500, gt_y=500, seed=14001):
    """Render 1000x1000 uint8 Grayscale image of Saddle-Fin DRAM Semiconductor Circuitry."""
    np.random.seed(seed)

    col_pitch = 24  # Bitline pitch
    bl_w = 6        # Bitline width
    row_pitch = 24  # Wordline pitch
    wl_h = 5        # Wordline width

    # Dark insulating substrate base
    img_gray = np.full((h, w), 32, dtype=np.uint8)

    # ─── 1. Vertical Bitlines (BL) ────────────────────────────────────
    for cx in range(col_pitch // 2, w, col_pitch):
        x1 = cx - bl_w // 2
        x2 = cx + bl_w // 2
        if 0 <= x1 and x2 < w:
            img_gray[:, x1:x2] = 105
            img_gray[:, x1] = 145
            img_gray[:, x2 - 1] = 145

    # ─── 2. Horizontal Wordlines (WL) ──────────────────────────────────
    for wy in range(row_pitch // 2, h, row_pitch):
        y1 = wy - wl_h // 2
        y2 = wy + wl_h // 2 + 1
        if 0 <= y1 and y2 < h:
            img_gray[y1:y2, :] = np.maximum(img_gray[y1:y2, :], 165)
            img_gray[y1, :] = 195
            img_gray[y2 - 1, :] = 195

    # ─── 3. Saddle-Fin Transistor Cells (Hourglass Channel + S/D Pads) ─
    for cx in range(col_pitch // 2, w, col_pitch):
        for wy in range(row_pitch // 2, h, row_pitch):
            cv2.rectangle(img_gray, (cx - 3, wy - 2), (cx + 3, wy + 2), 175, -1)
            cv2.rectangle(img_gray, (cx - 8, wy - 4), (cx - 3, wy + 4), 185, -1)
            cv2.rectangle(img_gray, (cx + 3, wy - 4), (cx + 8, wy + 4), 185, -1)

            sx1, sx2 = cx - 10, cx - 5
            if sx1 >= 0:
                cv2.rectangle(img_gray, (sx1, wy - 4), (sx2, wy + 4), 220, -1)
                cv2.rectangle(img_gray, (sx1, wy - 4), (sx2, wy + 4), 240, 1)
                cv2.circle(img_gray, (cx - 7, wy), 1, 250, -1)

            dx1, dx2 = cx + 5, cx + 10
            if dx2 < w:
                cv2.rectangle(img_gray, (dx1, wy - 4), (dx2, wy + 4), 220, -1)
                cv2.rectangle(img_gray, (dx1, wy - 4), (dx2, wy + 4), 240, 1)
                cv2.circle(img_gray, (cx + 7, wy), 1, 250, -1)

            cv2.circle(img_gray, (cx, wy), 1, 245, -1)

    # ─── 4. Sub-Array Logic Bands ──────────────────────────────────────
    sense_interval = row_pitch * 8  # 192px
    for sy in range(sense_interval, h - 10, sense_interval):
        cv2.rectangle(img_gray, (0, sy - 3), (w, sy + 3), 60, -1)
        cv2.line(img_gray, (0, sy - 4), (w, sy - 4), 140, 1)
        cv2.line(img_gray, (0, sy + 4), (w, sy + 4), 140, 1)

    # ─── 5. Embedded Double-Cell Box Landmark Centered at (gt_x, gt_y) ─
    cv2.rectangle(img_gray, (gt_x - 25, gt_y - 7), (gt_x + 25, gt_y + 7), 20, -1)
    cv2.line(img_gray, (gt_x - 24, gt_y), (gt_x + 24, gt_y), 185, 3)
    cv2.rectangle(img_gray, (gt_x - 22, gt_y - 4), (gt_x + 22, gt_y + 4), 225, -1)
    cv2.rectangle(img_gray, (gt_x - 22, gt_y - 4), (gt_x + 22, gt_y + 4), 245, 1)
    cv2.circle(img_gray, (gt_x - 12, gt_y), 2, 255, -1)
    cv2.circle(img_gray, (gt_x + 12, gt_y), 2, 255, -1)

    return img_gray


# [STRIPPED] def add_sem_noise_grayscale(image_uint8, seed=14001, is_target=False):
# [STRIPPED]     np.random.seed(seed)
# [STRIPPED]     std = 10 if not is_target else 4
# [STRIPPED]     noise = np.random.randint(-std, std + 1, image_uint8.shape, dtype=np.int16)
# [STRIPPED]     noisy = np.clip(image_uint8.astype(np.int16) + noise, 0, 255).astype(np.uint8)
# [STRIPPED]     return noisy
# [STRIPPED] 
# [STRIPPED] 
def generate_saddlefin_pair1(output_dir="generated_saddlefin/pair1", gt_x=500, gt_y=500, seed=14001):
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    search_clean = render_saddlefin_dram_scene(h=h, w=w, gt_x=gt_x, gt_y=gt_y, seed=seed)

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
        "target_name": "Grayscale Saddle-Fin DRAM - Center GT 500,500 (Pair 1)",
        "pair_id": "pair1",
        "scale_factor": 10.0
    }

    with open(os.path.join(output_dir, "groundtruth.json"), "w") as f:
        json.dump(gt_info, f, indent=2)

    cv2.imwrite(os.path.join(output_dir, "search.png"), search_out)
    cv2.imwrite(os.path.join(output_dir, "target.png"), target_out)
    shutil.copyfile(os.path.join(output_dir, "target.png"), os.path.join(output_dir, "reference.png"))

    print(f"[generate_saddlefin_pair1] Successfully generated Saddle-Fin DRAM pair 1 {output_dir} | GT: ({gt_x}, {gt_y})")
    return os.path.join(output_dir, "search.png"), os.path.join(output_dir, "target.png"), gt_info


if __name__ == "__main__":
    generate_saddlefin_pair1()
