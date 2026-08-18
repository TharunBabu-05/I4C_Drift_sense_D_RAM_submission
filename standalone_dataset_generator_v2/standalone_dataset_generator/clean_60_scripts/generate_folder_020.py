#!/usr/bin/env python3
"""
Generate Pair 2 for generated_new (Grayscale Dense Hexagonal Cell DRAM - Top GT 500,310)
============================================================================================
- Full-frame 1000x1000 dense Hexagonal Cell DRAM IC layout (NO external marks)
- Denser hex cell columns, bitlines, staggered 1T1C channels
- Heavy SEM noise (std=20 search, std=8 target)
- Unique landmark: a single thin inter-cell routing trace connecting the via contacts of two adjacent hex cells
  This is FULLY embedded in the design - same intensity as existing metal traces, not a filled patch
"""

import json
import os
import shutil
import cv2
import numpy as np


def draw_hexagon(img, cx, cy, radius, color, thickness=1):
    pts = []
    for i in range(6):
        a = np.pi / 3.0 * i - np.pi / 6.0
        pts.append([int(cx + radius * np.cos(a)), int(cy + radius * np.sin(a))])
    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
    if thickness == -1:
        cv2.fillPoly(img, [pts], color)
    else:
        cv2.polylines(img, [pts], True, color, thickness)


def render_hexagonal_dram_scene(h=1000, w=1000, gt_x=500, gt_y=310, seed=4002):
    np.random.seed(seed)
    img = np.full((h, w), 28, dtype=np.float32)

    col_pitch = 16
    row_pitch = 12
    hex_r     = 5
    via_r     = 2

    for cx in range(8, w - 8, col_pitch):
        col_type = (cx // col_pitch) % 3

        if col_type == 0:
            for cy in range(8, h - 8, row_pitch):
                draw_hexagon(img, cx, cy, hex_r, 155.0, thickness=1)
                draw_hexagon(img, cx, cy, via_r, 175.0, thickness=1)
                cv2.circle(img, (cx, cy), 1, 210.0, -1)

        elif col_type == 1:
            cv2.line(img, (cx - 2, 0), (cx - 2, h), 125.0, 1)
            cv2.line(img, (cx + 2, 0), (cx + 2, h), 125.0, 1)
            cv2.line(img, (cx, 0), (cx, h), 165.0, 2)
            for cy in range(6, h - 6, 12):
                cv2.rectangle(img, (cx - 3, cy), (cx + 3, cy + 2), 100.0, -1)

        else:
            offset = row_pitch // 2
            for cy in range(8 + offset, h - 8, row_pitch):
                draw_hexagon(img, cx, cy, hex_r - 1, 140.0, thickness=1)
                cv2.circle(img, (cx, cy), 1, 185.0, -1)

    for hy in range(20, h - 20, 40):
        cv2.line(img, (0, hy), (w, hy), 110.0, 2)
        for tx in range(8, w - 8, 32):
            cv2.rectangle(img, (tx - 2, hy - 2), (tx + 2, hy + 2), 180.0, -1)

    # --- Distinct High-Contrast Hexagonal DRAM Landmark Centered at (gt_x, gt_y) ---
    draw_hexagon(img, gt_x, gt_y, 10, 250.0, thickness=2)
    draw_hexagon(img, gt_x, gt_y, 5, 230.0, thickness=2)
    cv2.circle(img, (gt_x, gt_y), 4, 255.0, -1)
    cv2.line(img, (gt_x - 18, gt_y), (gt_x + 18, gt_y), 245.0, 2)
    cv2.line(img, (gt_x, gt_y - 18), (gt_x, gt_y + 18), 245.0, 2)

    return np.clip(img, 0, 255).astype(np.uint8)


# [STRIPPED] def add_heavy_sem_noise(image_uint8, seed=4002, is_target=False):
# [STRIPPED]     np.random.seed(seed)
# [STRIPPED]     std = 20 if not is_target else 8
# [STRIPPED]     noise = np.random.randint(-std, std + 1, image_uint8.shape, dtype=np.int16)
# [STRIPPED]     return np.clip(image_uint8.astype(np.int16) + noise, 0, 255).astype(np.uint8)
# [STRIPPED] 
# [STRIPPED] 
def generate_new_pair2(output_dir="generated_new/pair2", gt_x=500, gt_y=310, seed=4002):
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    search_clean = render_hexagonal_dram_scene(h=h, w=w, gt_x=gt_x, gt_y=gt_y, seed=seed)

    # Anchor crop to exact trace midpoint
    col_pitch_l, row_pitch_l = 16, 12
    col_idx = round((gt_x - 8) / col_pitch_l)
    if col_idx % 3 != 0:
        col_idx = (col_idx // 3) * 3
    row_idx = round((gt_y - 8) / row_pitch_l)
    anchor_x = 8 + col_idx * col_pitch_l
    anchor_y = 8 + row_idx * row_pitch_l + row_pitch_l // 2

    crop_size = 100
    x0 = max(0, min(w - crop_size, anchor_x - crop_size // 2))
    y0 = max(0, min(h - crop_size, anchor_y - crop_size // 2))
    crop_patch   = search_clean[y0:y0 + crop_size, x0:x0 + crop_size]
    target_clean = cv2.resize(crop_patch, (1000, 1000), interpolation=cv2.INTER_NEAREST)

    search_noisy = search_clean  # noise stripped by v2 pipeline
    target_noisy = target_clean  # noise stripped by v2 pipeline

    gt_info = {
        "center_x": anchor_x, "center_y": anchor_y,
        "target_name": "Grayscale Dense Hexagonal Cell DRAM (Embedded Inter-Cell Trace) - Top GT 500,310 (Pair 2)",
        "pair_id": "pair2", "scale_factor": 10.0
    }
    with open(os.path.join(output_dir, "groundtruth.json"), "w") as f:
        json.dump(gt_info, f, indent=2)

    cv2.imwrite(os.path.join(output_dir, "search.png"), search_noisy)
    cv2.imwrite(os.path.join(output_dir, "target.png"), target_noisy)
    shutil.copyfile(os.path.join(output_dir, "target.png"), os.path.join(output_dir, "reference.png"))

    for sub in ["generated_wafer/pair2"]:
        os.makedirs(sub, exist_ok=True)
        with open(os.path.join(sub, "groundtruth.json"), "w") as f:
            json.dump(gt_info, f, indent=2)
        cv2.imwrite(os.path.join(sub, "search.png"), search_noisy)
        cv2.imwrite(os.path.join(sub, "target.png"), target_noisy)
        shutil.copyfile(os.path.join(sub, "target.png"), os.path.join(sub, "reference.png"))

    print(f"[pair2] Done | GT: ({gt_x}, {gt_y})")
    return os.path.join(output_dir, "search.png"), os.path.join(output_dir, "target.png"), gt_info


if __name__ == "__main__":
    generate_new_pair2()
