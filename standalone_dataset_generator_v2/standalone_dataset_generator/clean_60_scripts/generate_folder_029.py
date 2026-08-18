#!/usr/bin/env python3
"""
Generate Pair 1 for generated_1tdram (Grayscale Capacitorless 1T DRAM - Center GT 500,500)
==========================================================================================
Synthesizes 1000x1000 Grayscale normal image matching SEM micro-photo of 1T DRAM IC structure:
  - Full-frame 1000x1000 IC layout
  - Medium-density single-gate planar transistor array (wide silicon tracks, single horizontal wordline)
  - Large square Source/Drain contact pads
  - Seamless Ground Truth Target Landmark at Center (500, 500): Cluster of adjacent 1T DRAM cells
  - Memory-efficient SEM noise pipeline
"""

import json
import os
import shutil
import cv2
import numpy as np


def render_1tdram_scene(h=1000, w=1000, gt_x=500, gt_y=500, seed=10001):
    """Render 1000x1000 uint8 Grayscale image of Capacitorless 1T DRAM Semiconductor Circuitry.

    1T DRAM key features:
      - Single-transistor cell with planar floating body
      - Wide silicon active tracks (width 8px, pitch 28px)
      - Single horizontal Wordline gate line per row (width 6px, pitch 24px)
      - Large square Source/Drain contact pads (6x6 px) between wordlines
    """
    np.random.seed(seed)

    col_pitch = 28  # Active silicon fin pitch (wide spacing)
    fin_w = 8       # Active fin width
    row_pitch = 24  # Wordline gate pitch (wide spacing)
    gate_h = 6      # Wordline gate width

    # Dark STI substrate base
    img_gray = np.full((h, w), 32, dtype=np.float32)

    # ─── 1. Active Silicon SOI Fins (Wide Vertical Tracks) ────────────
    for cx in range(col_pitch // 2, w, col_pitch):
        x1 = cx - fin_w // 2
        x2 = cx + fin_w // 2
        if 0 <= x1 and x2 < w:
            img_gray[:, x1:x2] = 95.0
            img_gray[:, x1] = 135.0
            img_gray[:, x2 - 1] = 135.0

    # ─── 2. Single Polysilicon Wordline Gate Stripes (Horizontal) ──────
    for wy in range(row_pitch // 2, h, row_pitch):
        y1 = wy - gate_h // 2
        y2 = wy + gate_h // 2 + 1
        if 0 <= y1 and y2 < h:
            img_gray[y1:y2, :] = np.maximum(img_gray[y1:y2, :], 160.0)
            img_gray[y1, :] = 190.0
            img_gray[y2 - 1, :] = 190.0

    # ─── 3. Single Transistor Channels & Large Square S/D Contacts ─────
    for cx in range(col_pitch // 2, w, col_pitch):
        x1 = cx - fin_w // 2
        x2 = cx + fin_w // 2
        if x1 < 0 or x2 >= w:
            continue

        for wy in range(row_pitch // 2, h, row_pitch):
            y1 = wy - gate_h // 2
            y2 = wy + gate_h // 2 + 1

            # Floating body region
            cv2.rectangle(img_gray, (x1 + 1, y1 + 1), (x2 - 2, y2 - 2), 125.0, -1)
            # Front gate contact via center dot
            cv2.circle(img_gray, (cx, wy), 1, 230.0, -1)

            # Large square Source / Drain contact pad
            sd_y = wy + row_pitch // 2
            if sd_y + 2 < h:
                cv2.rectangle(img_gray, (cx - 3, sd_y - 3), (cx + 3, sd_y + 3), 210.0, -1)

    # ─── 4. Peripheral Control Lines ──────────────────────────────────
    sense_interval = row_pitch * 6  # 144px
    for sy in range(sense_interval, h - 10, sense_interval):
        cv2.rectangle(img_gray, (0, sy - 3), (w, sy + 3), 60.0, -1)
        cv2.line(img_gray, (0, sy - 4), (w, sy - 4), 140.0, 1)
        cv2.line(img_gray, (0, sy + 4), (w, sy + 4), 140.0, 1)

    # ─── 5. Seamless In-Pattern Landmark at (gt_x, gt_y) ───────────────
    start_col = (gt_x // col_pitch) * col_pitch + col_pitch // 2
    start_row = (gt_y // row_pitch) * row_pitch + row_pitch // 2

    for dc in range(3):
        for dr in range(4):
            cur_x = start_col + dc * col_pitch
            cur_y = start_row + dr * row_pitch

            if (cur_x - fin_w // 2 >= 0 and cur_x + fin_w // 2 < w and
                    cur_y - gate_h // 2 >= 0 and cur_y + gate_h // 2 < h):

                x1 = cur_x - fin_w // 2
                x2 = cur_x + fin_w // 2
                y1 = cur_y - gate_h // 2
                y2 = cur_y + gate_h // 2 + 1

                cv2.rectangle(img_gray, (x1, y1), (x2 - 1, y2 - 1), 215.0, 1)
                cv2.circle(img_gray, (cur_x, cur_y), 1, 255.0, -1)

                if dc < 2:
                    next_x = cur_x + col_pitch
                    cv2.line(img_gray, (x2, cur_y), (next_x - fin_w // 2, cur_y), 235.0, 2)

                if dr < 3:
                    next_y = cur_y + row_pitch
                    cv2.line(img_gray, (cur_x, y2), (cur_x, next_y - gate_h // 2), 235.0, 2)

    return np.clip(img_gray, 0, 255).astype(np.uint8)


def apply_gaussian_noise(img_f32, sigma=10.0):
    noise = np.random.normal(0, sigma, img_f32.shape).astype(np.float32)
    return img_f32 + noise


def apply_poisson_noise(img_f32, scale=7.0):
    counts = np.maximum(0, img_f32) / 255.0 * scale
    noisy_counts = np.random.poisson(counts).astype(np.float32)
    return noisy_counts / scale * 255.0


def apply_speckle_noise(img_f32, intensity=0.15):
    speckle = np.random.randn(*img_f32.shape).astype(np.float32) * intensity
    return img_f32 * (1.0 + speckle)


def apply_vignette(img_f32, strength=0.25):
    h, w = img_f32.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    max_r = np.sqrt(cx ** 2 + cy ** 2)
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)
    vignette_map = 1.0 - strength * (r / max_r) ** 2
    return img_f32 * vignette_map


# [STRIPPED] def add_sem_noise_grayscale(image_uint8, seed=10001, is_target=False):
# [STRIPPED]     """Noise pipeline for SEM image."""
# [STRIPPED]     np.random.seed(seed)
# [STRIPPED]     img = image_uint8.astype(np.float32)
# [STRIPPED] 
# [STRIPPED]     if not is_target:
# [STRIPPED]         img = apply_poisson_noise(img, scale=7.0)
# [STRIPPED]         img = apply_gaussian_noise(img, sigma=10.0)
# [STRIPPED]         img = apply_speckle_noise(img, intensity=0.15)
# [STRIPPED]         img = apply_vignette(img, strength=0.25)
# [STRIPPED]         img = np.clip(img, 0, 255).astype(np.uint8)
# [STRIPPED]     else:
# [STRIPPED]         img = apply_gaussian_noise(img, sigma=4.0)
# [STRIPPED]         img = apply_speckle_noise(img, intensity=0.05)
# [STRIPPED]         img = apply_vignette(img, strength=0.10)
# [STRIPPED]         img = np.clip(img, 0, 255).astype(np.uint8)
# [STRIPPED] 
# [STRIPPED]     return np.clip(img, 0, 255).astype(np.uint8) if isinstance(img, np.floating) else img
# [STRIPPED] 
# [STRIPPED] 
def generate_1tdram_pair1(output_dir="generated_1tdram/pair1", gt_x=500, gt_y=500, seed=10001):
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    search_clean = render_1tdram_scene(h=h, w=w, gt_x=gt_x, gt_y=gt_y, seed=seed)

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
        "target_name": "Grayscale 1T DRAM - Center GT 500,500 (Pair 1)",
        "pair_id": "pair1",
        "scale_factor": 10.0
    }

    with open(os.path.join(output_dir, "groundtruth.json"), "w") as f:
        json.dump(gt_info, f, indent=2)

    cv2.imwrite(os.path.join(output_dir, "search.png"), search_out)
    cv2.imwrite(os.path.join(output_dir, "target.png"), target_out)
    shutil.copyfile(os.path.join(output_dir, "target.png"), os.path.join(output_dir, "reference.png"))

    print(f"[generate_1tdram_pair1] Successfully generated 1T DRAM pair 1 {output_dir} | GT: ({gt_x}, {gt_y})")
    return os.path.join(output_dir, "search.png"), os.path.join(output_dir, "target.png"), gt_info


if __name__ == "__main__":
    generate_1tdram_pair1()
