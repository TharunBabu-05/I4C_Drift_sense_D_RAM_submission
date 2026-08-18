#!/usr/bin/env python3
"""
Generate Pair 2 for Grayscale GIDL-based Capacitorless DRAM (Top GT 500,310)
==============================================================================
Synthesizes 1000x1000 Grayscale SEM image matching GIDL (Gate-Induced Drain Leakage) 1T1D Cell DRAM:
  - Full-frame 1000x1000 IC layout
  - 1T1D GIDL cell architecture: Central Gate flanked by Source (S) and Drain (D) pads
  - Horizontal Wordlines (WL) crossed by Vertical Bitlines (BL)
  - Subtle In-Pattern Embedded Landmark at Top (500, 310):
    Two adjacent cell boxes joined into a double-cell box directly in the grid (no big white mark!)
  - Full SEM noise pipeline (no barrel distortion)
"""

import json
import os
import shutil
import cv2
import numpy as np


def render_gidl_dram_scene(h=1000, w=1000, gt_x=500, gt_y=310, seed=13002):
    """Render 1000x1000 uint8 Grayscale image of GIDL-based Capacitorless DRAM Semiconductor Circuitry."""
    np.random.seed(seed)

    col_pitch = 26  # Bitline pitch
    bl_w = 6        # Bitline width
    row_pitch = 26  # Wordline pitch
    wl_h = 6        # Wordline width

    # Dark insulating substrate base
    img_gray = np.full((h, w), 32, dtype=np.float32)

    # ─── 1. Vertical Bitlines (BL) ────────────────────────────────────
    for cx in range(col_pitch // 2, w, col_pitch):
        x1 = cx - bl_w // 2
        x2 = cx + bl_w // 2
        if 0 <= x1 and x2 < w:
            img_gray[:, x1:x2] = 105.0
            img_gray[:, x1] = 145.0
            img_gray[:, x2 - 1] = 145.0

    # ─── 2. Horizontal Wordlines (WL) ──────────────────────────────────
    for wy in range(row_pitch // 2, h, row_pitch):
        y1 = wy - wl_h // 2
        y2 = wy + wl_h // 2 + 1
        if 0 <= y1 and y2 < h:
            img_gray[y1:y2, :] = np.maximum(img_gray[y1:y2, :], 160.0)
            img_gray[y1, :] = 190.0
            img_gray[y2 - 1, :] = 190.0

    # ─── 3. 1T1D GIDL Transistor Cells (Source, Gate, Drain) ───────────
    for cx in range(col_pitch // 2, w, col_pitch):
        for wy in range(row_pitch // 2, h, row_pitch):
            # Active Silicon Channel Trace
            cv2.line(img_gray, (cx - 10, wy), (cx + 10, wy), 165.0, 2)

            # Central Gate block
            cv2.rectangle(img_gray, (cx - 3, wy - 3), (cx + 3, wy + 3), 210.0, -1)
            cv2.circle(img_gray, (cx, wy), 1, 245.0, -1)

            # Source (S) Pad
            sx1, sx2 = cx - 11, cx - 5
            if sx1 >= 0:
                cv2.rectangle(img_gray, (sx1, wy - 4), (sx2, wy + 4), 225.0, -1)
                cv2.rectangle(img_gray, (sx1, wy - 4), (sx2, wy + 4), 245.0, 1)
                cv2.circle(img_gray, (cx - 8, wy), 1, 255.0, -1)

            # Drain (D) Pad
            dx1, dx2 = cx + 5, cx + 11
            if dx2 < w:
                cv2.rectangle(img_gray, (dx1, wy - 4), (dx2, wy + 4), 225.0, -1)
                cv2.rectangle(img_gray, (dx1, wy - 4), (dx2, wy + 4), 245.0, 1)
                cv2.circle(img_gray, (cx + 8, wy), 1, 255.0, -1)

    # ─── 4. Sub-Array & Driver Logic Bands ─────────────────────────────
    sense_interval = row_pitch * 8  # 208px
    for sy in range(sense_interval, h - 10, sense_interval):
        cv2.rectangle(img_gray, (0, sy - 3), (w, sy + 3), 60.0, -1)
        cv2.line(img_gray, (0, sy - 4), (w, sy - 4), 140.0, 1)
        cv2.line(img_gray, (0, sy + 4), (w, sy + 4), 140.0, 1)

    # ─── 5. Embedded Double-Cell Box Landmark at (gt_x, gt_y) ─────────
    start_col = (gt_x // col_pitch) * col_pitch + col_pitch // 2
    start_row = (gt_y // row_pitch) * row_pitch + row_pitch // 2

    # Dark oxide recessed border around joined double cell
    cv2.rectangle(img_gray, (start_col - 13, start_row - 6), (start_col + col_pitch + 13, start_row + 6), 20.0, -1)
    # Joined active silicon channel trace
    cv2.line(img_gray, (start_col - 12, start_row), (start_col + col_pitch + 12, start_row), 185.0, 3)
    # Joined double cell box (no separating line between two adjacent cell boxes)
    cv2.rectangle(img_gray, (start_col - 11, start_row - 4), (start_col + col_pitch + 11, start_row + 4), 225.0, -1)
    cv2.rectangle(img_gray, (start_col - 11, start_row - 4), (start_col + col_pitch + 11, start_row + 4), 245.0, 1)
    # Inner gate and via contacts
    cv2.circle(img_gray, (start_col, start_row), 2, 255.0, -1)
    cv2.circle(img_gray, (start_col + col_pitch, start_row), 2, 255.0, -1)

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


# [STRIPPED] def add_sem_noise_grayscale(image_uint8, seed=13002, is_target=False):
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
def generate_gidl_pair2(output_dir="generated_gidl/pair2", gt_x=500, gt_y=310, seed=13002):
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    search_clean = render_gidl_dram_scene(h=h, w=w, gt_x=gt_x, gt_y=gt_y, seed=seed)

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
        "target_name": "Grayscale GIDL-based Capacitorless DRAM - Top GT 500,310 (Pair 2)",
        "pair_id": "pair2",
        "scale_factor": 10.0
    }

    with open(os.path.join(output_dir, "groundtruth.json"), "w") as f:
        json.dump(gt_info, f, indent=2)

    cv2.imwrite(os.path.join(output_dir, "search.png"), search_out)
    cv2.imwrite(os.path.join(output_dir, "target.png"), target_out)
    shutil.copyfile(os.path.join(output_dir, "target.png"), os.path.join(output_dir, "reference.png"))

    print(f"[generate_gidl_pair2] Successfully generated GIDL DRAM pair 2 {output_dir} | GT: ({gt_x}, {gt_y})")
    return os.path.join(output_dir, "search.png"), os.path.join(output_dir, "target.png"), gt_info


if __name__ == "__main__":
    generate_gidl_pair2()
