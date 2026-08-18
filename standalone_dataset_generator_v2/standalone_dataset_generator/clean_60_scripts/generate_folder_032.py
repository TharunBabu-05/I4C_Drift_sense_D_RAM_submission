#!/usr/bin/env python3
"""
Generate Pair 2 for generated_fbc (Grayscale Floating-Body Cell DRAM - Top GT 500,310)
====================================================================================
Synthesizes 1000x1000 Grayscale SEM image matching FBC (Floating-Body Cell) DRAM structure:
  - Full-frame 1000x1000 IC layout
  - High-density ultra-narrow vertical SOI fins (width 4px, pitch 14px)
  - Dense front-gate wordlines (width 3px, pitch 14px) + dashed back-gate body bias lines
  - Oxide sidewall spacers along ultra-dense fins
  - Seamless Ground Truth Target Landmark at Top (500, 310): Cluster of adjacent FBC cells
    with interconnected metal straps and X-crosshair
  - Full SEM noise pipeline (no barrel distortion)
"""

import json
import os
import shutil
import cv2
import numpy as np


def render_fbc_dram_scene(h=1000, w=1000, gt_x=500, gt_y=310, seed=11002):
    """Render 1000x1000 uint8 Grayscale image of Floating-Body Cell (FBC) DRAM Semiconductor Circuitry."""
    np.random.seed(seed)

    col_pitch = 14  # Ultra-dense SOI fin pitch
    fin_w = 4       # Narrow fin width
    row_pitch = 14  # Dense wordline pitch
    gate_h = 3      # Narrow gate width

    # Dark Buried Oxide (BOX) substrate base
    img_gray = np.full((h, w), 30, dtype=np.float32)

    # ─── 1. Ultra-Dense Active Silicon SOI Fins (Vertical Tracks) ───────
    for cx in range(col_pitch // 2, w, col_pitch):
        x1 = cx - fin_w // 2
        x2 = cx + fin_w // 2
        if 0 <= x1 and x2 < w:
            img_gray[:, x1:x2] = 100.0
            img_gray[:, x1] = 155.0
            img_gray[:, x2 - 1] = 155.0

    # ─── 2. Front-Gate Wordlines & Dashed Back-Gate Bias Lines ─────────
    for wy in range(row_pitch // 2, h, row_pitch):
        y1 = wy - gate_h // 2
        y2 = wy + gate_h // 2 + 1
        if 0 <= y1 and y2 < h:
            bgy = wy - row_pitch // 2 + 1
            if 0 <= bgy < h:
                for bx in range(0, w, 8):
                    cv2.line(img_gray, (bx, bgy), (min(w, bx + 5), bgy), 120.0, 1)

            img_gray[y1:y2, :] = np.maximum(img_gray[y1:y2, :], 165.0)
            img_gray[y1, :] = 195.0

    # ─── 3. Dense Floating-Body Nodes & Contact Vias ──────────────────
    for cx in range(col_pitch // 2, w, col_pitch):
        for wy in range(row_pitch // 2, h, row_pitch):
            cv2.circle(img_gray, (cx, wy), 1, 235.0, -1)
            sd_y = wy + row_pitch // 2
            if sd_y + 1 < h:
                cv2.rectangle(img_gray, (cx - 1, sd_y - 1), (cx + 1, sd_y + 1), 215.0, -1)

    # ─── 4. Sense Amplifier & Control Bands ─────────────────────────────
    sense_interval = row_pitch * 10  # 140px
    for sy in range(sense_interval, h - 10, sense_interval):
        cv2.rectangle(img_gray, (0, sy - 2), (w, sy + 2), 65.0, -1)
        cv2.line(img_gray, (0, sy - 3), (w, sy - 3), 145.0, 1)
        cv2.line(img_gray, (0, sy + 3), (w, sy + 3), 145.0, 1)

    # ─── 5. High-Contrast Landmark at (gt_x, gt_y) ─────────────────────
    start_col = (gt_x // col_pitch) * col_pitch + col_pitch // 2
    start_row = (gt_y // row_pitch) * row_pitch + row_pitch // 2

    cluster_x1 = start_col - fin_w // 2
    cluster_x2 = start_col + 2 * col_pitch + fin_w // 2
    cluster_y1 = start_row - gate_h // 2
    cluster_y2 = start_row + 3 * row_pitch + gate_h // 2

    cv2.rectangle(img_gray, (cluster_x1 - 2, cluster_y1 - 2), (cluster_x2 + 2, cluster_y2 + 2), 215.0, -1)

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

                cv2.rectangle(img_gray, (x1, y1), (x2 - 1, y2 - 1), 235.0, 1)
                cv2.circle(img_gray, (cur_x, cur_y), 1, 255.0, -1)

                if dc < 2:
                    next_x = cur_x + col_pitch
                    cv2.line(img_gray, (x2, cur_y), (next_x - fin_w // 2, cur_y), 240.0, 2)

                if dr < 3:
                    next_y = cur_y + row_pitch
                    cv2.line(img_gray, (cur_x, y2), (cur_x, next_y - gate_h // 2), 240.0, 2)

    cv2.line(img_gray, (cluster_x1, cluster_y1), (cluster_x2, cluster_y2), 250.0, 2)
    cv2.line(img_gray, (cluster_x2, cluster_y1), (cluster_x1, cluster_y2), 250.0, 2)

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


# [STRIPPED] def add_sem_noise_grayscale(image_uint8, seed=11002, is_target=False):
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
def generate_fbc_pair2(output_dir="generated_fbc/pair2", gt_x=500, gt_y=310, seed=11002):
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    search_clean = render_fbc_dram_scene(h=h, w=w, gt_x=gt_x, gt_y=gt_y, seed=seed)

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
        "target_name": "Grayscale Floating-Body Cell (FBC) DRAM - Top GT 500,310 (Pair 2)",
        "pair_id": "pair2",
        "scale_factor": 10.0
    }

    with open(os.path.join(output_dir, "groundtruth.json"), "w") as f:
        json.dump(gt_info, f, indent=2)

    cv2.imwrite(os.path.join(output_dir, "search.png"), search_out)
    cv2.imwrite(os.path.join(output_dir, "target.png"), target_out)
    shutil.copyfile(os.path.join(output_dir, "target.png"), os.path.join(output_dir, "reference.png"))

    print(f"[generate_fbc_pair2] Successfully generated FBC DRAM pair 2 {output_dir} | GT: ({gt_x}, {gt_y})")
    return os.path.join(output_dir, "search.png"), os.path.join(output_dir, "target.png"), gt_info


if __name__ == "__main__":
    generate_fbc_pair2()
