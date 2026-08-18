#!/usr/bin/env python3
"""
Generate Pair 1 for Grayscale Vertical Capsule-Pillar DRAM (Center GT 500,500)
================================================================================
Synthesizes 1000x1000 Grayscale SEM image matching Vertical Capsule-Pillar DRAM:
  - High-density IC layout: increased line thickness, reduced black background gaps
  - Thicker vertical parallel Bitline (BL) tracks separating dense active cell columns
  - Wider vertical active cell pillars with top capsule units & elongated center storage capsules
  - Square transistor access via nodes with dark pinhole centers at Wordline (WL) intersections
  - Thicker horizontal Word Lines (WL)
  - Authentic SEM footer scale bar & metadata text ("Vertical Capsule-Pillar 3D DRAM Array (Top View)")
  - Creative In-Pattern Embedded Landmark at Center (500, 500):
    Fine interconnect bridge line linking two adjacent pillars with a central micro-via (NO big white box!)
  - Memory-optimized uint8 pipeline with full SEM noise
  - Stored in Script_and_their_images/055/
"""

import json
import os
import shutil
import cv2
import numpy as np


def render_capsulepillar_dram_scene(h=1000, w=1000, gt_x=500, gt_y=500, seed=23001):
    """Render 1000x1000 uint8 Grayscale image of Vertical Capsule-Pillar DRAM Circuitry."""
    np.random.seed(seed)

    col_pitch = 22  # Reduced column pitch for dense packing
    row_pitch = 28  # Reduced row pitch for dense packing

    # Silicon oxide substrate base
    img_gray = np.full((h, w), 45, dtype=np.uint8)

    # ─── 1. Thicker Vertical Parallel Bit Line (BL) Tracks ───────────────
    for cx in range(col_pitch // 2, w + col_pitch, col_pitch):
        cv2.line(img_gray, (cx - 8, 0), (cx - 8, h), 130, 3)
        cv2.line(img_gray, (cx - 8, 0), (cx - 8, h), 195, 2)
        cv2.line(img_gray, (cx + 8, 0), (cx + 8, h), 130, 3)
        cv2.line(img_gray, (cx + 8, 0), (cx + 8, h), 195, 2)

    # ─── 2. Dense Vertical Active Capsule Pillars & Storage Units ────────
    for cx in range(col_pitch // 2, w + col_pitch, col_pitch):
        for wy in range(row_pitch // 2, h - 40, row_pitch):
            # Vertical interconnect neck line
            cv2.line(img_gray, (cx, wy - 13), (cx, wy + 16), 190, 3)

            # Top Capsule Unit
            cv2.rectangle(img_gray, (cx - 6, wy - 13), (cx + 6, wy - 3), 175, -1)
            cv2.rectangle(img_gray, (cx - 6, wy - 13), (cx + 6, wy - 3), 225, 2)
            cv2.rectangle(img_gray, (cx - 4, wy - 11), (cx + 4, wy - 5), 115, -1)
            cv2.rectangle(img_gray, (cx - 2, wy - 9), (cx + 2, wy - 7), 245, -1)

            # Elongated Center Storage Capsule Unit
            cv2.rectangle(img_gray, (cx - 7, wy + 3), (cx + 7, wy + 16), 180, -1)
            cv2.rectangle(img_gray, (cx - 7, wy + 3), (cx + 7, wy + 16), 230, 2)
            cv2.rectangle(img_gray, (cx - 5, wy + 5), (cx + 5, wy + 14), 130, -1)
            cv2.rectangle(img_gray, (cx - 3, wy + 7), (cx + 3, wy + 12), 80, -1)
            cv2.rectangle(img_gray, (cx - 1, wy + 8), (cx + 1, wy + 11), 245, -1)

    # ─── 3. Thicker Horizontal Word Lines (WL) & Square Transistor Vias ──
    for wy in range(row_pitch // 2, h - 40, row_pitch):
        cv2.rectangle(img_gray, (0, wy - 4), (w, wy + 4), 165, -1)
        cv2.line(img_gray, (0, wy - 4), (w, wy - 4), 215, 2)
        cv2.line(img_gray, (0, wy + 4), (w, wy + 4), 215, 2)

        # Square Transistor Access Vias at WL-Pillar Intersections
        for cx in range(col_pitch // 2, w + col_pitch, col_pitch):
            cv2.rectangle(img_gray, (cx - 3, wy - 3), (cx + 3, wy + 3), 235, -1)
            cv2.rectangle(img_gray, (cx - 1, wy - 1), (cx + 1, wy + 1), 60, -1)

    # ─── 4. Sub-Array Sense Amplifier Logic Bands ──────────────────────
    sense_interval = row_pitch * 7  # 196px
    for sy in range(sense_interval, h - 50, sense_interval):
        cv2.rectangle(img_gray, (0, sy - 5), (w, sy + 5), 55, -1)
        cv2.line(img_gray, (0, sy - 6), (w, sy - 6), 150, 1)
        cv2.line(img_gray, (0, sy + 6), (w, sy + 6), 150, 1)

    # ─── 5. Creative In-Pattern Landmark Centered at (gt_x, gt_y) ──────
    cv2.line(img_gray, (gt_x - 14, gt_y), (gt_x + 14, gt_y), 245, 3)
    cv2.line(img_gray, (gt_x, gt_y - 8), (gt_x, gt_y + 8), 235, 3)
    cv2.circle(img_gray, (gt_x, gt_y), 3, 255, -1)
    cv2.rectangle(img_gray, (gt_x - 16, gt_y - 10), (gt_x + 16, gt_y + 10), 20, 1)

    # ─── 6. Authentic SEM Micrograph Footer Bar ────────────────────────
    cv2.rectangle(img_gray, (0, h - 35), (w, h), 15, -1)
    cv2.line(img_gray, (0, h - 35), (w, h - 35), 80, 1)

    cv2.putText(img_gray, "Vertical Capsule-Pillar 3D DRAM Array (Top View)", (w // 2 - 170, h - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, 220, 1, cv2.LINE_AA)
    # Scale bar line
    cv2.line(img_gray, (w - 200, h - 15), (w - 40, h - 15), 240, 2)
    cv2.line(img_gray, (w - 200, h - 18), (w - 200, h - 12), 240, 2)
    cv2.line(img_gray, (w - 40, h - 18), (w - 40, h - 12), 240, 2)
    cv2.putText(img_gray, "500 nm", (w - 35, h - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, 240, 1, cv2.LINE_AA)
    cv2.putText(img_gray, "Acc.V 5.00 kV   Spot 3.0   Magn 100.0 kx   WD 5.2 mm", (20, h - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, 200, 1, cv2.LINE_AA)

    return img_gray


# [STRIPPED] def add_unique_sem_noise(image_uint8, seed=23001, is_target=False):
# [STRIPPED]     np.random.seed(seed)
# [STRIPPED]     h, w = image_uint8.shape[:2]
# [STRIPPED]     img_f32 = image_uint8.astype(np.float32)
# [STRIPPED] 
# [STRIPPED]     if not is_target:
# [STRIPPED]         poisson = np.random.poisson(img_f32 / 255.0 * 14.0) / 14.0 * 255.0
# [STRIPPED]         gauss_fine = np.random.normal(0, 12.0, img_f32.shape)
# [STRIPPED]         gauss_coarse_raw = np.random.normal(0, 8.0, img_f32.shape)
# [STRIPPED]         gauss_coarse = cv2.GaussianBlur(gauss_coarse_raw, (7, 7), 0)
# [STRIPPED]         scanlines = np.sin(np.arange(h) * 1.5).astype(np.float32) * 4.0
# [STRIPPED]         scanline_map = np.tile(scanlines.reshape(-1, 1), (1, w))
# [STRIPPED]         speckle = np.random.randn(*img_f32.shape).astype(np.float32) * 0.10
# [STRIPPED]         noisy = (poisson + gauss_fine + gauss_coarse + scanline_map) * (1.0 + speckle)
# [STRIPPED]     else:
# [STRIPPED]         gauss = np.random.normal(0, 4.0, img_f32.shape)
# [STRIPPED]         noisy = img_f32 + gauss
# [STRIPPED] 
# [STRIPPED]     return np.clip(noisy, 0, 255).astype(np.uint8)
# [STRIPPED] 
# [STRIPPED] 
def generate_capsulepillar_pair1(output_dir="generated_capsulepillar/pair1", gt_x=500, gt_y=500, seed=23001):
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    search_clean = render_capsulepillar_dram_scene(h=h, w=w, gt_x=gt_x, gt_y=gt_y, seed=seed)

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
        "target_name": "Grayscale Vertical Capsule-Pillar DRAM - Center GT 500,500 (Pair 1)",
        "pair_id": "pair1",
        "scale_factor": 10.0
    }

    with open(os.path.join(output_dir, "groundtruth.json"), "w") as f:
        json.dump(gt_info, f, indent=2)

    cv2.imwrite(os.path.join(output_dir, "search.png"), search_out)
    cv2.imwrite(os.path.join(output_dir, "target.png"), target_out)
    shutil.copyfile(os.path.join(output_dir, "target.png"), os.path.join(output_dir, "reference.png"))

    print(f"[generate_capsulepillar_pair1] Successfully generated Capsule-Pillar DRAM pair 1 {output_dir} | GT: ({gt_x}, {gt_y})")
    return os.path.join(output_dir, "search.png"), os.path.join(output_dir, "target.png"), gt_info


if __name__ == "__main__":
    generate_capsulepillar_pair1()
