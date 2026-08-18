#!/usr/bin/env python3
"""
Generate Pair 2 for Grayscale BEST Buried-Strap Trench DRAM (Top GT 500,310)
================================================================================
Synthesizes 1000x1000 Grayscale SEM image matching BEST Buried-Strap Trench DRAM:
  - Exact repeating unit cell layout: 5 boxes (Far-Left, Mid-Left, Center Trench, Mid-Right, Far-Right) + 3 vertical lines (Left BL, Center Neck, Right BL)
  - Broad horizontal Word Lines (WL)
  - Authentic SEM footer scale bar & metadata text ("BEST Buried-Strap Trench DRAM Array (Top View)")
  - Creative In-Pattern Embedded Landmark at Top (500, 310):
    Fine interconnect bridge line linking two adjacent cells with a central micro-via (NO big white box!)
  - Memory-optimized uint8 pipeline with full SEM noise
  - Stored in Script_and_their_images/054/
"""

import json
import os
import shutil
import cv2
import numpy as np


def render_besttrench_dram_scene(h=1000, w=1000, gt_x=500, gt_y=310, seed=22002):
    """Render 1000x1000 uint8 Grayscale image of BEST Buried-Strap Trench DRAM with 5-box 3-line periodic cell layout."""
    np.random.seed(seed)

    unit_w = 48    # Horizontal pitch of repeating unit cell
    row_pitch = 32  # Vertical pitch between Word Lines

    # Dark silicon oxide substrate base
    img_gray = np.full((h, w), 30, dtype=np.uint8)

    # ─── 1. Repeat Unit Cells (5 Boxes + 3 Vertical Lines per Unit Cell) ───
    for cx in range(unit_w // 2, w + unit_w, unit_w):
        # ── 3 Vertical Lines (Line 1: Left BL, Line 2: Center Neck, Line 3: Right BL) ──
        cv2.line(img_gray, (cx - 10, 0), (cx - 10, h), 140, 2)
        cv2.line(img_gray, (cx - 10, 0), (cx - 10, h), 200, 1)

        cv2.line(img_gray, (cx, 0), (cx, h), 160, 2)

        cv2.line(img_gray, (cx + 10, 0), (cx + 10, h), 140, 2)
        cv2.line(img_gray, (cx + 10, 0), (cx + 10, h), 200, 1)

        # ── 5 Boxes per Unit Cell between Wordlines ──
        for wy in range(row_pitch // 2, h - 40, row_pitch):
            cs_y = wy + row_pitch // 2
            if cs_y + 10 < h - 40:
                # Box 1: Far-Left Rounded Box
                cv2.rectangle(img_gray, (cx - 21, cs_y - 6), (cx - 15, cs_y + 6), 165, -1)
                cv2.rectangle(img_gray, (cx - 21, cs_y - 6), (cx - 15, cs_y + 6), 210, 1)
                cv2.rectangle(img_gray, (cx - 19, cs_y - 4), (cx - 17, cs_y + 4), 85, -1)

                # Box 2: Mid-Left Narrow Box (Buried Strap 1)
                cv2.rectangle(img_gray, (cx - 14, cs_y - 5), (cx - 11, cs_y + 5), 185, -1)
                cv2.rectangle(img_gray, (cx - 14, cs_y - 5), (cx - 11, cs_y + 5), 225, 1)

                # Box 3: Center Trench Capacitor Box (Multi-ring concentric structure)
                cv2.rectangle(img_gray, (cx - 7, cs_y - 8), (cx + 7, cs_y + 8), 175, -1)
                cv2.rectangle(img_gray, (cx - 7, cs_y - 8), (cx + 7, cs_y + 8), 220, 1)
                cv2.rectangle(img_gray, (cx - 5, cs_y - 6), (cx + 5, cs_y + 6), 125, -1)
                cv2.rectangle(img_gray, (cx - 3, cs_y - 4), (cx + 3, cs_y + 4), 75, -1)
                cv2.rectangle(img_gray, (cx - 1, cs_y - 2), (cx + 1, cs_y + 2), 245, -1)

                # Box 4: Mid-Right Narrow Box (Buried Strap 2)
                cv2.rectangle(img_gray, (cx + 11, cs_y - 5), (cx + 14, cs_y + 5), 185, -1)
                cv2.rectangle(img_gray, (cx + 11, cs_y - 5), (cx + 14, cs_y + 5), 225, 1)

                # Box 5: Far-Right Rounded Box
                cv2.rectangle(img_gray, (cx + 15, cs_y - 6), (cx + 21, cs_y + 6), 165, -1)
                cv2.rectangle(img_gray, (cx + 15, cs_y - 6), (cx + 21, cs_y + 6), 210, 1)
                cv2.rectangle(img_gray, (cx + 17, cs_y - 4), (cx + 19, cs_y + 4), 85, -1)

    # ─── 2. Horizontal Word Lines (WL) ─────────────────────────────────
    for wy in range(row_pitch // 2, h - 40, row_pitch):
        cv2.rectangle(img_gray, (0, wy - 4), (w, wy + 4), 170, -1)
        cv2.line(img_gray, (0, wy - 4), (w, wy - 4), 215, 1)
        cv2.line(img_gray, (0, wy + 4), (w, wy + 4), 215, 1)

        for cx in range(unit_w // 2, w + unit_w, unit_w):
            cv2.circle(img_gray, (cx - 10, wy), 3, 230, -1)
            cv2.circle(img_gray, (cx - 10, wy), 1, 255, -1)
            cv2.circle(img_gray, (cx + 10, wy), 3, 230, -1)
            cv2.circle(img_gray, (cx + 10, wy), 1, 255, -1)

    # ─── 3. Sub-Array Sense Amplifier Logic Bands ──────────────────────
    sense_interval = row_pitch * 6  # 192px
    for sy in range(sense_interval, h - 50, sense_interval):
        cv2.rectangle(img_gray, (0, sy - 5), (w, sy + 5), 50, -1)
        cv2.line(img_gray, (0, sy - 6), (w, sy - 6), 145, 1)
        cv2.line(img_gray, (0, sy + 6), (w, sy + 6), 145, 1)

    # ─── 4. Creative In-Pattern Landmark Centered at (gt_x, gt_y) ──────
    cv2.line(img_gray, (gt_x - 16, gt_y), (gt_x + 16, gt_y), 245, 3)
    cv2.line(img_gray, (gt_x, gt_y - 10), (gt_x, gt_y + 10), 235, 3)
    cv2.circle(img_gray, (gt_x, gt_y), 3, 255, -1)
    cv2.rectangle(img_gray, (gt_x - 18, gt_y - 12), (gt_x + 18, gt_y + 12), 20, 1)

    # ─── 5. Authentic SEM Micrograph Footer Bar ────────────────────────
    cv2.rectangle(img_gray, (0, h - 35), (w, h), 15, -1)
    cv2.line(img_gray, (0, h - 35), (w, h - 35), 80, 1)

    cv2.putText(img_gray, "BEST Buried-Strap Trench DRAM Array (Top View)", (w // 2 - 170, h - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, 220, 1, cv2.LINE_AA)
    # Scale bar line
    cv2.line(img_gray, (w - 200, h - 15), (w - 40, h - 15), 240, 2)
    cv2.line(img_gray, (w - 200, h - 18), (w - 200, h - 12), 240, 2)
    cv2.line(img_gray, (w - 40, h - 18), (w - 40, h - 12), 240, 2)
    cv2.putText(img_gray, "500 nm", (w - 35, h - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, 240, 1, cv2.LINE_AA)
    cv2.putText(img_gray, "Acc.V 5.00 kV   Spot 3.0   Magn 100.0 kx   WD 5.1 mm", (20, h - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, 200, 1, cv2.LINE_AA)

    return img_gray


# [STRIPPED] def add_unique_sem_noise(image_uint8, seed=22002, is_target=False):
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
def generate_besttrench_pair2(output_dir="generated_besttrench/pair2", gt_x=500, gt_y=310, seed=22002):
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    search_clean = render_besttrench_dram_scene(h=h, w=w, gt_x=gt_x, gt_y=gt_y, seed=seed)

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
        "target_name": "Grayscale BEST Buried-Strap Trench DRAM - Top GT 500,310 (Pair 2)",
        "pair_id": "pair2",
        "scale_factor": 10.0
    }

    with open(os.path.join(output_dir, "groundtruth.json"), "w") as f:
        json.dump(gt_info, f, indent=2)

    cv2.imwrite(os.path.join(output_dir, "search.png"), search_out)
    cv2.imwrite(os.path.join(output_dir, "target.png"), target_out)
    shutil.copyfile(os.path.join(output_dir, "target.png"), os.path.join(output_dir, "reference.png"))

    print(f"[generate_besttrench_pair2] Successfully generated BEST Trench DRAM pair 2 {output_dir} | GT: ({gt_x}, {gt_y})")
    return os.path.join(output_dir, "search.png"), os.path.join(output_dir, "target.png"), gt_info


if __name__ == "__main__":
    generate_besttrench_pair2()
