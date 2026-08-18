#!/usr/bin/env python3
"""
Generate Pair 2 for Grayscale Asymmetric Dual-Block DRAM (Top GT 500,310)
============================================================================
Synthesizes 1000x1000 Grayscale SEM image matching Asymmetric Dual-Block DRAM:
  - Full-frame 1000x1000 IC layout matching reference SEM micrograph
  - Asymmetric Block A (Left Half): Dense vertical capsule pillars with left-aligned contact pads
  - Block Boundary (Center Seam x=500): Vertical Bitline boundary channel with central via nodes
  - Asymmetric Block B (Right Half): Staggered capsule pillars with right-aligned pads and different pitch
  - Horizontal Word Lines (WL) crossing both blocks
  - Authentic SEM footer scale bar & metadata text ("Asymmetric Dual-Block DRAM Array (Top View)")
  - Creative In-Pattern Embedded Landmark at Top (500, 310):
    Fine interconnect bridge line linking Block A to Block B at the boundary with a central micro-via (NO big white box!)
  - Memory-optimized uint8 pipeline with full SEM noise
  - Stored in Script_and_their_images/058/
"""

import json
import os
import shutil
import cv2
import numpy as np


def render_asymblock_dram_scene(h=1000, w=1000, gt_x=500, gt_y=310, seed=24002):
    """Render 1000x1000 uint8 Grayscale image of Asymmetric Dual-Block DRAM Circuitry."""
    np.random.seed(seed)

    row_pitch = 32  # Wordline pitch

    # Dark silicon oxide substrate base
    img_gray = np.full((h, w), 35, dtype=np.uint8)

    # ─── 1. Asymmetric Block A (Left Half: x = 0 to 480) ────────────────
    col_pitch_a = 24
    for cx in range(col_pitch_a // 2, 480, col_pitch_a):
        cv2.line(img_gray, (cx - 3, 0), (cx - 3, h), 120, 2)
        cv2.line(img_gray, (cx + 3, 0), (cx + 3, h), 120, 2)

        for wy in range(row_pitch // 2, h - 40, row_pitch):
            cv2.circle(img_gray, (cx, wy), 3, 225, -1)
            cv2.circle(img_gray, (cx, wy), 1, 255, -1)

            cs_y = wy + row_pitch // 2
            if cs_y + 8 < h - 40:
                cv2.rectangle(img_gray, (cx - 7, cs_y - 6), (cx + 4, cs_y + 6), 175, -1)
                cv2.rectangle(img_gray, (cx - 7, cs_y - 6), (cx + 4, cs_y + 6), 215, 1)
                cv2.rectangle(img_gray, (cx - 5, cs_y - 4), (cx + 2, cs_y + 4), 95, -1)
                cv2.rectangle(img_gray, (cx - 3, cs_y - 2), (cx + 1, cs_y + 2), 240, -1)

    # ─── 2. Asymmetric Block B (Right Half: x = 520 to 1000) ───────────
    col_pitch_b = 26
    for col_idx, cx in enumerate(range(520 + col_pitch_b // 2, w, col_pitch_b)):
        cv2.line(img_gray, (cx - 4, 0), (cx - 4, h), 135, 2)
        cv2.line(img_gray, (cx + 4, 0), (cx + 4, h), 135, 2)

        row_offset = (col_idx % 2) * (row_pitch // 2)
        for wy in range(row_pitch // 2 + row_offset, h - 40, row_pitch):
            cv2.rectangle(img_gray, (cx - 3, wy - 3), (cx + 3, wy + 3), 230, -1)
            cv2.rectangle(img_gray, (cx - 1, wy - 1), (cx + 1, wy + 1), 60, -1)

            cs_y = wy + row_pitch // 2
            if cs_y + 8 < h - 40:
                cv2.rectangle(img_gray, (cx - 4, cs_y - 7), (cx + 7, cs_y + 7), 185, -1)
                cv2.rectangle(img_gray, (cx - 4, cs_y - 7), (cx + 7, cs_y + 7), 225, 1)
                cv2.rectangle(img_gray, (cx - 2, cs_y - 5), (cx + 5, cs_y + 5), 105, -1)
                cv2.rectangle(img_gray, (cx, cs_y - 2), (cx + 3, cs_y + 2), 245, -1)

    # ─── 3. Central Block Boundary Seam (x = 480 to 520) ────────────────
    cv2.rectangle(img_gray, (484, 0), (516, h - 35), 25, -1)
    cv2.line(img_gray, (494, 0), (494, h - 35), 210, 3)
    cv2.line(img_gray, (506, 0), (506, h - 35), 210, 3)
    cv2.line(img_gray, (494, 0), (494, h - 35), 255, 1)
    cv2.line(img_gray, (506, 0), (506, h - 35), 255, 1)

    for wy in range(row_pitch // 2, h - 40, row_pitch):
        cv2.circle(img_gray, (500, wy), 4, 240, -1)
        cv2.circle(img_gray, (500, wy), 2, 255, -1)

    # ─── 4. Horizontal Word Lines (WL) Across Entire Array ─────────────
    for wy in range(row_pitch // 2, h - 40, row_pitch):
        cv2.rectangle(img_gray, (0, wy - 4), (w, wy + 4), 165, -1)
        cv2.line(img_gray, (0, wy - 4), (w, wy - 4), 215, 2)
        cv2.line(img_gray, (0, wy + 4), (w, wy + 4), 215, 2)

    # ─── 5. Sub-Array Sense Amplifier Logic Bands ──────────────────────
    sense_interval = row_pitch * 6  # 192px
    for sy in range(sense_interval, h - 50, sense_interval):
        cv2.rectangle(img_gray, (0, sy - 5), (w, sy + 5), 50, -1)
        cv2.line(img_gray, (0, sy - 6), (w, sy - 6), 145, 1)
        cv2.line(img_gray, (0, sy + 6), (w, sy + 6), 145, 1)

    # ─── 6. Creative In-Pattern Landmark Centered at (gt_x, gt_y) ──────
    cv2.line(img_gray, (gt_x - 16, gt_y), (gt_x + 16, gt_y), 245, 3)
    cv2.line(img_gray, (gt_x, gt_y - 10), (gt_x, gt_y + 10), 235, 3)
    cv2.circle(img_gray, (gt_x, gt_y), 3, 255, -1)
    cv2.rectangle(img_gray, (gt_x - 18, gt_y - 12), (gt_x + 18, gt_y + 12), 20, 1)

    # ─── 7. Authentic SEM Micrograph Footer Bar ────────────────────────
    cv2.rectangle(img_gray, (0, h - 35), (w, h), 15, -1)
    cv2.line(img_gray, (0, h - 35), (w, h - 35), 80, 1)

    cv2.putText(img_gray, "Asymmetric Dual-Block DRAM Array (Top View)", (w // 2 - 170, h - 11),
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


# [STRIPPED] def add_unique_sem_noise(image_uint8, seed=24002, is_target=False):
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
def generate_asymblock_pair2(output_dir="generated_asymblock/pair2", gt_x=500, gt_y=310, seed=24002):
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    search_clean = render_asymblock_dram_scene(h=h, w=w, gt_x=gt_x, gt_y=gt_y, seed=seed)

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
        "target_name": "Grayscale Asymmetric Dual-Block DRAM - Top GT 500,310 (Pair 2)",
        "pair_id": "pair2",
        "scale_factor": 10.0
    }

    with open(os.path.join(output_dir, "groundtruth.json"), "w") as f:
        json.dump(gt_info, f, indent=2)

    cv2.imwrite(os.path.join(output_dir, "search.png"), search_out)
    cv2.imwrite(os.path.join(output_dir, "target.png"), target_out)
    shutil.copyfile(os.path.join(output_dir, "target.png"), os.path.join(output_dir, "reference.png"))

    print(f"[generate_asymblock_pair2] Successfully generated Asymmetric Block DRAM pair 2 {output_dir} | GT: ({gt_x}, {gt_y})")
    return os.path.join(output_dir, "search.png"), os.path.join(output_dir, "target.png"), gt_info


if __name__ == "__main__":
    generate_asymblock_pair2()
