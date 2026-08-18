#!/usr/bin/env python3
"""
Generate Pair 1 for Hybrid Single-Zigzag Line + Box Capacitor DRAM (Center GT 500,500)
=====================================================================================
Synthesizes 1000x1000 Grayscale SEM image matching Hybrid DRAM Architecture:
  - Alternating Columns:
    - Odd Columns (Col 1, 3, 5...): Single plain continuous zigzag/twisted line (no box capacitors!)
    - Even Columns (Col 2, 4, 6...): Straight bit lines with Box Capacitors (rectangular box storage capacitors)
  - Horizontal Word Lines (WL)
  - Unique SEM Noise Signature: Multi-scale grain noise + Poisson shot noise + electron-beam scanline ripple
  - Creative In-Pattern Embedded Landmark at Center (500, 500):
    Fine interconnect bridge line linking a zigzag line column to a box capacitor cell with a central micro-via (NO big white box!)
  - Stored in Script_and_their_images/051/
"""

import json
import os
import shutil
import cv2
import numpy as np


def render_hybridtwist_dram_scene(h=1000, w=1000, gt_x=500, gt_y=500, seed=21001):
    """Render 1000x1000 uint8 Grayscale image of Hybrid Single-Zigzag Line + Box Capacitor DRAM."""
    np.random.seed(seed)

    col_pitch = 24  # Column pitch
    row_pitch = 24  # Wordline pitch

    # Dark silicon oxide substrate base
    img_gray = np.full((h, w), 35, dtype=np.uint8)

    # ─── 1. Alternating Columns: Single Zigzag Line vs Box Capacitors ──
    for col_idx, cx in enumerate(range(col_pitch // 2, w, col_pitch)):
        if col_idx % 2 == 0:
            # ── Single Plain Continuous Zigzag Twisted Line ──
            pts = []
            phase = (col_idx % 4) * np.pi / 2.0
            for y in range(0, h - 35, 4):
                offset = 5.0 * np.sin(2.0 * np.pi * y / (row_pitch * 2) + phase)
                x = int(cx + offset)
                pts.append([x, y])

            pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_gray, [pts], isClosed=False, color=180, thickness=2)

            # Micro contact dots at WL intersections along the zigzag line
            for wy in range(row_pitch // 2, h - 40, row_pitch):
                offset = int(5.0 * np.sin(2.0 * np.pi * wy / (row_pitch * 2) + phase))
                cv2.circle(img_gray, (cx + offset, wy), 2, 220, -1)
                cv2.circle(img_gray, (cx + offset, wy), 1, 255, -1)
        else:
            # ── Straight BL Track with Box Capacitors ──
            cv2.line(img_gray, (cx - 3, 0), (cx - 3, h), 110, 2)
            cv2.line(img_gray, (cx + 3, 0), (cx + 3, h), 110, 2)
            cv2.line(img_gray, (cx - 4, 0), (cx - 4, h), 75, 1)
            cv2.line(img_gray, (cx + 4, 0), (cx + 4, h), 75, 1)

            # Box Capacitors between WL rows
            for wy in range(row_pitch // 2, h - 40, row_pitch):
                cv2.circle(img_gray, (cx, wy), 3, 220, -1)
                cv2.circle(img_gray, (cx, wy), 1, 255, -1)

                cs_y = wy + row_pitch // 2
                if cs_y + 8 < h - 40:
                    cv2.rectangle(img_gray, (cx - 6, cs_y - 5), (cx + 6, cs_y + 5), 185, -1)
                    cv2.rectangle(img_gray, (cx - 6, cs_y - 5), (cx + 6, cs_y + 5), 225, 1)
                    cv2.rectangle(img_gray, (cx - 4, cs_y - 3), (cx + 4, cs_y + 3), 90, -1)
                    cv2.rectangle(img_gray, (cx - 2, cs_y - 1), (cx + 2, cs_y + 1), 245, -1)

    # ─── 2. Horizontal Word Lines (WL) ─────────────────────────────────
    for wy in range(row_pitch // 2, h - 40, row_pitch):
        cv2.rectangle(img_gray, (0, wy - 3), (w, wy + 3), 160, -1)
        cv2.line(img_gray, (0, wy - 3), (w, wy - 3), 205, 1)
        cv2.line(img_gray, (0, wy + 3), (w, wy + 3), 205, 1)

    # ─── 3. Sub-Array Sense Amplifier Logic Bands ──────────────────────
    sense_interval = row_pitch * 8  # 192px
    for sy in range(sense_interval, h - 50, sense_interval):
        cv2.rectangle(img_gray, (0, sy - 4), (w, sy + 4), 55, -1)
        cv2.line(img_gray, (0, sy - 5), (w, sy - 5), 145, 1)
        cv2.line(img_gray, (0, sy + 5), (w, sy + 5), 145, 1)

    # ─── 4. Creative In-Pattern Landmark Centered at (gt_x, gt_y) ──────
    # Fine horizontal bridge line connecting zigzag line column to box capacitor cell
    cv2.line(img_gray, (gt_x - 14, gt_y), (gt_x + 14, gt_y), 245, 3)
    # Small cross-bridge interconnect line
    cv2.line(img_gray, (gt_x, gt_y - 8), (gt_x, gt_y + 8), 235, 3)
    # Central micro-via dot at the line intersection
    cv2.circle(img_gray, (gt_x, gt_y), 3, 255, -1)
    # Dark oxide border framing the interconnect line bridge
    cv2.rectangle(img_gray, (gt_x - 16, gt_y - 10), (gt_x + 16, gt_y + 10), 20, 1)

    # ─── 5. Authentic SEM Micrograph Footer Bar ────────────────────────
    cv2.rectangle(img_gray, (0, h - 35), (w, h), 15, -1)
    cv2.line(img_gray, (0, h - 35), (w, h - 35), 80, 1)

    cv2.putText(img_gray, "Hybrid Single-Zigzag & Box-Capacitor DRAM (Top View)", (w // 2 - 170, h - 11),
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


# [STRIPPED] def add_unique_sem_noise(image_uint8, seed=21001, is_target=False):
# [STRIPPED]     """Unique SEM Noise Signature: Multi-scale Gaussian grain + Poisson shot noise + Electron-beam scanline ripple."""
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
def generate_hybridtwist_pair1(output_dir="generated_hybridtwist/pair1", gt_x=500, gt_y=500, seed=21001):
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    search_clean = render_hybridtwist_dram_scene(h=h, w=w, gt_x=gt_x, gt_y=gt_y, seed=seed)

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
        "target_name": "Grayscale Hybrid Single-Zigzag & Box-Capacitor DRAM - Center GT 500,500 (Pair 1)",
        "pair_id": "pair1",
        "scale_factor": 10.0
    }

    with open(os.path.join(output_dir, "groundtruth.json"), "w") as f:
        json.dump(gt_info, f, indent=2)

    cv2.imwrite(os.path.join(output_dir, "search.png"), search_out)
    cv2.imwrite(os.path.join(output_dir, "target.png"), target_out)
    shutil.copyfile(os.path.join(output_dir, "target.png"), os.path.join(output_dir, "reference.png"))

    print(f"[generate_hybridtwist_pair1] Successfully generated Hybrid DRAM pair 1 {output_dir} | GT: ({gt_x}, {gt_y})")
    return os.path.join(output_dir, "search.png"), os.path.join(output_dir, "target.png"), gt_info


if __name__ == "__main__":
    generate_hybridtwist_pair1()
