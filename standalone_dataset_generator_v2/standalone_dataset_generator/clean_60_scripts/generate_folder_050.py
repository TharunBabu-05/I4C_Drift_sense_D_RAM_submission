#!/usr/bin/env python3
"""
Generate Pair 2 for Grayscale Vertical BL-Twist DRAM (Top GT 500,310)
====================================================================
Synthesizes 1000x1000 Grayscale SEM image matching Vertical BL-Twist DRAM:
  - Full-frame 1000x1000 IC layout matching reference SEM micrograph
  - Twisted vertical Bitlines (BL Twist) weaving in zigzag S-curves around Storage Capacitors
  - Horizontal Word Lines (WL) & vertical capsule-shaped Storage Capacitors (Cs)
  - Authentic SEM footer scale bar & metadata text ("Vertical BL-Twist DRAM Array (Top View)")
  - No random noise; applies 2-degree rotation & soft vignette shading as requested
  - Creative In-Pattern Embedded Landmark at Top (500, 310):
    Fine interconnect bridge line linking two adjacent cells with a central micro-via (NO big white box!)
"""

import json
import os
import shutil
import cv2
import numpy as np


def render_bltwist_dram_scene(h=1000, w=1000, gt_x=500, gt_y=310, seed=20002):
    """Render 1000x1000 uint8 Grayscale image of Vertical BL-Twist DRAM Circuitry."""
    np.random.seed(seed)

    col_pitch = 24  # Column pitch
    row_pitch = 24  # Wordline pitch

    # Dark silicon oxide substrate base
    img_gray = np.full((h, w), 35, dtype=np.uint8)

    # ─── 1. Twisted Vertical Bit Lines (BL Twist S-Curves) ─────────────
    for col_idx, cx in enumerate(range(col_pitch // 2, w, col_pitch)):
        pts_left = []
        pts_right = []
        phase = (col_idx % 2) * np.pi

        for y in range(0, h - 35, 4):
            offset = 4.0 * np.sin(2.0 * np.pi * y / (row_pitch * 2) + phase)
            x_l = int(cx - 3 + offset)
            x_r = int(cx + 3 - offset)
            pts_left.append([x_l, y])
            pts_right.append([x_r, y])

        pts_left = np.array(pts_left, dtype=np.int32).reshape((-1, 1, 2))
        pts_right = np.array(pts_right, dtype=np.int32).reshape((-1, 1, 2))

        cv2.polylines(img_gray, [pts_left], isClosed=False, color=115, thickness=2)
        cv2.polylines(img_gray, [pts_right], isClosed=False, color=115, thickness=2)

    # ─── 2. Horizontal Word Lines (WL) ─────────────────────────────────
    for wy in range(row_pitch // 2, h - 40, row_pitch):
        cv2.rectangle(img_gray, (0, wy - 3), (w, wy + 3), 160, -1)
        cv2.line(img_gray, (0, wy - 3), (w, wy - 3), 205, 1)
        cv2.line(img_gray, (0, wy + 3), (w, wy + 3), 205, 1)

    # ─── 3. Storage Capacitors (Cs) & BL Twist Nodes ───────────────────
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
    start_col = (gt_x // col_pitch) * col_pitch + col_pitch // 2
    start_row = (gt_y // row_pitch) * row_pitch + row_pitch // 2
    cs_y = start_row + row_pitch // 2

    # Fine horizontal bridge line connecting two adjacent cells
    cv2.line(img_gray, (start_col - 4, cs_y), (start_col + col_pitch + 4, cs_y), 235, 2)
    # Small cross-bridge interconnect line
    cv2.line(img_gray, (start_col + col_pitch // 2, cs_y - 6), (start_col + col_pitch // 2, cs_y + 6), 225, 2)
    # Central micro-via dot at the line intersection
    cv2.circle(img_gray, (start_col + col_pitch // 2, cs_y), 2, 255, -1)

    # ─── 6. Authentic SEM Micrograph Footer Bar ────────────────────────
    cv2.rectangle(img_gray, (0, h - 35), (w, h), 15, -1)
    cv2.line(img_gray, (0, h - 35), (w, h - 35), 80, 1)

    cv2.putText(img_gray, "Vertical BL-Twist DRAM Array (Top View)", (w // 2 - 140, h - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, 220, 1, cv2.LINE_AA)
    # Scale bar line
    cv2.line(img_gray, (w - 200, h - 15), (w - 40, h - 15), 240, 2)
    cv2.line(img_gray, (w - 200, h - 18), (w - 200, h - 12), 240, 2)
    cv2.line(img_gray, (w - 40, h - 18), (w - 40, h - 12), 240, 2)
    cv2.putText(img_gray, "500 nm", (w - 35, h - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, 240, 1, cv2.LINE_AA)
    cv2.putText(img_gray, "Acc.V 5.00 kV   Spot 3.0   Magn 100.0 kx   WD 5.2 mm", (20, h - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, 200, 1, cv2.LINE_AA)

    # ─── 7. Apply 2-Degree Turn (Rotation Transform) ───────────────────
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 2.0, 1.0)
    img_rotated = cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    return img_rotated


def apply_vignette_clean(img_uint8, strength=0.25):
    """Apply soft vignette shading gradient without random noise."""
    h, w = img_uint8.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    max_r = np.sqrt(cx ** 2 + cy ** 2)
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)
    vignette_map = 1.0 - strength * (r / max_r) ** 2
    img_f32 = img_uint8.astype(np.float32) * vignette_map
    return np.clip(img_f32, 0, 255).astype(np.uint8)


def generate_bltwist_pair2(output_dir="generated_bltwist/pair2", gt_x=500, gt_y=310, seed=20002):
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    search_clean = render_bltwist_dram_scene(h=h, w=w, gt_x=gt_x, gt_y=gt_y, seed=seed)

    crop_size = 100
    x0 = max(0, min(w - crop_size, gt_x - crop_size // 2))
    y0 = max(0, min(h - crop_size, gt_y - crop_size // 2))
    x1, y1 = x0 + crop_size, y0 + crop_size

    crop_patch = search_clean[y0:y1, x0:x1]
    target_clean = cv2.resize(crop_patch, (1000, 1000), interpolation=cv2.INTER_NEAREST)

    # Clean vignette output without noise
    search_out = apply_vignette_clean(search_clean, strength=0.25)
    target_out = apply_vignette_clean(target_clean, strength=0.10)

    gt_info = {
        "center_x": gt_x,
        "center_y": gt_y,
        "target_name": "Grayscale Vertical BL-Twist DRAM - Top GT 500,310 (Pair 2)",
        "pair_id": "pair2",
        "scale_factor": 10.0
    }

    with open(os.path.join(output_dir, "groundtruth.json"), "w") as f:
        json.dump(gt_info, f, indent=2)

    cv2.imwrite(os.path.join(output_dir, "search.png"), search_out)
    cv2.imwrite(os.path.join(output_dir, "target.png"), target_out)
    shutil.copyfile(os.path.join(output_dir, "target.png"), os.path.join(output_dir, "reference.png"))

    print(f"[generate_bltwist_pair2] Successfully generated Vertical BL-Twist DRAM pair 2 {output_dir} | GT: ({gt_x}, {gt_y})")
    return os.path.join(output_dir, "search.png"), os.path.join(output_dir, "target.png"), gt_info


if __name__ == "__main__":
    generate_bltwist_pair2()
