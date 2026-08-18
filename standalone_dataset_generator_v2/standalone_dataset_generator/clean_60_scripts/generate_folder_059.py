#!/usr/bin/env python3
"""
Generate Pair 1 for Grayscale Concentric Ring-Capacitor DRAM (Center GT 500,500)
=================================================================================
Synthesizes 1000x1000 Grayscale SEM image matching Concentric Ring-Capacitor DRAM:
  - Full-frame 1000x1000 IC layout matching reference SEM micrograph
  - Concentric circular storage capacitors with multi-ring walls and central solid node core
  - Spoke interconnect tabs connecting circular capacitors to orthogonal grid tracks
  - Square transistor via nodes with dark pinhole centers at Wordline-Bitline intersections
  - Horizontal Word Lines (WL) & vertical Bit Lines (BL) forming a rectangular mesh frame
  - Authentic SEM footer scale bar & metadata text ("Concentric Ring-Capacitor 3D DRAM (Top View)")
  - Creative In-Pattern Embedded Landmark at Center (500, 500):
    Fine interconnect bridge line linking two adjacent circular capacitors with a central micro-via (NO big white box!)
  - Memory-optimized uint8 pipeline with full SEM noise
  - Stored in Script_and_their_images/059/
"""

import json
import os
import shutil
import cv2
import numpy as np


def render_ringcap_dram_scene(h=1000, w=1000, gt_x=500, gt_y=500, seed=25001):
    """Render 1000x1000 uint8 Grayscale image of Concentric Ring-Capacitor DRAM Circuitry."""
    np.random.seed(seed)

    col_pitch = 32  # Column pitch
    row_pitch = 32  # Row pitch

    # Dark silicon oxide substrate base
    img_gray = np.full((h, w), 35, dtype=np.uint8)

    # ─── 1. Vertical Bit Lines (BL Tracks) ─────────────────────────────
    for cx in range(col_pitch // 2, w + col_pitch, col_pitch):
        cv2.rectangle(img_gray, (cx - 3, 0), (cx + 3, h), 130, -1)
        cv2.line(img_gray, (cx - 3, 0), (cx - 3, h), 190, 1)
        cv2.line(img_gray, (cx + 3, 0), (cx + 3, h), 190, 1)

    # ─── 2. Horizontal Word Lines (WL Tracks) ───────────────────────────
    for wy in range(row_pitch // 2, h - 40, row_pitch):
        cv2.rectangle(img_gray, (0, wy - 3), (w, wy + 3), 160, -1)
        cv2.line(img_gray, (0, wy - 3), (w, wy - 3), 215, 1)
        cv2.line(img_gray, (0, wy + 3), (w, wy + 3), 215, 1)

    # ─── 3. Concentric Circular Storage Capacitors & Spoke Tabs ─────────
    for cx in range(col_pitch // 2, w + col_pitch, col_pitch):
        for wy in range(row_pitch // 2, h - 40, row_pitch):
            cy = wy + row_pitch // 2
            if cy + 14 < h - 40:
                # Horizontal spoke interconnect tabs
                cv2.line(img_gray, (cx - 15, cy), (cx + 15, cy), 165, 2)
                cv2.line(img_gray, (cx, cy - 15), (cx, cy + 15), 165, 2)

                # Outer Circular Ring Rim
                cv2.circle(img_gray, (cx, cy), 13, 180, 2)
                # Dielectric Liner Ring
                cv2.circle(img_gray, (cx, cy), 10, 130, 2)
                # Recessed Annular Trench Cavity
                cv2.circle(img_gray, (cx, cy), 7, 75, -1)
                # Central Solid Circular Node Core
                cv2.circle(img_gray, (cx, cy), 4, 245, -1)

    # ─── 4. Square Transistor Access Vias with Dark Pinhole Centers ──────
    for cx in range(col_pitch // 2, w + col_pitch, col_pitch):
        for wy in range(row_pitch // 2, h - 40, row_pitch):
            cv2.rectangle(img_gray, (cx - 4, wy - 4), (cx + 4, wy + 4), 220, -1)
            cv2.rectangle(img_gray, (cx - 4, wy - 4), (cx + 4, wy + 4), 250, 1)
            cv2.rectangle(img_gray, (cx - 1, wy - 1), (cx + 1, wy + 1), 50, -1)

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

    cv2.putText(img_gray, "Concentric Ring-Capacitor 3D DRAM (Top View)", (w // 2 - 170, h - 11),
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


# [STRIPPED] def add_unique_sem_noise(image_uint8, seed=25001, is_target=False):
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
def generate_ringcap_pair1(output_dir="generated_ringcap/pair1", gt_x=500, gt_y=500, seed=25001):
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    search_clean = render_ringcap_dram_scene(h=h, w=w, gt_x=gt_x, gt_y=gt_y, seed=seed)

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
        "target_name": "Grayscale Concentric Ring-Capacitor DRAM - Center GT 500,500 (Pair 1)",
        "pair_id": "pair1",
        "scale_factor": 10.0
    }

    with open(os.path.join(output_dir, "groundtruth.json"), "w") as f:
        json.dump(gt_info, f, indent=2)

    cv2.imwrite(os.path.join(output_dir, "search.png"), search_out)
    cv2.imwrite(os.path.join(output_dir, "target.png"), target_out)
    shutil.copyfile(os.path.join(output_dir, "target.png"), os.path.join(output_dir, "reference.png"))

    print(f"[generate_ringcap_pair1] Successfully generated Ring-Capacitor DRAM pair 1 {output_dir} | GT: ({gt_x}, {gt_y})")
    return os.path.join(output_dir, "search.png"), os.path.join(output_dir, "target.png"), gt_info


if __name__ == "__main__":
    generate_ringcap_pair1()
