#!/usr/bin/env python3
"""
Generate Pair 2 for generated_7fsquare (Grayscale 7F² DRAM Cell Layout - Top GT 500,310)
=========================================================================================
Synthesizes 1000x1000 Grayscale normal image matching SEM micro-photo of 7F² DRAM IC structure:
  - Full-frame 1000x1000 IC layout
  - 7F² folded-bitline DRAM cell arrays with staggered active area rectangles
  - Alternating storage and passing wordline (gate) stripes
  - Bitline contacts (square) and storage node contacts (circular) per cell
  - Sense amplifier bands at regular intervals
  - Seamless Ground Truth Target Landmark at Top (500, 310): Bright cluster of 7F² cells
    with interconnected metal strap overlay
  - Heavy SEM noise pipeline: Gaussian + Poisson + Speckle + Vignette + Barrel distortion
"""

import json
import os
import shutil
import cv2
import numpy as np


def render_7fsquare_dram_scene(h=1000, w=1000, gt_x=500, gt_y=310, seed=7002):
    """Render 1000x1000 uint8 Grayscale Normal Image of 7F² DRAM Semiconductor Circuitry.

    7F² cell layout key features:
      - F (feature size) = 4px  →  cell footprint = 4F x 1.75F = 16 x 7 px
      - Folded bitline architecture: bitlines run vertically, wordlines horizontally
      - Staggered active areas between adjacent bitline columns (offset by half cell height)
      - Two wordlines per cell row: one storage WL (connects to capacitor) and one passing WL
      - Bitline contacts (BL-C) as small squares, storage node contacts (SN-C) as circles
    """
    np.random.seed(seed)
    F = 4  # Minimum feature size in pixels

    # Cell dimensions: 4F wide × 1.75F tall  →  16 × 7 px
    cell_w = 4 * F       # 16 px
    cell_h = int(1.75 * F + 0.5)  # 7 px

    # Full-frame dark IC substrate base
    img_gray = np.full((h, w), 28, dtype=np.float32)

    # ─── 1. Wordline (Gate) Stripes ─────────────────────────────────────
    wl_pitch = cell_h
    for wy in range(0, h, wl_pitch):
        storage_wl_y = wy + 2
        if 0 <= storage_wl_y < h:
            cv2.line(img_gray, (0, storage_wl_y), (w, storage_wl_y), 105.0, 1)
        passing_wl_y = wy + 5
        if 0 <= passing_wl_y < h:
            cv2.line(img_gray, (0, passing_wl_y), (w, passing_wl_y), 80.0, 1)

    # ─── 2. Staggered Active Area Rectangles ────────────────────────────
    for col_idx, cx in enumerate(range(0, w, cell_w)):
        y_offset = (cell_h // 2) if (col_idx % 2 == 1) else 0
        for ry in range(y_offset, h, cell_h):
            aa_x1 = cx + 2
            aa_y1 = ry + 1
            aa_x2 = cx + cell_w - 2
            aa_y2 = ry + cell_h - 1
            if aa_x2 < w and aa_y2 < h:
                cv2.rectangle(img_gray, (aa_x1, aa_y1), (aa_x2, aa_y2), 130.0, 1)
                cv2.rectangle(img_gray, (aa_x1 + 1, aa_y1 + 1), (aa_x2 - 1, aa_y2 - 1), 65.0, -1)

                blc_x = cx + cell_w // 2
                blc_y = ry + 1
                cv2.rectangle(img_gray, (blc_x - 1, blc_y), (blc_x + 1, blc_y + 1), 200.0, -1)

                snc_x = cx + cell_w // 2
                snc_y = ry + cell_h - 2
                cv2.circle(img_gray, (snc_x, snc_y), 1, 185.0, -1)

    # ─── 3. Vertical Bitlines (Metal-1 layer) ──────────────────────────
    for bx in range(cell_w // 2, w, cell_w):
        cv2.line(img_gray, (bx, 0), (bx, h), 145.0, 1)

    # ─── 4. Sense Amplifier Bands ───────────────────────────────────────
    sa_interval = cell_h * 18
    for sa_y in range(sa_interval, h - 10, sa_interval):
        cv2.rectangle(img_gray, (0, sa_y - 3), (w, sa_y + 3), 55.0, -1)
        for sa_x in range(4, w - 4, cell_w):
            cv2.rectangle(img_gray, (sa_x, sa_y - 2), (sa_x + 6, sa_y + 2), 160.0, 1)
            cv2.circle(img_gray, (sa_x + 3, sa_y), 1, 190.0, -1)
        cv2.line(img_gray, (0, sa_y - 4), (w, sa_y - 4), 115.0, 1)
        cv2.line(img_gray, (0, sa_y + 4), (w, sa_y + 4), 115.0, 1)

    # ─── 5. Sub-Wordline Driver (SWD) Columns ──────────────────────────
    swd_pitch = cell_w * 8
    for swd_x in range(swd_pitch, w - 4, swd_pitch):
        cv2.line(img_gray, (swd_x, 0), (swd_x, h), 100.0, 2)
        cv2.line(img_gray, (swd_x - 1, 0), (swd_x - 1, h), 70.0, 1)
        cv2.line(img_gray, (swd_x + 1, 0), (swd_x + 1, h), 70.0, 1)
        for swd_y in range(12, h - 12, cell_h * 4):
            cv2.rectangle(img_gray, (swd_x - 3, swd_y - 2), (swd_x + 3, swd_y + 2), 175.0, -1)

    # ─── 6. Subtle In-Pattern Semiconductor Landmark at (gt_x, gt_y) ───
    start_col = (gt_x // cell_w) * cell_w
    start_row = (gt_y // cell_h) * cell_h

    for dc in range(3):
        for dr in range(4):
            cur_x = start_col + dc * cell_w
            cur_y = start_row + dr * cell_h
            y_off = (cell_h // 2) if (((cur_x // cell_w) % 2) == 1) else 0
            cur_y_adj = cur_y + y_off

            aa_x1 = cur_x + 2
            aa_y1 = cur_y_adj + 1
            aa_x2 = cur_x + cell_w - 2
            aa_y2 = cur_y_adj + cell_h - 1

            if 0 <= aa_x1 and aa_x2 < w and 0 <= aa_y1 and aa_y2 < h:
                cv2.rectangle(img_gray, (aa_x1, aa_y1), (aa_x2, aa_y2), 220.0, 1)
                cv2.rectangle(img_gray, (aa_x1 + 1, aa_y1 + 1), (aa_x2 - 1, aa_y2 - 1), 200.0, -1)

                blc_x = cur_x + cell_w // 2
                cv2.rectangle(img_gray, (blc_x - 1, aa_y1), (blc_x + 1, aa_y1 + 1), 245.0, -1)

                snc_x = cur_x + cell_w // 2
                cv2.circle(img_gray, (snc_x, aa_y2 - 1), 1, 240.0, -1)

                if dc < 2:
                    next_x = cur_x + cell_w + 2
                    mid_y = (aa_y1 + aa_y2) // 2
                    cv2.line(img_gray, (aa_x2, mid_y), (next_x, mid_y), 230.0, 1)

                if dr < 3:
                    cv2.line(img_gray, (cur_x + cell_w // 2, aa_y2),
                             (cur_x + cell_w // 2, aa_y2 + 4), 230.0, 1)

    return np.clip(img_gray, 0, 255).astype(np.uint8)


# =============================================================================
# Enhanced Multi-Noise SEM Pipeline
# =============================================================================

def apply_gaussian_noise(img_f32, sigma=12.0):
    """Additive Gaussian read noise (sensor electronics)."""
    noise = np.random.normal(0, sigma, img_f32.shape).astype(np.float32)
    return img_f32 + noise


def apply_poisson_noise(img_f32, scale=6.0):
    """Poisson shot noise (electron counting statistics in SEM)."""
    counts = np.maximum(0, img_f32) / 255.0 * scale
    noisy_counts = np.random.poisson(counts).astype(np.float32)
    return noisy_counts / scale * 255.0


def apply_speckle_noise(img_f32, intensity=0.15):
    """Multiplicative speckle noise (granular interference in electron beam)."""
    speckle = np.random.randn(*img_f32.shape).astype(np.float32) * intensity
    return img_f32 * (1.0 + speckle)


def apply_vignette(img_f32, strength=0.35):
    """Radial vignette darkening (SEM column aberration / detector falloff)."""
    h, w = img_f32.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    max_r = np.sqrt(cx ** 2 + cy ** 2)
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)
    vignette_map = 1.0 - strength * (r / max_r) ** 2
    return img_f32 * vignette_map


def apply_barrel_distortion(img_u8, k1=0.12, k2=0.04):
    """Barrel/pincushion distortion (SEM electromagnetic lens aberration)."""
    h, w = img_u8.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    camera_matrix = np.array([
        [cx, 0, cx],
        [0, cy, cy],
        [0, 0,  1]
    ], dtype=np.float64)

    dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float64)

    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_32FC1
    )
    distorted = cv2.remap(img_u8, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return distorted


def apply_scan_line_noise(img_f32, intensity=3.0, freq=0.02):
    """Horizontal scan line intensity ripple (SEM raster scan jitter)."""
    h, w = img_f32.shape[:2]
    y_vals = np.arange(h, dtype=np.float32)
    ripple = intensity * np.sin(2 * np.pi * freq * y_vals + np.random.uniform(0, 2 * np.pi))
    ripple += np.random.normal(0, intensity * 0.3, h).astype(np.float32)
    return img_f32 + ripple[:, np.newaxis]


# [STRIPPED] def add_heavy_sem_noise_grayscale(image_uint8, seed=7002, is_target=False):
# [STRIPPED]     """Full SEM noise pipeline: Gaussian + Poisson + Speckle + Scan Lines + Vignette + Barrel Distortion."""
# [STRIPPED]     np.random.seed(seed)
# [STRIPPED]     img = image_uint8.astype(np.float32)
# [STRIPPED] 
# [STRIPPED]     if not is_target:
# [STRIPPED]         img = apply_poisson_noise(img, scale=6.0)
# [STRIPPED]         img = apply_gaussian_noise(img, sigma=14.0)
# [STRIPPED]         img = apply_speckle_noise(img, intensity=0.18)
# [STRIPPED]         img = apply_scan_line_noise(img, intensity=3.5, freq=0.015)
# [STRIPPED]         img = apply_vignette(img, strength=0.30)
# [STRIPPED]         img = np.clip(img, 0, 255).astype(np.uint8)
# [STRIPPED]         img = apply_barrel_distortion(img, k1=0.10, k2=0.03)
# [STRIPPED]     else:
# [STRIPPED]         img = apply_gaussian_noise(img, sigma=5.0)
# [STRIPPED]         img = apply_speckle_noise(img, intensity=0.06)
# [STRIPPED]         img = apply_vignette(img, strength=0.12)
# [STRIPPED]         img = np.clip(img, 0, 255).astype(np.uint8)
# [STRIPPED] 
# [STRIPPED]     return np.clip(img, 0, 255).astype(np.uint8) if isinstance(img, np.floating) else img
# [STRIPPED] 
# [STRIPPED] 
def generate_7fsquare_pair2(output_dir="generated_7fsquare/pair2", gt_x=500, gt_y=310, seed=7002):
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    search_clean = render_7fsquare_dram_scene(h=h, w=w, gt_x=gt_x, gt_y=gt_y, seed=seed)

    crop_size = 100
    x0 = max(0, min(w - crop_size, gt_x - crop_size // 2))
    y0 = max(0, min(h - crop_size, gt_y - crop_size // 2))
    x1, y1 = x0 + crop_size, y0 + crop_size

    crop_patch = search_clean[y0:y1, x0:x1]
    target_clean = cv2.resize(crop_patch, (1000, 1000), interpolation=cv2.INTER_NEAREST)

    search_noisy = search_clean  # noise stripped by v2 pipeline
    target_noisy = target_clean  # noise stripped by v2 pipeline

    gt_info = {
        "center_x": gt_x,
        "center_y": gt_y,
        "target_name": "Grayscale 7F² DRAM Cell Layout - Top GT 500,310 (Pair 2)",
        "pair_id": "pair2",
        "scale_factor": 10.0
    }

    with open(os.path.join(output_dir, "groundtruth.json"), "w") as f:
        json.dump(gt_info, f, indent=2)

    cv2.imwrite(os.path.join(output_dir, "search.png"), search_noisy)
    cv2.imwrite(os.path.join(output_dir, "target.png"), target_noisy)
    shutil.copyfile(os.path.join(output_dir, "target.png"), os.path.join(output_dir, "reference.png"))

    print(f"[generate_7fsquare_pair2] Successfully generated 7F² DRAM pair 2 {output_dir} | GT: ({gt_x}, {gt_y})")
    return os.path.join(output_dir, "search.png"), os.path.join(output_dir, "target.png"), gt_info


if __name__ == "__main__":
    generate_7fsquare_pair2()
