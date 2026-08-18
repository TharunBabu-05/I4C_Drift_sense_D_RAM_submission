#!/usr/bin/env python3
"""
Generate Pair 1 for generated_trench (Grayscale Trench-Cell DRAM - Center GT 500,500)
======================================================================================
Synthesizes 1000x1000 Grayscale normal image matching SEM cross-section of Trench-Cell DRAM IC:
  - Full-frame 1000x1000 IC layout
  - Deep trench capacitor wells (dark oval pits etched into silicon substrate)
  - Bright silicon substrate walls and Shallow Trench Isolation (STI) ridges
  - Collar oxide rings around trench openings
  - Buried strap connections between trench capacitor and access transistor
  - Horizontal passing wordlines and vertical bitline contacts
  - Seamless Ground Truth Target Landmark at Center (500, 500): Cluster of bright-collar
    trench cells with metallic strap overlay
  - Heavy SEM noise pipeline: Gaussian + Poisson + Speckle + Vignette + Barrel distortion
"""

import json
import os
import shutil
import cv2
import numpy as np


def render_trench_cell_dram_scene(h=1000, w=1000, gt_x=500, gt_y=500, seed=8001):
    """Render 1000x1000 uint8 Grayscale Normal Image of Trench-Cell DRAM Semiconductor Circuitry.

    Trench-Cell DRAM key features:
      - Capacitor is etched DOWN into the silicon substrate (opposite of stacked)
      - Trenches appear as dark pits in SEM top-down view
      - Collar oxide visible as bright ring around each trench opening
      - Buried strap connects trench to access transistor drain
      - STI (Shallow Trench Isolation) ridges separate active area rows
      - Cell layout: staggered grid with ~20px col pitch, ~14px row pitch
    """
    np.random.seed(seed)

    # Trench cell geometry
    col_pitch = 20   # Horizontal spacing between trench centers
    row_pitch = 14   # Vertical spacing between trench centers
    trench_rx = 6    # Trench ellipse semi-axis X (horizontal)
    trench_ry = 4    # Trench ellipse semi-axis Y (vertical)

    # Bright silicon substrate base (trenches are dark holes in bright substrate)
    img_gray = np.full((h, w), 115, dtype=np.float32)

    # ─── 1. STI (Shallow Trench Isolation) Ridge Rows ───────────────────
    # Horizontal isolation ridges separating active-area rows
    for sty in range(0, h, row_pitch * 3):
        cv2.line(img_gray, (0, sty), (w, sty), 145.0, 2)
        cv2.line(img_gray, (0, sty + 1), (w, sty + 1), 135.0, 1)

    # ─── 2. Deep Trench Capacitor Wells ─────────────────────────────────
    # Staggered grid of dark elliptical trench pits
    for col_idx, cx in enumerate(range(col_pitch // 2, w, col_pitch)):
        y_offset = (row_pitch // 2) if (col_idx % 2 == 1) else 0
        for cy in range(row_pitch // 2 + y_offset, h, row_pitch):
            if cx - trench_rx < 0 or cx + trench_rx >= w:
                continue
            if cy - trench_ry < 0 or cy + trench_ry >= h:
                continue

            # Collar oxide ring (bright ring around trench opening)
            cv2.ellipse(img_gray, (cx, cy), (trench_rx + 2, trench_ry + 2),
                        0, 0, 360, 165.0, 2)

            # Trench pit wall gradient (medium gray rim)
            cv2.ellipse(img_gray, (cx, cy), (trench_rx + 1, trench_ry + 1),
                        0, 0, 360, 85.0, 1)

            # Deep trench interior (dark elliptical pit)
            cv2.ellipse(img_gray, (cx, cy), (trench_rx, trench_ry),
                        0, 0, 360, 35.0, -1)

            # Trench bottom plate electrode (slightly brighter center dot)
            cv2.circle(img_gray, (cx, cy), 2, 55.0, -1)

            # Buried strap connection (small bright rectangle on one side)
            strap_side = 1 if (col_idx % 2 == 0) else -1
            strap_x = cx + strap_side * (trench_rx + 1)
            cv2.rectangle(img_gray, (strap_x - 1, cy - 1),
                          (strap_x + 1, cy + 1), 175.0, -1)

    # ─── 3. Horizontal Wordlines (Gate Conductors) ──────────────────────
    # Polysilicon wordlines running horizontally across the trench array
    for wl_y in range(row_pitch // 2 - 2, h, row_pitch):
        cv2.line(img_gray, (0, wl_y), (w, wl_y), 100.0, 1)
    # Second set of wordlines (passing gates)
    for wl_y in range(row_pitch // 2 + 2, h, row_pitch):
        cv2.line(img_gray, (0, wl_y), (w, wl_y), 90.0, 1)

    # ─── 4. Vertical Bitline Contacts & Metal-1 Bitlines ───────────────
    # Bitline contacts between trench pairs
    for bx in range(col_pitch, w - col_pitch, col_pitch):
        # Thin vertical bitline trace
        cv2.line(img_gray, (bx, 0), (bx, h), 130.0, 1)
        # Contact pads at regular intervals
        for by in range(row_pitch, h - row_pitch, row_pitch * 2):
            cv2.rectangle(img_gray, (bx - 2, by - 1), (bx + 2, by + 1), 190.0, -1)

    # ─── 5. Array Edge & Power Bus Straps ───────────────────────────────
    # Thicker horizontal bus lines at major intervals
    bus_interval = row_pitch * 8  # ~112 px
    for bus_y in range(bus_interval, h - 10, bus_interval):
        cv2.rectangle(img_gray, (0, bus_y - 2), (w, bus_y + 2), 70.0, -1)
        cv2.line(img_gray, (0, bus_y - 3), (w, bus_y - 3), 140.0, 1)
        cv2.line(img_gray, (0, bus_y + 3), (w, bus_y + 3), 140.0, 1)
        # Bus strap contact pads
        for bsx in range(col_pitch, w - col_pitch, col_pitch * 4):
            cv2.rectangle(img_gray, (bsx - 4, bus_y - 3), (bsx + 4, bus_y + 3), 180.0, -1)
            cv2.circle(img_gray, (bsx, bus_y), 1, 210.0, -1)

    # ─── 6. Subtle In-Pattern Semiconductor Landmark at (gt_x, gt_y) ───
    # Highlight a 3×4 cluster of trench cells with bright collar + metal strap overlay
    start_col = (gt_x // col_pitch) * col_pitch + col_pitch // 2
    start_row = (gt_y // row_pitch) * row_pitch + row_pitch // 2

    for dc in range(3):
        for dr in range(4):
            cur_x = start_col + dc * col_pitch
            cur_y_base = start_row + dr * row_pitch
            # Stagger offset
            col_idx_eff = (cur_x - col_pitch // 2) // col_pitch
            y_off = (row_pitch // 2) if (col_idx_eff % 2 == 1) else 0
            cur_y = cur_y_base + y_off

            if (cur_x - trench_rx - 3 >= 0 and cur_x + trench_rx + 3 < w and
                    cur_y - trench_ry - 3 >= 0 and cur_y + trench_ry + 3 < h):

                # Bright highlighted collar oxide (extra bright)
                cv2.ellipse(img_gray, (cur_x, cur_y), (trench_rx + 3, trench_ry + 3),
                            0, 0, 360, 230.0, 2)
                cv2.ellipse(img_gray, (cur_x, cur_y), (trench_rx + 2, trench_ry + 2),
                            0, 0, 360, 220.0, 2)

                # Bright trench interior (inverted contrast for landmark)
                cv2.ellipse(img_gray, (cur_x, cur_y), (trench_rx, trench_ry),
                            0, 0, 360, 200.0, -1)
                cv2.circle(img_gray, (cur_x, cur_y), 2, 240.0, -1)

                # Metal strap interconnect to right
                if dc < 2:
                    next_x = cur_x + col_pitch
                    cv2.line(img_gray, (cur_x + trench_rx + 2, cur_y),
                             (next_x - trench_rx - 2, cur_y), 235.0, 1)

                # Metal strap interconnect downward
                if dr < 3:
                    cv2.line(img_gray, (cur_x, cur_y + trench_ry + 2),
                             (cur_x, cur_y + trench_ry + 8), 235.0, 1)

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
        [cx, 0, cx], [0, cy, cy], [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float64)
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_32FC1
    )
    return cv2.remap(img_u8, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def apply_scan_line_noise(img_f32, intensity=3.0, freq=0.02):
    """Horizontal scan line intensity ripple (SEM raster scan jitter)."""
    h = img_f32.shape[0]
    y_vals = np.arange(h, dtype=np.float32)
    ripple = intensity * np.sin(2 * np.pi * freq * y_vals + np.random.uniform(0, 2 * np.pi))
    ripple += np.random.normal(0, intensity * 0.3, h).astype(np.float32)
    return img_f32 + ripple[:, np.newaxis]


# [STRIPPED] def add_heavy_sem_noise_grayscale(image_uint8, seed=8001, is_target=False):
# [STRIPPED]     """Full SEM noise pipeline: Gaussian + Poisson + Speckle + Scan Lines + Vignette + Barrel."""
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
def generate_trench_pair1(output_dir="generated_trench/pair1", gt_x=500, gt_y=500, seed=8001):
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    search_clean = render_trench_cell_dram_scene(h=h, w=w, gt_x=gt_x, gt_y=gt_y, seed=seed)

    crop_size = 100
    x0 = max(0, min(w - crop_size, gt_x - crop_size // 2))
    y0 = max(0, min(h - crop_size, gt_y - crop_size // 2))
    x1, y1 = x0 + crop_size, y0 + crop_size

    crop_patch = search_clean[y0:y1, x0:x1]
    target_clean = cv2.resize(crop_patch, (1000, 1000), interpolation=cv2.INTER_NEAREST)

    # No noise — clean images
    gt_info = {
        "center_x": gt_x,
        "center_y": gt_y,
        "target_name": "Grayscale Trench-Cell DRAM - Center GT 500,500 (Pair 1)",
        "pair_id": "pair1",
        "scale_factor": 10.0
    }

    with open(os.path.join(output_dir, "groundtruth.json"), "w") as f:
        json.dump(gt_info, f, indent=2)

    cv2.imwrite(os.path.join(output_dir, "search.png"), search_clean)
    cv2.imwrite(os.path.join(output_dir, "target.png"), target_clean)
    shutil.copyfile(os.path.join(output_dir, "target.png"), os.path.join(output_dir, "reference.png"))

    print(f"[generate_trench_pair1] Successfully generated Trench-Cell DRAM pair 1 {output_dir} | GT: ({gt_x}, {gt_y})")
    return os.path.join(output_dir, "search.png"), os.path.join(output_dir, "target.png"), gt_info


if __name__ == "__main__":
    generate_trench_pair1()
