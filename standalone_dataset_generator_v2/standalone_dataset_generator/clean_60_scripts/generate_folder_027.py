#!/usr/bin/env python3
"""
Generate Pair 1 for generated_spt (Grayscale Substrate-Plate Trench DRAM - Center GT 500,500)
==============================================================================================
Synthesizes 1000x1000 Grayscale SEM image of SPT DRAM IC structure:
  - Full-frame 1000x1000 IC layout
  - SPT cell architecture: substrate acts as capacitor plate, storage node (polysilicon)
    fills INSIDE the trench, thin node dielectric lines the trench wall
  - Concentric ring cell cross-section: bright poly-Si storage node center → dark dielectric
    ring → medium substrate plate → collar oxide rim
  - Buried strap connections from storage node to access transistor
  - Horizontal wordlines (gate conductors) and vertical bitline contacts
  - Seamless GT landmark at Center (500, 500): cluster with bright storage node overlay
  - Light noise: speckle + vignette only
"""

import json
import os
import shutil
import cv2
import numpy as np


def render_spt_dram_scene(h=1000, w=1000, gt_x=500, gt_y=500, seed=9001):
    """Render 1000x1000 uint8 Grayscale SPT DRAM.

    Substrate-Plate Trench (SPT) DRAM key features:
      - Substrate = one capacitor plate (medium gray, doped Si)
      - Deep trench etched into substrate
      - Thin node dielectric (oxide/nitride/oxide) lines trench walls (dark ring)
      - Storage node = polysilicon fill INSIDE the trench (bright center)
      - Collar oxide at trench top prevents parasitic leakage (bright rim)
      - Buried strap connects poly-Si storage node to transistor drain
      - Cell pitch: 22px col × 16px row, staggered layout
    """
    np.random.seed(seed)

    col_pitch = 22   # Horizontal cell spacing
    row_pitch = 16   # Vertical cell spacing
    # Trench radii for concentric ring structure
    r_collar = 8     # Outer collar oxide radius
    r_substrate = 7  # Substrate plate edge
    r_dielectric = 5 # Node dielectric (ONO) ring
    r_storage = 3    # Inner polysilicon storage node

    # Medium gray substrate base (doped silicon = capacitor plate)
    img_gray = np.full((h, w), 95, dtype=np.float32)

    # ─── 1. Shallow Trench Isolation (STI) grid ────────────────────────
    # Horizontal STI ridges separating active area rows
    for sty in range(0, h, row_pitch * 4):
        cv2.rectangle(img_gray, (0, sty), (w, sty + 2), 140.0, -1)
    # Vertical STI ridges separating column pairs
    for stx in range(0, w, col_pitch * 4):
        cv2.rectangle(img_gray, (stx, 0), (stx + 2, h), 135.0, -1)

    # ─── 2. SPT Trench Capacitor Cells (concentric ring cross-section) ─
    for col_idx, cx in enumerate(range(col_pitch // 2, w, col_pitch)):
        y_offset = (row_pitch // 2) if (col_idx % 2 == 1) else 0
        for cy in range(row_pitch // 2 + y_offset, h, row_pitch):
            if (cx - r_collar - 1 < 0 or cx + r_collar + 1 >= w or
                    cy - r_collar - 1 < 0 or cy + r_collar + 1 >= h):
                continue

            # Layer 1 (outermost): Collar oxide rim — bright ring preventing leakage
            cv2.circle(img_gray, (cx, cy), r_collar, 160.0, 2)

            # Layer 2: Substrate plate region (doped Si surrounding trench)
            cv2.circle(img_gray, (cx, cy), r_substrate, 110.0, 1)

            # Layer 3: Node dielectric (ONO stack) — dark ring lining trench wall
            cv2.circle(img_gray, (cx, cy), r_dielectric, 45.0, 2)

            # Layer 4: Trench interior fill — dark trench cavity
            cv2.circle(img_gray, (cx, cy), r_dielectric - 1, 55.0, -1)

            # Layer 5 (innermost): Polysilicon storage node — bright center fill
            cv2.circle(img_gray, (cx, cy), r_storage, 195.0, -1)
            # Storage node contact dot (top of poly-Si)
            cv2.circle(img_gray, (cx, cy), 1, 220.0, -1)

            # Buried strap: connects poly-Si storage node to transistor drain
            # Small bright rectangle on alternating sides
            strap_dir = 1 if (col_idx % 2 == 0) else -1
            strap_x = cx + strap_dir * (r_collar + 1)
            if 0 <= strap_x - 1 and strap_x + 1 < w:
                cv2.rectangle(img_gray, (strap_x - 1, cy - 1),
                              (strap_x + 1, cy + 1), 180.0, -1)

    # ─── 3. Wordlines (polysilicon gate conductors) ─────────────────────
    # Two wordlines per cell row: active WL and passing WL
    for wl_y in range(row_pitch // 2 - 3, h, row_pitch):
        cv2.line(img_gray, (0, wl_y), (w, wl_y), 120.0, 1)
    for wl_y in range(row_pitch // 2 + 3, h, row_pitch):
        cv2.line(img_gray, (0, wl_y), (w, wl_y), 85.0, 1)

    # ─── 4. Vertical Bitlines (Metal-1) & contacts ─────────────────────
    for bx in range(col_pitch, w - col_pitch, col_pitch):
        cv2.line(img_gray, (bx, 0), (bx, h), 130.0, 1)
        # Bitline contact pads between trench pairs
        for by in range(row_pitch, h - row_pitch, row_pitch * 2):
            cv2.rectangle(img_gray, (bx - 2, by - 1), (bx + 2, by + 1), 185.0, -1)

    # ─── 5. Array peripheral bus straps ─────────────────────────────────
    bus_interval = row_pitch * 10  # ~160px
    for bus_y in range(bus_interval, h - 10, bus_interval):
        cv2.rectangle(img_gray, (0, bus_y - 2), (w, bus_y + 2), 65.0, -1)
        cv2.line(img_gray, (0, bus_y - 3), (w, bus_y - 3), 145.0, 1)
        cv2.line(img_gray, (0, bus_y + 3), (w, bus_y + 3), 145.0, 1)
        for bsx in range(col_pitch * 2, w - col_pitch, col_pitch * 5):
            cv2.rectangle(img_gray, (bsx - 5, bus_y - 3), (bsx + 5, bus_y + 3), 175.0, -1)
            cv2.circle(img_gray, (bsx, bus_y), 1, 210.0, -1)

    # ─── 6. Subtle landmark at (gt_x, gt_y) ────────────────────────────
    # Highlight 3×4 cluster: bright storage nodes + interconnect straps
    start_col = (gt_x // col_pitch) * col_pitch + col_pitch // 2
    start_row = (gt_y // row_pitch) * row_pitch + row_pitch // 2

    for dc in range(3):
        for dr in range(4):
            cur_x = start_col + dc * col_pitch
            col_idx_eff = (cur_x - col_pitch // 2) // col_pitch
            y_off = (row_pitch // 2) if (col_idx_eff % 2 == 1) else 0
            cur_y = start_row + dr * row_pitch + y_off

            if (cur_x - r_collar - 2 >= 0 and cur_x + r_collar + 2 < w and
                    cur_y - r_collar - 2 >= 0 and cur_y + r_collar + 2 < h):

                # Draw identical cell circle geometry (same radii as standard cells throughout)
                cv2.circle(img_gray, (cur_x, cur_y), r_collar, 160.0, 2)
                cv2.circle(img_gray, (cur_x, cur_y), r_substrate, 110.0, 1)
                cv2.circle(img_gray, (cur_x, cur_y), r_dielectric, 45.0, 2)
                cv2.circle(img_gray, (cur_x, cur_y), r_dielectric - 1, 55.0, -1)
                cv2.circle(img_gray, (cur_x, cur_y), r_storage, 220.0, -1)
                cv2.circle(img_gray, (cur_x, cur_y), 1, 245.0, -1)

                # Horizontal metal interconnect to right cell
                if dc < 2:
                    next_x = cur_x + col_pitch
                    cv2.line(img_gray, (cur_x + r_collar, cur_y),
                             (next_x - r_collar, cur_y), 235.0, 2)
                # Vertical metal interconnect downward
                if dr < 3:
                    cv2.line(img_gray, (cur_x, cur_y + r_collar),
                             (cur_x, cur_y + r_collar + 6), 235.0, 2)

    return np.clip(img_gray, 0, 255).astype(np.uint8)


# =============================================================================
# Light Noise: Speckle + Vignette only
# =============================================================================

def apply_speckle_noise(img_f32, intensity=0.10):
    """Multiplicative speckle noise (light grain)."""
    speckle = np.random.randn(*img_f32.shape).astype(np.float32) * intensity
    return img_f32 * (1.0 + speckle)


def apply_vignette(img_f32, strength=0.20):
    """Radial vignette darkening."""
    h, w = img_f32.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    max_r = np.sqrt(cx ** 2 + cy ** 2)
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)
    vignette_map = 1.0 - strength * (r / max_r) ** 2
    return img_f32 * vignette_map


# [STRIPPED] def add_light_noise(image_uint8, seed=9001, is_target=False):
# [STRIPPED]     """Light noise pipeline: speckle + vignette only."""
# [STRIPPED]     np.random.seed(seed)
# [STRIPPED]     img = image_uint8.astype(np.float32)
# [STRIPPED] 
# [STRIPPED]     if not is_target:
# [STRIPPED]         img = apply_speckle_noise(img, intensity=0.12)
# [STRIPPED]         img = apply_vignette(img, strength=0.20)
# [STRIPPED]     else:
# [STRIPPED]         img = apply_speckle_noise(img, intensity=0.05)
# [STRIPPED]         img = apply_vignette(img, strength=0.08)
# [STRIPPED] 
# [STRIPPED]     return np.clip(img, 0, 255).astype(np.uint8)
# [STRIPPED] 
# [STRIPPED] 
def generate_spt_pair1(output_dir="generated_spt/pair1", gt_x=500, gt_y=500, seed=9001):
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    search_clean = render_spt_dram_scene(h=h, w=w, gt_x=gt_x, gt_y=gt_y, seed=seed)

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
        "target_name": "Grayscale SPT DRAM (Substrate-Plate Trench) - Center GT 500,500 (Pair 1)",
        "pair_id": "pair1",
        "scale_factor": 10.0
    }

    with open(os.path.join(output_dir, "groundtruth.json"), "w") as f:
        json.dump(gt_info, f, indent=2)

    cv2.imwrite(os.path.join(output_dir, "search.png"), search_out)
    cv2.imwrite(os.path.join(output_dir, "target.png"), target_out)
    shutil.copyfile(os.path.join(output_dir, "target.png"), os.path.join(output_dir, "reference.png"))

    print(f"[generate_spt_pair1] Successfully generated SPT DRAM pair 1 {output_dir} | GT: ({gt_x}, {gt_y})")
    return os.path.join(output_dir, "search.png"), os.path.join(output_dir, "target.png"), gt_info


if __name__ == "__main__":
    generate_spt_pair1()
