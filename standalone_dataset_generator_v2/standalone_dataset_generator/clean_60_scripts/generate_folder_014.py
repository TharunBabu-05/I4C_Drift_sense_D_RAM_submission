#!/usr/bin/env python3
"""
Generate Pair 4 for generated_new (50-Chip Silicon Wafer Array - 484-Point Joined Matrix - Left 310,500)
======================================================================================================
Synthesizes a 1000x1000 px uint8 grayscale search image and 1000x1000 px target reference image.

Design Features:
  - 50 Microchips on a single circular silicon wafer disk (9x9 grid array filtered by wafer radius)
  - Top-Left section of every chip features a 22x22 grid of 484 points (via pads / dots)
  - Regular chips: All 484 points are separate individual dots
  - UNIQUE TARGET LANDMARK (Pair 4 - Left Chip 310,500 [ANOTHER CHIP!]): The 484 points are JOINED together by
    thin metallic interconnect bus lines running through the point matrix!
  - Heavy SEM Noise (Speckle sigma=0.15, Shot noise=18.0, Read noise=12.0, Vignette=0.35)
"""

import json
import os
import shutil
import cv2
import numpy as np


def render_semiconductor_wafer_scene(h=1000, w=1000, gt_x=310, gt_y=500, seed=2004):
    """Render 1000x1000 uint8 grayscale Silicon Wafer containing ~50 microchips with 484 points test section."""
    np.random.seed(seed)
    img = np.full((h, w), 20, dtype=np.float32)

    # 1. Circular Silicon Wafer Disk (Radius 470)
    center_wafer = (w // 2, h // 2)
    radius_wafer = 470
    cv2.circle(img, center_wafer, radius_wafer, 40, -1)

    y_indices, x_indices = np.indices((h, w))
    dist_from_center = np.sqrt((x_indices - center_wafer[0])**2 + (y_indices - center_wafer[1])**2)
    wafer_mask = dist_from_center <= radius_wafer
    img[wafer_mask] += 15.0 * (1.0 - dist_from_center[wafer_mask] / radius_wafer)

    cv2.rectangle(img, (w // 2 - 15, h // 2 - radius_wafer - 5), (w // 2 + 15, h // 2 - radius_wafer + 15), 15, -1)

    # 2. Render ~50 Microchips on Silicon Wafer Disk (9x9 grid on pitch 95px)
    chip_w, chip_h = 80, 80
    pitch_x, pitch_y = 95, 95
    
    cols = [500 + i * pitch_x for i in range(-4, 5)] # [120, 215, 310, 405, 500, 595, 690, 785, 880]
    rows = [500 + j * pitch_y for j in range(-4, 5)] # [120, 215, 310, 405, 500, 595, 690, 785, 880]

    for cy in rows:
        for cx in cols:
            if np.sqrt((cx - center_wafer[0])**2 + (cy - center_wafer[1])**2) <= radius_wafer - 40:
                x1, y1 = cx - chip_w // 2, cy - chip_h // 2
                x2, y2 = cx + chip_w // 2, cy + chip_h // 2

                # Crisp Scribe line outer die frame
                cv2.rectangle(img, (x1, y1), (x2, y2), 220, 1)

                # Inner die substrate
                cv2.rectangle(img, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), 60, -1)

                # Top-Right Section: Word Lines (horizontal parallel stripes)
                for wy in range(y1 + 6, y1 + 32, 3):
                    cv2.line(img, (cx - 2, wy), (x2 - 5, wy), 235, 1)

                # Middle Section: Bit Lines & Memory Cell Matrix (crosshatch grid)
                for bx in range(x1 + 6, x2 - 6, 4):
                    cv2.line(img, (bx, y1 + 35), (bx, y2 - 20), 190, 1)
                for my in range(y1 + 37, y2 - 22, 6):
                    cv2.line(img, (x1 + 6, my), (x2 - 6, my), 150, 1)

                # Bottom Section: Sense Amplifier & I/O Circuitry
                cv2.rectangle(img, (x1 + 6, y2 - 18), (x2 - 6, y2 - 4), 120, -1)
                for px in range(x1 + 8, x2 - 8, 7):
                    cv2.rectangle(img, (px, y2 - 16), (px + 3, y2 - 6), 245, -1)

                # Top-Left Section: 22x22 Grid of 484 Points (via pads / dots)
                start_px = cx - 36
                start_py = cy - 36
                grid_dim = 22 # 22 x 22 = 484 points total!
                spacing = 32.0 / (grid_dim - 1)

                is_gt_chip = (cx == gt_x and cy == gt_y)

                if is_gt_chip:
                    # TARGET CHIP: 484 Points JOINED together by thin metallic trace lines
                    # 1. Background glow area behind joined 484-point matrix
                    cv2.rectangle(img, (start_px - 2, start_py - 2), (start_px + 34, start_py + 34), 180, -1)

                    # 2. Horizontal trace lines joining the 484 points across rows
                    for r in range(grid_dim):
                        ry = int(start_py + r * spacing)
                        rx1 = int(start_px)
                        rx2 = int(start_px + 32)
                        cv2.line(img, (rx1, ry), (rx2, ry), 255, 1)

                    # 3. Vertical trace lines joining the 484 points across columns
                    for c in range(grid_dim):
                        cx_pt = int(start_px + c * spacing)
                        cy1 = int(start_py)
                        cy2 = int(start_py + 32)
                        cv2.line(img, (cx_pt, cy1), (cx_pt, cy2), 255, 1)

                    # 4. Draw the 484 bright points at grid intersections
                    for r in range(grid_dim):
                        for c in range(grid_dim):
                            pt_x = int(start_px + c * spacing)
                            pt_y = int(start_py + r * spacing)
                            cv2.circle(img, (pt_x, pt_y), 1, 255, -1)
                else:
                    # REGULAR CHIP: 484 separate individual dots/points
                    for r in range(grid_dim):
                        for c in range(grid_dim):
                            pt_x = int(start_px + c * spacing)
                            pt_y = int(start_py + r * spacing)
                            cv2.circle(img, (pt_x, pt_y), 1, 210, -1)

    return np.clip(img, 0, 255).astype(np.uint8)


def apply_vignette_noise(image_float, strength=0.35):
    h, w = image_float.shape
    v1 = (1.0 - strength * (np.linspace(-1.0, 1.0, w, dtype=np.float32)**2)).astype(np.float32)
    v2 = (1.0 - strength * (np.linspace(-1.0, 1.0, h, dtype=np.float32)**2)).astype(np.float32)
    vignette = np.outer(v2, v1)
    return image_float * vignette


def apply_speckle_noise(image_float, sigma=0.15):
    noise = np.random.normal(0, sigma, image_float.shape).astype(np.float32)
    return image_float * (1.0 + noise)


# [STRIPPED] def add_heavy_multi_sem_noise(image_uint8, seed=2004, is_target=False):
# [STRIPPED]     """Apply Heavy SEM Noise Pipeline (Speckle sigma=0.15, Shot=18.0, Read=12.0, Vignette=0.35)."""
# [STRIPPED]     np.random.seed(seed if not is_target else seed + 9999)
# [STRIPPED]     img_float = image_uint8.astype(np.float32)
# [STRIPPED] 
# [STRIPPED]     vignetted = apply_vignette_noise(img_float, strength=0.35 if not is_target else 0.20)
# [STRIPPED]     speckled = apply_speckle_noise(vignetted, sigma=0.15 if not is_target else 0.08)
# [STRIPPED] 
# [STRIPPED]     shot_noise = np.random.normal(0, 18.0 if not is_target else 10.0, img_float.shape).astype(np.float32)
# [STRIPPED]     read_noise = np.random.normal(0, 12.0 if not is_target else 6.0, img_float.shape).astype(np.float32)
# [STRIPPED] 
# [STRIPPED]     noisy_img = np.clip(speckled + shot_noise + read_noise, 0, 255).astype(np.uint8)
# [STRIPPED]     return noisy_img
# [STRIPPED] 
# [STRIPPED] 
def generate_new_pair4(output_dir="generated_new/pair4", gt_x=310, gt_y=500, seed=2004):
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    search_clean = render_semiconductor_wafer_scene(h=h, w=w, gt_x=gt_x, gt_y=gt_y, seed=seed)

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
        "target_name": "50-Chip Silicon Wafer Array (484-Point Joined Matrix - Left GT 310,500) 4",
        "pair_id": "pair4",
        "scale_factor": 10.0
    }

    with open(os.path.join(output_dir, "groundtruth.json"), "w") as f:
        json.dump(gt_info, f, indent=2)

    cv2.imwrite(os.path.join(output_dir, "search.png"), search_noisy)
    cv2.imwrite(os.path.join(output_dir, "target.png"), target_noisy)
    shutil.copyfile(os.path.join(output_dir, "target.png"), os.path.join(output_dir, "reference.png"))

    wafer_dir = os.path.join("generated_wafer", "pair4")
    os.makedirs(wafer_dir, exist_ok=True)
    with open(os.path.join(wafer_dir, "groundtruth.json"), "w") as f:
        json.dump(gt_info, f, indent=2)
    cv2.imwrite(os.path.join(wafer_dir, "search.png"), search_noisy)
    cv2.imwrite(os.path.join(wafer_dir, "target.png"), target_noisy)
    shutil.copyfile(os.path.join(wafer_dir, "target.png"), os.path.join(wafer_dir, "reference.png"))

    print(f"[generate_new_pair4] Successfully generated pair 4 {output_dir} | GT: ({gt_x}, {gt_y})")
    return os.path.join(output_dir, "search.png"), os.path.join(output_dir, "target.png"), gt_info


if __name__ == "__main__":
    generate_new_pair4()
