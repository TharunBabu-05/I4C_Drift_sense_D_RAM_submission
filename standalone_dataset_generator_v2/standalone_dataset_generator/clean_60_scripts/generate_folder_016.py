#!/usr/bin/env python3
"""
Generate Pair 4 for generated_new (RGB Silicon Wafer with Thin-Film Rainbow Gradient - Left GT 310,500)
========================================================================================================
Synthesizes 1000x1000x3 uint8 BGR color search image and target reference image matching real optical wafer photo:
  - Deep Sapphire Blue silicon wafer disk substrate with bevel ring and alignment notches
  - Thin-film optical interference rainbow color gradient across ~50 microchip dies (copper/orange -> lime green -> cyan)
  - Top-Left section of every chip features a 22x22 grid of 484 points
  - Ground Truth Target Chip at Left (310, 500) [ANOTHER CHIP!]: Features a highly visible bright metallic 484-point joined bus net + X-cross!
  - Memory-efficient RGB noise pipeline
"""

import json
import os
import shutil
import cv2
import numpy as np


def render_semiconductor_wafer_scene(h=1000, w=1000, gt_x=310, gt_y=500, seed=2004):
    """Render 1000x1000x3 uint8 BGR Silicon Wafer containing ~50 microchips with rainbow optical gradient."""
    np.random.seed(seed)
    img_bgr = np.full((h, w, 3), 15, dtype=np.float32)

    # 1. Circular Silicon Wafer Disk (Radius 470)
    center_wafer = (w // 2, h // 2)
    radius_wafer = 470

    y_indices, x_indices = np.indices((h, w))
    dist_wafer = np.sqrt((x_indices - center_wafer[0])**2 + (y_indices - center_wafer[1])**2)
    wafer_mask = dist_wafer <= radius_wafer

    # Deep Sapphire Blue base wafer substrate (BGR = 80, 40, 15)
    img_bgr[wafer_mask, 0] = 80.0  # Blue
    img_bgr[wafer_mask, 1] = 40.0  # Green
    img_bgr[wafer_mask, 2] = 15.0  # Red

    # Outer bevel ring (dark midnight blue)
    bevel_mask = (dist_wafer > radius_wafer - 15) & wafer_mask
    img_bgr[bevel_mask, 0] = 120.0
    img_bgr[bevel_mask, 1] = 30.0
    img_bgr[bevel_mask, 2] = 10.0

    # Wafer alignment notch
    cv2.rectangle(img_bgr, (w // 2 - 15, h // 2 - radius_wafer - 5), (w // 2 + 15, h // 2 - radius_wafer + 15), (10, 10, 10), -1)

    # 2. Render ~50 Microchip Dies with Thin-Film Rainbow Gradient
    chip_w, chip_h = 80, 80
    pitch_x, pitch_y = 95, 95
    
    cols = [500 + i * pitch_x for i in range(-4, 5)] # [120, 215, 310, 405, 500, 595, 690, 785, 880]
    rows = [500 + j * pitch_y for j in range(-4, 5)] # [120, 215, 310, 405, 500, 595, 690, 785, 880]

    for cy in rows:
        for cx in cols:
            if np.sqrt((cx - center_wafer[0])**2 + (cy - center_wafer[1])**2) <= radius_wafer - 40:
                x1, y1 = cx - chip_w // 2, cy - chip_h // 2
                x2, y2 = cx + chip_w // 2, cy + chip_h // 2

                # Optical thin-film rainbow iridescence gradient
                t_diag = np.clip(((cx - 100.0) / 800.0 * 0.7 + (cy - 100.0) / 800.0 * 0.3), 0.0, 1.0)
                hue_val = int(12 + t_diag * 70) # OpenCV Hue [0..179]: 12 (copper/orange) -> 82 (cyan/green)
                hsv_die = np.uint8([[[hue_val, 210, 160]]])
                bgr_raw = cv2.cvtColor(hsv_die, cv2.COLOR_HSV2BGR)[0, 0].astype(float)

                # Normalize luminance so baseline grayscale is uniform across all dies
                gray_lum = 0.114 * bgr_raw[0] + 0.587 * bgr_raw[1] + 0.299 * bgr_raw[2]
                bgr_die = bgr_raw * (120.0 / max(1.0, gray_lum))

                # Die Substrate
                for ch in range(3):
                    img_bgr[y1 + 2:y2 - 2, x1 + 2:x2 - 2, ch] = bgr_die[ch]

                # Crisp Scribe line outer die frame
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (200.0, 220.0, 220.0), 1)

                c_stripe = (float(bgr_die[0] + 60), float(bgr_die[1] + 60), float(bgr_die[2] + 60))
                c_grid1  = (float(bgr_die[0] + 35), float(bgr_die[1] + 35), float(bgr_die[2] + 35))
                c_grid2  = (float(bgr_die[0] + 15), float(bgr_die[1] + 15), float(bgr_die[2] + 15))

                # Top-Right Section: Word Lines (horizontal parallel stripes)
                for wy in range(y1 + 6, y1 + 32, 3):
                    cv2.line(img_bgr, (cx - 2, wy), (x2 - 5, wy), c_stripe, 1)

                # Middle Section: Bit Lines & Memory Cell Matrix
                for bx in range(x1 + 6, x2 - 6, 4):
                    cv2.line(img_bgr, (bx, y1 + 35), (bx, y2 - 20), c_grid1, 1)
                for my in range(y1 + 37, y2 - 22, 6):
                    cv2.line(img_bgr, (x1 + 6, my), (x2 - 6, my), c_grid2, 1)

                # Bottom Section: Sense Amplifier & I/O Circuitry
                cv2.rectangle(img_bgr, (x1 + 6, y2 - 18), (x2 - 6, y2 - 4), (50.0, 50.0, 50.0), -1)
                for px in range(x1 + 8, x2 - 8, 7):
                    cv2.rectangle(img_bgr, (px, y2 - 16), (px + 3, y2 - 6), (220.0, 240.0, 255.0), -1)

                # Top-Left Section: 22x22 Grid of 484 Points (via pads / dots)
                start_px = cx - 36
                start_py = cy - 36
                grid_dim = 22
                spacing = 32.0 / (grid_dim - 1)

                is_gt_chip = (cx == gt_x and cy == gt_y)

                if is_gt_chip:
                    # TARGET CHIP: HIGHLY VISIBLE UNIQUE FEATURE (484 Points JOINED + Metallic X-Crosshair)
                    cv2.rectangle(img_bgr, (start_px - 2, start_py - 2), (start_px + 34, start_py + 34), (250.0, 250.0, 250.0), -1)

                    # Horizontal interconnect trace lines
                    for r in range(grid_dim):
                        ry = int(start_py + r * spacing)
                        cv2.line(img_bgr, (int(start_px), ry), (int(start_px + 32), ry), (255.0, 255.0, 255.0), 2)

                    # Vertical interconnect trace lines
                    for c in range(grid_dim):
                        cx_pt = int(start_px + c * spacing)
                        cv2.line(img_bgr, (cx_pt, int(start_py)), (cx_pt, int(start_py + 32)), (255.0, 255.0, 255.0), 2)

                    # Diagonal metallic X-cross lines
                    cv2.line(img_bgr, (int(start_px), int(start_py)), (int(start_px + 32), int(start_py + 32)), (255.0, 255.0, 255.0), 3)
                    cv2.line(img_bgr, (int(start_px + 32), int(start_py)), (int(start_px), int(start_py + 32)), (255.0, 255.0, 255.0), 3)

                    # 484 bright via points at grid intersections
                    for r in range(grid_dim):
                        for c in range(grid_dim):
                            pt_x = int(start_px + c * spacing)
                            pt_y = int(start_py + r * spacing)
                            cv2.circle(img_bgr, (pt_x, pt_y), 1, (255.0, 255.0, 255.0), -1)
                else:
                    # REGULAR CHIP: 484 separate dots/points
                    for r in range(grid_dim):
                        for c in range(grid_dim):
                            pt_x = int(start_px + c * spacing)
                            pt_y = int(start_py + r * spacing)
                            cv2.circle(img_bgr, (pt_x, pt_y), 1, (160.0, 160.0, 160.0), -1)

    return np.clip(img_bgr, 0, 255).astype(np.uint8)


# [STRIPPED] def add_heavy_multi_sem_noise_rgb(image_uint8, seed=2004, is_target=False):
# [STRIPPED]     """Memory-efficient RGB Optical Noise pipeline."""
# [STRIPPED]     np.random.seed(seed if not is_target else seed + 9999)
# [STRIPPED]     img = image_uint8.copy()
# [STRIPPED] 
# [STRIPPED]     std = 12 if not is_target else 6
# [STRIPPED]     noise = np.random.randint(-std, std + 1, img.shape, dtype=np.int16)
# [STRIPPED]     noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
# [STRIPPED]     return noisy
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
        "target_name": "RGB Silicon Wafer Array (Visible Unique Feature - Left GT 310,500) 4",
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

    print(f"[generate_new_pair4] Successfully generated RGB wafer pair 4 {output_dir} | GT: ({gt_x}, {gt_y})")
    return os.path.join(output_dir, "search.png"), os.path.join(output_dir, "target.png"), gt_info


if __name__ == "__main__":
    generate_new_pair4()
