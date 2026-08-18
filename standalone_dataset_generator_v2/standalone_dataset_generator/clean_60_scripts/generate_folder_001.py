#!/usr/bin/env python3
"""
Generate Dataset Pair 8 (Real SEM Reference Target 2)
===================================================
Generates:
  generated_pairs/pair8/groundtruth.json
  generated_pairs/pair8/search.png   (1000x1000 px, 10x scale with SEM noise)
  generated_pairs/pair8/target.png   (1000x1000 px, 100x high-mag reference crop)
  generated_pairs/pair8/reference.png (1000x1000 px, identical to target.png)
"""

import json
import os
import shutil
import cv2
import numpy as np


def add_sem_noise(image_uint8, poisson_scale=35.0, gauss_sigma=4.0, blur_kernel=3):
    """Apply realistic SEM imaging noise (Poisson shot noise, Gaussian read noise, defocus blur)."""
    img_float = image_uint8.astype(np.float32)

    # 1. Poisson shot noise
    counts = np.maximum(0, img_float) / 255.0 * poisson_scale
    noisy_counts = np.random.poisson(counts).astype(np.float32)
    img_poisson = noisy_counts / poisson_scale * 255.0

    # 2. Gaussian read noise
    gauss_noise = np.random.normal(0, gauss_sigma, img_float.shape).astype(np.float32)
    noisy_img = img_poisson + gauss_noise

    # 3. Light defocus blur
    if blur_kernel > 1:
        noisy_img = cv2.GaussianBlur(noisy_img, (blur_kernel, blur_kernel), 0.6)

    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def generate_pair8(output_dir="generated_pairs/pair8", gt_x=650, gt_y=350, seed=108):
    """Generate pair8 dataset files."""
    h, w = 1000, 1000
    os.makedirs(output_dir, exist_ok=True)

    # Source reference image path candidates
    candidate_paths = [
        r"C:\Users\linga\Downloads\I4C_hackathon - Copy\I4C_hackathon - Copy\research_images\image1.png",
        r"C:\Users\linga\Downloads\I4C_hackathon - Copy\I4C_hackathon - Copy\real_sem_user_dataset\reference.png"
    ]

    loaded_img = None
    ref_used_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            loaded_img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if loaded_img is not None:
                ref_used_path = p
                break

    if loaded_img is None:
        np.random.seed(seed)
        search_img = np.full((h, w), 50, dtype=np.float32)
        for y in range(50, h, 80):
            for x in range(50, w, 80):
                cv2.rectangle(search_img, (x-25, y-25), (x+25, y+25), 180, 2)
                cv2.circle(search_img, (x, y), 10, 220, -1)
        cv2.circle(search_img, (gt_x, gt_y), 30, 255, -1)
        search_clean = np.clip(search_img, 0, 255).astype(np.uint8)
        ref_name = "Synthetic SEM Microstructure"
    else:
        search_clean = cv2.resize(loaded_img, (1000, 1000), interpolation=cv2.INTER_AREA)
        ref_name = os.path.basename(ref_used_path)

    # 100x100 crop upscaled 10x to 1000x1000 for target.png
    crop_size = 100
    x0 = max(0, min(w - crop_size, gt_x - crop_size // 2))
    y0 = max(0, min(h - crop_size, gt_y - crop_size // 2))
    x1, y1 = x0 + crop_size, y0 + crop_size

    crop_patch = search_clean[y0:y1, x0:x1]
    target_clean = cv2.resize(crop_patch, (1000, 1000), interpolation=cv2.INTER_CUBIC)

    # Apply SEM noise to search image
    np.random.seed(seed)
    search_noisy = add_sem_noise(search_clean, poisson_scale=35.0, gauss_sigma=4.0, blur_kernel=3)

    gt_info = {
        "center_x": gt_x,
        "center_y": gt_y,
        "target_name": f"Real SEM Reference Target 2 from {ref_name}",
        "pair_id": "pair8",
        "scale_factor": 10.0
    }

    # Save pair8 files
    gt_path = os.path.join(output_dir, "groundtruth.json")
    search_path = os.path.join(output_dir, "search.png")
    target_path = os.path.join(output_dir, "target.png")
    ref_path = os.path.join(output_dir, "reference.png")

    with open(gt_path, "w") as f:
        json.dump(gt_info, f, indent=2)

    cv2.imwrite(search_path, search_noisy)
    cv2.imwrite(target_path, target_clean)
    shutil.copyfile(target_path, ref_path)

    print(f"[generate_pair8] Successfully generated {output_dir} | GT: ({gt_x}, {gt_y})")
    return search_path, target_path, gt_info


if __name__ == "__main__":
    generate_pair8()
