"""
Phase-2 Pyramidal Multi-Scale & Multi-Rotation Inference Engine
================================================================
Backup before EXP-19 Adaptive High-Res Rescue experiment
"""

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import math
import time
import cv2
import gc
import numpy as np
import torch
import torchvision.transforms.functional as TF

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pyramid_siamese import PyramidSiameseNetwork
from phase2.phase2_config import Phase2Config

def load_grayscale_image(img_input):
    if isinstance(img_input, str):
        if not os.path.exists(img_input):
            raise FileNotFoundError(f"Image file not found: {img_input}")
        img = cv2.imread(img_input, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image from {img_input}")
    elif isinstance(img_input, np.ndarray):
        img = img_input.copy()
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise TypeError(f"Unsupported image input type: {type(img_input)}")
        
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def fit_parabola_subpixel(grid_3x3, peak_x, peak_y):
    try:
        val_c = grid_3x3[1, 1]
        dx = (grid_3x3[1, 2] - grid_3x3[1, 0]) / 2.0
        dy = (grid_3x3[2, 1] - grid_3x3[0, 1]) / 2.0
        dxx = grid_3x3[1, 2] - 2.0 * val_c + grid_3x3[1, 0]
        dyy = grid_3x3[2, 1] - 2.0 * val_c + grid_3x3[0, 1]
        offset_x = -dx / dxx if abs(dxx) > 1e-5 else 0.0
        offset_y = -dy / dyy if abs(dyy) > 1e-5 else 0.0
        offset_x = max(-0.8, min(0.8, offset_x))
        offset_y = max(-0.8, min(0.8, offset_y))
        return peak_x + offset_x, peak_y + offset_y
    except Exception:
        return float(peak_x), float(peak_y)

def compute_periodicity_count(match_matrix, threshold_ratio=0.85):
    if match_matrix is None or match_matrix.size == 0:
        return 1
    _, max_val, _, _ = cv2.minMaxLoc(match_matrix)
    if max_val <= 0:
        return 1
    thresh = threshold_ratio * max_val
    high_mask = (match_matrix >= thresh).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(high_mask)
    return max(1, num_labels - 1)

class Phase2InferenceEngine:
    def __init__(self, checkpoint_path=None, config=None, device="cpu"):
        torch.set_num_threads(1)
        self.device = torch.device(device)
        self.config = config if config is not None else Phase2Config()
        
        resolved_path = None
        candidate_paths = [
            checkpoint_path,
            "checkpoints_phase2_v2_sunday/best_model_phase2.pth",
            "phase2_checkpoints/best_model_level1.pth",
            "best_model_level1.pth"
        ]
        for p in candidate_paths:
            if p and os.path.exists(p):
                resolved_path = p
                break
                
        self.model = PyramidSiameseNetwork(encoder_type="resnet").to(self.device)
        if resolved_path and os.path.exists(resolved_path):
            self.model.load_state_dict(torch.load(resolved_path, map_location=self.device))
        self.model.eval()
