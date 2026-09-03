"""
Phase-2 Pyramidal Multi-Scale & Multi-Rotation Inference Engine
================================================================
Extends Round-1 Hybrid NCC + Custom 4-Layer ResNet Siamese system to handle:
1. Scale variation s in [8x, 12x]
2. Rotation variation theta in [-5°, +5°]
3. Absent target rejection (Set C)
4. Pose estimation (x, y, theta, scale)
5. Sigmoid calibrated confidence scoring
6. RGB optical 3-channel auto-conversion
"""

import os
import sys

# Limit thread allocation to prevent Windows paging file / memory errors
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
    """Load image input (path or numpy array) as single-channel uint8 grayscale."""
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
    """Refine peak position using 2D 2nd-order Taylor expansion / parabola fitting."""
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
    """Computes number of distinct local peak components exceeding threshold_ratio * max_val."""
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
        
        # Priority order for checkpoint resolution
        resolved_path = None
        candidate_paths = [
            checkpoint_path,
            "checkpoints_phase2_v2_sunday/best_model_phase2.pth",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints_phase2_v2_sunday", "best_model_phase2.pth"),
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
            print(f"[OK] Phase2InferenceEngine loaded checkpoint: '{resolved_path}'")
        else:
            print(f"WARNING: No valid checkpoint found. Using initialized weights.")
        self.model.eval()
        
    def extract_siamese_embedding(self, patch_100x100):
        """Extract 128-D L2-normalized feature embedding using Round-1 SiameseEncoder."""
        patch_resized = cv2.resize(patch_100x100, (100, 100), interpolation=cv2.INTER_AREA)
        tensor = TF.to_tensor(patch_resized).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encoder(tensor)
        return emb

    def extract_batch_embeddings(self, patch_list_100x100):
        """Batch extract embeddings for speed and memory efficiency."""
        if len(patch_list_100x100) == 0:
            return torch.empty((0, 128), device=self.device)
            
        tensors = []
        for p in patch_list_100x100:
            p_res = cv2.resize(p, (100, 100), interpolation=cv2.INTER_AREA)
            tensors.append(TF.to_tensor(p_res))
        batch_tensor = torch.stack(tensors, dim=0).to(self.device)
        with torch.no_grad():
            embs = self.model.encoder(batch_tensor)
        return embs

    def localize_pair(
        self,
        ref_input,
        search_input,
        ncc_weight=None,
        rejection_thresh=None,
        scale_step=0.25,
        theta_step=1.0,
        top_k_coarse=None,
        return_diagnostics=False
    ):
        """
        Executes Phase-2 pyramidal multi-scale & multi-rotation search with hybrid NCC + Siamese fusion.
        Returns dict with: (x, y, theta, scale, found, score, fused_score, raw_ncc, raw_siamese, runtime_ms)
        """
        ref_img = load_grayscale_image(ref_input)
        search_img = load_grayscale_image(search_input)
        
        w_alpha = ncc_weight if ncc_weight is not None else self.config.NCC_WEIGHT
        tau = rejection_thresh if rejection_thresh is not None else self.config.REJECTION_THRESHOLD
        
        h_s, w_s = search_img.shape
        
        # 1. Reference Landmark Template
        if ref_img.shape != (100, 100):
            ref_template = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
        else:
            ref_template = ref_img.copy()
            
        ref_emb = self.extract_siamese_embedding(ref_template)
        
        # 2. Pyramidal Downsampling for Coarse Search (500x500)
        search_coarse = cv2.resize(search_img, (500, 500), interpolation=cv2.INTER_AREA)
        
        best_candidates = []
        coarse_scales = self.config.COARSE_SCALES
        coarse_thetas = self.config.COARSE_THETAS
        
        for scale in coarse_scales:
            patch_size = int(round(1000.0 / scale))
            if patch_size < 30 or patch_size > 300:
                continue
                
            ref_scaled = cv2.resize(ref_template, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
            
            for theta in coarse_thetas:
                if abs(theta) > 0.1:
                    M_rot = cv2.getRotationMatrix2D((patch_size / 2.0, patch_size / 2.0), theta, 1.0)
                    ref_rot = cv2.warpAffine(ref_scaled, M_rot, (patch_size, patch_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                else:
                    ref_rot = ref_scaled
                    
                ref_coarse = cv2.resize(ref_rot, (max(10, patch_size // 2), max(10, patch_size // 2)), interpolation=cv2.INTER_AREA)
                
                if ref_coarse.shape[0] >= search_coarse.shape[0] - 10 or ref_coarse.shape[1] >= search_coarse.shape[1] - 10:
                    continue
                if ref_coarse.shape[0] > 120 or ref_coarse.shape[1] > 120:
                    continue
                if ref_coarse.shape[0] < 5 or ref_coarse.shape[1] < 5:
                    continue
                    
                res_ncc = cv2.matchTemplate(search_coarse, ref_coarse, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_ncc)
                
                cx = (max_loc[0] + ref_coarse.shape[1] / 2.0) * 2.0
                cy = (max_loc[1] + ref_coarse.shape[0] / 2.0) * 2.0
                
                best_candidates.append({
                    "coarse_ncc": float(max_val),
                    "x": cx,
                    "y": cy,
                    "scale": scale,
                    "theta": theta
                })
                del res_ncc
                
        best_candidates.sort(key=lambda c: -c["coarse_ncc"])
        k_top = top_k_coarse if top_k_coarse is not None else self.config.TOP_K_COARSE
        top_candidates = best_candidates[:k_top]
        
        # 3. Fine Grid Refinement around Top Candidates
        refined_results = []
        patch_list = []
        cand_meta = []
        
        for cand in top_candidates:
            c_scale = cand["scale"]
            c_theta = cand["theta"]
            c_x = cand["x"]
            c_y = cand["y"]
            
            fine_scales = [c_scale - scale_step, c_scale, c_scale + scale_step]
            fine_thetas = [c_theta - theta_step, c_theta, c_theta + theta_step]
            
            for f_sc in fine_scales:
                if f_sc < 7.8 or f_sc > 12.2:
                    continue
                p_size = int(round(1000.0 / f_sc))
                ref_s = cv2.resize(ref_template, (p_size, p_size), interpolation=cv2.INTER_LINEAR)
                
                for f_th in fine_thetas:
                    if abs(f_th) > 5.5:
                        continue
                    if abs(f_th) > 0.05:
                        M_r = cv2.getRotationMatrix2D((p_size / 2.0, p_size / 2.0), f_th, 1.0)
                        ref_r = cv2.warpAffine(ref_s, M_r, (p_size, p_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    else:
                        ref_r = ref_s
                        
                    win_r = int(round(p_size / 2.0 + 80))
                    rx0, rx1 = max(0, int(c_x - win_r)), min(w_s, int(c_x + win_r))
                    ry0, ry1 = max(0, int(c_y - win_r)), min(h_s, int(c_y + win_r))
                    
                    search_sub = search_img[ry0:ry1, rx0:rx1]
                    if search_sub.shape[0] <= ref_r.shape[0] or search_sub.shape[1] <= ref_r.shape[1]:
                        continue
                        
                    match_res = cv2.matchTemplate(search_sub, ref_r, cv2.TM_CCOEFF_NORMED)
                    _, f_max_val, _, f_max_loc = cv2.minMaxLoc(match_res)
                    
                    px = rx0 + f_max_loc[0] + ref_r.shape[1] / 2.0
                    py = ry0 + f_max_loc[1] + ref_r.shape[0] / 2.0
                    
                    x0_crop = max(0, min(w_s - 100, int(round(px - 50))))
                    y0_crop = max(0, min(h_s - 100, int(round(py - 50))))
                    cand_patch = search_img[y0_crop:y0_crop+100, x0_crop:x0_crop+100]
                    
                    patch_list.append(cand_patch)
                    cand_meta.append({
                        "x": px, "y": py, "scale": f_sc, "theta": f_th,
                        "ncc_norm": max(0.0, min(1.0, (float(f_max_val) + 1.0) / 2.0)),
                        "match_matrix": match_res
                    })
                    
        if len(patch_list) == 0:
            return {
                "x": 0.0, "y": 0.0, "theta": 0.0, "scale": 0.0,
                "found": 0, "score": 0.0, "fused_score": 0.0,
                "raw_ncc": 0.0, "raw_siamese": 0.0
            }
            
        cand_embs = self.extract_batch_embeddings(patch_list)
        siamese_sims = torch.sum(ref_emb * cand_embs, dim=1).cpu().numpy()
        
        for idx, meta in enumerate(cand_meta):
            s_sim = float(siamese_sims[idx])
            n_norm = meta["ncc_norm"]
            f_score = w_alpha * n_norm + (1.0 - w_alpha) * s_sim
            dist_center = math.sqrt((meta["x"] - 500.0)**2 + (meta["y"] - 500.0)**2)
            adj_score = f_score - self.config.CENTER_BIAS_WEIGHT * (dist_center / 707.0)
            
            # EXP-13 PROMOTION: Periodicity Penalty
            p_count = compute_periodicity_count(meta.get("match_matrix", None))
            p_penalty = 0.05 * (p_count - 1)
            adj_ncc = n_norm - p_penalty

            refined_results.append({
                "x": meta["x"], "y": meta["y"], "scale": meta["scale"], "theta": meta["theta"],
                "fused_score": f_score, "adjusted_score": adj_score,
                "ncc_norm": n_norm, "adjusted_ncc": adj_ncc, "siamese_sim": s_sim,
                "periodicity_count": p_count, "match_matrix": meta["match_matrix"]
            })
            
        # EXP-13 PROMOTION: Pure NCC + Periodicity Penalty Primary Ranking
        # Ranks candidates primarily by adjusted_ncc descending; ties broken by adjusted_score.
        refined_results.sort(key=lambda r: (-r["adjusted_ncc"], -r["adjusted_score"]))
        best_cand = refined_results[0]
        
        try:
            m_mat = best_cand["match_matrix"]
            if m_mat.shape[0] >= 3 and m_mat.shape[1] >= 3:
                sub_3x3 = m_mat[:3, :3]
                fine_x, fine_y = fit_parabola_subpixel(sub_3x3, best_cand["x"], best_cand["y"])
            else:
                fine_x, fine_y = best_cand["x"], best_cand["y"]
        except Exception:
            fine_x, fine_y = best_cand["x"], best_cand["y"]
            
        final_fused = best_cand["fused_score"]
        raw_ncc = best_cand["ncc_norm"]
        ncc_min_tau = getattr(self.config, "NCC_MIN_THRESHOLD", 0.72)
        
        if final_fused >= tau and raw_ncc >= ncc_min_tau:
            found = 1
            pred_x = float(round(fine_x, 2))
            pred_y = float(round(fine_y, 2))
            pred_theta = float(round(best_cand["theta"], 2))
            pred_scale = float(round(best_cand["scale"], 2))
        else:
            found = 0
            pred_x = 0.0
            pred_y = 0.0
            pred_theta = 0.0
            pred_scale = 0.0
            
        conf_score = 1.0 / (1.0 + math.exp(-self.config.CONFIDENCE_SLOPE * (final_fused - tau)))
        conf_score = float(round(max(0.0001, min(0.9999, conf_score)), 4))
        
        gc.collect()
        res_dict = {
            "x": pred_x, "y": pred_y, "theta": pred_theta, "scale": pred_scale,
            "found": found, "score": conf_score, "fused_score": float(round(final_fused, 4)),
            "raw_ncc": float(round(best_cand["ncc_norm"], 4)),
            "raw_siamese": float(round(best_cand["siamese_sim"], 4))
        }
        if return_diagnostics:
            return res_dict, best_candidates, refined_results
        return res_dict
