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
def compute_psr(m_mat, py, px):
    """Computes Peak-to-Sidelobe Ratio (PSR) on 31x31 neighborhood around peak."""
    try:
        if m_mat is None or m_mat.size < 25: return 10.0
        h_m, w_m = m_mat.shape
        py_i = int(round(py))
        px_i = int(round(px))
        y0, y1 = max(0, py_i - 15), min(h_m, py_i + 16)
        x0, x1 = max(0, px_i - 15), min(w_m, px_i + 16)
        patch = m_mat[y0:y1, x0:x1]
        if patch.size < 25: return 10.0
        cy, cx = py_i - y0, px_i - x0
        cy = max(0, min(patch.shape[0] - 1, cy))
        cx = max(0, min(patch.shape[1] - 1, cx))
        peak_val = patch[cy, cx]
        mask = np.ones(patch.shape, dtype=bool)
        cy0, cy1 = max(0, cy - 2), min(patch.shape[0], cy + 3)
        cx0, cx1 = max(0, cx - 2), min(patch.shape[1], cx + 3)
        mask[cy0:cy1, cx0:cx1] = False
        sidelobes = patch[mask]
        if len(sidelobes) == 0: return 10.0
        s_std = np.std(sidelobes)
        if s_std < 1e-6: return 10.0
        return float((peak_val - np.mean(sidelobes)) / s_std)
    except Exception:
        return 10.0

def perform_deep_rescan_verification(ref_img, search_img, cand_x, cand_y, cand_scale, cand_theta):
    """
    In-depth Re-Scan Verification Pass:
    Verifies candidate structural alignment via Sobel edge correlation.
    Returns True if target is genuinely present, False if it's an absent decoy.
    """
    try:
        ref_h, ref_w = ref_img.shape[:2]
        ref_eq = cv2.equalizeHist(ref_img)
        
        inv_scale = 1.0 / max(1.0, cand_scale)
        M = cv2.getRotationMatrix2D((cand_x, cand_y), -cand_theta, inv_scale)
        M[0, 2] += (ref_w / 2.0 - cand_x)
        M[1, 2] += (ref_h / 2.0 - cand_y)
        
        warped_crop = cv2.warpAffine(search_img, M, (ref_w, ref_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        crop_eq = cv2.equalizeHist(warped_crop)
        
        ref_dx = cv2.Sobel(ref_eq, cv2.CV_32F, 1, 0, ksize=3)
        crop_dx = cv2.Sobel(crop_eq, cv2.CV_32F, 1, 0, ksize=3)
        edge_ncc = float(cv2.matchTemplate(crop_dx, ref_dx, cv2.TM_CCOEFF_NORMED)[0, 0])
        
        if edge_ncc < 0.25:
            return False
        return True
    except Exception:
        return True

def spatial_uniqueness_rescan(model, device, search_img, ref_emb, matched_x, matched_y, n_samples=10, crop_size=100):
    """
    Spatial Uniqueness Re-Scan:
    Samples n_samples random 100x100 crops from the search image AWAY from the matched location.
    Computes mean Siamese cosine similarity of those random crops to the reference embedding.
    
    Logic:
    - DECOY (Set C): The whole canvas is from the same DRAM architecture -> random crops ALSO
      look similar to the reference -> mean_random_sim will be HIGH (>= threshold).
    - GENUINE PRESENT (Set A/B/D): Only the TRUE match location looks like the reference ->
      random crops will be dissimilar -> mean_random_sim will be LOW (< threshold).
    
    Returns: (mean_random_sim, is_decoy)
    """
    try:
        h, w = search_img.shape[:2]
        if h < crop_size + 20 or w < crop_size + 20:
            return 0.0, False  # Image too small to sample away
        
        mx, my = int(matched_x), int(matched_y)
        guard = crop_size + 20  # Exclusion zone radius around matched location
        
        sims = []
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        attempts = 0
        while len(sims) < n_samples and attempts < n_samples * 5:
            attempts += 1
            rx = rng.randint(0, w - crop_size)
            ry = rng.randint(0, h - crop_size)
            # Skip if too close to matched location
            if abs(rx + crop_size // 2 - mx) < guard and abs(ry + crop_size // 2 - my) < guard:
                continue
            crop = search_img[ry:ry + crop_size, rx:rx + crop_size]
            if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
                continue
            t = torch.from_numpy(crop).float().unsqueeze(0).unsqueeze(0) / 255.0
            t = t.to(device)
            with torch.no_grad():
                emb = model.encoder(t).cpu()
            ref_cpu = ref_emb.cpu()
            sim = float(torch.nn.functional.cosine_similarity(emb, ref_cpu).item())
            sims.append(sim)
        
        if len(sims) < 3:
            return 0.0, False  # Not enough samples
        
        mean_sim = float(np.mean(sims))
        # Threshold: if random crops average > 0.88 Siamese sim -> canvas-wide match -> decoy
        is_decoy = mean_sim >= 0.88
        return mean_sim, is_decoy
    except Exception:
        return 0.0, False

class Phase2InferenceEngine:
    def __init__(self, checkpoint_path=None, config=None, device="cpu"):
        torch.set_num_threads(1)
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.config = config if config is not None else Phase2Config()
        
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
        
    def extract_siamese_embedding(self, img_patch_100x100):
        if img_patch_100x100.ndim == 3:
            img_gray = cv2.cvtColor(img_patch_100x100, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_patch_100x100
        tensor = torch.from_numpy(img_gray).float().unsqueeze(0).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)
        with torch.no_grad():
            emb = self.model.encoder(tensor)
        return emb

    def extract_batch_embeddings(self, patch_list):
        if len(patch_list) == 0:
            return torch.empty((0, 128), device=self.device)
        batch_tensors = []
        for p in patch_list:
            if p.shape[:2] != (100, 100):
                p = cv2.resize(p, (100, 100), interpolation=cv2.INTER_LINEAR)
            if p.ndim == 3:
                p_gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
            else:
                p_gray = p
            batch_tensors.append(torch.from_numpy(p_gray).float().unsqueeze(0) / 255.0)
        
        all_embs = []
        batch_size = 16
        for i in range(0, len(batch_tensors), batch_size):
            sub_batch = torch.stack(batch_tensors[i:i+batch_size]).to(self.device)
            with torch.no_grad():
                sub_emb = self.model.encoder(sub_batch).cpu()
                all_embs.append(sub_emb)
            del sub_batch
        return torch.cat(all_embs, dim=0)

    def localize_pair(self, reference_input, search_input, scale_step=0.10, theta_step=0.25, top_k_coarse=None, ncc_weight=None, rejection_thresh=None, return_diagnostics=False):
        ref_img = load_grayscale_image(reference_input)
        search_img = load_grayscale_image(search_input)
        
        w_alpha = ncc_weight if ncc_weight is not None else self.config.NCC_WEIGHT
        tau = rejection_thresh if rejection_thresh is not None else self.config.REJECTION_THRESHOLD
        
        h_s, w_s = search_img.shape
        
        if ref_img.shape != (100, 100):
            ref_template = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
        else:
            ref_template = ref_img.copy()
            
        ref_emb = self.extract_siamese_embedding(ref_template)
        
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
                    
                    ml_x, ml_y = f_max_loc[0], f_max_loc[1]
                    sub_x, sub_y = float(ml_x), float(ml_y)
                    if ml_y >= 1 and ml_y < match_res.shape[0] - 1 and ml_x >= 1 and ml_x < match_res.shape[1] - 1:
                        sub_3x3 = match_res[ml_y-1:ml_y+2, ml_x-1:ml_x+2]
                        sub_x, sub_y = fit_parabola_subpixel(sub_3x3, float(ml_x), float(ml_y))
                        
                    px = rx0 + sub_x + ref_r.shape[1] / 2.0
                    py = ry0 + sub_y + ref_r.shape[0] / 2.0
                    
                    x0_crop = max(0, min(w_s - 100, int(round(px - 50))))
                    y0_crop = max(0, min(h_s - 100, int(round(py - 50))))
                    cand_patch = search_img[y0_crop:y0_crop+100, x0_crop:x0_crop+100]
                    
                    patch_list.append(cand_patch)
                    cand_meta.append({
                        "x": px, "y": py, "scale": f_sc, "theta": f_th,
                        "local_px": ml_x, "local_py": ml_y,
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
            
            p_count = compute_periodicity_count(meta.get("match_matrix", None)) or 1
            p_penalty = 0.05 * (p_count - 1)
            adj_ncc = n_norm - p_penalty

            refined_results.append({
                "x": meta["x"], "y": meta["y"], "scale": meta["scale"], "theta": meta["theta"],
                "local_px": meta["local_px"], "local_py": meta["local_py"],
                "fused_score": f_score, "adjusted_score": adj_score,
                "ncc_norm": n_norm, "adjusted_ncc": adj_ncc, "siamese_sim": s_sim,
                "periodicity_count": p_count, "match_matrix": meta["match_matrix"]
            })
            
        refined_results.sort(key=lambda r: (-r["adjusted_ncc"], -r["adjusted_score"]))
        best_cand = refined_results[0]
        
        fine_x, fine_y = best_cand["x"], best_cand["y"]
        c_scale = best_cand["scale"]
        c_theta = best_cand["theta"]
        
        # 1D Parabolic Continuous Scale Search Interpolation
        same_theta_cands = [r for r in refined_results if abs(r["theta"] - c_theta) < 0.01]
        sc_plus = [r for r in same_theta_cands if abs(r["scale"] - (c_scale + 0.10)) < 0.02]
        sc_minus = [r for r in same_theta_cands if abs(r["scale"] - (c_scale - 0.10)) < 0.02]
        
        if sc_plus and sc_minus:
            y_c = best_cand["ncc_norm"]
            y_p = sc_plus[0]["ncc_norm"]
            y_m = sc_minus[0]["ncc_norm"]
            denom = 2.0 * (2.0 * y_c - y_p - y_m)
            if abs(denom) > 1e-5:
                delta_s = 0.10 * (y_p - y_m) / denom
                delta_s = max(-0.08, min(0.08, delta_s))
                opt_scale = c_scale + delta_s
            else:
                opt_scale = c_scale
        else:
            opt_scale = c_scale

        final_fused = best_cand["fused_score"]
        raw_ncc = best_cand["ncc_norm"]
        raw_siamese = best_cand["siamese_sim"]
        
        lap_var = float(cv2.Laplacian(search_img, cv2.CV_8U).var())
        dynamic_ncc_min = 0.68 if lap_var > 2200.0 else 0.78
        siamese_min_tau = getattr(self.config, "SIAMESE_MIN_THRESHOLD", 0.75)
        
        best_psr = compute_psr(best_cand["match_matrix"], best_cand["local_py"], best_cand["local_px"])
        psr_min_tau = 3.2 if raw_ncc < 0.78 else 1.8
        
        is_ncc_confident = (raw_ncc >= max(0.72, dynamic_ncc_min) and best_psr >= psr_min_tau)
        is_fused_confident = (final_fused >= tau and raw_ncc >= dynamic_ncc_min and raw_siamese >= siamese_min_tau and best_psr >= psr_min_tau)
        
        if is_ncc_confident or is_fused_confident:
            found = 1
            pred_x = float(round(fine_x, 2))
            pred_y = float(round(fine_y, 2))
            pred_theta = float(round(c_theta, 2))
            pred_scale = float(round(opt_scale, 2))
        else:
            found = 0
            pred_x = 0.0
            pred_y = 0.0
            pred_theta = 0.0
            pred_scale = 0.0
            
        conf_slope = getattr(self.config, "CONFIDENCE_SLOPE", 15.0)
        conf_score = 1.0 / (1.0 + math.exp(-conf_slope * (final_fused - 0.45)))
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
