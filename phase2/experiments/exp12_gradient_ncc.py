#!/usr/bin/env python3
"""
EXP-12 — Gradient-Normalized NCC Only
======================================

STRICT SINGLE-CHANGE EXPERIMENT

Hypothesis:
    Raw intensity NCC is vulnerable to periodic DRAM texture similarity where repetitive
    brightness patterns produce high correlation (0.95+). Gradient-normalized NCC
    (Sobel gradient magnitude G = sqrt(Gx^2 + Gy^2)) measures structural edge transitions
    rather than intensity levels, allowing landmark structural features to outperform
    smooth periodic decoys.

Single Change:
    Replace raw intensity template matching (cv2.matchTemplate on grayscale I)
    with gradient magnitude template matching (cv2.matchTemplate on Sobel gradient magnitude G).

All downstream logic (Strategy A Pure NCC Primary Ranking, refinement, sigmoid confidence,
rejection tau=0.42, 500x500 coarse search) remains 100% UNCHANGED.

Production files: UNMODIFIED. Checkpoint: UNMODIFIED.
"""

import os
import sys
import json
import time
import math
import hashlib
import csv
import gc
import cv2
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from phase2.phase2_inference import Phase2InferenceEngine, load_grayscale_image, fit_parabola_subpixel
from phase2.phase2_config import Phase2Config

import torch


def compute_gradient_magnitude(img):
    """
    Computes 32-bit float Sobel gradient magnitude: G = sqrt(Gx^2 + Gy^2).
    """
    if img.dtype != np.float32:
        img_f = img.astype(np.float32)
    else:
        img_f = img
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    g = cv2.magnitude(gx, gy)
    return g


def localize_pair_gradient_ncc(engine, ref_input, search_input, ncc_weight=0.5, rejection_thresh=0.42,
                               scale_step=0.25, theta_step=1.0, top_k_coarse=10):
    """
    Executes Phase-2 search replacing raw intensity NCC with Gradient-Normalized NCC.
    Uses PROMOTED Strategy A ranking (argmax(gradient_ncc_norm)).
    """
    ref_img = load_grayscale_image(ref_input)
    search_img = load_grayscale_image(search_input)

    w_alpha = ncc_weight if ncc_weight is not None else engine.config.NCC_WEIGHT
    tau = rejection_thresh if rejection_thresh is not None else engine.config.REJECTION_THRESHOLD

    h_s, w_s = search_img.shape

    # 1. Reference Landmark Template & Gradient
    if ref_img.shape != (100, 100):
        ref_template = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
    else:
        ref_template = ref_img.copy()

    ref_emb = engine.extract_siamese_embedding(ref_template)

    # 2. Pyramidal Downsampling & Gradient Image
    search_coarse = cv2.resize(search_img, (500, 500), interpolation=cv2.INTER_AREA)
    search_coarse_g = compute_gradient_magnitude(search_coarse)

    best_candidates = []
    coarse_scales = engine.config.COARSE_SCALES
    coarse_thetas = engine.config.COARSE_THETAS

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

            ref_coarse_g = compute_gradient_magnitude(ref_coarse)

            res_ncc = cv2.matchTemplate(search_coarse_g, ref_coarse_g, cv2.TM_CCOEFF_NORMED)
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
            del res_ncc, ref_coarse_g

    best_candidates.sort(key=lambda c: -c["coarse_ncc"])
    k_top = top_k_coarse if top_k_coarse is not None else engine.config.TOP_K_COARSE
    top_candidates = best_candidates[:k_top]

    # Pre-compute full search gradient image for fine search
    search_img_g = compute_gradient_magnitude(search_img)

    # 3. Fine Grid Refinement around Top Candidates (Gradient NCC)
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

                search_sub_g = search_img_g[ry0:ry1, rx0:rx1]
                if search_sub_g.shape[0] <= ref_r.shape[0] or search_sub_g.shape[1] <= ref_r.shape[1]:
                    continue

                ref_r_g = compute_gradient_magnitude(ref_r)

                match_res = cv2.matchTemplate(search_sub_g, ref_r_g, cv2.TM_CCOEFF_NORMED)
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
        }, best_candidates, []

    cand_embs = engine.extract_batch_embeddings(patch_list)
    siamese_sims = torch.sum(ref_emb * cand_embs, dim=1).cpu().numpy()

    for idx, meta in enumerate(cand_meta):
        s_sim = float(siamese_sims[idx])
        n_norm = meta["ncc_norm"]
        f_score = w_alpha * n_norm + (1.0 - w_alpha) * s_sim
        dist_center = math.sqrt((meta["x"] - 500.0)**2 + (meta["y"] - 500.0)**2)
        adj_score = f_score - engine.config.CENTER_BIAS_WEIGHT * (dist_center / 707.0)

        refined_results.append({
            "x": meta["x"], "y": meta["y"], "scale": meta["scale"], "theta": meta["theta"],
            "fused_score": f_score, "adjusted_score": adj_score,
            "ncc_norm": n_norm, "siamese_sim": s_sim, "match_matrix": meta["match_matrix"]
        })

    # Strategy A: Pure NCC Primary Ranking on Gradient NCC
    refined_results.sort(key=lambda r: (-r["ncc_norm"], -r["adjusted_score"]))
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

    if final_fused >= tau:
        found = 1
        pred_x = float(round(fine_x, 2))
        pred_y = float(round(fine_y, 2))
        pred_theta = float(round(best_cand["theta"], 2))
        pred_scale = float(round(best_cand["scale"], 2))
    else:
        found = 0
        pred_x = pred_y = pred_theta = pred_scale = 0.0

    conf_score = 1.0 / (1.0 + math.exp(-engine.config.CONFIDENCE_SLOPE * (final_fused - tau)))
    conf_score = float(round(max(0.0001, min(0.9999, conf_score)), 4))

    clean_refined = []
    for r in refined_results:
        clean_refined.append({
            "x": r["x"], "y": r["y"], "scale": r["scale"], "theta": r["theta"],
            "fused_score": r["fused_score"], "adjusted_score": r["adjusted_score"],
            "ncc_norm": r["ncc_norm"], "siamese_sim": r["siamese_sim"]
        })

    res_dict = {
        "x": pred_x, "y": pred_y, "theta": pred_theta, "scale": pred_scale,
        "found": found, "score": conf_score, "fused_score": float(round(final_fused, 4)),
        "raw_ncc": float(round(best_cand["ncc_norm"], 4)),
        "raw_siamese": float(round(best_cand["siamese_sim"], 4))
    }

    del cand_embs, siamese_sims, patch_list, refined_results, search_img_g
    gc.collect()

    return res_dict, best_candidates, clean_refined


def calculate_auc(y_true, y_scores):
    pos_scores = [s for t, s in zip(y_true, y_scores) if t == 1]
    neg_scores = [s for t, s in zip(y_true, y_scores) if t == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5
    count = 0.0
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                count += 1.0
            elif p == n:
                count += 0.5
    return count / (len(pos_scores) * len(neg_scores))


def compute_100pt_breakdown(results):
    sets_data = {"Set A": [], "Set B": [], "Set C": [], "Set D": []}
    for r in results:
        sets_data[r["set"]].append(r)

    def calc_loc_credit(entries):
        present = [e for e in entries if e["gt_found"] == 1]
        n = len(present)
        if n == 0:
            return 0.0, 0, 0, 0, 0, 0
        credits = []
        c1 = c2 = c3 = c5 = 0
        for e in present:
            if e["pred_found"] == 1:
                err = e["loc_err"]
                if err <= 1.0:   credits.append(1.00); c1 += 1
                elif err <= 2.0: credits.append(0.80); c2 += 1
                elif err <= 3.0: credits.append(0.60); c3 += 1
                elif err <= 5.0: credits.append(0.40); c5 += 1
                else:            credits.append(0.00)
            else:
                credits.append(0.00)
        return np.mean(credits), c1, c2, c3, c5, n

    credit_a, _, _, _, _, _ = calc_loc_credit(sets_data["Set A"])
    credit_b, _, _, _, _, _ = calc_loc_credit(sets_data["Set B"])
    loc_score = (0.45 * credit_a + 0.55 * credit_b) * 40.0

    total_present = sum(1 for r in results if r["gt_found"] == 1)
    scale_credits = []
    theta_credits = []
    for r in results:
        if r["gt_found"] == 1:
            if r["pred_found"] == 1 and r["loc_err"] <= 5.0:
                s_err = r["scale_err"]
                t_err = r["theta_err"]
                scale_credits.append(1.0 if s_err <= 0.25 else (0.5 if s_err <= 0.50 else 0.0))
                theta_credits.append(1.0 if t_err <= 0.5 else (0.5 if t_err <= 1.5 else 0.0))
            else:
                scale_credits.append(0.0)
                theta_credits.append(0.0)
    scale_score = (sum(scale_credits) / total_present) * 10.0 if total_present > 0 else 0.0
    theta_score = (sum(theta_credits) / total_present) * 10.0 if total_present > 0 else 0.0
    pose_score = scale_score + theta_score

    tp = sum(1 for r in results if r["gt_found"] == 1 and r["pred_found"] == 1)
    tn = sum(1 for r in results if r["gt_found"] == 0 and r["pred_found"] == 0)
    fp = sum(1 for r in results if r["gt_found"] == 0 and r["pred_found"] == 1)
    fn = sum(1 for r in results if r["gt_found"] == 1 and r["pred_found"] == 0)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    rejection_score = f1 * 15.0

    y_true = [r["gt_found"] for r in results]
    y_scores = [r["pred_score"] for r in results]
    auc = calculate_auc(y_true, y_scores)
    confidence_score = auc * 10.0

    runtimes = [r["runtime_ms"] for r in results]
    med_rt = float(np.median(runtimes))
    eff_score = 5.0 if med_rt <= 5000.0 else (2.5 if med_rt <= 10000.0 else 0.0)
    gen_score = 10.0

    total = loc_score + pose_score + rejection_score + confidence_score + eff_score + gen_score

    return {
        "total_100_score": total,
        "loc_score": loc_score,
        "scale_score": scale_score,
        "theta_score": theta_score,
        "pose_score": pose_score,
        "rejection_score": rejection_score,
        "confidence_score": confidence_score,
        "eff_score": eff_score,
        "gen_score": gen_score,
        "f1": f1, "auc": auc, "med_rt": med_rt,
        "p90_rt": float(np.percentile(runtimes, 90)),
        "p99_rt": float(np.percentile(runtimes, 99)),
    }


def run_experiment():
    print("=" * 70)
    print("EXP-12: GRADIENT-NORMALIZED NCC ONLY")
    print("=" * 70)

    # 1. Checkpoint Integrity Check
    ckpt_path = "phase2_checkpoints/best_model_level1.pth"
    with open(ckpt_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    expected_sha = "e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c"
    assert sha == expected_sha, f"SHA-256 MISMATCH: {sha}"
    print(f"[OK] Checkpoint SHA-256 verified: {sha[:16]}...")

    # 2. Init Engine (Production code)
    engine = Phase2InferenceEngine(checkpoint_path=ckpt_path, device="cpu")
    print(f"[OK] Engine initialized.")

    data_dir = "local_phase2_60gen_200_pairs"
    manifest_path = os.path.join(data_dir, "phase2_60generator_manifest.csv")
    assert os.path.exists(manifest_path), f"Manifest not found: {manifest_path}"

    pairs = []
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(row)
    print(f"[OK] Loaded {len(pairs)} pairs from manifest")

    target_pair_ids = {"pair_006", "pair_066", "pair_116", "pair_186"}

    # 3. Evaluate Baseline (Raw Intensity NCC Strategy A via production engine)
    print(f"\n{'='*70}")
    print("RUNNING PRODUCTION BASELINE EVALUATION (Raw Intensity NCC)...")
    print(f"{'='*70}")

    base_results = []
    base_target_debug = {}

    for pi, row in enumerate(pairs):
        pair_id = row["pair_id"]
        ref_path = row["reference_path"]
        search_path = row["search_path"]
        gt_x = float(row["x_gt"])
        gt_y = float(row["y_gt"])
        gt_theta = float(row["theta_gt"])
        gt_scale = float(row["scale_gt"])
        gt_found = int(row["found_gt"])
        set_name = row["set"]
        gen_id = row.get("generator_id", "generic")

        t0 = time.time()
        res_dict, best_coarse, clean_refined = engine.localize_pair(
            ref_path, search_path,
            ncc_weight=0.5, rejection_thresh=0.42, scale_step=0.25, theta_step=1.0,
            return_diagnostics=True
        )
        t1 = time.time()
        runtime_ms = (t1 - t0) * 1000.0

        pred_x, pred_y = res_dict["x"], res_dict["y"]
        pred_theta, pred_scale = res_dict["theta"], res_dict["scale"]
        pred_found, pred_score = res_dict["found"], res_dict["score"]

        if gt_found == 1 and pred_found == 1:
            loc_err = math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
            scale_err = abs(pred_scale - gt_scale)
            theta_err = abs(pred_theta - gt_theta)
        elif gt_found == 0 and pred_found == 0:
            loc_err = scale_err = theta_err = 0.0
        else:
            loc_err = scale_err = theta_err = 999.0

        base_results.append({
            "pair_id": pair_id, "set": set_name, "gen_id": gen_id,
            "gt_x": gt_x, "gt_y": gt_y, "gt_theta": gt_theta, "gt_scale": gt_scale, "gt_found": gt_found,
            "pred_x": pred_x, "pred_y": pred_y, "pred_theta": pred_theta, "pred_scale": pred_scale,
            "pred_found": pred_found, "pred_score": pred_score,
            "loc_err": loc_err, "scale_err": scale_err, "theta_err": theta_err, "runtime_ms": runtime_ms,
            "raw_ncc": res_dict.get("raw_ncc", 0.0), "raw_siamese": res_dict.get("raw_siamese", 0.0)
        })

        if pair_id in target_pair_ids:
            def min_d(cands):
                if not cands: return 9999.0
                return min(math.sqrt((c["x"] - gt_x)**2 + (c["y"] - gt_y)**2) for c in cands)

            d_ref = min_d(clean_refined)
            base_target_debug[pair_id] = {
                "gt_x": gt_x, "gt_y": gt_y, "loc_err": round(loc_err, 2),
                "pred_found": pred_found, "min_d_refined": round(d_ref, 2),
                "gt_in_refined": d_ref <= 15.0, "raw_ncc": res_dict.get("raw_ncc", 0.0)
            }

        gc.collect()

    base_metrics = compute_100pt_breakdown(base_results)
    print(f"\n---> PRODUCTION BASELINE TOTAL SCORE: {base_metrics['total_100_score']:.2f} / 100")
    print(f"     Loc: {base_metrics['loc_score']:.2f}/40 | Pose: {base_metrics['pose_score']:.2f}/20 | Rej: {base_metrics['rejection_score']:.2f}/15 | Conf: {base_metrics['confidence_score']:.2f}/10 | Eff: {base_metrics['eff_score']:.2f}/5")

    # 4. Evaluate EXP-12 Gradient-Normalized NCC
    print(f"\n{'='*70}")
    print("RUNNING EXP-12 EVALUATION (Gradient-Normalized NCC)...")
    print(f"{'='*70}")

    exp12_results = []
    exp12_target_debug = {}

    for pi, row in enumerate(pairs):
        pair_id = row["pair_id"]
        ref_path = row["reference_path"]
        search_path = row["search_path"]
        gt_x = float(row["x_gt"])
        gt_y = float(row["y_gt"])
        gt_theta = float(row["theta_gt"])
        gt_scale = float(row["scale_gt"])
        gt_found = int(row["found_gt"])
        set_name = row["set"]
        gen_id = row.get("generator_id", "generic")

        t0 = time.time()
        res_dict, best_coarse, clean_refined = localize_pair_gradient_ncc(
            engine, ref_path, search_path,
            ncc_weight=0.5, rejection_thresh=0.42, scale_step=0.25, theta_step=1.0
        )
        t1 = time.time()
        runtime_ms = (t1 - t0) * 1000.0

        pred_x, pred_y = res_dict["x"], res_dict["y"]
        pred_theta, pred_scale = res_dict["theta"], res_dict["scale"]
        pred_found, pred_score = res_dict["found"], res_dict["score"]

        if gt_found == 1 and pred_found == 1:
            loc_err = math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
            scale_err = abs(pred_scale - gt_scale)
            theta_err = abs(pred_theta - gt_theta)
        elif gt_found == 0 and pred_found == 0:
            loc_err = scale_err = theta_err = 0.0
        else:
            loc_err = scale_err = theta_err = 999.0

        exp12_results.append({
            "pair_id": pair_id, "set": set_name, "gen_id": gen_id,
            "gt_x": gt_x, "gt_y": gt_y, "gt_theta": gt_theta, "gt_scale": gt_scale, "gt_found": gt_found,
            "pred_x": pred_x, "pred_y": pred_y, "pred_theta": pred_theta, "pred_scale": pred_scale,
            "pred_found": pred_found, "pred_score": pred_score,
            "loc_err": loc_err, "scale_err": scale_err, "theta_err": theta_err, "runtime_ms": runtime_ms,
            "grad_ncc": res_dict.get("raw_ncc", 0.0), "raw_siamese": res_dict.get("raw_siamese", 0.0)
        })

        if pair_id in target_pair_ids:
            def min_d(cands):
                if not cands: return 9999.0
                return min(math.sqrt((c["x"] - gt_x)**2 + (c["y"] - gt_y)**2) for c in cands)

            d_ref = min_d(clean_refined)
            
            # Find GT candidate in clean_refined
            gt_cand_rank = 99
            gt_grad_ncc = 0.0
            decoy_grad_ncc = clean_refined[0]["ncc_norm"] if len(clean_refined) > 0 else 0.0
            
            for rank_idx, c in enumerate(clean_refined):
                d = math.sqrt((c["x"] - gt_x)**2 + (c["y"] - gt_y)**2)
                if d <= 15.0:
                    gt_cand_rank = rank_idx + 1
                    gt_grad_ncc = c["ncc_norm"]
                    break

            exp12_target_debug[pair_id] = {
                "gt_x": gt_x, "gt_y": gt_y, "loc_err": round(loc_err, 2),
                "pred_found": pred_found, "min_d_refined": round(d_ref, 2),
                "gt_in_refined": d_ref <= 15.0,
                "gt_cand_rank": gt_cand_rank,
                "gt_grad_ncc": round(gt_grad_ncc, 4),
                "decoy_grad_ncc": round(decoy_grad_ncc, 4),
                "selected_grad_ncc": res_dict.get("raw_ncc", 0.0)
            }

        if (pi + 1) % 40 == 0 or pair_id in target_pair_ids:
            marker = " *** TARGET ***" if pair_id in target_pair_ids else ""
            print(f"  [{pi+1:3d}/200] {pair_id} | loc_err={loc_err:.2f}px | {runtime_ms:.0f}ms{marker}")

        gc.collect()

    exp12_metrics = compute_100pt_breakdown(exp12_results)

    print(f"\n---> EXP-12 GRADIENT-NCC TOTAL SCORE: {exp12_metrics['total_100_score']:.2f} / 100")
    print(f"     Loc: {exp12_metrics['loc_score']:.2f}/40 | Pose: {exp12_metrics['pose_score']:.2f}/20 | Rej: {exp12_metrics['rejection_score']:.2f}/15 | Conf: {exp12_metrics['confidence_score']:.2f}/10 | Eff: {exp12_metrics['eff_score']:.2f}/5")

    # Recall Audit Metrics
    n_present = sum(1 for r in pairs if int(r["found_gt"]) == 1)

    def calc_recall(results_list):
        rec = {1: 0, 5: 0, 15: 0, 50: 0}
        for r in results_list:
            if r["gt_found"] == 1 and r["pred_found"] == 1:
                err = r["loc_err"]
                if err <= 1.0: rec[1] += 1
                if err <= 5.0: rec[5] += 1
                if err <= 15.0: rec[15] += 1
                if err <= 50.0: rec[50] += 1
        return {t: round(count / float(n_present) * 100.0, 2) for t, count in rec.items()}

    base_recall = calc_recall(base_results)
    exp12_recall = calc_recall(exp12_results)

    # Comparison Table
    print(f"\n{'='*70}")
    print("EXP-12 vs BASELINE COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'Total':>7} {'Loc':>7} {'Pose':>7} {'Rej':>7} {'Conf':>7} {'Eff':>5} {'Rec@5px':>10} {'Rec@15px':>11} {'Med_RT':>8}")
    print("-" * 90)
    print(f"{'Baseline (Raw NCC)':<25} {base_metrics['total_100_score']:7.2f} {base_metrics['loc_score']:7.2f} {base_metrics['pose_score']:7.2f} {base_metrics['rejection_score']:7.2f} {base_metrics['confidence_score']:7.2f} {base_metrics['eff_score']:5.1f} {base_recall[5]:9.1f}% {base_recall[15]:10.1f}% {base_metrics['med_rt']:7.0f}ms")
    print(f"{'EXP-12 (Gradient NCC)':<25} {exp12_metrics['total_100_score']:7.2f} {exp12_metrics['loc_score']:7.2f} {exp12_metrics['pose_score']:7.2f} {exp12_metrics['rejection_score']:7.2f} {exp12_metrics['confidence_score']:7.2f} {exp12_metrics['eff_score']:5.1f} {exp12_recall[5]:9.1f}% {exp12_recall[15]:10.1f}% {exp12_metrics['med_rt']:7.0f}ms")

    # Target pair deep-dive
    print(f"\n{'='*70}")
    print("TARGET PAIRS DEEP-DIVE (Raw NCC vs Gradient NCC)")
    print(f"{'='*70}")
    for pid in sorted(target_pair_ids):
        bd = base_target_debug[pid]
        ed = exp12_target_debug[pid]
        print(f"\n--- {pid} ---")
        print(f"  GT location: ({bd['gt_x']}, {bd['gt_y']})")
        print(f"  Baseline (Raw NCC):  LocErr = {bd['loc_err']:>6.2f}px | GT in Refined = {bd['gt_in_refined']} | Raw NCC = {bd['raw_ncc']:.4f}")
        print(f"  EXP-12 (Grad NCC):  LocErr = {ed['loc_err']:>6.2f}px | GT in Refined = {ed['gt_in_refined']} | GT Rank = {ed['gt_cand_rank']} | GT GradNCC = {ed['gt_grad_ncc']:.4f} vs Decoy GradNCC = {ed['decoy_grad_ncc']:.4f}")

    # Regression analysis
    recovered = []
    regressed = []
    unchanged = []

    for idx, b_r in enumerate(base_results):
        e_r = exp12_results[idx]
        pid = b_r["pair_id"]
        b_err = b_r["loc_err"]
        e_err = e_r["loc_err"]

        if b_err > 5.0 and e_err <= 5.0:
            recovered.append((pid, b_err, e_err))
        elif b_err <= 5.0 and e_err > 5.0:
            regressed.append((pid, b_err, e_err))
        else:
            unchanged.append(pid)

    print(f"\n{'='*70}")
    print("REGRESSION ANALYSIS (EXP-12 vs Baseline 60.99)")
    print(f"{'='*70}")
    print(f"  Recovered pairs (Baseline failed >5px, EXP-12 passed <=5px): {len(recovered)}")
    for r in recovered:
        print(f"    + {r[0]}: Baseline err = {r[1]:.2f}px -> EXP-12 err = {r[2]:.2f}px")
    print(f"  Regressed pairs (Baseline passed <=5px, EXP-12 failed >5px): {len(regressed)}")
    for r in regressed:
        print(f"    - {r[0]}: Baseline err = {r[1]:.2f}px -> EXP-12 err = {r[2]:.2f}px")
    print(f"  Unchanged pairs: {len(unchanged)}")

    base_total = base_metrics["total_100_score"]
    base_loc = base_metrics["loc_score"]
    exp_total = exp12_metrics["total_100_score"]
    exp_loc = exp12_metrics["loc_score"]

    delta_total = exp_total - base_total
    delta_loc = exp_loc - base_loc

    print(f"\n{'='*70}")
    print("DECISION EVALUATION")
    print(f"{'='*70}")
    print(f"  Baseline Total: {base_total:.2f} | EXP-12 Total: {exp_total:.2f} (Delta = {delta_total:+.2f})")
    print(f"  Baseline Loc:   {base_loc:.2f} | EXP-12 Loc:   {exp_loc:.2f} (Delta = {delta_loc:+.2f})")

    if delta_total > 1.0 and delta_loc >= 0 and len(regressed) == 0:
        verdict = "PROMOTE"
        print(f"\n  [PASS] PROMOTE: EXP-12 improved total score by {delta_total:.2f} and localization by {delta_loc:.2f} with zero regressions!")
    elif delta_total > 1.0 and delta_loc >= 0:
        verdict = "PROMOTE"
        print(f"\n  [PASS] PROMOTE: Total score improved by {delta_total:.2f}.")
    elif delta_total <= 1.0 and delta_total > 0:
        verdict = "REJECT"
        print(f"\n  [FAIL] REJECT: Total score gain ({delta_total:+.2f}) is <= +1.0 threshold.")
    elif delta_loc < 0:
        verdict = "REJECT"
        print(f"\n  [FAIL] REJECT: Localization score regressed by {abs(delta_loc):.2f}.")
    else:
        verdict = "REJECT"
        print(f"\n  [FAIL] REJECT: Total score did not improve over baseline.")

    # Save CSV
    os.makedirs("phase2/results", exist_ok=True)
    csv_path = "phase2/results/exp12_gradient_ncc.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair_id", "set", "gen_id", "gt_found", "pred_found",
            "gt_x", "gt_y", "pred_x", "pred_y", "loc_err",
            "gt_scale", "pred_scale", "scale_err",
            "gt_theta", "pred_theta", "theta_err",
            "pred_score", "grad_ncc", "raw_siamese", "runtime_ms"
        ])
        for r in exp12_results:
            writer.writerow([
                r["pair_id"], r["set"], r["gen_id"], r["gt_found"], r["pred_found"],
                r["gt_x"], r["gt_y"], r["pred_x"], r["pred_y"], round(r["loc_err"], 2),
                r["gt_scale"], r["pred_scale"], round(r["scale_err"], 2),
                r["gt_theta"], r["pred_theta"], round(r["theta_err"], 2),
                r["pred_score"], r["grad_ncc"], r["raw_siamese"], round(r["runtime_ms"], 2)
            ])
    print(f"\n[OK] Saved CSV to {csv_path}")

    # Write Markdown Report
    report_path = "phase2/reports/EXP12_GRADIENT_NCC_ANALYSIS.md"
    os.makedirs("phase2/reports", exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"""# EXP-12 — GRADIENT-NORMALIZED NCC ANALYSIS REPORT

## Executive Summary

- **Baseline Total Score (Raw NCC)**: {base_total:.2f} / 100
- **EXP-12 Total Score (Gradient NCC)**: {exp_total:.2f} / 100
- **Delta Total Score**: {delta_total:+.2f}
- **Baseline Localization Score**: {base_loc:.2f} / 40
- **EXP-12 Localization Score**: {exp_loc:.2f} / 40
- **Delta Localization Score**: {delta_loc:+.2f}
- **Decision**: **{verdict}**

---

## 100-Point Score Breakdown

| Metric | Baseline (Raw Intensity NCC) | EXP-12 (Gradient-Normalized NCC) |
|---|---|---|
| **Localization /40** | {base_metrics['loc_score']:.2f} | {exp12_metrics['loc_score']:.2f} |
| **Scale /10** | {base_metrics['scale_score']:.2f} | {exp12_metrics['scale_score']:.2f} |
| **Rotation /10** | {base_metrics['theta_score']:.2f} | {exp12_metrics['theta_score']:.2f} |
| **Pose Total /20** | {base_metrics['pose_score']:.2f} | {exp12_metrics['pose_score']:.2f} |
| **Rejection /15** | {base_metrics['rejection_score']:.2f} | {exp12_metrics['rejection_score']:.2f} |
| **Confidence /10** | {base_metrics['confidence_score']:.2f} | {exp12_metrics['confidence_score']:.2f} |
| **Efficiency /5** | {base_metrics['eff_score']:.2f} | {exp12_metrics['eff_score']:.2f} |
| **Generator/Citations /10** | 10.00 | 10.00 |
| **TOTAL SCORE /100** | **{base_total:.2f}** | **{exp_total:.2f}** |

---

## Candidate Recall Audit

| Metric / Threshold | Baseline (Raw NCC) | EXP-12 (Gradient NCC) |
|---|---|---|
| **Final Selected Recall @1px** | {base_recall[1]}% | {exp12_recall[1]}% |
| **Final Selected Recall @5px** | {base_recall[5]}% | {exp12_recall[5]}% |
| **Final Selected Recall @15px** | {base_recall[15]}% | {exp12_recall[15]}% |
| **Final Selected Recall @50px** | {base_recall[50]}% | {exp12_recall[50]}% |

---

## Target Pairs Forensic Breakdown

""")
        for pid in sorted(target_pair_ids):
            bd = base_target_debug[pid]
            ed = exp12_target_debug[pid]
            f.write(f"### {pid}\n")
            f.write(f"- **GT Location**: ({bd['gt_x']}, {bd['gt_y']})\n")
            f.write(f"- **Baseline (Raw NCC)**: LocErr = {bd['loc_err']}px, GT in Refined = {bd['gt_in_refined']}, Raw NCC = {bd['raw_ncc']:.4f}\n")
            f.write(f"- **EXP-12 (Grad NCC)**: LocErr = {ed['loc_err']}px, GT in Refined = {ed['gt_in_refined']}, GT Rank = {ed['gt_cand_rank']}, GT GradNCC = {ed['gt_grad_ncc']:.4f} vs Decoy GradNCC = {ed['decoy_grad_ncc']:.4f}\n\n")

        f.write(f"""---

## Regression Analysis (EXP-12 vs Baseline 60.99)

- **Recovered Pairs**: {len(recovered)}
""")
        for r in recovered:
            f.write(f"  - `{r[0]}`: Baseline error {r[1]:.2f}px -> EXP-12 error {r[2]:.2f}px\n")

        f.write(f"""- **Regressed Pairs**: {len(regressed)}\n""")
        for r in regressed:
            f.write(f"  - `{r[0]}`: Baseline error {r[1]:.2f}px -> EXP-12 error {r[2]:.2f}px\n")

        f.write(f"""- **Unchanged Pairs**: {len(unchanged)}

---

## Runtime Performance

| Metric | Baseline (Raw NCC) | EXP-12 (Gradient NCC) |
|---|---|---|
| **Median Runtime** | {base_metrics['med_rt']:.0f} ms | {exp12_metrics['med_rt']:.0f} ms |
| **P90 Runtime** | {base_metrics['p90_rt']:.0f} ms | {exp12_metrics['p90_rt']:.0f} ms |
| **P99 Runtime** | {base_metrics['p99_rt']:.0f} ms | {exp12_metrics['p99_rt']:.0f} ms |

---

## Final Decision: {verdict}
""")

    print(f"[OK] Report saved to {report_path}")


if __name__ == "__main__":
    run_experiment()
