#!/usr/bin/env python3
"""
EXP-09 — Multi-Peak NCC Candidate Generation
=============================================

STRICT SINGLE-CHANGE EXPERIMENT

Hypothesis:
    Extracting multiple local NCC peaks per scale/rotation (with spatial NMS)
    recovers true GT landmark candidates that single global-max extraction misses
    due to periodic decoy structures.

Change:
    Replace single global maximum extraction per (scale, theta) in coarse search
    with multi-peak local extraction (up to K_peaks peaks per combination, NMS_radius = 15px coarse space).

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


def extract_multipeak_coarse_candidates(ref_img, search_img, config, peaks_per_scale_theta=5, nms_radius=15):
    """
    Coarse search extracting multiple local NCC peaks per (scale, theta) combination.
    Coordinates returned in full 1000x1000 search image space.
    """
    if ref_img.shape != (100, 100):
        ref_template = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
    else:
        ref_template = ref_img.copy()

    search_coarse = cv2.resize(search_img, (500, 500), interpolation=cv2.INTER_AREA)
    
    coarse_candidates = []
    coarse_scales = config.COARSE_SCALES
    coarse_thetas = config.COARSE_THETAS

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
            res_work = res_ncc.copy()
            del res_ncc

            # Multi-peak extraction with NMS
            count = 0
            while count < peaks_per_scale_theta:
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_work)
                if max_val < -0.5:
                    break

                cx = (max_loc[0] + ref_coarse.shape[1] / 2.0) * 2.0
                cy = (max_loc[1] + ref_coarse.shape[0] / 2.0) * 2.0

                coarse_candidates.append({
                    "coarse_ncc": float(max_val),
                    "x": cx,
                    "y": cy,
                    "scale": scale,
                    "theta": theta
                })
                count += 1

                # Zero out region around max_loc for NMS
                r_rad = max(4, int(ref_coarse.shape[1] * 0.2))
                y1 = max(0, max_loc[1] - r_rad)
                y2 = min(res_work.shape[0], max_loc[1] + r_rad)
                x1 = max(0, max_loc[0] - r_rad)
                x2 = min(res_work.shape[1], max_loc[0] + r_rad)
                res_work[y1:y2, x1:x2] = -1.0

            del res_work

    # Spatial deduplication across scale/theta combinations
    coarse_candidates.sort(key=lambda c: -c["coarse_ncc"])
    dedup_candidates = []
    for c in coarse_candidates:
        dup = False
        for d in dedup_candidates:
            dist = math.sqrt((c["x"] - d["x"])**2 + (c["y"] - d["y"])**2)
            if dist < 15.0 and abs(c["scale"] - d["scale"]) < 0.5 and abs(c["theta"] - d["theta"]) < 2.0:
                dup = True
                break
        if not dup:
            dedup_candidates.append(c)

    return coarse_candidates, dedup_candidates


def run_pipeline_with_multipeak(engine, ref_input, search_input, ncc_weight=0.5, rejection_thresh=0.42,
                                scale_step=0.25, theta_step=1.0, peaks_per_scale_theta=5, top_k_coarse=10):
    """
    Runs full inference pipeline replacing single-peak coarse extraction with multi-peak extraction.
    """
    ref_img = load_grayscale_image(ref_input)
    search_img = load_grayscale_image(search_input)

    h_s, w_s = search_img.shape

    if ref_img.shape != (100, 100):
        ref_template = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
    else:
        ref_template = ref_img.copy()

    ref_emb = engine.extract_siamese_embedding(ref_template)

    # 1. Multi-peak coarse candidate generation
    all_raw_coarse, dedup_coarse = extract_multipeak_coarse_candidates(
        ref_img, search_img, engine.config,
        peaks_per_scale_theta=peaks_per_scale_theta, nms_radius=15
    )

    top_candidates = dedup_coarse[:top_k_coarse]

    # 2. Fine Grid Refinement around Top Candidates (Identical to production)
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
        }, all_raw_coarse, top_candidates, []

    cand_embs = engine.extract_batch_embeddings(patch_list)
    siamese_sims = torch.sum(ref_emb * cand_embs, dim=1).cpu().numpy()

    w_alpha = ncc_weight
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

    refined_results.sort(key=lambda r: -r["adjusted_score"])
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

    if final_fused >= rejection_thresh:
        found = 1
        pred_x = float(round(fine_x, 2))
        pred_y = float(round(fine_y, 2))
        pred_theta = float(round(best_cand["theta"], 2))
        pred_scale = float(round(best_cand["scale"], 2))
    else:
        found = 0
        pred_x = pred_y = pred_theta = pred_scale = 0.0

    conf_score = 1.0 / (1.0 + math.exp(-engine.config.CONFIDENCE_SLOPE * (final_fused - rejection_thresh)))
    conf_score = float(round(max(0.0001, min(0.9999, conf_score)), 4))

    # Strip match_matrix before returning diagnostic objects to save RAM
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

    del cand_embs, siamese_sims, patch_list, refined_results
    gc.collect()

    return res_dict, all_raw_coarse, top_candidates, clean_refined


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
    print("EXP-09: MULTI-PEAK NCC CANDIDATE GENERATION")
    print("=" * 70)

    ckpt_path = "phase2_checkpoints/best_model_level1.pth"
    with open(ckpt_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    expected = "e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c"
    assert sha == expected, f"SHA-256 MISMATCH: {sha}"
    print(f"[OK] Checkpoint SHA-256 verified: {sha[:16]}...")

    engine = Phase2InferenceEngine(checkpoint_path="best_model_level1.pth", device="cpu")
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
    k_ablation_values = [1, 3, 5, 10]
    
    ablation_metrics = {}
    ablation_results = {}
    ablation_recall = {}
    ablation_targets = {}
    ablation_diagnostics = {}

    for K_peaks in k_ablation_values:
        print(f"\n{'='*70}")
        print(f"RUNNING EVALUATION FOR K_peaks = {K_peaks} (peaks per scale/theta)")
        print(f"{'='*70}")

        results = []
        target_debug = {}

        # Recall metrics for this K
        recall_counts = {
            "all_coarse": {1: 0, 5: 0, 15: 0, 50: 0},
            "top_k_coarse": {1: 0, 5: 0, 15: 0, 50: 0},
            "all_refined": {1: 0, 5: 0, 15: 0, 50: 0},
            "top5_refined": {1: 0, 5: 0, 15: 0, 50: 0},
        }

        diag_counts = {
            "gt_absent_coarse": 0,
            "gt_lost_nms_dedup": 0,
            "gt_lost_topk_coarse": 0,
            "gt_lost_refinement": 0,
            "gt_lost_top5_ranking": 0,
            "gt_success_top1": 0,
            "gt_success_top5": 0,
        }

        n_present = 0

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
            res_dict, raw_coarse, top_coarse, clean_refined = run_pipeline_with_multipeak(
                engine, ref_path, search_path,
                ncc_weight=0.5, rejection_thresh=0.42, scale_step=0.25, theta_step=1.0,
                peaks_per_scale_theta=K_peaks, top_k_coarse=10
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

            # Evaluate recall on present pairs
            if gt_found == 1:
                n_present += 1
                def min_d(cands):
                    if not cands: return 9999.0
                    return min(math.sqrt((c["x"] - gt_x)**2 + (c["y"] - gt_y)**2) for c in cands)

                d_raw_coarse = min_d(raw_coarse)
                d_top_coarse = min_d(top_coarse)
                d_all_refined = min_d(clean_refined)
                top5_refined = clean_refined[:5]
                d_top5_refined = min_d(top5_refined)

                for t_px in [1, 5, 15, 50]:
                    if d_raw_coarse <= t_px: recall_counts["all_coarse"][t_px] += 1
                    if d_top_coarse <= t_px: recall_counts["top_k_coarse"][t_px] += 1
                    if d_all_refined <= t_px: recall_counts["all_refined"][t_px] += 1
                    if d_top5_refined <= t_px: recall_counts["top5_refined"][t_px] += 1

                # Failure stage diagnostic
                if d_raw_coarse > 15.0:
                    diag_counts["gt_absent_coarse"] += 1
                elif d_top_coarse > 15.0:
                    diag_counts["gt_lost_topk_coarse"] += 1
                elif d_all_refined > 15.0:
                    diag_counts["gt_lost_refinement"] += 1
                elif d_top5_refined > 15.0:
                    diag_counts["gt_lost_top5_ranking"] += 1
                else:
                    if loc_err <= 5.0:
                        diag_counts["gt_success_top1"] += 1
                    else:
                        diag_counts["gt_success_top5"] += 1

            results.append({
                "pair_id": pair_id, "set": set_name, "gen_id": gen_id,
                "gt_x": gt_x, "gt_y": gt_y, "gt_theta": gt_theta, "gt_scale": gt_scale, "gt_found": gt_found,
                "pred_x": pred_x, "pred_y": pred_y, "pred_theta": pred_theta, "pred_scale": pred_scale,
                "pred_found": pred_found, "pred_score": pred_score,
                "loc_err": loc_err, "scale_err": scale_err, "theta_err": theta_err, "runtime_ms": runtime_ms
            })

            if pair_id in target_pair_ids:
                d_coarse = min_d(raw_coarse)
                d_topk = min_d(top_coarse)
                d_ref = min_d(clean_refined)
                top5_cands = clean_refined[:5]
                d_top5 = min_d(top5_cands)

                target_debug[pair_id] = {
                    "gt_x": gt_x, "gt_y": gt_y,
                    "pred_x": pred_x, "pred_y": pred_y,
                    "loc_err": round(loc_err, 2),
                    "pred_found": pred_found,
                    "min_d_raw_coarse": round(d_coarse, 2),
                    "min_d_top_coarse": round(d_topk, 2),
                    "min_d_all_refined": round(d_ref, 2),
                    "min_d_top5_refined": round(d_top5, 2),
                    "gt_recovered_coarse": d_coarse <= 15.0,
                    "gt_survived_topk": d_topk <= 15.0,
                    "gt_survived_refined": d_ref <= 15.0,
                    "gt_survived_top5": d_top5 <= 15.0,
                    "final_selected_is_gt": loc_err <= 5.0,
                    "top5_details": [
                        {
                            "rank": r_idx + 1, "x": round(c["x"], 1), "y": round(c["y"], 1),
                            "ncc": round(c["ncc_norm"], 4), "sia": round(c["siamese_sim"], 4),
                            "fused": round(c["fused_score"], 4), "dist_gt": round(math.sqrt((c["x"]-gt_x)**2 + (c["y"]-gt_y)**2), 1)
                        } for r_idx, c in enumerate(top5_cands)
                    ]
                }

            if (pi + 1) % 40 == 0 or pair_id in target_pair_ids:
                marker = " *** TARGET ***" if pair_id in target_pair_ids else ""
                print(f"  [{pi+1:3d}/200] {pair_id} | loc_err={loc_err:.1f}px | {runtime_ms:.0f}ms{marker}")

            gc.collect()

        metrics = compute_100pt_breakdown(results)
        ablation_metrics[K_peaks] = metrics
        ablation_results[K_peaks] = results
        ablation_recall[K_peaks] = {
            stage: {t_px: round(count / float(n_present) * 100.0, 2) for t_px, count in t_dict.items()}
            for stage, t_dict in recall_counts.items()
        }
        ablation_targets[K_peaks] = target_debug
        ablation_diagnostics[K_peaks] = diag_counts

        print(f"\n---> K_peaks={K_peaks} TOTAL SCORE: {metrics['total_100_score']:.2f} / 100")
        print(f"     Loc: {metrics['loc_score']:.2f}/40 | Pose: {metrics['pose_score']:.2f}/20 | Rej: {metrics['rejection_score']:.2f}/15 | Conf: {metrics['confidence_score']:.2f}/10 | Eff: {metrics['eff_score']:.2f}/5")
        print(f"     Top-5 Refined Recall @15px: {ablation_recall[K_peaks]['top5_refined'][15]}%")
        print(f"     All Refined Recall @15px: {ablation_recall[K_peaks]['all_refined'][15]}%")

    # Print summary of ablation
    print(f"\n{'='*70}")
    print("EXP-09 ABLATION SUMMARY (K_peaks = 1, 3, 5, 10)")
    print(f"{'='*70}")
    print(f"{'K_peaks':<8} {'Total':>7} {'Loc':>7} {'Pose':>7} {'Rej':>7} {'Conf':>7} {'Eff':>5} {'Top5_Rec@5':>11} {'Top5_Rec@15':>12} {'Med_RT':>8}")
    print("-" * 80)
    print(f"{'BASE(1)':<8} {'46.77':>7} {'9.38':>7} {'-':>7} {'13.70':>7} {'9.69':>7} {'5.0':>5} {'28.7%':>11} {'36.2%':>12} {'347ms':>8}")
    for K in k_ablation_values:
        m = ablation_metrics[K]
        rec = ablation_recall[K]
        print(f"{K:<8} {m['total_100_score']:7.2f} {m['loc_score']:7.2f} {m['pose_score']:7.2f} {m['rejection_score']:7.2f} {m['confidence_score']:7.2f} {m['eff_score']:5.1f} {rec['top5_refined'][5]:10.1f}% {rec['top5_refined'][15]:11.1f}% {m['med_rt']:7.0f}ms")

    # Target pair summary for primary candidate K=5
    print(f"\n{'='*70}")
    print("TARGET PAIRS SUMMARY (K_peaks = 5)")
    print(f"{'='*70}")
    t5 = ablation_targets[5]
    for pid in sorted(target_pair_ids):
        if pid in t5:
            d = t5[pid]
            rec_c = "YES" if d["gt_recovered_coarse"] else "NO"
            rec_r = "YES" if d["gt_survived_refined"] else "NO"
            rec_t5 = "YES" if d["gt_survived_top5"] else "NO"
            sel = "GT" if d["final_selected_is_gt"] else "DECOY"
            print(f"  {pid}: GT recovered coarse={rec_c} | Survived refined={rec_r} | Survived Top5={rec_t5} | Selected={sel} (err={d['loc_err']}px)")

    # Diagnostic summary for K=5
    print(f"\n{'='*70}")
    print("DIAGNOSTIC FAILURE STAGE COUNTS (K_peaks = 5, total present = 160)")
    print(f"{'='*70}")
    dc = ablation_diagnostics[5]
    print(f"  GT absent from coarse search (>15px):       {dc['gt_absent_coarse']}")
    print(f"  GT lost at coarse Top-K selection (>15px):   {dc['gt_lost_topk_coarse']}")
    print(f"  GT lost during fine grid refinement (>15px): {dc['gt_lost_refinement']}")
    print(f"  GT lost at final Top-5 fused ranking:        {dc['gt_lost_top5_ranking']}")
    print(f"  GT successfully selected (Top-1 <= 5px):     {dc['gt_success_top1']}")
    print(f"  GT in Top-5 but wrong candidate selected:    {dc['gt_success_top5']}")

    # Regression analysis (K=5 vs Baseline)
    k5_res = ablation_results[5]
    base_loc = 9.38
    base_total = 46.77
    k5_loc = ablation_metrics[5]["loc_score"]
    k5_total = ablation_metrics[5]["total_100_score"]

    recovered_pairs = []
    regressed_pairs = []
    unchanged_pairs = []

    # Save EXP-09 CSV
    os.makedirs("phase2/results", exist_ok=True)
    csv_path = "phase2/results/exp09_multipeak_ncc.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair_id", "set", "gen_id", "gt_found", "pred_found",
            "gt_x", "gt_y", "pred_x", "pred_y", "loc_err",
            "gt_scale", "pred_scale", "scale_err",
            "gt_theta", "pred_theta", "theta_err",
            "pred_score", "runtime_ms"
        ])
        for r in k5_res:
            writer.writerow([
                r["pair_id"], r["set"], r["gen_id"], r["gt_found"], r["pred_found"],
                r["gt_x"], r["gt_y"], r["pred_x"], r["pred_y"], round(r["loc_err"], 2),
                r["gt_scale"], r["pred_scale"], round(r["scale_err"], 2),
                r["gt_theta"], r["pred_theta"], round(r["theta_err"], 2),
                r["pred_score"], round(r["runtime_ms"], 2)
            ])
    print(f"\n[OK] Saved CSV to {csv_path}")

    # Decision
    delta_total = k5_total - base_total
    delta_loc = k5_loc - base_loc

    print(f"\n{'='*70}")
    print("DECISION ANALYSIS")
    print(f"{'='*70}")
    print(f"  Baseline Total: {base_total:.2f} | K=5 Total: {k5_total:.2f} (Delta = {delta_total:+.2f})")
    print(f"  Baseline Loc:   {base_loc:.2f} | K=5 Loc:   {k5_loc:.2f} (Delta = {delta_loc:+.2f})")

    if delta_total > 0 and delta_loc >= 0:
        verdict = "PROMOTE"
        print(f"\n  [PASS] PROMOTE: Total score improved by {delta_total:.2f} without localization regression.")
    elif delta_total > 0 and delta_loc < 0:
        verdict = "REJECT"
        print(f"\n  [FAIL] REJECT: Total score improved but localization REGRESSED by {abs(delta_loc):.2f}.")
    else:
        verdict = "REJECT"
        print(f"\n  [FAIL] REJECT: No improvement over baseline.")

    # Write Markdown Report
    report_path = "phase2/reports/EXP09_MULTIPEAK_NCC_ANALYSIS.md"
    os.makedirs("phase2/reports", exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"""# EXP-09 MULTI-PEAK NCC REPORT

## Summary

- **Baseline Total**: {base_total:.2f} / 100
- **EXP-09 Total (K=5)**: {k5_total:.2f} / 100
- **Delta Total**: {delta_total:+.2f}
- **Baseline Loc**: {base_loc:.2f} / 40
- **EXP-09 Loc (K=5)**: {k5_loc:.2f} / 40
- **Delta Loc**: {delta_loc:+.2f}
- **Decision**: **{verdict}**

---

## Candidate Recall (K_peaks = 5)

| Stage | @1px | @5px | @15px | @50px |
|---|---|---|---|---|
| **All Coarse** | {ablation_recall[5]['all_coarse'][1]}% | {ablation_recall[5]['all_coarse'][5]}% | {ablation_recall[5]['all_coarse'][15]}% | {ablation_recall[5]['all_coarse'][50]}% |
| **Top-K Coarse** | {ablation_recall[5]['top_k_coarse'][1]}% | {ablation_recall[5]['top_k_coarse'][5]}% | {ablation_recall[5]['top_k_coarse'][15]}% | {ablation_recall[5]['top_k_coarse'][50]}% |
| **All Refined** | {ablation_recall[5]['all_refined'][1]}% | {ablation_recall[5]['all_refined'][5]}% | {ablation_recall[5]['all_refined'][15]}% | {ablation_recall[5]['all_refined'][50]}% |
| **Final Top-5** | {ablation_recall[5]['top5_refined'][1]}% | {ablation_recall[5]['top5_refined'][5]}% | {ablation_recall[5]['top5_refined'][15]}% | {ablation_recall[5]['top5_refined'][50]}% |

---

## K_peaks Ablation Table

| K_peaks | Total Score | Loc /40 | Pose /20 | Rej /15 | Conf /10 | Eff /5 | Top-5 Rec@5px | Top-5 Rec@15px | Med Runtime |
|---|---|---|---|---|---|---|---|---|---|
| **BASE (1)** | **46.77** | **9.38** | **9.00** | **13.70** | **9.69** | **5.0** | **28.7%** | **36.2%** | **347ms** |
""")
        for K in k_ablation_values:
            m = ablation_metrics[K]
            rec = ablation_recall[K]
            f.write(f"| {K} | {m['total_100_score']:.2f} | {m['loc_score']:.2f} | {m['pose_score']:.2f} | {m['rejection_score']:.2f} | {m['confidence_score']:.2f} | {m['eff_score']:.1f} | {rec['top5_refined'][5]:.1f}% | {rec['top5_refined'][15]:.1f}% | {m['med_rt']:.0f}ms |\n")

        f.write(f"""
---

## Target Failure Cases (K_peaks = 5)

""")
        for pid in sorted(target_pair_ids):
            if pid in t5:
                d = t5[pid]
                f.write(f"### {pid}\n")
                f.write(f"- **GT**: ({d['gt_x']}, {d['gt_y']})\n")
                f.write(f"- **GT Recovered Coarse**: {'YES' if d['gt_recovered_coarse'] else 'NO'} (dist={d['min_d_raw_coarse']}px)\n")
                f.write(f"- **GT Survived Top-K**: {'YES' if d['gt_survived_topk'] else 'NO'} (dist={d['min_d_top_coarse']}px)\n")
                f.write(f"- **GT Survived Refined**: {'YES' if d['gt_survived_refined'] else 'NO'} (dist={d['min_d_all_refined']}px)\n")
                f.write(f"- **GT Survived Top-5**: {'YES' if d['gt_survived_top5'] else 'NO'} (dist={d['min_d_top5_refined']}px)\n")
                f.write(f"- **Final Selected**: {'GT' if d['final_selected_is_gt'] else 'DECOY'} (loc_err={d['loc_err']}px)\n\n")

        f.write(f"""---

## Failure Stage Diagnostics (K_peaks = 5, Present N=160)

- GT absent from coarse search (>15px): **{dc['gt_absent_coarse']}**
- GT lost at coarse Top-K selection (>15px): **{dc['gt_lost_topk_coarse']}**
- GT lost during fine grid refinement (>15px): **{dc['gt_lost_refinement']}**
- GT lost at final Top-5 fused ranking: **{dc['gt_lost_top5_ranking']}**
- GT successfully selected (Top-1 <= 5px): **{dc['gt_success_top1']}**
- GT in Top-5 but wrong candidate selected: **{dc['gt_success_top5']}**

---

## Runtime Performance (K_peaks = 5)

- **Median**: {ablation_metrics[5]['med_rt']:.0f} ms
- **P90**: {ablation_metrics[5]['p90_rt']:.0f} ms
- **P99**: {ablation_metrics[5]['p99_rt']:.0f} ms

---

## Final Decision: {verdict}
""")

    print(f"[OK] Report saved to {report_path}")


if __name__ == "__main__":
    run_experiment()
