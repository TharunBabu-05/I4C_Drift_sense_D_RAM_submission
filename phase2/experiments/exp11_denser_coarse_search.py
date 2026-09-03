#!/usr/bin/env python3
"""
EXP-11 — Denser Coarse Search Only
===================================

STRICT SINGLE-CHANGE EXPERIMENT

Hypothesis:
    Increasing the coarse spatial search resolution (from 500x500 to 750x750 or 1000x1000)
    will improve coarse candidate recall by preserving fine structural landmarks that are blurred
    or coarsened at 500x500 resolution.

Tested Configurations:
    - 500x500 (Current Production Baseline: 60.99 / 100)
    - 750x750 (1.5x spatial resolution)
    - 1000x1000 (2.0x spatial resolution / Full Native)

Downstream pipeline:
    PROMOTED Strategy A (Pure NCC Primary Ranking)
    Subpixel refinement
    Sigmoid calibrated confidence
    Production rejection (tau = 0.42)

Production files: UNMODIFIED during experiment.
Checkpoint: UNMODIFIED.
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


def localize_pair_denser_coarse(engine, ref_input, search_input, coarse_dim=750,
                                ncc_weight=0.5, rejection_thresh=0.42,
                                scale_step=0.25, theta_step=1.0, top_k_coarse=10):
    """
    Executes Phase-2 search with configurable coarse search spatial resolution (coarse_dim x coarse_dim).
    Uses PROMOTED Strategy A (Pure NCC Primary Ranking).
    """
    ref_img = load_grayscale_image(ref_input)
    search_img = load_grayscale_image(search_input)

    w_alpha = ncc_weight if ncc_weight is not None else engine.config.NCC_WEIGHT
    tau = rejection_thresh if rejection_thresh is not None else engine.config.REJECTION_THRESHOLD

    h_s, w_s = search_img.shape
    scale_factor = float(w_s) / float(coarse_dim)

    # 1. Reference Landmark Template
    if ref_img.shape != (100, 100):
        ref_template = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
    else:
        ref_template = ref_img.copy()

    ref_emb = engine.extract_siamese_embedding(ref_template)

    # 2. Pyramidal Downsampling for Coarse Search (coarse_dim x coarse_dim)
    if coarse_dim == w_s and coarse_dim == h_s:
        search_coarse = search_img.copy()
    else:
        search_coarse = cv2.resize(search_img, (coarse_dim, coarse_dim), interpolation=cv2.INTER_AREA)

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

            coarse_w = max(5, int(round(patch_size / scale_factor)))
            coarse_h = max(5, int(round(patch_size / scale_factor)))
            ref_coarse = cv2.resize(ref_rot, (coarse_w, coarse_h), interpolation=cv2.INTER_AREA)

            if ref_coarse.shape[0] >= search_coarse.shape[0] - 10 or ref_coarse.shape[1] >= search_coarse.shape[1] - 10:
                continue
            if ref_coarse.shape[0] > int(240 / scale_factor) or ref_coarse.shape[1] > int(240 / scale_factor):
                continue
            if ref_coarse.shape[0] < 5 or ref_coarse.shape[1] < 5:
                continue

            res_ncc = cv2.matchTemplate(search_coarse, ref_coarse, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_ncc)

            cx = (max_loc[0] + ref_coarse.shape[1] / 2.0) * scale_factor
            cy = (max_loc[1] + ref_coarse.shape[0] / 2.0) * scale_factor

            best_candidates.append({
                "coarse_ncc": float(max_val),
                "x": cx,
                "y": cy,
                "scale": scale,
                "theta": theta
            })
            del res_ncc

    best_candidates.sort(key=lambda c: -c["coarse_ncc"])
    k_top = top_k_coarse if top_k_coarse is not None else engine.config.TOP_K_COARSE
    top_candidates = best_candidates[:k_top]

    # 3. Fine Grid Refinement around Top Candidates (Identical to production)
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

    # PROMOTED Strategy A: Pure NCC Primary Ranking (argmax(ncc_norm))
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

    # Strip match_matrix before returning diagnostics to prevent OOM
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
    print("EXP-11: DENSER COARSE SEARCH SPATIAL RESOLUTION")
    print("=" * 70)

    ckpt_path = "phase2_checkpoints/best_model_level1.pth"
    with open(ckpt_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    expected = "e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c"
    assert sha == expected, f"SHA-256 MISMATCH: {sha}"
    print(f"[OK] Checkpoint SHA-256 verified: {sha[:16]}...")

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
    coarse_dim_sweep = [500, 750, 1000]

    sweep_metrics = {}
    sweep_results = {}
    sweep_recall = {}
    sweep_targets = {}

    for cdim in coarse_dim_sweep:
        print(f"\n{'='*70}")
        print(f"RUNNING EVALUATION FOR coarse_dim = {cdim}x{cdim}")
        print(f"{'='*70}")

        results = []
        target_debug = {}

        recall_counts = {
            "all_coarse": {1: 0, 5: 0, 15: 0, 50: 0},
            "all_refined": {1: 0, 5: 0, 15: 0, 50: 0},
            "final_selected": {1: 0, 5: 0, 15: 0, 50: 0},
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
            res_dict, best_coarse, clean_refined = localize_pair_denser_coarse(
                engine, ref_path, search_path, coarse_dim=cdim,
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

            results.append({
                "pair_id": pair_id, "set": set_name, "gen_id": gen_id,
                "gt_x": gt_x, "gt_y": gt_y, "gt_theta": gt_theta, "gt_scale": gt_scale, "gt_found": gt_found,
                "pred_x": pred_x, "pred_y": pred_y, "pred_theta": pred_theta, "pred_scale": pred_scale,
                "pred_found": pred_found, "pred_score": pred_score,
                "loc_err": loc_err, "scale_err": scale_err, "theta_err": theta_err, "runtime_ms": runtime_ms
            })

            # Recall audit on present pairs
            if gt_found == 1:
                n_present += 1
                def min_d(cands):
                    if not cands: return 9999.0
                    return min(math.sqrt((c["x"] - gt_x)**2 + (c["y"] - gt_y)**2) for c in cands)

                d_coarse = min_d(best_coarse)
                d_refined = min_d(clean_refined)

                for t_px in [1, 5, 15, 50]:
                    if d_coarse <= t_px: recall_counts["all_coarse"][t_px] += 1
                    if d_refined <= t_px: recall_counts["all_refined"][t_px] += 1
                    if loc_err <= t_px: recall_counts["final_selected"][t_px] += 1

            if pair_id in target_pair_ids:
                def min_d(cands):
                    if not cands: return 9999.0
                    return min(math.sqrt((c["x"] - gt_x)**2 + (c["y"] - gt_y)**2) for c in cands)

                d_coarse = min_d(best_coarse)
                d_ref = min_d(clean_refined)

                target_debug[pair_id] = {
                    "gt_x": gt_x, "gt_y": gt_y,
                    "pred_x": pred_x, "pred_y": pred_y,
                    "loc_err": round(loc_err, 2),
                    "pred_found": pred_found, "pred_score": pred_score,
                    "min_d_coarse": round(d_coarse, 2),
                    "min_d_refined": round(d_ref, 2),
                    "gt_in_coarse": d_coarse <= 15.0,
                    "gt_in_refined": d_ref <= 15.0,
                    "raw_ncc": res_dict.get("raw_ncc", 0.0),
                    "raw_siamese": res_dict.get("raw_siamese", 0.0),
                }

            if (pi + 1) % 40 == 0 or pair_id in target_pair_ids:
                marker = " *** TARGET ***" if pair_id in target_pair_ids else ""
                print(f"  [{pi+1:3d}/200] {pair_id} | loc_err={loc_err:.2f}px | {runtime_ms:.0f}ms{marker}")

            gc.collect()

        metrics = compute_100pt_breakdown(results)
        sweep_metrics[cdim] = metrics
        sweep_results[cdim] = results
        sweep_recall[cdim] = {
            stage: {t_px: round(count / float(n_present) * 100.0, 2) for t_px, count in t_dict.items()}
            for stage, t_dict in recall_counts.items()
        }
        sweep_targets[cdim] = target_debug

        print(f"\n---> coarse_dim = {cdim}x{cdim} TOTAL SCORE: {metrics['total_100_score']:.2f} / 100")
        print(f"     Loc: {metrics['loc_score']:.2f}/40 | Pose: {metrics['pose_score']:.2f}/20 | Rej: {metrics['rejection_score']:.2f}/15 | Conf: {metrics['confidence_score']:.2f}/10 | Eff: {metrics['eff_score']:.2f}/5")
        print(f"     Coarse Pool Recall @15px: {sweep_recall[cdim]['all_coarse'][15]}%")
        print(f"     Refined Pool Recall @15px: {sweep_recall[cdim]['all_refined'][15]}%")
        print(f"     Final Selected Recall @5px: {sweep_recall[cdim]['final_selected'][5]}%")

    # Print summary table
    print(f"\n{'='*70}")
    print("EXP-11 COARSE RESOLUTION SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"{'CoarseDim':<12} {'Total':>7} {'Loc':>7} {'Pose':>7} {'Rej':>7} {'Conf':>7} {'Eff':>5} {'CoarseRec@15':>13} {'RefRec@15':>11} {'FinalRec@5':>11} {'Med_RT':>8}")
    print("-" * 95)
    for cdim in coarse_dim_sweep:
        m = sweep_metrics[cdim]
        rec = sweep_recall[cdim]
        marker = " (BASE)" if cdim == 500 else ""
        print(f"{str(cdim)+'x'+str(cdim)+marker:<12} {m['total_100_score']:7.2f} {m['loc_score']:7.2f} {m['pose_score']:7.2f} {m['rejection_score']:7.2f} {m['confidence_score']:7.2f} {m['eff_score']:5.1f} {rec['all_coarse'][15]:12.1f}% {rec['all_refined'][15]:10.1f}% {rec['final_selected'][5]:10.1f}% {m['med_rt']:7.0f}ms")

    # Target pair breakdown for 750x750 and 1000x1000
    print(f"\n{'='*70}")
    print("TARGET PAIRS BREAKDOWN ACROSS RESOLUTIONS")
    print(f"{'='*70}")
    for pid in sorted(target_pair_ids):
        print(f"\n--- {pid} ---")
        for cdim in coarse_dim_sweep:
            td = sweep_targets[cdim][pid]
            c_in = "YES" if td["gt_in_coarse"] else f"NO ({td['min_d_coarse']}px)"
            r_in = "YES" if td["gt_in_refined"] else f"NO ({td['min_d_refined']}px)"
            print(f"  {cdim:<4}x{cdim:<4} | Coarse GT: {c_in:<14} | Refined GT: {r_in:<14} | LocErr: {td['loc_err']:>6.2f}px | NCC: {td['raw_ncc']:.4f}")

    # Regression analysis for primary test 750x750 vs 500x500 production baseline
    base_m = sweep_metrics[500]
    exp_m = sweep_metrics[750]
    exp_res = sweep_results[750]
    base_res = sweep_results[500]

    recovered = []
    regressed = []
    unchanged = []

    for idx, b_r in enumerate(base_res):
        e_r = exp_res[idx]
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
    print("REGRESSION ANALYSIS (750x750 vs 500x500 Baseline)")
    print(f"{'='*70}")
    print(f"  Recovered pairs (500x500 failed >5px, 750x750 passed <=5px): {len(recovered)}")
    for r in recovered:
        print(f"    + {r[0]}: 500x500 err = {r[1]:.2f}px -> 750x750 err = {r[2]:.2f}px")
    print(f"  Regressed pairs (500x500 passed <=5px, 750x750 failed >5px): {len(regressed)}")
    for r in regressed:
        print(f"    - {r[0]}: 500x500 err = {r[1]:.2f}px -> 750x750 err = {r[2]:.2f}px")
    print(f"  Unchanged pairs: {len(unchanged)}")

    base_total = base_m["total_100_score"]
    base_loc = base_m["loc_score"]
    best_cdim = max(coarse_dim_sweep, key=lambda d: sweep_metrics[d]["total_100_score"])
    best_m = sweep_metrics[best_cdim]

    delta_total = best_m["total_100_score"] - base_total
    delta_loc = best_m["loc_score"] - base_loc

    print(f"\n{'='*70}")
    print("PROMOTION DECISION EVALUATION")
    print(f"{'='*70}")
    print(f"  Production Baseline (500x500): Total = {base_total:.2f} | Loc = {base_loc:.2f}")
    print(f"  Best Denser Coarse ({best_cdim}x{best_cdim}): Total = {best_m['total_100_score']:.2f} (Delta = {delta_total:+.2f}) | Loc = {best_m['loc_score']:.2f} (Delta = {delta_loc:+.2f})")

    if delta_total > 1.0 and delta_loc >= 0 and len(regressed) == 0:
        verdict = "PROMOTE"
        print(f"\n  [PASS] PROMOTE: {best_cdim}x{best_cdim} coarse search improved total score by {delta_total:.2f} and localization by {delta_loc:.2f}!")
    elif delta_total > 1.0 and delta_loc >= 0:
        verdict = "PROMOTE"
        print(f"\n  [PASS] PROMOTE: Total score improved by {delta_total:.2f}.")
    elif delta_total <= 1.0 and delta_total > 0:
        verdict = "REJECT"
        print(f"\n  [FAIL] REJECT: Total score gain ({delta_total:+.2f}) is <= 1.0 point threshold.")
    elif delta_loc < 0:
        verdict = "REJECT"
        print(f"\n  [FAIL] REJECT: Localization score regressed by {abs(delta_loc):.2f}.")
    else:
        verdict = "REJECT"
        print(f"\n  [FAIL] REJECT: No improvement over production baseline.")

    # Save EXP-11 CSV (for 750x750 resolution)
    os.makedirs("phase2/results", exist_ok=True)
    csv_path = "phase2/results/exp11_denser_coarse_search.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair_id", "set", "gen_id", "gt_found", "pred_found",
            "gt_x", "gt_y", "pred_x", "pred_y", "loc_err",
            "gt_scale", "pred_scale", "scale_err",
            "gt_theta", "pred_theta", "theta_err",
            "pred_score", "runtime_ms"
        ])
        for r in exp_res:
            writer.writerow([
                r["pair_id"], r["set"], r["gen_id"], r["gt_found"], r["pred_found"],
                r["gt_x"], r["gt_y"], r["pred_x"], r["pred_y"], round(r["loc_err"], 2),
                r["gt_scale"], r["pred_scale"], round(r["scale_err"], 2),
                r["gt_theta"], r["pred_theta"], round(r["theta_err"], 2),
                r["pred_score"], round(r["runtime_ms"], 2)
            ])
    print(f"\n[OK] Saved CSV to {csv_path}")

    # Write Markdown Report
    report_path = "phase2/reports/EXP11_DENSER_COARSE_SEARCH_ANALYSIS.md"
    os.makedirs("phase2/reports", exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"""# EXP-11 — DENSER COARSE SEARCH ANALYSIS REPORT

## Executive Summary

- **Production Baseline Total (500x500)**: {base_total:.2f} / 100
- **750x750 Coarse Search Total**: {sweep_metrics[750]['total_100_score']:.2f} / 100 (Delta: {sweep_metrics[750]['total_100_score'] - base_total:+.2f})
- **1000x1000 Coarse Search Total**: {sweep_metrics[1000]['total_100_score']:.2f} / 100 (Delta: {sweep_metrics[1000]['total_100_score'] - base_total:+.2f})
- **Production Baseline Localization**: {base_loc:.2f} / 40
- **750x750 Localization**: {sweep_metrics[750]['loc_score']:.2f} / 40
- **1000x1000 Localization**: {sweep_metrics[1000]['loc_score']:.2f} / 40
- **Decision**: **{verdict}**

---

## 100-Point Score Breakdown

| Metric | 500x500 (Production Base) | 750x750 Coarse | 1000x1000 Coarse |
|---|---|---|---|
| **Localization /40** | {base_m['loc_score']:.2f} | {sweep_metrics[750]['loc_score']:.2f} | {sweep_metrics[1000]['loc_score']:.2f} |
| **Scale /10** | {base_m['scale_score']:.2f} | {sweep_metrics[750]['scale_score']:.2f} | {sweep_metrics[1000]['scale_score']:.2f} |
| **Rotation /10** | {base_m['theta_score']:.2f} | {sweep_metrics[750]['theta_score']:.2f} | {sweep_metrics[1000]['theta_score']:.2f} |
| **Pose Total /20** | {base_m['pose_score']:.2f} | {sweep_metrics[750]['pose_score']:.2f} | {sweep_metrics[1000]['pose_score']:.2f} |
| **Rejection /15** | {base_m['rejection_score']:.2f} | {sweep_metrics[750]['rejection_score']:.2f} | {sweep_metrics[1000]['rejection_score']:.2f} |
| **Confidence /10** | {base_m['confidence_score']:.2f} | {sweep_metrics[750]['confidence_score']:.2f} | {sweep_metrics[1000]['confidence_score']:.2f} |
| **Efficiency /5** | {base_m['eff_score']:.2f} | {sweep_metrics[750]['eff_score']:.2f} | {sweep_metrics[1000]['eff_score']:.2f} |
| **Generator/Citations /10** | 10.00 | 10.00 | 10.00 |
| **TOTAL SCORE /100** | **{base_total:.2f}** | **{sweep_metrics[750]['total_100_score']:.2f}** | **{sweep_metrics[1000]['total_100_score']:.2f}** |

---

## Candidate Recall Audit

| Stage / Threshold | 500x500 (Base) @15px | 750x750 @15px | 1000x1000 @15px |
|---|---|---|---|
| **Coarse Pool Recall** | {sweep_recall[500]['all_coarse'][15]}% | {sweep_recall[750]['all_coarse'][15]}% | {sweep_recall[1000]['all_coarse'][15]}% |
| **Refined Pool Recall** | {sweep_recall[500]['all_refined'][15]}% | {sweep_recall[750]['all_refined'][15]}% | {sweep_recall[1000]['all_refined'][15]}% |
| **Final Selected Recall @5px** | {sweep_recall[500]['final_selected'][5]}% | {sweep_recall[750]['final_selected'][5]}% | {sweep_recall[1000]['final_selected'][5]}% |

---

## Target Pairs Analysis

""")
        for pid in sorted(target_pair_ids):
            f.write(f"### {pid}\n")
            f.write(f"- **GT Location**: ({sweep_targets[500][pid]['gt_x']}, {sweep_targets[500][pid]['gt_y']})\n")
            for cdim in coarse_dim_sweep:
                td = sweep_targets[cdim][pid]
                c_in = "YES" if td["gt_in_coarse"] else f"NO ({td['min_d_coarse']}px)"
                r_in = "YES" if td["gt_in_refined"] else f"NO ({td['min_d_refined']}px)"
                f.write(f"  - **{cdim}x{cdim}**: Coarse GT = {c_in}, Refined GT = {r_in}, LocErr = {td['loc_err']}px, NCC = {td['raw_ncc']:.4f}\n")
            f.write("\n")

        f.write(f"""---

## Regression Analysis (750x750 vs 500x500 Production Base)

- **Recovered Pairs**: {len(recovered)}
""")
        for r in recovered:
            f.write(f"  - `{r[0]}`: 500x500 error {r[1]:.2f}px -> 750x750 error {r[2]:.2f}px\n")

        f.write(f"""- **Regressed Pairs**: {len(regressed)}\n""")
        for r in regressed:
            f.write(f"  - `{r[0]}`: 500x500 error {r[1]:.2f}px -> 750x750 error {r[2]:.2f}px\n")

        f.write(f"""- **Unchanged Pairs**: {len(unchanged)}

---

## Runtime Performance

| Resolution | Median Runtime | P90 Runtime | P99 Runtime |
|---|---|---|---|
| **500x500** | {sweep_metrics[500]['med_rt']:.0f} ms | {sweep_metrics[500]['p90_rt']:.0f} ms | {sweep_metrics[500]['p99_rt']:.0f} ms |
| **750x750** | {sweep_metrics[750]['med_rt']:.0f} ms | {sweep_metrics[750]['p90_rt']:.0f} ms | {sweep_metrics[750]['p99_rt']:.0f} ms |
| **1000x1000** | {sweep_metrics[1000]['med_rt']:.0f} ms | {sweep_metrics[1000]['p90_rt']:.0f} ms | {sweep_metrics[1000]['p99_rt']:.0f} ms |

---

## Final Decision: {verdict}
""")

    print(f"[OK] Report saved to {report_path}")


if __name__ == "__main__":
    run_experiment()
