#!/usr/bin/env python3
"""
EXP-19 — Adaptive High-Resolution Rescue for Degraded Set-B
============================================================

STRICT SINGLE-CHANGE EXPERIMENT — ISOLATED EXPERIMENT ONLY

Hypothesis:
    In degraded SEM search images (Set B), 500x500 downsampling wipes out low-contrast landmark features.
    When coarse search shows evidence of an ambiguous/unreliable match (low Top-1 NCC < 0.65 or small Top-1/Top-2 gap < 0.03),
    executing a localized high-resolution rescue search on full-resolution (1000x1000) search crops will recover missing GT candidates into the refinement pool.

Baseline to Beat: 72.80 / 100.0 (EXP-13 + Sunday Checkpoint).
Promotion Threshold: Total Score >= 73.80 / 100.0 (+1.0 point gain).
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from phase2.phase2_inference import Phase2InferenceEngine, load_grayscale_image, compute_periodicity_count, fit_parabola_subpixel
from phase2.phase2_config import Phase2Config

import torch
import torchvision.transforms.functional as TF

class Phase2InferenceEngineEXP19(Phase2InferenceEngine):
    def localize_pair_exp19(
        self,
        ref_input,
        search_input,
        ncc_weight=None,
        rejection_thresh=None,
        scale_step=0.25,
        theta_step=1.0,
        top_k_coarse=10,
        return_diagnostics=False
    ):
        """
        Executes EXP-19 Adaptive High-Res Rescue Search:
        - Primary Path: Standard 500x500 coarse search -> Top-K candidates.
        - Ambiguity Trigger: If Top-1 coarse NCC < 0.65 or (Top-1 - Top-2 gap) < 0.03.
        - Triggered Action: Perform localized high-res 1000x1000 template matching around top candidate regions.
        - All refinement & EXP-13 Periodicity Penalization remain 100% identical.
        """
        ref_img = load_grayscale_image(ref_input)
        search_img = load_grayscale_image(search_input)

        w_alpha = ncc_weight if ncc_weight is not None else self.config.NCC_WEIGHT
        tau = rejection_thresh if rejection_thresh is not None else self.config.REJECTION_THRESHOLD

        h_s, w_s = search_img.shape

        if ref_img.shape != (100, 100):
            ref_template = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
        else:
            ref_template = ref_img.copy()

        ref_emb = self.extract_siamese_embedding(ref_template)

        # 1. Primary 500x500 Coarse Search
        search_coarse = cv2.resize(search_img, (500, 500), interpolation=cv2.INTER_AREA)
        coarse_candidates = []

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
                if ref_coarse.shape[0] > 120 or ref_coarse.shape[1] > 120 or ref_coarse.shape[0] < 5 or ref_coarse.shape[1] < 5:
                    continue

                res_coarse = cv2.matchTemplate(search_coarse, ref_coarse, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res_coarse)

                cx = (max_loc[0] + ref_coarse.shape[1] / 2.0) * 2.0
                cy = (max_loc[1] + ref_coarse.shape[0] / 2.0) * 2.0

                coarse_candidates.append({
                    "coarse_ncc": float(max_val),
                    "x": cx, "y": cy, "scale": scale, "theta": theta
                })

        coarse_candidates.sort(key=lambda c: -c["coarse_ncc"])

        top1_ncc = coarse_candidates[0]["coarse_ncc"] if len(coarse_candidates) > 0 else 0.0
        top2_ncc = coarse_candidates[1]["coarse_ncc"] if len(coarse_candidates) > 1 else 0.0
        top_gap = top1_ncc - top2_ncc

        # Objective Ambiguity Trigger
        is_triggered = (top1_ncc < 0.65) or (top_gap < 0.03)

        top_candidates = coarse_candidates[:top_k_coarse]

        # 2. Localized High-Resolution Rescue (If Triggered)
        if is_triggered:
            rescue_candidates = []
            for cand in top_candidates[:5]:
                cx, cy = cand["x"], cand["y"]
                c_scale = cand["scale"]
                c_theta = cand["theta"]

                win = 220
                rx0, rx1 = max(0, int(cx - win)), min(w_s, int(cx + win))
                ry0, ry1 = max(0, int(cy - win)), min(h_s, int(cy + win))

                search_crop = search_img[ry0:ry1, rx0:rx1]
                p_size = int(round(1000.0 / c_scale))

                if search_crop.shape[0] <= p_size + 10 or search_crop.shape[1] <= p_size + 10:
                    continue

                ref_s = cv2.resize(ref_template, (p_size, p_size), interpolation=cv2.INTER_LINEAR)
                if abs(c_theta) > 0.05:
                    M_r = cv2.getRotationMatrix2D((p_size / 2.0, p_size / 2.0), c_theta, 1.0)
                    ref_r = cv2.warpAffine(ref_s, M_r, (p_size, p_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                else:
                    ref_r = ref_s

                res_high = cv2.matchTemplate(search_crop, ref_r, cv2.TM_CCOEFF_NORMED)
                _, max_v, _, max_l = cv2.minMaxLoc(res_high)

                hx = rx0 + max_l[0] + ref_r.shape[1] / 2.0
                hy = ry0 + max_l[1] + ref_r.shape[0] / 2.0

                rescue_candidates.append({
                    "coarse_ncc": float(max_v),
                    "x": hx, "y": hy, "scale": c_scale, "theta": c_theta
                })

            # Merge rescue candidates into top_candidates
            merged = top_candidates + rescue_candidates
            merged.sort(key=lambda c: -c["coarse_ncc"])
            dedup = []
            for c in merged:
                is_dup = False
                for d in dedup:
                    if math.sqrt((c["x"] - d["x"])**2 + (c["y"] - d["y"])**2) < 15.0 and abs(c["scale"] - d["scale"]) <= 0.5:
                        is_dup = True
                        break
                if not is_dup:
                    dedup.append(c)
                if len(dedup) >= top_k_coarse:
                    break
            top_candidates = dedup[:top_k_coarse]

        # 3. Fine Refinement (Identical to production)
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

            # EXP-13 Periodicity Penalty
            p_count = compute_periodicity_count(meta.get("match_matrix", None))
            p_penalty = 0.05 * (p_count - 1)
            adj_ncc = n_norm - p_penalty

            refined_results.append({
                "x": meta["x"], "y": meta["y"], "scale": meta["scale"], "theta": meta["theta"],
                "fused_score": f_score, "adjusted_score": adj_score,
                "ncc_norm": n_norm, "adjusted_ncc": adj_ncc, "siamese_sim": s_sim,
                "periodicity_count": p_count, "match_matrix": meta["match_matrix"]
            })

        # EXP-13 Primary Ranking
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

        if final_fused >= tau:
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
            "raw_siamese": float(round(best_cand["siamese_sim"], 4)),
            "is_triggered": is_triggered
        }
        if return_diagnostics:
            return res_dict, top_candidates, refined_results
        return res_dict


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


def main():
    print("=" * 70)
    print("EXP-19: ADAPTIVE HIGH-RESOLUTION RESCUE FOR DEGRADED SET-B")
    print("=" * 70)

    ckpt_path = "checkpoints_phase2_v2_sunday/best_model_phase2.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = "phase2_checkpoints/best_model_level1.pth"

    with open(ckpt_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    print(f"[OK] Production Checkpoint Loaded: {ckpt_path} (SHA-256: {sha[:16]}...)")

    engine_base = Phase2InferenceEngine(checkpoint_path=ckpt_path, device="cpu")
    engine_exp19 = Phase2InferenceEngineEXP19(checkpoint_path=ckpt_path, device="cpu")

    data_dir = "local_phase2_60gen_200_pairs"
    manifest_path = os.path.join(data_dir, "phase2_60generator_manifest.csv")
    assert os.path.exists(manifest_path), f"Manifest not found: {manifest_path}"

    pairs = []
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(row)
    print(f"[OK] Loaded {len(pairs)} pairs from manifest")

    target_pair_ids = {"pair_006", "pair_066", "pair_116", "pair_160", "pair_186"}
    cat_a_pairs = {"pair_080", "pair_116", "pair_120", "pair_143", "pair_144", "pair_153", "pair_154", "pair_169"}

    results_base = []
    results_exp19 = []

    target_details_base = {}
    target_details_exp19 = {}

    triggered_count = 0
    cat_a_recovered_count = 0
    cat_b_recovered_count = 0

    print(f"\nRunning 200-Pair Evaluation for BASE (Production 72.80) and EXP-19...")

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

        # 1. BASE Run
        t0 = time.time()
        res_b = engine_base.localize_pair(
            ref_path, search_path,
            ncc_weight=0.5, rejection_thresh=0.42, scale_step=0.25, theta_step=1.0
        )
        t1 = time.time()
        rt_b = (t1 - t0) * 1000.0

        if gt_found == 1 and res_b["found"] == 1:
            err_b = math.sqrt((res_b["x"] - gt_x)**2 + (res_b["y"] - gt_y)**2)
            serr_b = abs(res_b["scale"] - gt_scale)
            terr_b = abs(res_b["theta"] - gt_theta)
        elif gt_found == 0 and res_b["found"] == 0:
            err_b = serr_b = terr_b = 0.0
        else:
            err_b = serr_b = terr_b = 999.0

        results_base.append({
            "pair_id": pair_id, "set": set_name, "gen_id": gen_id,
            "gt_x": gt_x, "gt_y": gt_y, "gt_theta": gt_theta, "gt_scale": gt_scale, "gt_found": gt_found,
            "pred_x": res_b["x"], "pred_y": res_b["y"], "pred_theta": res_b["theta"], "pred_scale": res_b["scale"],
            "pred_found": res_b["found"], "pred_score": res_b["score"],
            "loc_err": err_b, "scale_err": serr_b, "theta_err": terr_b,
            "runtime_ms": rt_b
        })

        # 2. EXP-19 Run
        t0 = time.time()
        res_e = engine_exp19.localize_pair_exp19(
            ref_path, search_path,
            ncc_weight=0.5, rejection_thresh=0.42, scale_step=0.25, theta_step=1.0,
            top_k_coarse=10
        )
        t1 = time.time()
        rt_e = (t1 - t0) * 1000.0

        if res_e.get("is_triggered", False):
            triggered_count += 1

        if gt_found == 1 and res_e["found"] == 1:
            err_e = math.sqrt((res_e["x"] - gt_x)**2 + (res_e["y"] - gt_y)**2)
            serr_e = abs(res_e["scale"] - gt_scale)
            terr_e = abs(res_e["theta"] - gt_theta)
        elif gt_found == 0 and res_e["found"] == 0:
            err_e = serr_e = terr_e = 0.0
        else:
            err_e = serr_e = terr_e = 999.0

        results_exp19.append({
            "pair_id": pair_id, "set": set_name, "gen_id": gen_id,
            "gt_x": gt_x, "gt_y": gt_y, "gt_theta": gt_theta, "gt_scale": gt_scale, "gt_found": gt_found,
            "pred_x": res_e["x"], "pred_y": res_e["y"], "pred_theta": res_e["theta"], "pred_scale": res_e["scale"],
            "pred_found": res_e["found"], "pred_score": res_e["score"],
            "loc_err": err_e, "scale_err": serr_e, "theta_err": terr_e,
            "is_triggered": res_e.get("is_triggered", False),
            "runtime_ms": rt_e
        })

        if pair_id in target_pair_ids:
            target_details_base[pair_id] = {"err": round(err_b, 2), "found": res_b["found"], "score": res_b["score"]}
            target_details_exp19[pair_id] = {"err": round(err_e, 2), "found": res_e["found"], "score": res_e["score"]}

        if (pi + 1) % 40 == 0 or pair_id in target_pair_ids:
            trig_str = " [RESCUE TRIGGERED]" if res_e.get("is_triggered", False) else ""
            marker = " *** TARGET ***" if pair_id in target_pair_ids else ""
            print(f"  [{pi+1:3d}/200] {pair_id} | base_err={err_b:.2f}px | exp19_err={err_e:.2f}px | rt={rt_e:.0f}ms{trig_str}{marker}")

        gc.collect()

    m_b = compute_100pt_breakdown(results_base)
    m_e = compute_100pt_breakdown(results_exp19)

    recovered_pairs = []
    regressed_pairs = []
    unchanged_pairs = []

    for idx, r_b in enumerate(results_base):
        r_e = results_exp19[idx]
        pid = r_b["pair_id"]
        eb = r_b["loc_err"]
        ee = r_e["loc_err"]

        if eb > 5.0 and ee <= 5.0:
            recovered_pairs.append((pid, r_b["set"], eb, ee))
            if pid in cat_a_pairs:
                cat_a_recovered_count += 1
            if r_b["set"] == "Set B":
                cat_b_recovered_count += 1
        elif eb <= 5.0 and ee > 5.0:
            regressed_pairs.append((pid, r_b["set"], eb, ee))
        else:
            unchanged_pairs.append(pid)

    delta_total = m_e["total_100_score"] - m_b["total_100_score"]
    delta_loc = m_e["loc_score"] - m_b["loc_score"]

    print(f"\n{'='*70}")
    print("EXP-19 OFFICIAL 100-POINT SCORE COMPARISON")
    print(f"{'='*70}")
    print(f"Baseline (Production 72.80): Total = {m_b['total_100_score']:.2f} / 100 (Loc {m_b['loc_score']:.2f}, Pose {m_b['pose_score']:.2f}, Rej {m_b['rejection_score']:.2f}, Conf {m_b['confidence_score']:.2f}, RT {m_b['med_rt']:.0f}ms)")
    print(f"EXP-19 (Adaptive High-Res)  : Total = {m_e['total_100_score']:.2f} / 100 (Loc {m_e['loc_score']:.2f}, Pose {m_e['pose_score']:.2f}, Rej {m_e['rejection_score']:.2f}, Conf {m_e['confidence_score']:.2f}, RT {m_e['med_rt']:.0f}ms)")
    print(f"DELTA: Total = {delta_total:+.2f} points | Localization = {delta_loc:+.2f} points")

    print(f"\n{'='*70}")
    print("RESCUE TRIGGER & RECOVERY AUDIT")
    print(f"{'='*70}")
    print(f"  - Pairs Triggering Adaptive Rescue : {triggered_count} / 200")
    print(f"  - Cat-A Baseline Pairs Recovered   : {cat_a_recovered_count} / 8")
    print(f"  - Set-B Baseline Pairs Recovered    : {cat_b_recovered_count} / 40")
    print(f"  - Total Recovered Pairs             : {len(recovered_pairs)}")
    for r in recovered_pairs:
        print(f"    + {r[0]} ({r[1]}): Base err = {r[2]:.2f}px -> EXP-19 err = {r[3]:.2f}px")

    print(f"\n{'='*70}")
    print("REGRESSION ANALYSIS")
    print(f"{'='*70}")
    print(f"  - Total Regressed Pairs: {len(regressed_pairs)}")
    for r in regressed_pairs:
        print(f"    - {r[0]} ({r[1]}): Base err = {r[2]:.2f}px -> EXP-19 err = {r[3]:.2f}px")

    # Decision evaluation against 72.80 baseline
    if delta_total >= 1.0 and delta_loc >= 0 and len(regressed_pairs) == 0 and m_e['total_100_score'] >= 73.80:
        verdict = "PROMOTE"
        print(f"\n  [PASS] PROMOTE: Score reached {m_e['total_100_score']:.2f} (>= 73.80) with 0 regressions!")
    else:
        verdict = "REJECT"
        print(f"\n  [FAIL] REJECT: Score gain ({delta_total:+.2f}) failed promotion threshold (must reach >= 73.80). Production remains 72.80.")

    # Save CSV
    os.makedirs("phase2/results", exist_ok=True)
    csv_path = "phase2/results/exp19_adaptive_highres_rescue.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair_id", "set", "gen_id", "gt_found", "base_pred_found", "exp19_pred_found",
            "gt_x", "gt_y", "base_pred_x", "base_pred_y", "exp19_pred_x", "exp19_pred_y",
            "base_loc_err", "exp19_loc_err", "is_triggered", "base_score", "exp19_score", "runtime_ms"
        ])
        for idx, r_e in enumerate(results_exp19):
            r_b = results_base[idx]
            writer.writerow([
                r_e["pair_id"], r_e["set"], r_e["gen_id"], r_e["gt_found"], r_b["pred_found"], r_e["pred_found"],
                r_e["gt_x"], r_e["gt_y"], r_b["pred_x"], r_b["pred_y"], r_e["pred_x"], r_e["pred_y"],
                r_b["loc_err"], r_e["loc_err"], r_e["is_triggered"], r_b["pred_score"], r_e["pred_score"], round(r_e["runtime_ms"], 2)
            ])
    print(f"\n[OK] Saved CSV to {csv_path}")

    # Write Markdown Report
    os.makedirs("phase2/reports", exist_ok=True)
    report_path = "phase2/reports/EXP19_ADAPTIVE_HIGHRES_RESCUE_ANALYSIS.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""# EXP-19 ADAPTIVE HIGH-RESOLUTION RESCUE REPORT

## Executive Summary

- **Production Baseline Score**: **{m_b['total_100_score']:.2f} / 100**
- **EXP-19 Score**: **{m_e['total_100_score']:.2f} / 100**
- **Delta Total Score**: **{delta_total:+.2f}**
- **Production Baseline Localization**: {m_b['loc_score']:.2f} / 40
- **EXP-19 Localization**: **{m_e['loc_score']:.2f} / 40** ({delta_loc:+.2f})
- **Pairs Triggering Rescue**: {triggered_count} / 200
- **Decision**: **{verdict}**

---

## 100-Point Score Breakdown

| Category | Production Baseline (72.80) | EXP-19 | Delta |
|---|---|---|---|
| **Localization /40** | {m_b['loc_score']:.2f} | **{m_e['loc_score']:.2f}** | **{delta_loc:+.2f}** |
| **Scale /10** | {m_b['scale_score']:.2f} | **{m_e['scale_score']:.2f}** | **{m_e['scale_score'] - m_b['scale_score']:+.2f}** |
| **Rotation /10** | {m_b['theta_score']:.2f} | **{m_e['theta_score']:.2f}** | **{m_e['theta_score'] - m_b['theta_score']:+.2f}** |
| **Pose Total /20** | {m_b['pose_score']:.2f} | **{m_e['pose_score']:.2f}** | **{m_e['pose_score'] - m_b['pose_score']:+.2f}** |
| **Rejection /15** | {m_b['rejection_score']:.2f} | **{m_e['rejection_score']:.2f}** | **{m_e['rejection_score'] - m_b['rejection_score']:+.2f}** |
| **Confidence /10** | {m_b['confidence_score']:.2f} | **{m_e['confidence_score']:.2f}** | **{m_e['confidence_score'] - m_b['confidence_score']:+.2f}** |
| **Efficiency /5** | {m_b['eff_score']:.2f} | **{m_e['eff_score']:.2f}** | **0.00** |
| **Generator/Citations /10** | 10.00 | **10.00** | **0.00** |
| **TOTAL SCORE /100** | **{m_b['total_100_score']:.2f}** | **{m_e['total_100_score']:.2f}** | **{delta_total:+.2f}** |

---

## Set-B & Cat-A Candidate Recovery Audit

- **Pairs Triggering High-Res Rescue**: {triggered_count} / 200
- **Cat-A Baseline Coarse-Missing Pairs Recovered**: {cat_a_recovered_count} / 8
- **Set-B Degraded SEM Baseline Pairs Recovered**: {cat_b_recovered_count} / 40
- **Total Recovered Pairs**: {len(recovered_pairs)}
- **Total Regressed Pairs**: {len(regressed_pairs)}

---

## Target Pairs Comparison

""")
        for pid in sorted(target_pair_ids):
            tb = target_details_base[pid]
            te = target_details_exp19[pid]
            f.write(f"### {pid}\n")
            f.write(f"- **Production Baseline**: LocErr = {tb['err']}px, Found = {tb['found']}, Score = {tb['score']}\n")
            f.write(f"- **EXP-19**             : LocErr = {te['err']}px, Found = {te['found']}, Score = {te['score']}\n\n")

        f.write(f"""---

## Runtime Performance

- **Production Baseline Median Runtime**: {m_b['med_rt']:.0f} ms
- **EXP-19 Median Runtime**: **{m_e['med_rt']:.0f} ms** (Delta: {m_e['med_rt'] - m_b['med_rt']:+.0f} ms)
- **EXP-19 P90 Runtime**: {m_e['p90_rt']:.0f} ms
- **EXP-19 P99 Runtime**: {m_e['p99_rt']:.0f} ms

---

## Decision & Technical Conclusion: {verdict}
""")

    print(f"[OK] Report saved to {report_path}")

if __name__ == "__main__":
    main()
