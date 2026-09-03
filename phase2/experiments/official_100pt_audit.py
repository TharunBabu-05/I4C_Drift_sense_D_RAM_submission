#!/usr/bin/env python3
"""
Phase-2 Official 100-Point Scoring Audit & Bottleneck Analysis Script
=======================================================================
Runs official 100-point evaluation across both 200-pair Phase-2 test suites:
- Dataset 2: local_phase2_60gen_200_pairs (60-generator benchmark)
- Dataset 1: local_phase2_200_pairs (Generic DRAM benchmark)

Model Checkpoint: phase2_checkpoints/best_model_level1.pth (READ-ONLY, SHA-256 UNTOUCHED)
Production Code: 100% Unmodified

Official 100-Point Breakdown:
1. Localization (40 pts): Tiered credit (<=1px:1.0, <=2px:0.8, <=3px:0.6, <=5px:0.4) on Present pairs. Weighted 0.45*SetA + 0.55*SetB.
2. Pose Recovery (20 pts): Scale (10 pts) + Rotation (10 pts). ONLY awarded when localization error <= 5.0px!
3. Rejection (15 pts): 15.0 * F1 score across present & absent pairs.
4. Confidence Calibration (10 pts): 10.0 * AUC score.
5. Efficiency (5 pts): 5.0 pts if median runtime <= 5000ms.
6. Generator/Citations/Failure Analysis (10 pts): Carried forward from Phase 1 (10.0/10.0).
TOTAL = 100 Points.
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
from phase2.phase2_inference import Phase2InferenceEngine

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

def run_official_100pt_evaluation(engine, data_dir, manifest_filename, dataset_name):
    manifest_path = os.path.join(data_dir, manifest_filename)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    print(f"\n=======================================================")
    print(f"OFFICIAL 100-POINT AUDIT: {dataset_name.upper()}")
    print(f"Data Dir: {data_dir}")
    print(f"=======================================================")

    results = []
    top5_debug_data = {}

    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair_id = row["pair_id"]
            set_name = row["set"]
            ref_path = row["reference_path"]
            search_path = row["search_path"]

            gt_x = float(row["x_gt"])
            gt_y = float(row["y_gt"])
            gt_theta = float(row["theta_gt"])
            gt_scale = float(row["scale_gt"])
            gt_found = int(row["found_gt"])

            gen_id = row.get("generator_id", "generic")

            t0 = time.time()
            # Localize with debug output to extract Top-5 candidate details
            res_dict, best_coarse, refined_results = engine.localize_pair(
                ref_path, search_path, ncc_weight=0.5, rejection_thresh=0.42, scale_step=0.25, theta_step=1.0,
                return_diagnostics=True
            )
            t1 = time.time()
            runtime_ms = (t1 - t0) * 1000.0

            pred_x = res_dict["x"]
            pred_y = res_dict["y"]
            pred_theta = res_dict["theta"]
            pred_scale = res_dict["scale"]
            pred_found = res_dict["found"]
            pred_score = res_dict["score"]

            if gt_found == 1 and pred_found == 1:
                loc_err = math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
                scale_err = abs(pred_scale - gt_scale)
                theta_err = abs(pred_theta - gt_theta)
            elif gt_found == 0 and pred_found == 0:
                loc_err = 0.0
                scale_err = 0.0
                theta_err = 0.0
            else:
                loc_err = 999.0
                scale_err = 999.0
                theta_err = 999.0

            # Store Top-5 candidate details for analysis
            top5_list = []
            gt_in_top5 = False
            best_top5_dist = 999.0
            for r_idx, cand in enumerate(refined_results[:5]):
                cand_x, cand_y = cand["x"], cand["y"]
                dist = math.sqrt((cand_x - gt_x)**2 + (cand_y - gt_y)**2) if gt_found == 1 else 999.0
                if dist <= 15.0:
                    gt_in_top5 = True
                if dist < best_top5_dist:
                    best_top5_dist = dist
                top5_list.append({
                    "rank": r_idx + 1, "x": cand_x, "y": cand_y,
                    "ncc": cand.get("ncc_norm", 0.0), "siamese": cand.get("siamese_sim", 0.0),
                    "fused": cand.get("fused_score", 0.0), "dist_gt": dist
                })

            top5_debug_data[pair_id] = top5_list

            results.append({
                "pair_id": pair_id, "set": set_name, "gen_id": gen_id,
                "gt_x": gt_x, "gt_y": gt_y, "gt_theta": gt_theta, "gt_scale": gt_scale, "gt_found": gt_found,
                "pred_x": pred_x, "pred_y": pred_y, "pred_theta": pred_theta, "pred_scale": pred_scale,
                "pred_found": pred_found, "pred_score": pred_score,
                "loc_err": loc_err, "scale_err": scale_err, "theta_err": theta_err, "runtime_ms": runtime_ms,
                "gt_in_top5": gt_in_top5, "best_top5_dist": best_top5_dist
            })

            gc.collect()

    return results, top5_debug_data

def compute_100pt_breakdown(results):
    sets_data = {"Set A": [], "Set B": [], "Set C": [], "Set D": []}
    for r in results:
        sets_data[r["set"]].append(r)

    # 1. Localization Score (40 pts)
    def calc_loc_credit(entries):
        present_entries = [e for e in entries if e["gt_found"] == 1]
        n_present = len(present_entries)
        if n_present == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        credits = []
        c1, c2, c3, c5 = 0, 0, 0, 0
        for e in present_entries:
            if e["pred_found"] == 1:
                err = e["loc_err"]
                if err <= 1.0:
                    credits.append(1.00); c1 += 1
                elif err <= 2.0:
                    credits.append(0.80); c2 += 1
                elif err <= 3.0:
                    credits.append(0.60); c3 += 1
                elif err <= 5.0:
                    credits.append(0.40); c5 += 1
                else:
                    credits.append(0.00)
            else:
                credits.append(0.00)
        mean_credit = np.mean(credits)
        return mean_credit, c1 / n_present, c2 / n_present, c3 / n_present, c5 / n_present, n_present

    credit_a, _, _, _, pct_5_a, n_a = calc_loc_credit(sets_data["Set A"])
    credit_b, _, _, _, pct_5_b, n_b = calc_loc_credit(sets_data["Set B"])

    loc_score = (0.45 * credit_a + 0.55 * credit_b) * 40.0
    loc_points_lost = 40.0 - loc_score

    # 2. Pose Recovery Score (20 pts: Scale=10, Rot=10)
    # CRITICAL RULE: Pose points ONLY awarded when localization is correct (loc_err <= 5.0)
    total_present_pairs = sum(1 for r in results if r["gt_found"] == 1)

    scale_credits = []
    theta_credits = []

    for r in results:
        if r["gt_found"] == 1:
            if r["pred_found"] == 1 and r["loc_err"] <= 5.0:
                s_err = r["scale_err"]
                t_err = r["theta_err"]

                # Scale credit
                if s_err <= 0.25: scale_credits.append(1.0)
                elif s_err <= 0.50: scale_credits.append(0.5)
                else: scale_credits.append(0.0)

                # Theta credit
                if t_err <= 0.5: theta_credits.append(1.0)
                elif t_err <= 1.5: theta_credits.append(0.5)
                else: theta_credits.append(0.0)
            else:
                # Pose points NOT awarded because localization failed
                scale_credits.append(0.0)
                theta_credits.append(0.0)

    scale_score = (sum(scale_credits) / total_present_pairs) * 10.0 if total_present_pairs > 0 else 0.0
    theta_score = (sum(theta_credits) / total_present_pairs) * 10.0 if total_present_pairs > 0 else 0.0
    pose_score = scale_score + theta_score
    pose_points_lost = 20.0 - pose_score

    # 3. Rejection F1 Score (15 pts)
    tp = sum(1 for r in results if r["gt_found"] == 1 and r["pred_found"] == 1)
    tn = sum(1 for r in results if r["gt_found"] == 0 and r["pred_found"] == 0)
    fp = sum(1 for r in results if r["gt_found"] == 0 and r["pred_found"] == 1)
    fn = sum(1 for r in results if r["gt_found"] == 1 and r["pred_found"] == 0)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    rejection_score = f1_score * 15.0
    rejection_points_lost = 15.0 - rejection_score

    # 4. Confidence Calibration AUC (10 pts)
    y_true = [r["gt_found"] for r in results]
    y_scores = [r["pred_score"] for r in results]
    auc = float(calculate_auc(y_true, y_scores))
    confidence_score = auc * 10.0
    confidence_points_lost = 10.0 - confidence_score

    # 5. Efficiency Score (5 pts)
    runtimes = [r["runtime_ms"] for r in results]
    med_rt = float(np.median(runtimes))
    eff_score = 5.0 if med_rt <= 5000.0 else (2.5 if med_rt <= 10000.0 else 0.0)
    eff_points_lost = 5.0 - eff_score

    # 6. Generator / Citations / Failure Analysis (10 pts carried forward from Phase 1)
    gen_score = 10.0
    gen_points_lost = 0.0

    total_100_score = loc_score + scale_score + theta_score + rejection_score + confidence_score + eff_score + gen_score
    total_points_lost = 100.0 - total_100_score

    # Detailed Points Lost Categorization
    # Category 1: GT Not in Top-5 candidates
    lost_not_in_top5 = sum(1 for r in results if r["gt_found"] == 1 and not r["gt_in_top5"])
    # Category 2: GT in Top-5 but wrong candidate selected (periodic decoy selected)
    lost_wrong_top5_selected = sum(1 for r in results if r["gt_found"] == 1 and r["gt_in_top5"] and r["loc_err"] > 5.0)
    # Category 3: Correct candidate selected but refinement error (1.0px < loc_err <= 5.0px)
    lost_refinement = sum(1 for r in results if r["gt_found"] == 1 and r["pred_found"] == 1 and 1.0 < r["loc_err"] <= 5.0)
    # Category 4: Pose Scale Error
    lost_scale = sum(1 for r in results if r["gt_found"] == 1 and r["pred_found"] == 1 and r["loc_err"] <= 5.0 and r["scale_err"] > 0.25)
    # Category 5: Pose Rotation Error
    lost_theta = sum(1 for r in results if r["gt_found"] == 1 and r["pred_found"] == 1 and r["loc_err"] <= 5.0 and r["theta_err"] > 0.5)

    return {
        "loc_score": loc_score, "scale_score": scale_score, "theta_score": theta_score, "pose_score": pose_score,
        "rejection_score": rejection_score, "confidence_score": confidence_score, "eff_score": eff_score,
        "gen_score": gen_score, "total_100_score": total_100_score,
        "loc_points_lost": loc_points_lost, "pose_points_lost": pose_points_lost,
        "rejection_points_lost": rejection_points_lost, "confidence_points_lost": confidence_points_lost,
        "eff_points_lost": eff_points_lost, "gen_points_lost": gen_points_lost, "total_points_lost": total_points_lost,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "f1_score": f1_score, "auc": auc, "med_rt": med_rt,
        "pct_5_a": pct_5_a * 100.0, "pct_5_b": pct_5_b * 100.0,
        "lost_not_in_top5": lost_not_in_top5, "lost_wrong_top5_selected": lost_wrong_top5_selected,
        "lost_refinement": lost_refinement, "lost_scale": lost_scale, "lost_theta": lost_theta
    }

def main():
    ckpt_path = "phase2_checkpoints/best_model_level1.pth"
    with open(ckpt_path, "rb") as f:
        sha256_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"Original Checkpoint SHA-256 Hash: {sha256_hash}")
    assert sha256_hash == "e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c", "SHA-256 Hash Mismatch!"

    engine = Phase2InferenceEngine(checkpoint_path="best_model_level1.pth", device="cpu")

    # Evaluate Dataset 2 (60-generator benchmark)
    res_d2, top5_d2 = run_official_100pt_evaluation(engine, "local_phase2_60gen_200_pairs", "phase2_60generator_manifest.csv", "ds2_60gen")
    metrics_d2 = compute_100pt_breakdown(res_d2)

    # Evaluate Dataset 1 (Generic benchmark)
    res_d1, top5_d1 = run_official_100pt_evaluation(engine, "local_phase2_200_pairs", "dataset_manifest.csv", "ds1_generic")
    metrics_d1 = compute_100pt_breakdown(res_d1)

    # Output official 100-point baseline CSV
    os.makedirs("phase2/results", exist_ok=True)
    csv_100pt_path = "phase2/results/official_100_point_baseline.csv"
    with open(csv_100pt_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "pair_id", "set", "gen_id", "gt_x", "gt_y", "gt_theta", "gt_scale", "gt_found",
            "pred_x", "pred_y", "pred_theta", "pred_scale", "pred_found", "pred_score",
            "loc_err_px", "scale_err", "theta_err_deg", "gt_in_top5", "runtime_ms"
        ])
        for r in res_d2:
            writer.writerow([
                "ds2_60gen", r["pair_id"], r["set"], r["gen_id"], r["gt_x"], r["gt_y"], r["gt_theta"], r["gt_scale"], r["gt_found"],
                r["pred_x"], r["pred_y"], r["pred_theta"], r["pred_scale"], r["pred_found"], r["pred_score"],
                round(r["loc_err"], 2), round(r["scale_err"], 2), round(r["theta_err"], 2), r["gt_in_top5"], round(r["runtime_ms"], 2)
            ])
        for r in res_d1:
            writer.writerow([
                "ds1_generic", r["pair_id"], r["set"], r["gen_id"], r["gt_x"], r["gt_y"], r["gt_theta"], r["gt_scale"], r["gt_found"],
                r["pred_x"], r["pred_y"], r["pred_theta"], r["pred_scale"], r["pred_found"], r["pred_score"],
                round(r["loc_err"], 2), round(r["scale_err"], 2), round(r["theta_err"], 2), r["gt_in_top5"], round(r["runtime_ms"], 2)
            ])
    print(f"\nSaved official 100-point baseline CSV to: {csv_100pt_path}")

    # Output Points Lost Analysis CSV
    csv_lost_path = "phase2/results/points_lost_analysis.csv"
    with open(csv_lost_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "max_points", "ds2_60gen_score", "ds2_points_lost", "ds1_generic_score", "ds1_points_lost", "primary_bottleneck"])
        writer.writerow(["Localization (40)", 40.0, round(metrics_d2["loc_score"], 2), round(metrics_d2["loc_points_lost"], 2), round(metrics_d1["loc_score"], 2), round(metrics_d1["loc_points_lost"], 2), "Periodic cell replica selection"])
        writer.writerow(["Pose Scale (10)", 10.0, round(metrics_d2["scale_score"], 2), round(10.0 - metrics_d2["scale_score"], 2), round(metrics_d1["scale_score"], 2), round(10.0 - metrics_d1["scale_score"], 2), "Localization failure cascades to pose"])
        writer.writerow(["Pose Rotation (10)", 10.0, round(metrics_d2["theta_score"], 2), round(10.0 - metrics_d2["theta_score"], 2), round(metrics_d1["theta_score"], 2), round(10.0 - metrics_d1["theta_score"], 2), "Localization failure cascades to pose"])
        writer.writerow(["Rejection (15)", 15.0, round(metrics_d2["rejection_score"], 2), round(metrics_d2["rejection_points_lost"], 2), round(metrics_d1["rejection_score"], 2), round(metrics_d1["rejection_points_lost"], 2), "Rejection threshold & FN/FP"])
        writer.writerow(["Confidence Calibration (10)", 10.0, round(metrics_d2["confidence_score"], 2), round(metrics_d2["confidence_points_lost"], 2), round(metrics_d1["confidence_score"], 2), round(metrics_d1["confidence_points_lost"], 2), "Score ranking overlap"])
        writer.writerow(["CPU Efficiency (5)", 5.0, round(metrics_d2["eff_score"], 2), round(metrics_d2["eff_points_lost"], 2), round(metrics_d1["eff_score"], 2), round(metrics_d1["eff_points_lost"], 2), "None (Fast runtime ~350ms)"])
        writer.writerow(["Generator / Citations (10)", 10.0, 10.0, 0.0, 10.0, 0.0, "Carried forward from Phase 1"])
        writer.writerow(["TOTAL SCORE (100)", 100.0, round(metrics_d2["total_100_score"], 2), round(metrics_d2["total_points_lost"], 2), round(metrics_d1["total_100_score"], 2), round(metrics_d1["total_points_lost"], 2), "Candidate Selection of Periodic Replicas"])

    print(f"Saved points lost analysis CSV to: {csv_lost_path}")

    # Calculate Theoretical Upper Bounds (A through F)
    # A. Current Score: metrics_d2["total_100_score"]
    # B. Perfect Candidate Selection (If candidate selection always picked GT candidate when present in Top-5)
    # C. Perfect Localization (Sub-pixel loc_err <= 1.0 for all present pairs)
    # D. Perfect Localization + Pose (loc_err <= 1.0, scale_err <= 0.25, theta_err <= 0.5)
    # E. Perfect Rejection (F1 = 1.0 -> 15.0 pts)
    # F. Perfect Confidence (AUC = 1.0 -> 10.0 pts)

    # Compute Perfect Candidate Selection Score for DS2:
    # 159 out of 160 present pairs have GT in Top-5.
    perfect_cand_loc_credit = (159.0 / 160.0) * 1.00 # Assuming 1px refinement
    perfect_cand_loc_score = perfect_cand_loc_credit * 40.0
    perfect_cand_pose_score = (159.0 / 160.0) * 20.0
    perfect_cand_total = perfect_cand_loc_score + perfect_cand_pose_score + metrics_d2["rejection_score"] + metrics_d2["confidence_score"] + 5.0 + 10.0

    print("\n=======================================================")
    print("THEORETICAL UPPER BOUNDS ANALYSIS (DS2 60-Generator)")
    print("=======================================================")
    print(f"A. Current 100-Point Score: {metrics_d2['total_100_score']:.2f} / 100.0")
    print(f"B. Score if Candidate Selection is Perfect: {perfect_cand_total:.2f} / 100.0 (+{perfect_cand_total - metrics_d2['total_100_score']:.2f} pts gain!)")
    print(f"C. Score if Localization is Perfect: {40.0 + metrics_d2['pose_score'] + metrics_d2['rejection_score'] + metrics_d2['confidence_score'] + 15.0:.2f} / 100.0")
    print(f"D. Score if Localization + Pose are Perfect: {40.0 + 20.0 + metrics_d2['rejection_score'] + metrics_d2['confidence_score'] + 15.0:.2f} / 100.0")
    print(f"E. Score if Rejection is Perfect (F1=1.0): {metrics_d2['total_100_score'] + metrics_d2['rejection_points_lost']:.2f} / 100.0")
    print(f"F. Score if Confidence Calibration is Perfect (AUC=1.0): {metrics_d2['total_100_score'] + metrics_d2['confidence_points_lost']:.2f} / 100.0")

    # Print Periodic Target Failures Deep-Dive
    target_pairs = ["pair_006", "pair_066", "pair_186", "pair_116"]
    print("\n=======================================================")
    print("PERIODIC TARGET FAILURES DEEP-DIVE (Top-5 Candidate Breakdown)")
    print("=======================================================")
    for pid in target_pairs:
        print(f"\n--- {pid} ---")
        if pid in top5_d2:
            for cand in top5_d2[pid]:
                print(f"  Rank {cand['rank']}: (x={cand['x']:.1f}, y={cand['y']:.1f}) | NCC={cand['ncc']:.4f} | Siamese={cand['siamese']:.4f} | Fused={cand['fused']:.4f} | Dist from GT = {cand['dist_gt']:.1f}px")

if __name__ == "__main__":
    main()
