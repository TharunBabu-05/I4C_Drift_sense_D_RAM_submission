#!/usr/bin/env python3
"""
Phase-2 Day-1 Diagnostic Experiment: Global Landmark Boundary & Asymmetric Context Alignment
=============================================================================================
Tests whether lightweight non-neural global boundary and asymmetric context features can
distinguish true GT landmarks from periodic DRAM cell decoys in NCC Top-5 candidates.

CONSTRAINTS:
- Checkpoint: best_model_level1.pth (READ ONLY - SHA256 UNTOUCHED)
- Production Code: 100% UNTOUCHED
- No GT information used for inference decisions (GT used solely for evaluation)

Features Evaluated per Top-5 Candidate:
1. Coarse NCC Score
2. Boundary Distance Score (Alignment relative to macro-cell edge contours in 300x300 context)
3. Asymmetric Radial Context Score (Quadrant edge orientation histogram variance in 300x300 context)
4. Local Structural Score (Inner/Outer ring edge sharpness ratio)
5. Combined Score
"""

import os
import sys
import math
import time
import hashlib
import csv
import gc
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from phase2.phase2_inference import Phase2InferenceEngine, load_grayscale_image
from phase2.experiments.official_100pt_audit import compute_100pt_breakdown

def extract_boundary_and_context_features(search_img, cx, cy, patch_size=300):
    """
    Extracts global boundary and asymmetric context features from a context window around (cx, cy).
    """
    h_img, w_img = search_img.shape[:2]
    half = patch_size // 2
    x0, x1 = max(0, int(round(cx - half))), min(w_img, int(round(cx + half)))
    y0, y1 = max(0, int(round(cy - half))), min(h_img, int(round(cy + half)))

    crop = search_img[y0:y1, x0:x1]
    if crop.shape[0] != patch_size or crop.shape[1] != patch_size:
        crop = cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

    # 1. Macro Boundary Score (Distance to high-magnitude edge contours in 300x300 context)
    sobelx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    _, strong_edges = cv2.threshold(mag, 100.0, 255.0, cv2.THRESH_BINARY)
    strong_edges = strong_edges.astype(np.uint8)

    # Distance map from strong macro edges
    dist_map = cv2.distanceTransform(255 - strong_edges, cv2.DIST_L2, 5)
    center_dist = float(dist_map[half, half])
    boundary_score = 1.0 / (1.0 + center_dist / 50.0) # Higher if near macro boundary

    # 2. Asymmetric Radial Context Score (Variance across 4 quadrants in 300x300 context)
    q_tl = mag[0:half, 0:half]
    q_tr = mag[0:half, half:patch_size]
    q_bl = mag[half:patch_size, 0:half]
    q_br = mag[half:patch_size, half:patch_size]

    q_means = [np.mean(q_tl), np.mean(q_tr), np.mean(q_bl), np.mean(q_br)]
    asymmetric_context_score = float(np.std(q_means) / (np.mean(q_means) + 1e-5))

    # 3. Local Structural Score (Inner 50px core vs Outer 150px ring contrast ratio)
    inner_mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
    cv2.circle(inner_mask, (half, half), 25, 255, -1)
    outer_mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
    cv2.circle(outer_mask, (half, half), 75, 255, -1)
    outer_ring_mask = cv2.bitwise_and(outer_mask, cv2.bitwise_not(inner_mask))

    inner_std = float(np.std(crop[inner_mask == 255]))
    outer_std = float(np.std(crop[outer_ring_mask == 255]))
    local_structural_score = float(inner_std / (outer_std + 1e-5))

    return boundary_score, asymmetric_context_score, local_structural_score

def run_day1_experiment(w_boundary=0.30, w_asym=0.20, w_struct=0.10):
    ckpt_path = "phase2_checkpoints/best_model_level1.pth"
    with open(ckpt_path, "rb") as f:
        sha256_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"Original Checkpoint SHA-256 Hash: {sha256_hash}")
    assert sha256_hash == "e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c", "SHA-256 Mismatch!"

    engine = Phase2InferenceEngine(checkpoint_path="best_model_level1.pth", device="cpu")
    manifest_path = "local_phase2_60gen_200_pairs/phase2_60generator_manifest.csv"

    print("\n===========================================================================")
    print("PHASE-2 DAY-1 DIAGNOSTIC EXPERIMENT: GLOBAL BOUNDARY & ASYMMETRIC CONTEXT")
    print("===========================================================================")

    diagnostic_records = []
    results_exp = []
    target_pairs = ["pair_006", "pair_066", "pair_186", "pair_116"]
    target_diagnostics = {pid: [] for pid in target_pairs}

    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair_id = row["pair_id"]
            set_name = row["set"]
            ref_path = os.path.abspath(row["reference_path"])
            search_path = os.path.abspath(row["search_path"])

            gt_x = float(row["x_gt"])
            gt_y = float(row["y_gt"])
            gt_theta = float(row["theta_gt"])
            gt_scale = float(row["scale_gt"])
            gt_found = int(row["found_gt"])
            gen_id = row.get("generator_id", "generic")

            search_img = load_grayscale_image(search_path)
            t0 = time.time()

            res_dict, best_coarse, refined_results = engine.localize_pair(
                ref_path, search_path, ncc_weight=0.5, rejection_thresh=0.42, scale_step=0.25, theta_step=1.0,
                return_diagnostics=True
            )

            # Evaluate features on Top-5 candidates
            cand_eval = []
            for r_idx, cand in enumerate(refined_results[:5]):
                cand_x, cand_y = cand["x"], cand["y"]
                dist_gt = math.sqrt((cand_x - gt_x)**2 + (cand_y - gt_y)**2) if gt_found == 1 else 999.0
                is_gt = (dist_gt <= 15.0)

                b_score, a_score, s_score = extract_boundary_and_context_features(search_img, cand_x, cand_y)

                # Combined Verification Score
                # fused_orig = 0.5 * NCC + 0.5 * Siamese
                fused_orig = cand.get("fused_score", 0.0)
                comb_score = fused_orig + w_boundary * b_score + w_asym * a_score + w_struct * s_score

                cand_rec = {
                    "rank_orig": r_idx + 1, "x": cand_x, "y": cand_y,
                    "scale": cand["scale"], "theta": cand["theta"],
                    "ncc": cand.get("ncc_norm", 0.0), "siamese": cand.get("siamese_sim", 0.0),
                    "fused_orig": fused_orig, "b_score": b_score, "a_score": a_score, "s_score": s_score,
                    "comb_score": comb_score, "is_gt": is_gt, "dist_gt": dist_gt
                }
                cand_eval.append(cand_rec)
                diagnostic_records.append({"pair_id": pair_id, **cand_rec})

            # Re-rank Top-5 candidates based ONLY on non-GT comb_score
            cand_eval.sort(key=lambda c: -c["comb_score"])
            selected_cand = cand_eval[0]
            t1 = time.time()
            runtime_ms = (t1 - t0) * 1000.0

            if pair_id in target_diagnostics:
                target_diagnostics[pair_id] = cand_eval

            pred_x, pred_y = selected_cand["x"], selected_cand["y"]
            pred_scale, pred_theta = selected_cand["scale"], selected_cand["theta"]
            pred_fused = selected_cand["comb_score"]

            tau = 0.42
            pred_found = 1 if pred_fused >= tau else 0
            pred_score = float(round(1.0 / (1.0 + math.exp(-6.0 * (pred_fused - tau))), 4))

            if gt_found == 1 and pred_found == 1:
                loc_err = math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
                scale_err = abs(pred_scale - gt_scale)
                theta_err = abs(pred_theta - gt_theta)
            elif gt_found == 0 and pred_found == 0:
                loc_err = 0.0; scale_err = 0.0; theta_err = 0.0
            else:
                loc_err = 999.0; scale_err = 999.0; theta_err = 999.0

            results_exp.append({
                "pair_id": pair_id, "set": set_name, "gen_id": gen_id,
                "gt_x": gt_x, "gt_y": gt_y, "gt_theta": gt_theta, "gt_scale": gt_scale, "gt_found": gt_found,
                "pred_x": pred_x, "pred_y": pred_y, "pred_theta": pred_theta, "pred_scale": pred_scale,
                "pred_found": pred_found, "pred_score": pred_score,
                "loc_err": loc_err, "scale_err": scale_err, "theta_err": theta_err, "runtime_ms": runtime_ms,
                "gt_in_top5": any(c["is_gt"] for c in cand_eval)
            })

            gc.collect()

    # Save diagnostic candidate feature CSV
    os.makedirs("phase2/results", exist_ok=True)
    diag_csv_path = "phase2/results/day1_top5_candidate_diagnostics.csv"
    with open(diag_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair_id", "rank_orig", "x", "y", "scale", "theta", "ncc", "siamese", "fused_orig",
            "b_score", "a_score", "s_score", "comb_score", "is_gt", "dist_gt"
        ])
        writer.writeheader()
        writer.writerows(diagnostic_records)
    print(f"\nSaved candidate diagnostic feature CSV to: {diag_csv_path}")

    # Print Diagnostic Case Deep-Dive for Periodic Failure Cases
    print("\n===========================================================================")
    print("CRITICAL PERIODIC FAILURE CASES DIAGNOSTIC BREAKDOWN")
    print("===========================================================================")
    for pid in target_pairs:
        print(f"\n--- {pid} ---")
        for cand in target_diagnostics[pid]:
            gt_str = "GT LANDMARK" if cand["is_gt"] else "DECOY"
            print(f"  [{gt_str}] Orig Rank {cand['rank_orig']}: (x={cand['x']:.1f}, y={cand['y']:.1f}) | NCC={cand['ncc']:.4f} | Siam={cand['siamese']:.4f} | Boundary={cand['b_score']:.4f} | Asym={cand['a_score']:.4f} | Struct={cand['s_score']:.4f} | COMBINED={cand['comb_score']:.4f} | Dist GT={cand['dist_gt']:.1f}px")

    # Evaluate Official 100-Point Score
    metrics_exp = compute_100pt_breakdown(results_exp)
    print("\n===========================================================================")
    print("OFFICIAL 100-POINT BENCHMARK SCORE COMPARISON (DS2 60-GENERATOR)")
    print("===========================================================================")
    print(f"Baseline Score  : 46.77 / 100.0")
    print(f"New Day-1 Score : {metrics_exp['total_100_score']:.2f} / 100.0 (Delta: {metrics_exp['total_100_score'] - 46.77:+.2f} pts)")
    print(f"  - Localization Score : {metrics_exp['loc_score']:.2f} / 40.0")
    print(f"  - Pose Recovery Score: {metrics_exp['pose_score']:.2f} / 20.0 (Scale: {metrics_exp['scale_score']:.2f}, Rot: {metrics_exp['theta_score']:.2f})")
    print(f"  - Rejection F1 Score : {metrics_exp['rejection_score']:.2f} / 15.0 (F1: {metrics_exp['f1_score']:.4f})")
    print(f"  - Confidence AUC     : {metrics_exp['confidence_score']:.2f} / 10.0 (AUC: {metrics_exp['auc']:.4f})")
    print(f"  - Efficiency Score   : {metrics_exp['eff_score']:.2f} / 5.0 (Median RT: {metrics_exp['med_rt']:.1f} ms)")
    print(f"  - Generator Citations: {metrics_exp['gen_score']:.2f} / 10.0")

    return metrics_exp, target_diagnostics

if __name__ == "__main__":
    run_day1_experiment()
