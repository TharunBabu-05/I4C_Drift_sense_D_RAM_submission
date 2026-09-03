#!/usr/bin/env python3
"""
Phase-2 EXP-03: Isotropic Symmetry Penalization (Single Change Experiment)
==========================================================================
Hypothesis: Periodic DRAM cell matrix decoys exhibit highly isotropic / spatially symmetric
local gradient variance across sub-blocks, whereas true inserted landmark targets exhibit
asymmetric structural gradient distributions.

Single Change Tested:
Penalize candidates based on Isotropic Grid Symmetry:
  P_iso = std(subblock_variances) / (mean(subblock_variances) + 1e-5)
  New Candidate Score = Fused_Orig * (1.0 + alpha * P_iso)

CONSTRAINTS:
- Original Checkpoint: best_model_level1.pth (READ ONLY - SHA256 UNTOUCHED)
- Production Code: 100% UNTOUCHED
- No GT information used for candidate selection or inference decisions
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

def compute_isotropic_asymmetry_penalty(search_img, cx, cy, patch_size=100):
    """
    Computes local subblock gradient variance asymmetry P_iso for candidate crop centered at (cx, cy).
    """
    h_img, w_img = search_img.shape[:2]
    half = patch_size // 2
    x0, x1 = max(0, int(round(cx - half))), min(w_img, int(round(cx + half)))
    y0, y1 = max(0, int(round(cy - half))), min(h_img, int(round(cy + half)))

    crop = search_img[y0:y1, x0:x1]
    if crop.shape[0] != patch_size or crop.shape[1] != patch_size:
        crop = cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

    # Divide 100x100 crop into 4x4 sub-blocks (25x25 each)
    sub_vars = []
    sb_size = patch_size // 4
    for r in range(4):
        for c in range(4):
            sb = crop[r*sb_size:(r+1)*sb_size, c*sb_size:(c+1)*sb_size]
            sub_vars.append(np.var(sb))

    mean_v = np.mean(sub_vars)
    std_v = np.std(sub_vars)

    p_iso = float(std_v / (mean_v + 1e-5))
    return p_iso

def run_exp03():
    ckpt_path = "phase2_checkpoints/best_model_level1.pth"
    with open(ckpt_path, "rb") as f:
        sha256_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"Original Checkpoint SHA-256 Hash: {sha256_hash}")
    assert sha256_hash == "e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c", "SHA-256 Mismatch!"

    engine = Phase2InferenceEngine(checkpoint_path="best_model_level1.pth", device="cpu")
    manifest_path = "local_phase2_60gen_200_pairs/phase2_60generator_manifest.csv"

    print("\n===========================================================================")
    print("PHASE-2 EXP-03: ISOTROPIC SYMMETRY PENALIZATION (SINGLE CHANGE TEST)")
    print("===========================================================================")

    pair_records = []
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
            t1 = time.time()
            runtime_ms = (t1 - t0) * 1000.0

            cand_eval = []
            for r_idx, cand in enumerate(refined_results[:5]):
                cand_x, cand_y = cand["x"], cand["y"]
                fused_orig = cand.get("fused_score", 0.0)

                p_iso = compute_isotropic_asymmetry_penalty(search_img, cand_x, cand_y)

                # Single modification: scale fused score by (1 + 0.25 * p_iso)
                alpha = 0.25
                score_new = fused_orig * (1.0 + alpha * p_iso)

                dist_gt = math.sqrt((cand_x - gt_x)**2 + (cand_y - gt_y)**2) if gt_found == 1 else 999.0
                is_gt = (dist_gt <= 15.0)

                cand_eval.append({
                    "rank_orig": r_idx + 1, "x": cand_x, "y": cand_y,
                    "scale": cand["scale"], "theta": cand["theta"],
                    "fused_orig": fused_orig, "p_iso": p_iso, "score_new": score_new,
                    "is_gt": is_gt, "dist_gt": dist_gt
                })

            pair_records.append({
                "pair_id": pair_id, "set": set_name, "gen_id": gen_id,
                "gt_x": gt_x, "gt_y": gt_y, "gt_theta": gt_theta, "gt_scale": gt_scale, "gt_found": gt_found,
                "runtime_ms": runtime_ms, "candidates": cand_eval,
                "gt_in_top5": any(c["is_gt"] for c in cand_eval)
            })

            gc.collect()

    # Evaluate Baseline vs EXP-03
    results_base = []
    results_exp03 = []
    tau = 0.42

    target_pairs = ["pair_006", "pair_066", "pair_186", "pair_116"]
    diag_targets = {}

    for rec in pair_records:
        gt_found = rec["gt_found"]
        cands = rec["candidates"]

        # 1. Baseline Selection (Original Fused Score)
        cands_base = sorted(cands, key=lambda c: -c["fused_orig"])
        sel_base = cands_base[0]

        pred_found_b = 1 if sel_base["fused_orig"] >= tau else 0
        pred_score_b = float(round(1.0 / (1.0 + math.exp(-6.0 * (sel_base["fused_orig"] - tau))), 4))

        if gt_found == 1 and pred_found_b == 1:
            loc_err_b = math.sqrt((sel_base["x"] - rec["gt_x"])**2 + (sel_base["y"] - rec["gt_y"])**2)
            scale_err_b = abs(sel_base["scale"] - rec["gt_scale"])
            theta_err_b = abs(sel_base["theta"] - rec["gt_theta"])
        elif gt_found == 0 and pred_found_b == 0:
            loc_err_b = 0.0; scale_err_b = 0.0; theta_err_b = 0.0
        else:
            loc_err_b = 999.0; scale_err_b = 999.0; theta_err_b = 999.0

        results_base.append({
            "pair_id": rec["pair_id"], "set": rec["set"], "gen_id": rec["gen_id"],
            "gt_x": rec["gt_x"], "gt_y": rec["gt_y"], "gt_theta": rec["gt_theta"], "gt_scale": rec["gt_scale"],
            "gt_found": gt_found, "pred_x": sel_base["x"], "pred_y": sel_base["y"],
            "pred_theta": sel_base["theta"], "pred_scale": sel_base["scale"],
            "pred_found": pred_found_b, "pred_score": pred_score_b,
            "loc_err": loc_err_b, "scale_err": scale_err_b, "theta_err": theta_err_b,
            "runtime_ms": rec["runtime_ms"], "gt_in_top5": rec["gt_in_top5"]
        })

        # 2. EXP-03 Selection (New Asymmetry Weighted Score)
        cands_exp03 = sorted(cands, key=lambda c: -c["score_new"])
        sel_exp03 = cands_exp03[0]

        pred_found_e = 1 if sel_exp03["score_new"] >= tau else 0
        pred_score_e = float(round(1.0 / (1.0 + math.exp(-6.0 * (sel_exp03["score_new"] - tau))), 4))

        if gt_found == 1 and pred_found_e == 1:
            loc_err_e = math.sqrt((sel_exp03["x"] - rec["gt_x"])**2 + (sel_exp03["y"] - rec["gt_y"])**2)
            scale_err_e = abs(sel_exp03["scale"] - rec["gt_scale"])
            theta_err_e = abs(sel_exp03["theta"] - rec["gt_theta"])
        elif gt_found == 0 and pred_found_e == 0:
            loc_err_e = 0.0; scale_err_e = 0.0; theta_err_e = 0.0
        else:
            loc_err_e = 999.0; scale_err_e = 999.0; theta_err_e = 999.0

        results_exp03.append({
            "pair_id": rec["pair_id"], "set": rec["set"], "gen_id": rec["gen_id"],
            "gt_x": rec["gt_x"], "gt_y": rec["gt_y"], "gt_theta": rec["gt_theta"], "gt_scale": rec["gt_scale"],
            "gt_found": gt_found, "pred_x": sel_exp03["x"], "pred_y": sel_exp03["y"],
            "pred_theta": sel_exp03["theta"], "pred_scale": sel_exp03["scale"],
            "pred_found": pred_found_e, "pred_score": pred_score_e,
            "loc_err": loc_err_e, "scale_err": scale_err_e, "theta_err": theta_err_e,
            "runtime_ms": rec["runtime_ms"], "gt_in_top5": rec["gt_in_top5"]
        })

        if rec["pair_id"] in target_pairs:
            diag_targets[rec["pair_id"]] = cands_exp03

    m_base = compute_100pt_breakdown(results_base)
    m_exp03 = compute_100pt_breakdown(results_exp03)

    print("\n===========================================================================")
    print("TARGET PERIODIC FAILURE CASES DIAGNOSTIC BREAKDOWN (EXP-03)")
    print("===========================================================================")
    for pid in target_pairs:
        print(f"\n--- Pair: {pid} ---")
        for cand in diag_targets[pid]:
            gt_tag = "GT LANDMARK" if cand["is_gt"] else "DECOY"
            print(f"  [{gt_tag}] Orig Rank {cand['rank_orig']}: (x={cand['x']:.1f}, y={cand['y']:.1f}) | Fused={cand['fused_orig']:.4f} | P_iso={cand['p_iso']:.4f} | ScoreNew={cand['score_new']:.4f} | DistGT={cand['dist_gt']:.1f}px")

    print("\n===========================================================================")
    print("OFFICIAL 100-POINT BENCHMARK SCORE COMPARISON (EXP-03 vs BASELINE)")
    print("===========================================================================")
    print(f"Baseline Score : {m_base['total_100_score']:.2f} / 100.0 (Loc: {m_base['loc_score']:.2f}, Pose: {m_base['pose_score']:.2f})")
    print(f"EXP-03 Score   : {m_exp03['total_100_score']:.2f} / 100.0 (Loc: {m_exp03['loc_score']:.2f}, Pose: {m_exp03['pose_score']:.2f})")
    print(f"Score Delta    : {m_exp03['total_100_score'] - m_base['total_100_score']:+.2f} Points")
    print(f"  - Localization Score: {m_exp03['loc_score']:.2f} / 40.0 (Delta: {m_exp03['loc_score'] - m_base['loc_score']:+.2f})")
    print(f"  - Pose Scale Score  : {m_exp03['scale_score']:.2f} / 10.0")
    print(f"  - Pose Rot Score    : {m_exp03['theta_score']:.2f} / 10.0")
    print(f"  - Rejection F1      : {m_exp03['rejection_score']:.2f} / 15.0 (F1: {m_exp03['f1_score']:.4f})")
    print(f"  - Confidence AUC    : {m_exp03['confidence_score']:.2f} / 10.0 (AUC: {m_exp03['auc']:.4f})")
    print(f"  - Median Runtime    : {m_exp03['med_rt']:.1f} ms")
    sys.stdout.flush()

if __name__ == "__main__":
    run_exp03()
