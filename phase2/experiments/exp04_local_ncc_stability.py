#!/usr/bin/env python3
"""
Phase-2 EXP-04: Local NCC Peak Stability & Refinement Analysis (Vectorized OpenCV Version)
=======================================================================================
Single-Change Experiment testing whether evaluating local NCC peak stability / sharpness
in a small 9x9 translation neighborhood around existing Top-5 candidates can improve candidate ranking.

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

def compute_local_ncc_stability_fast(search_img, ref_img, cx, cy, scale, theta, radius=4):
    """
    Vectorized OpenCV matchTemplate evaluation over a (2*radius+1) x (2*radius+1) neighborhood.
    """
    h_img, w_img = search_img.shape[:2]

    # 1. Prepare rotated and scaled reference template
    ref_curr = ref_img
    if abs(theta) > 0.01:
        h_r, w_r = ref_curr.shape[:2]
        M_rot = cv2.getRotationMatrix2D((w_r / 2.0, h_r / 2.0), -theta, 1.0)
        ref_curr = cv2.warpAffine(ref_curr, M_rot, (w_r, h_r), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    if abs(scale - 1.0) > 0.01:
        new_w = max(4, int(round(ref_curr.shape[1] * scale)))
        new_h = max(4, int(round(ref_curr.shape[0] * scale)))
        ref_curr = cv2.resize(ref_curr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    th, tw = ref_curr.shape[:2]
    half_w = tw / 2.0
    half_h = th / 2.0

    # 2. Extract bounding box from search image enclosing the 9x9 neighborhood
    margin = radius + 2
    x0 = int(math.floor(cx - half_w - margin))
    y0 = int(math.floor(cy - half_h - margin))
    x1 = int(math.ceil(cx + half_w + margin))
    y1 = int(math.ceil(cy + half_h + margin))

    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(w_img, x1); y1 = min(h_img, y1)

    search_patch = search_img[y0:y1, x0:x1]
    if search_patch.shape[0] <= th or search_patch.shape[1] <= tw:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # 3. Vectorized MatchTemplate call (C++ execution)
    res_map = cv2.matchTemplate(search_patch, ref_curr, cv2.TM_CCOEFF_NORMED)

    # 4. Extract local neighborhood centered at candidate position
    map_cx = int(round((cx - half_w) - x0))
    map_cy = int(round((cy - half_h) - y0))

    r0 = max(0, map_cy - radius); r1 = min(res_map.shape[0], map_cy + radius + 1)
    c0 = max(0, map_cx - radius); c1 = min(res_map.shape[1], map_cx + radius + 1)

    local_box = res_map[r0:r1, c0:c1]
    if local_box.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    c_ncc = float(res_map[map_cy, map_cx]) if (0 <= map_cy < res_map.shape[0] and 0 <= map_cx < res_map.shape[1]) else float(local_box[0,0])
    loc_max = float(np.max(local_box))
    loc_mean = float(np.mean(local_box))
    loc_std = float(np.std(local_box))
    loc_margin = loc_max - loc_mean
    peak_sharpness = float(loc_margin / (loc_std + 1e-5))

    return c_ncc, loc_max, loc_mean, loc_std, loc_margin, peak_sharpness

def run_single_pass_dataset(engine, data_dir, manifest_filename):
    manifest_path = os.path.join(data_dir, manifest_filename)
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

            ref_img = load_grayscale_image(ref_path)
            search_img = load_grayscale_image(search_path)

            t0 = time.time()
            res_dict, best_coarse, refined_results = engine.localize_pair(
                ref_path, search_path, ncc_weight=0.5, rejection_thresh=0.42, scale_step=0.25, theta_step=1.0,
                return_diagnostics=True
            )
            t1 = time.time()
            runtime_ms = (t1 - t0) * 1000.0

            cand_details = []
            for r_idx, cand in enumerate(refined_results[:5]):
                cand_x, cand_y = cand["x"], cand["y"]
                cand_scale, cand_theta = cand["scale"], cand["theta"]

                c_ncc, loc_max, loc_mean, loc_std, loc_margin, peak_sharp = compute_local_ncc_stability_fast(
                    search_img, ref_img, cand_x, cand_y, cand_scale, cand_theta
                )

                dist_gt = math.sqrt((cand_x - gt_x)**2 + (cand_y - gt_y)**2) if gt_found == 1 else 999.0
                is_gt = (dist_gt <= 15.0)

                cand_details.append({
                    "rank_orig": r_idx + 1, "x": cand_x, "y": cand_y,
                    "scale": cand_scale, "theta": cand_theta,
                    "ncc_orig": cand.get("ncc_norm", 0.0), "siamese": cand.get("siamese_sim", 0.0),
                    "fused_orig": cand.get("fused_score", 0.0),
                    "center_ncc": c_ncc, "loc_max": loc_max, "loc_mean": loc_mean,
                    "loc_std": loc_std, "loc_margin": loc_margin, "peak_sharpness": peak_sharp,
                    "is_gt": is_gt, "dist_gt": dist_gt
                })

            pair_records.append({
                "pair_id": pair_id, "set": set_name, "gen_id": gen_id,
                "gt_x": gt_x, "gt_y": gt_y, "gt_theta": gt_theta, "gt_scale": gt_scale, "gt_found": gt_found,
                "runtime_ms": runtime_ms, "candidates": cand_details,
                "gt_in_top5": any(c["is_gt"] for c in cand_details)
            })

            gc.collect()

    return pair_records

def evaluate_stability_strategy(pair_records, mode="BASE", lam=0.05):
    results = []
    tau = 0.42

    for rec in pair_records:
        cand_list = rec["candidates"]
        eval_cands = []
        for cand in cand_list:
            if mode == "BASE":
                score = cand["fused_orig"]
            elif mode == "EXP04_REF":
                norm_sharp = min(1.0, max(0.0, cand["peak_sharpness"] / 10.0))
                score = cand["fused_orig"] + lam * norm_sharp
            elif mode == "EXP04_MARGIN":
                score = cand["fused_orig"] + lam * cand["loc_margin"]
            elif mode == "EXP04_MAX":
                score = 0.5 * cand["loc_max"] + 0.5 * cand["siamese"]
            else:
                score = cand["fused_orig"]

            eval_cands.append({**cand, "eval_score": score})

        eval_cands.sort(key=lambda c: -c["eval_score"])
        selected = eval_cands[0]

        pred_x, pred_y = selected["x"], selected["y"]
        pred_scale, pred_theta = selected["scale"], selected["theta"]
        pred_fused = selected["eval_score"]

        gt_found = rec["gt_found"]
        pred_found = 1 if pred_fused >= tau else 0
        pred_score = float(round(1.0 / (1.0 + math.exp(-6.0 * (pred_fused - tau))), 4))

        if gt_found == 1 and pred_found == 1:
            loc_err = math.sqrt((pred_x - rec["gt_x"])**2 + (pred_y - rec["gt_y"])**2)
            scale_err = abs(pred_scale - rec["gt_scale"])
            theta_err = abs(pred_theta - rec["gt_theta"])
        elif gt_found == 0 and pred_found == 0:
            loc_err = 0.0; scale_err = 0.0; theta_err = 0.0
        else:
            loc_err = 999.0; scale_err = 999.0; theta_err = 999.0

        results.append({
            "pair_id": rec["pair_id"], "set": rec["set"], "gen_id": rec["gen_id"],
            "gt_x": rec["gt_x"], "gt_y": rec["gt_y"], "gt_theta": rec["gt_theta"], "gt_scale": rec["gt_scale"],
            "gt_found": gt_found, "pred_x": pred_x, "pred_y": pred_y, "pred_theta": pred_theta, "pred_scale": pred_scale,
            "pred_found": pred_found, "pred_score": pred_score, "loc_err": loc_err, "scale_err": scale_err,
            "theta_err": theta_err, "runtime_ms": rec["runtime_ms"], "gt_in_top5": rec["gt_in_top5"],
            "eval_candidates": eval_cands
        })

    metrics = compute_100pt_breakdown(results)
    return metrics, results

def main():
    ckpt_path = "phase2_checkpoints/best_model_level1.pth"
    with open(ckpt_path, "rb") as f:
        sha256_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"Original Checkpoint SHA-256 Hash: {sha256_hash}")
    assert sha256_hash == "e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c", "SHA-256 Mismatch!"

    engine = Phase2InferenceEngine(checkpoint_path="best_model_level1.pth", device="cpu")

    print("\n===========================================================================")
    print("PHASE-2 EXP-04: LOCAL NCC PEAK STABILITY & REFINEMENT ANALYSIS (FAST)")
    print("===========================================================================")

    print("Running fast single-pass feature extraction on DS2 (local_phase2_60gen_200_pairs)...")
    sys.stdout.flush()
    ds2_records = run_single_pass_dataset(engine, "local_phase2_60gen_200_pairs", "phase2_60generator_manifest.csv")

    print("Running fast single-pass feature extraction on DS1 (local_phase2_200_pairs)...")
    sys.stdout.flush()
    ds1_records = run_single_pass_dataset(engine, "local_phase2_200_pairs", "dataset_manifest.csv")

    modes = [
        ("BASE", "Baseline_Fused"),
        ("EXP04_REF", "EXP04_Refined_PeakSharpness_lambda0.05"),
        ("EXP04_MARGIN", "EXP04_Margin_lambda0.05"),
        ("EXP04_MAX", "EXP04_LocalMax_Siamese_Fusion")
    ]

    target_pairs = ["pair_006", "pair_066", "pair_186", "pair_116"]
    summary_records = []

    print("\nEvaluating EXP-04 stability formulations...")
    for mode_code, mode_desc in modes:
        m_ds2, res_ds2 = evaluate_stability_strategy(ds2_records, mode=mode_code, lam=0.05)
        m_ds1, res_ds1 = evaluate_stability_strategy(ds1_records, mode=mode_code, lam=0.05)

        runtimes = [r["runtime_ms"] for r in res_ds2]
        p90_rt = float(np.percentile(runtimes, 90))

        rec = {
            "mode": mode_code, "strategy": mode_desc,
            "ds2_total": round(m_ds2["total_100_score"], 2),
            "ds2_loc": round(m_ds2["loc_score"], 2),
            "ds2_scale": round(m_ds2["scale_score"], 2),
            "ds2_theta": round(m_ds2["theta_score"], 2),
            "ds2_rejection": round(m_ds2["rejection_score"], 2),
            "ds2_conf": round(m_ds2["confidence_score"], 2),
            "ds2_eff": round(m_ds2["eff_score"], 2),
            "ds2_gen": round(m_ds2["gen_score"], 2),
            "ds2_set_a_5px": round(m_ds2["pct_5_a"], 1),
            "ds2_set_b_5px": round(m_ds2["pct_5_b"], 1),
            "ds1_total": round(m_ds1["total_100_score"], 2),
            "ds1_loc": round(m_ds1["loc_score"], 2),
            "med_rt": round(m_ds2["med_rt"], 1),
            "p90_rt": round(p90_rt, 1)
        }
        summary_records.append(rec)
        print(f"Strategy {mode_code} ({mode_desc}): DS2 Total = {m_ds2['total_100_score']:.2f} / 100.0 | Loc = {m_ds2['loc_score']:.2f} / 40.0 | DS1 Total = {m_ds1['total_100_score']:.2f} / 100.0")
        sys.stdout.flush()

        if mode_code == "EXP04_REF":
            print(f"\n===========================================================================")
            print(f"EXP-04 TARGET PERIODIC FAILURES DEEP-DIVE TABLE")
            print(f"===========================================================================")
            for r in res_ds2:
                if r["pair_id"] in target_pairs:
                    print(f"\n--- Pair: {r['pair_id']} (GT: x={r['gt_x']}, y={r['gt_y']}) ---")
                    for cand in r["eval_candidates"]:
                        gt_tag = "GT LANDMARK" if cand["is_gt"] else "DECOY"
                        print(f"  [{gt_tag}] Orig Rank {cand['rank_orig']}: (x={cand['x']:.1f}, y={cand['y']:.1f}) | NCC={cand['ncc_orig']:.4f} | LocalMax={cand['loc_max']:.4f} | LocalStd={cand['loc_std']:.4f} | Sharpness={cand['peak_sharpness']:.4f} | EvalScore={cand['eval_score']:.4f} | DistGT={cand['dist_gt']:.1f}px")
            sys.stdout.flush()

    os.makedirs("phase2/results", exist_ok=True)
    csv_path = "phase2/results/exp04_local_ncc_stability.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mode", "strategy", "ds2_total", "ds2_loc", "ds2_scale", "ds2_theta",
            "ds2_rejection", "ds2_conf", "ds2_eff", "ds2_gen", "ds2_set_a_5px", "ds2_set_b_5px",
            "ds1_total", "ds1_loc", "med_rt", "p90_rt"
        ])
        writer.writeheader()
        writer.writerows(summary_records)
    print(f"\nSaved EXP-04 stability CSV to: {csv_path}")

    # Re-verify SHA256 Hash
    with open(ckpt_path, "rb") as f:
        sha256_hash_after = hashlib.sha256(f.read()).hexdigest()
    print(f"Post-run Checkpoint SHA-256 Hash: {sha256_hash_after}")
    assert sha256_hash_after == "e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c", "SHA-256 Altered!"

if __name__ == "__main__":
    main()
