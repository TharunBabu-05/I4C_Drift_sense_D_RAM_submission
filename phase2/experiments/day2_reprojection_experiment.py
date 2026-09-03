#!/usr/bin/env python3
"""
Phase-2 Day-2 Experiment: Reference Reprojection / Candidate Consistency Verification
=======================================================================================
Single-pass evaluation of Top-5 candidate reference-reprojection / unwarping consistency.

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

def unwarp_candidate_crop(search_img, cx, cy, scale, theta_deg, out_size=100):
    half_out = out_size / 2.0
    rad = math.radians(theta_deg)
    cos_a = math.cos(rad) * scale
    sin_a = math.sin(rad) * scale

    M = np.array([
        [cos_a, -sin_a, cx - (cos_a * half_out - sin_a * half_out)],
        [sin_a,  cos_a, cy - (sin_a * half_out + cos_a * half_out)]
    ], dtype=np.float32)

    unwarped = cv2.warpAffine(search_img, M, (out_size, out_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return unwarped

def compute_reprojection_metrics(ref_crop, unwarped_crop):
    ref_f = ref_crop.astype(np.float32)
    unw_f = unwarped_crop.astype(np.float32)

    ref_std = np.std(ref_f)
    unw_std = np.std(unw_f)
    if ref_std > 1e-5 and unw_std > 1e-5:
        ref_norm = (ref_f - np.mean(ref_f)) / ref_std
        unw_norm = (unw_f - np.mean(unw_f)) / unw_std
        pixel_ncc = float(np.mean(ref_norm * unw_norm))
    else:
        pixel_ncc = 0.0

    ref_gx = cv2.Sobel(ref_f, cv2.CV_32F, 1, 0, ksize=3)
    ref_gy = cv2.Sobel(ref_f, cv2.CV_32F, 0, 1, ksize=3)
    ref_mag = cv2.magnitude(ref_gx, ref_gy)

    unw_gx = cv2.Sobel(unw_f, cv2.CV_32F, 1, 0, ksize=3)
    unw_gy = cv2.Sobel(unw_f, cv2.CV_32F, 0, 1, ksize=3)
    unw_mag = cv2.magnitude(unw_gx, unw_gy)

    rm_std = np.std(ref_mag)
    um_std = np.std(unw_mag)
    if rm_std > 1e-5 and um_std > 1e-5:
        rm_norm = (ref_mag - np.mean(ref_mag)) / rm_std
        um_norm = (unw_mag - np.mean(unw_mag)) / um_std
        grad_ncc = float(np.mean(rm_norm * um_norm))
    else:
        grad_ncc = 0.0

    pixel_err = float(np.mean(np.abs(unw_f - ref_f)) / 255.0)
    grad_err = float(np.mean(np.abs(unw_mag - ref_mag)) / (np.max(ref_mag) + 1e-5))

    ref_2x = cv2.resize(ref_crop, (50, 50), interpolation=cv2.INTER_AREA).astype(np.float32)
    unw_2x = cv2.resize(unwarped_crop, (50, 50), interpolation=cv2.INTER_AREA).astype(np.float32)
    err_2x = float(np.mean(np.abs(unw_2x - ref_2x)) / 255.0)

    ref_4x = cv2.resize(ref_crop, (25, 25), interpolation=cv2.INTER_AREA).astype(np.float32)
    unw_4x = cv2.resize(unwarped_crop, (25, 25), interpolation=cv2.INTER_AREA).astype(np.float32)
    err_4x = float(np.mean(np.abs(unw_4x - ref_4x)) / 255.0)

    multi_res_err = float((pixel_err + err_2x + err_4x) / 3.0)
    reproj_score = float((pixel_ncc + grad_ncc + (1.0 - multi_res_err)) / 3.0)

    return {
        "pixel_ncc": pixel_ncc,
        "grad_ncc": grad_ncc,
        "pixel_err": pixel_err,
        "grad_err": grad_err,
        "multi_res_err": multi_res_err,
        "reproj_score": reproj_score
    }

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
            ref_crop = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)

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

                unw_crop = unwarp_candidate_crop(search_img, cand_x, cand_y, cand_scale, cand_theta, out_size=100)
                m = compute_reprojection_metrics(ref_crop, unw_crop)

                dist_gt = math.sqrt((cand_x - gt_x)**2 + (cand_y - gt_y)**2) if gt_found == 1 else 999.0
                is_gt = (dist_gt <= 15.0)

                cand_details.append({
                    "rank_orig": r_idx + 1, "x": cand_x, "y": cand_y,
                    "scale": cand_scale, "theta": cand_theta,
                    "ncc": cand.get("ncc_norm", 0.0), "siamese": cand.get("siamese_sim", 0.0),
                    "fused_orig": cand.get("fused_score", 0.0),
                    "pixel_ncc": m["pixel_ncc"], "grad_ncc": m["grad_ncc"],
                    "pixel_err": m["pixel_err"], "grad_err": m["grad_err"],
                    "multi_res_err": m["multi_res_err"], "reproj_score": m["reproj_score"],
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

def evaluate_strategy_on_records(pair_records, mode="A"):
    results = []
    tau = 0.42

    for rec in pair_records:
        cand_list = rec["candidates"]
        eval_cands = []
        for cand in cand_list:
            if mode == "A": # NCC Only
                score = cand["ncc"]
            elif mode == "B": # NCC + Pixel Reprojection
                score = 0.5 * cand["ncc"] + 0.5 * cand["pixel_ncc"]
            elif mode == "C": # NCC + Gradient Reprojection
                score = 0.5 * cand["ncc"] + 0.5 * cand["grad_ncc"]
            elif mode == "D": # NCC + Multi-Resolution Reprojection
                score = 0.5 * cand["ncc"] + 0.5 * (1.0 - cand["multi_res_err"])
            elif mode == "E": # NCC + Pixel + Gradient
                score = 0.4 * cand["ncc"] + 0.3 * cand["pixel_ncc"] + 0.3 * cand["grad_ncc"]
            elif mode == "F": # NCC + All Reprojection Signals
                score = 0.5 * cand["fused_orig"] + 0.5 * cand["reproj_score"]
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

    modes = [
        ("A", "A_NCC_Only"),
        ("B", "B_NCC_PixelReproj"),
        ("C", "C_NCC_GradReproj"),
        ("D", "D_NCC_MultiResReproj"),
        ("E", "E_NCC_Pixel_Grad"),
        ("F", "F_NCC_AllReproj")
    ]

    print("\n===========================================================================")
    print("PHASE-2 DAY-2 REPROJECTION ABLATION EXPERIMENT (SINGLE-PASS HIGH-SPEED)")
    print("===========================================================================")

    print("Running single-pass feature extraction on DS2 (local_phase2_60gen_200_pairs)...")
    sys.stdout.flush()
    ds2_records = run_single_pass_dataset(engine, "local_phase2_60gen_200_pairs", "phase2_60generator_manifest.csv")

    print("Running single-pass feature extraction on DS1 (local_phase2_200_pairs)...")
    sys.stdout.flush()
    ds1_records = run_single_pass_dataset(engine, "local_phase2_200_pairs", "dataset_manifest.csv")

    ablation_summary = []
    target_pairs = ["pair_006", "pair_066", "pair_186", "pair_116"]

    print("\nEvaluating all 6 ablation strategies...")
    for mode, name in modes:
        m_ds2, res_ds2 = evaluate_strategy_on_records(ds2_records, mode=mode)
        m_ds1, res_ds1 = evaluate_strategy_on_records(ds1_records, mode=mode)

        runtimes = [r["runtime_ms"] for r in res_ds2]
        p90_rt = float(np.percentile(runtimes, 90))

        ablation_summary.append({
            "mode": mode, "strategy": name,
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
        })
        print(f"Strategy {mode} ({name}): DS2 Total = {m_ds2['total_100_score']:.2f} / 100.0 | Loc = {m_ds2['loc_score']:.2f} / 40.0 | DS1 Total = {m_ds1['total_100_score']:.2f} / 100.0")
        sys.stdout.flush()

        if mode == "F":
            print(f"\n===========================================================================")
            print(f"STRATEGY {mode} PERIODIC TARGET FAILURES DEEP-DIVE TABLE")
            print(f"===========================================================================")
            for r in res_ds2:
                if r["pair_id"] in target_pairs:
                    print(f"\n--- Pair: {r['pair_id']} (GT: x={r['gt_x']}, y={r['gt_y']}) ---")
                    for cand in r["eval_candidates"]:
                        gt_tag = "GT LANDMARK" if cand["is_gt"] else "DECOY"
                        print(f"  [{gt_tag}] Orig Rank {cand['rank_orig']}: (x={cand['x']:.1f}, y={cand['y']:.1f}, s={cand['scale']:.2f}, th={cand['theta']:.1f}) | NCC={cand['ncc']:.4f} | PixelErr={cand['pixel_err']:.4f} | GradErr={cand['grad_err']:.4f} | MultiResErr={cand['multi_res_err']:.4f} | ReprojScore={cand['reproj_score']:.4f} | FinalScore={cand['eval_score']:.4f} | DistGT={cand['dist_gt']:.1f}px")
            sys.stdout.flush()

    os.makedirs("phase2/results", exist_ok=True)
    csv_ablation_path = "phase2/results/day2_reprojection_ablation.csv"
    with open(csv_ablation_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mode", "strategy", "ds2_total", "ds2_loc", "ds2_scale", "ds2_theta",
            "ds2_rejection", "ds2_conf", "ds2_eff", "ds2_gen", "ds2_set_a_5px", "ds2_set_b_5px",
            "ds1_total", "ds1_loc", "med_rt", "p90_rt"
        ])
        writer.writeheader()
        writer.writerows(ablation_summary)
    print(f"\nSaved Day-2 reprojection ablation CSV to: {csv_ablation_path}")

    print("\n===========================================================================")
    print("DAY-2 REPROJECTION EXPERIMENT ABLATION SUMMARY")
    print("===========================================================================")
    print(f"Official Baseline Score: 46.77 / 100.0 (DS2 60-Generator)")
    for s in ablation_summary:
        print(f"Strategy {s['mode']} ({s['strategy']}): DS2 Total = {s['ds2_total']} | Loc = {s['ds2_loc']} | Pose = {s['ds2_scale'] + s['ds2_theta']:.2f} | DS1 Total = {s['ds1_total']} | Med RT = {s['med_rt']}ms")

if __name__ == "__main__":
    main()
