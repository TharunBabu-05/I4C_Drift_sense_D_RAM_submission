#!/usr/bin/env python3
"""
Phase-2 EXP-04: Top-5 High-Resolution Local NCC Re-Ranking
=========================================================
Single-Change Experiment testing whether high-resolution local NCC re-ranking among the
existing Top-5 candidates improves candidate selection on the official Phase-2 200-pair benchmark.

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

def compute_highres_local_ncc(search_img, ref_img, cx, cy, scale, theta, patch_size=200):
    """
    Computes normalized cross-correlation between a high-resolution 200x200 cropped patch
    around (cx, cy) in search_img and the rotated/scaled 200x200 ref_img patch.
    """
    h_img, w_img = search_img.shape[:2]

    # Rotate reference template if theta != 0
    ref_curr = ref_img
    if abs(theta) > 0.01:
        h_r, w_r = ref_curr.shape[:2]
        M_rot = cv2.getRotationMatrix2D((w_r / 2.0, h_r / 2.0), -theta, 1.0)
        ref_curr = cv2.warpAffine(ref_curr, M_rot, (w_r, h_r), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Resize reference template if scale != 1.0
    if abs(scale - 1.0) > 0.01:
        new_w = max(4, int(round(ref_curr.shape[1] * scale)))
        new_h = max(4, int(round(ref_curr.shape[0] * scale)))
        ref_curr = cv2.resize(ref_curr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    th, tw = ref_curr.shape[:2]
    half_w = tw / 2.0
    half_h = th / 2.0

    x0 = int(round(cx - half_w))
    y0 = int(round(cy - half_h))
    x1 = x0 + tw
    y1 = y0 + th

    if x0 < 0 or y0 < 0 or x1 > w_img or y1 > h_img:
        return 0.0

    crop_search = search_img[y0:y1, x0:x1].astype(np.float32)
    crop_ref = ref_curr.astype(np.float32)

    crop_search_norm = crop_search - np.mean(crop_search)
    crop_ref_norm = crop_ref - np.mean(crop_ref)

    sq_s = np.sum(crop_search_norm ** 2)
    sq_r = np.sum(crop_ref_norm ** 2)

    if sq_s < 1e-5 or sq_r < 1e-5:
        return 0.0

    ncc_val = np.sum(crop_search_norm * crop_ref_norm) / math.sqrt(sq_s * sq_r)
    return float(np.clip(ncc_val, 0.0, 1.0))

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

                local_ncc = compute_highres_local_ncc(
                    search_img, ref_img, cand_x, cand_y, cand_scale, cand_theta
                )

                dist_gt = math.sqrt((cand_x - gt_x)**2 + (cand_y - gt_y)**2) if gt_found == 1 else 999.0
                is_gt = (dist_gt <= 15.0)

                cand_details.append({
                    "rank_orig": r_idx + 1, "x": cand_x, "y": cand_y,
                    "scale": cand_scale, "theta": cand_theta,
                    "ncc_orig": cand.get("ncc_norm", 0.0), "siamese": cand.get("siamese_sim", 0.0),
                    "fused_orig": cand.get("fused_score", 0.0),
                    "local_ncc": local_ncc, "is_gt": is_gt, "dist_gt": dist_gt
                })

            gt_rank_top5 = None
            for c in cand_details:
                if c["is_gt"]:
                    gt_rank_top5 = c["rank_orig"]
                    break

            pair_records.append({
                "pair_id": pair_id, "set": set_name, "gen_id": gen_id,
                "gt_x": gt_x, "gt_y": gt_y, "gt_theta": gt_theta, "gt_scale": gt_scale, "gt_found": gt_found,
                "runtime_ms": runtime_ms, "candidates": cand_details,
                "gt_in_top5": gt_rank_top5 is not None, "gt_rank_top5": gt_rank_top5
            })

            gc.collect()

    return pair_records

def evaluate_lambda_strategy(pair_records, lam=0.05):
    results = []
    tau = 0.42

    gt_in_top5_count = 0
    gt_selected_count = 0

    for rec in pair_records:
        cand_list = rec["candidates"]
        eval_cands = []
        for cand in cand_list:
            score = cand["fused_orig"] + lam * cand["local_ncc"]
            eval_cands.append({**cand, "eval_score": score})

        eval_cands.sort(key=lambda c: -c["eval_score"])
        selected = eval_cands[0]

        if rec["gt_in_top5"]:
            gt_in_top5_count += 1
            if selected["is_gt"]:
                gt_selected_count += 1

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
    metrics["gt_in_top5_count"] = gt_in_top5_count
    metrics["gt_selected_count"] = gt_selected_count
    return metrics, results

def main():
    ckpt_path = "phase2_checkpoints/best_model_level1.pth"
    with open(ckpt_path, "rb") as f:
        sha256_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"Original Checkpoint SHA-256 Hash: {sha256_hash}")
    assert sha256_hash == "e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c", "SHA-256 Mismatch!"

    engine = Phase2InferenceEngine(checkpoint_path="best_model_level1.pth", device="cpu")

    print("\n===========================================================================")
    print("PHASE-2 EXP-04: TOP-5 HIGH-RESOLUTION LOCAL NCC RE-RANKING EVALUATION")
    print("===========================================================================")

    print("Running feature extraction on DS2 (local_phase2_60gen_200_pairs)...")
    sys.stdout.flush()
    ds2_records = run_single_pass_dataset(engine, "local_phase2_60gen_200_pairs", "phase2_60generator_manifest.csv")

    print("Running feature extraction on DS1 (local_phase2_200_pairs)...")
    sys.stdout.flush()
    ds1_records = run_single_pass_dataset(engine, "local_phase2_200_pairs", "dataset_manifest.csv")

    lambdas = [0.0, 0.05, 0.10, 0.20]
    summary_records = []

    print("\nEvaluating lambda sweeps on DS2 and DS1...")
    target_pairs = ["pair_006", "pair_066", "pair_186", "pair_116"]

    for lam in lambdas:
        m_ds2, res_ds2 = evaluate_lambda_strategy(ds2_records, lam=lam)
        m_ds1, res_ds1 = evaluate_lambda_strategy(ds1_records, lam=lam)

        runtimes = [r["runtime_ms"] for r in res_ds2]
        p90_rt = float(np.percentile(runtimes, 90))
        p99_rt = float(np.percentile(runtimes, 99))

        tot_present_ds2 = sum(1 for r in ds2_records if r["gt_found"] == 1)
        top5_recall_pct = (m_ds2["gt_in_top5_count"] / tot_present_ds2) * 100.0 if tot_present_ds2 > 0 else 0.0

        rec = {
            "lambda": lam, "strategy": f"EXP04_lambda_{lam}",
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
            "top5_recall_pct": round(top5_recall_pct, 1),
            "gt_selected_count": m_ds2["gt_selected_count"],
            "med_rt": round(m_ds2["med_rt"], 1),
            "p90_rt": round(p90_rt, 1),
            "p99_rt": round(p99_rt, 1)
        }
        summary_records.append(rec)
        print(f"Lambda = {lam:.2f}: DS2 Total = {m_ds2['total_100_score']:.2f} / 100.0 | Loc = {m_ds2['loc_score']:.2f} / 40.0 | Top5 Recall = {top5_recall_pct:.1f}% | GT Selected = {m_ds2['gt_selected_count']}/{tot_present_ds2}")
        sys.stdout.flush()

        if lam == 0.05:
            print(f"\n===========================================================================")
            print(f"TARGET PERIODIC FAILURES DIAGNOSTIC TABLE (Lambda = 0.05)")
            print(f"===========================================================================")
            for r in res_ds2:
                if r["pair_id"] in target_pairs:
                    print(f"\n--- Pair: {r['pair_id']} (GT: x={r['gt_x']}, y={r['gt_y']}) ---")
                    for cand in r["eval_candidates"]:
                        gt_tag = "GT LANDMARK" if cand["is_gt"] else "DECOY"
                        print(f"  [{gt_tag}] Orig Rank {cand['rank_orig']}: (x={cand['x']:.1f}, y={cand['y']:.1f}) | FusedGlobal={cand['fused_orig']:.4f} | LocalNCC={cand['local_ncc']:.4f} | FinalEvalScore={cand['eval_score']:.4f} | DistGT={cand['dist_gt']:.1f}px")
            sys.stdout.flush()

    os.makedirs("phase2/results", exist_ok=True)
    csv_path = "phase2/results/exp04_top5_local_ncc.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "lambda", "strategy", "ds2_total", "ds2_loc", "ds2_scale", "ds2_theta",
            "ds2_rejection", "ds2_conf", "ds2_eff", "ds2_gen", "ds2_set_a_5px", "ds2_set_b_5px",
            "ds1_total", "ds1_loc", "top5_recall_pct", "gt_selected_count", "med_rt", "p90_rt", "p99_rt"
        ])
        writer.writeheader()
        writer.writerows(summary_records)
    print(f"\nSaved EXP-04 CSV artifact to: {csv_path}")

    # Re-verify SHA256 Hash
    with open(ckpt_path, "rb") as f:
        sha256_hash_after = hashlib.sha256(f.read()).hexdigest()
    print(f"Post-run Checkpoint SHA-256 Hash: {sha256_hash_after}")
    assert sha256_hash_after == "e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c", "SHA-256 Altered!"

if __name__ == "__main__":
    main()
