#!/usr/bin/env python3
"""
Phase-2 EXP-06: Multi-Scale / Multi-Rotation NCC Transform Consensus Analysis
=============================================================================
Single-Change Experiment testing whether transform consensus (hypothesis stability across
coarse multi-scale/rotation NCC evaluations) can discriminate true GT landmarks from periodic DRAM cell decoys.

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

def compute_transform_consensus(cand_x, cand_y, cand_scale, cand_theta, coarse_hypotheses, spatial_tol=25.0, scale_tol=1.5, rot_tol=3.0):
    """
    Computes transform consensus metrics for candidate (cand_x, cand_y, cand_scale, cand_theta)
    across all coarse search hypotheses.
    """
    supporting = []
    tot_hypotheses = max(1, len(coarse_hypotheses))

    for h in coarse_hypotheses:
        hx, hy = h["x"], h["y"]
        hs, hth = h["scale"], h["theta"]
        hncc = h["coarse_ncc"]

        spatial_dist = math.sqrt((hx - cand_x)**2 + (hy - cand_y)**2)
        scale_diff = abs(hs - cand_scale)
        rot_diff = abs(hth - cand_theta)

        if spatial_dist <= spatial_tol and scale_diff <= scale_tol and rot_diff <= rot_tol:
            supporting.append((spatial_dist, hncc))

    supp_count = len(supporting)
    supp_ratio = supp_count / float(tot_hypotheses)

    if supp_count > 0:
        mean_supp_ncc = float(np.mean([s[1] for s in supporting]))
        # Spatial Gaussian weighting
        w_scores = [s[1] * math.exp(-(s[0]**2) / (2 * (15.0**2))) for s in supporting]
        weighted_supp = float(np.sum(w_scores))
    else:
        mean_supp_ncc = 0.0
        weighted_supp = 0.0

    return supp_count, supp_ratio, mean_supp_ncc, weighted_supp

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

                supp_cnt, supp_ratio, mean_supp, weighted_supp = compute_transform_consensus(
                    cand_x, cand_y, cand_scale, cand_theta, best_coarse
                )

                dist_gt = math.sqrt((cand_x - gt_x)**2 + (cand_y - gt_y)**2) if gt_found == 1 else 999.0
                is_gt = (dist_gt <= 15.0)

                cand_details.append({
                    "rank_orig": r_idx + 1, "x": cand_x, "y": cand_y,
                    "scale": cand_scale, "theta": cand_theta,
                    "ncc_orig": cand.get("ncc_norm", 0.0), "siamese": cand.get("siamese_sim", 0.0),
                    "fused_orig": cand.get("fused_score", 0.0),
                    "supp_count": supp_cnt, "supp_ratio": supp_ratio,
                    "mean_supp_ncc": mean_supp, "weighted_supp": weighted_supp,
                    "is_gt": is_gt, "dist_gt": dist_gt
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

def evaluate_consensus_strategy(pair_records, mode="BASE", lam=0.05):
    results = []
    tau = 0.42

    gt_in_top5_count = 0
    gt_selected_count = 0

    for rec in pair_records:
        cand_list = rec["candidates"]
        eval_cands = []
        for cand in cand_list:
            if mode == "BASE":
                score = cand["fused_orig"]
            elif mode == "EXP06B": # NCC + support count
                score = cand["fused_orig"] + lam * cand["supp_count"]
            elif mode == "EXP06C": # NCC + mean support NCC
                score = cand["fused_orig"] + lam * cand["mean_supp_ncc"]
            elif mode == "EXP06D": # NCC + support ratio
                score = cand["fused_orig"] + lam * cand["supp_ratio"]
            elif mode == "EXP06E": # NCC + weighted transform stability
                score = cand["fused_orig"] + lam * cand["weighted_supp"]
            else:
                score = cand["fused_orig"]

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
    print("PHASE-2 EXP-06: MULTI-SCALE / MULTI-ROTATION NCC CONSENSUS EVALUATION")
    print("===========================================================================")

    print("Running single-pass feature extraction on DS2 (local_phase2_60gen_200_pairs)...")
    sys.stdout.flush()
    ds2_records = run_single_pass_dataset(engine, "local_phase2_60gen_200_pairs", "phase2_60generator_manifest.csv")

    modes = [
        ("BASE", "d_base", 0.00, "Baseline_Fused"),
        ("EXP06B", "supp_count", 0.05, "NCC_plus_SupportCount_lambda0.05"),
        ("EXP06C", "mean_supp_ncc", 0.05, "NCC_plus_MeanSupportNCC_lambda0.05"),
        ("EXP06D", "supp_ratio", 0.05, "NCC_plus_SupportRatio_lambda0.05"),
        ("EXP06E", "weighted_supp", 0.05, "NCC_plus_WeightedTransformStability_lambda0.05")
    ]

    target_pairs = ["pair_006", "pair_066", "pair_186", "pair_116"]
    summary_records = []

    print("\nEvaluating EXP-06 consensus formulations...")
    for exp_id, layer_key, lam, desc in modes:
        m_ds2, res_ds2 = evaluate_consensus_strategy(ds2_records, mode=exp_id, lam=lam)

        runtimes = [r["runtime_ms"] for r in res_ds2]
        p90_rt = float(np.percentile(runtimes, 90))
        p99_rt = float(np.percentile(runtimes, 99))

        tot_present = sum(1 for r in ds2_records if r["gt_found"] == 1)
        top5_recall = (m_ds2["gt_in_top5_count"] / tot_present) * 100.0 if tot_present > 0 else 0.0

        rec = {
            "exp_id": exp_id, "feature_key": layer_key, "lambda": lam, "strategy": desc,
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
            "top5_recall_pct": round(top5_recall, 1),
            "gt_selected_count": m_ds2["gt_selected_count"],
            "med_rt": round(m_ds2["med_rt"], 1),
            "p90_rt": round(p90_rt, 1),
            "p99_rt": round(p99_rt, 1)
        }
        summary_records.append(rec)
        print(f"Strategy {exp_id} ({desc}): DS2 Total = {m_ds2['total_100_score']:.2f} / 100.0 | Loc = {m_ds2['loc_score']:.2f} / 40.0 | GT Selected = {m_ds2['gt_selected_count']}/{tot_present}")
        sys.stdout.flush()

        if exp_id == "EXP06E":
            print(f"\n===========================================================================")
            print(f"EXP-06E TARGET PERIODIC FAILURES DIAGNOSTIC TABLE (Weighted Transform Stability)")
            print(f"===========================================================================")
            for r in res_ds2:
                if r["pair_id"] in target_pairs:
                    print(f"\n--- Pair: {r['pair_id']} (GT: x={r['gt_x']}, y={r['gt_y']}) ---")
                    for cand in r["eval_candidates"]:
                        gt_tag = "GT LANDMARK" if cand["is_gt"] else "DECOY"
                        print(f"  [{gt_tag}] Orig Rank {cand['rank_orig']}: (x={cand['x']:.1f}, y={cand['y']:.1f}) | FusedOrig={cand['fused_orig']:.4f} | SuppCount={cand['supp_count']} | MeanSuppNCC={cand['mean_supp_ncc']:.4f} | EvalScore={cand['eval_score']:.4f} | DistGT={cand['dist_gt']:.1f}px")
            sys.stdout.flush()

    os.makedirs("phase2/results", exist_ok=True)
    csv_path = "phase2/results/exp06_ncc_transform_consensus.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "exp_id", "feature_key", "lambda", "strategy", "ds2_total", "ds2_loc", "ds2_scale", "ds2_theta",
            "ds2_rejection", "ds2_conf", "ds2_eff", "ds2_gen", "ds2_set_a_5px", "ds2_set_b_5px",
            "top5_recall_pct", "gt_selected_count", "med_rt", "p90_rt", "p99_rt"
        ])
        writer.writeheader()
        writer.writerows(summary_records)
    print(f"\nSaved EXP-06 CSV artifact to: {csv_path}")

    # Re-verify SHA256 Hash
    with open(ckpt_path, "rb") as f:
        sha256_hash_after = hashlib.sha256(f.read()).hexdigest()
    print(f"Post-run Checkpoint SHA-256 Hash: {sha256_hash_after}")
    assert sha256_hash_after == "e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c", "SHA-256 Altered!"

if __name__ == "__main__":
    main()
