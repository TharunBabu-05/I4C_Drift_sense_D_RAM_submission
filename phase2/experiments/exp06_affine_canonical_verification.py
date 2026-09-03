#!/usr/bin/env python3
"""
Phase-2 EXP-06: Post-Top-5 Affine-Canonical Candidate Verification Analysis
=============================================================================
Single-Change Experiment testing whether fine candidate-local affine canonicalization and fine pose verification
(scale +/- 0.005, rotation +/- 0.25 deg) can discriminate true GT landmarks from periodic DRAM cell decoys.

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

def verify_candidate_affine_canonical(search_img, ref_template, cand_x, cand_y, cand_scale, cand_theta):
    """
    Performs fine candidate-local affine canonicalization and fine pose verification.
    1. Extracts candidate local patch.
    2. Sweeps fine scale (cand_scale +/- 0.005) and fine rotation (cand_theta +/- 0.25 deg).
    3. Computes canonical local normalized cross correlation against clean 100x100 reference template.
    """
    h_s, w_s = search_img.shape[:2]
    fine_scales = [cand_scale - 0.005, cand_scale, cand_scale + 0.005]
    fine_thetas = [cand_theta - 0.25, cand_theta, cand_theta + 0.25]

    best_local_ncc = -1.0
    best_scale = cand_scale
    best_theta = cand_theta

    for f_sc in fine_scales:
        if f_sc < 7.5 or f_sc > 12.5:
            continue
        p_size = int(round(1000.0 / f_sc))
        ref_s = cv2.resize(ref_template, (p_size, p_size), interpolation=cv2.INTER_LINEAR)

        for f_th in fine_thetas:
            if abs(f_th) > 5.5:
                continue
            if abs(f_th) > 0.01:
                M_r = cv2.getRotationMatrix2D((p_size / 2.0, p_size / 2.0), f_th, 1.0)
                ref_r = cv2.warpAffine(ref_s, M_r, (p_size, p_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            else:
                ref_r = ref_s

            # Extract window around candidate
            win_r = int(round(p_size / 2.0 + 40))
            rx0, rx1 = max(0, int(round(cand_x - win_r))), min(w_s, int(round(cand_x + win_r)))
            ry0, ry1 = max(0, int(round(cand_y - win_r))), min(h_s, int(round(cand_y + win_r)))

            search_sub = search_img[ry0:ry1, rx0:rx1]
            if search_sub.shape[0] < ref_r.shape[0] or search_sub.shape[1] < ref_r.shape[1]:
                continue

            res_local = cv2.matchTemplate(search_sub, ref_r, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res_local)

            if max_val > best_local_ncc:
                best_local_ncc = float(max_val)
                best_scale = f_sc
                best_theta = f_th

    return max(0.0, best_local_ncc), best_scale, best_theta

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

            if ref_img.shape != (100, 100):
                ref_template = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
            else:
                ref_template = ref_img.copy()

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

                canon_ncc, fine_scale, fine_theta = verify_candidate_affine_canonical(
                    search_img, ref_template, cand_x, cand_y, cand_scale, cand_theta
                )

                dist_gt = math.sqrt((cand_x - gt_x)**2 + (cand_y - gt_y)**2) if gt_found == 1 else 999.0
                is_gt = (dist_gt <= 15.0)

                cand_details.append({
                    "rank_orig": r_idx + 1, "x": cand_x, "y": cand_y,
                    "scale": cand_scale, "theta": cand_theta,
                    "fine_scale": fine_scale, "fine_theta": fine_theta,
                    "ncc_orig": cand.get("ncc_norm", 0.0), "siamese": cand.get("siamese_sim", 0.0),
                    "fused_orig": cand.get("fused_score", 0.0),
                    "canonical_ncc": canon_ncc,
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

def evaluate_canonical_strategy(pair_records, mode="BASE", w_verif=0.20):
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
            elif mode == "EXP06_CANONICAL_NCC":
                score = cand["canonical_ncc"]
            elif mode == "EXP06_FUSED_CANONICAL":
                score = (1.0 - w_verif) * cand["fused_orig"] + w_verif * cand["canonical_ncc"]
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
        pred_scale = selected["fine_scale"] if mode != "BASE" else selected["scale"]
        pred_theta = selected["fine_theta"] if mode != "BASE" else selected["theta"]
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
    print("PHASE-2 EXP-06: AFFINE-CANONICAL CANDIDATE VERIFICATION EVALUATION")
    print("===========================================================================")

    print("Running single-pass feature extraction on DS2 (local_phase2_60gen_200_pairs)...")
    sys.stdout.flush()
    ds2_records = run_single_pass_dataset(engine, "local_phase2_60gen_200_pairs", "phase2_60generator_manifest.csv")

    modes = [
        ("BASE", "fused_orig", 0.00, "Baseline_Fused"),
        ("EXP06_CANONICAL_NCC", "canonical_ncc", 1.00, "Canonical_Local_NCC_Only"),
        ("EXP06_W10", "canonical_ncc", 0.10, "Fused_80_Canonical_10"),
        ("EXP06_W20", "canonical_ncc", 0.20, "Fused_80_Canonical_20"),
        ("EXP06_W30", "canonical_ncc", 0.30, "Fused_70_Canonical_30")
    ]

    target_pairs = ["pair_006", "pair_066", "pair_186", "pair_116"]
    summary_records = []

    print("\nEvaluating EXP-06 affine canonical verification formulations...")
    for exp_id, layer_key, w_verif, desc in modes:
        m_ds2, res_ds2 = evaluate_canonical_strategy(ds2_records, mode=exp_id, w_verif=w_verif)

        runtimes = [r["runtime_ms"] for r in res_ds2]
        p90_rt = float(np.percentile(runtimes, 90))
        p99_rt = float(np.percentile(runtimes, 99))

        tot_present = sum(1 for r in ds2_records if r["gt_found"] == 1)
        top5_recall = (m_ds2["gt_in_top5_count"] / tot_present) * 100.0 if tot_present > 0 else 0.0

        rec = {
            "exp_id": exp_id, "feature_key": layer_key, "w_verif": w_verif, "strategy": desc,
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

        if exp_id == "EXP06_W20":
            print(f"\n===========================================================================")
            print(f"EXP-06_W20 TARGET PERIODIC FAILURES DIAGNOSTIC TABLE (W_verif = 0.20)")
            print(f"===========================================================================")
            for r in res_ds2:
                if r["pair_id"] in target_pairs:
                    print(f"\n--- Pair: {r['pair_id']} (GT: x={r['gt_x']}, y={r['gt_y']}) ---")
                    for cand in r["eval_candidates"]:
                        gt_tag = "GT LANDMARK" if cand["is_gt"] else "DECOY"
                        print(f"  [{gt_tag}] Orig Rank {cand['rank_orig']}: (x={cand['x']:.1f}, y={cand['y']:.1f}) | FusedOrig={cand['fused_orig']:.4f} | CanonNCC={cand['canonical_ncc']:.4f} | EvalScore={cand['eval_score']:.4f} | DistGT={cand['dist_gt']:.1f}px")
            sys.stdout.flush()

    os.makedirs("phase2/results", exist_ok=True)
    csv_path = "phase2/results/exp06_affine_canonical_verification.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "exp_id", "feature_key", "w_verif", "strategy", "ds2_total", "ds2_loc", "ds2_scale", "ds2_theta",
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
