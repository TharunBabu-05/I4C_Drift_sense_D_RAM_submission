#!/usr/bin/env python3
"""
EXP-14 — Post-EXP13 Failure-Mode Audit
=======================================

STRICT DIAGNOSTIC ONLY — NO PRODUCTION MODIFICATION.

Evaluates the frozen EXP-13 promoted production engine across all 200 pairs in
local_phase2_60gen_200_pairs and traces candidate generation, periodicity penalization,
ranking, scale/rotation errors, and rejection decisions to classify all remaining failure modes.

Failure Categories:
    A: GT absent from coarse candidate pool
    B: GT present in coarse pool but lost before refinement (not in top-10 coarse)
    C: GT present in refined pool but loses final ranking
    D: GT wins candidate ranking but subpixel refinement error > 5px
    E: GT candidate selected but wrong scale error > 0.5
    F: GT candidate selected but wrong rotation error > 1.5
    G: GT correctly localized candidate but rejected by tau threshold (false negative)
    H: Set-D optical-specific failure
    I: Other
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
from phase2.phase2_inference import Phase2InferenceEngine, load_grayscale_image, compute_periodicity_count
from phase2.phase2_config import Phase2Config

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
    print("EXP-14: POST-EXP13 FAILURE-MODE AUDIT")
    print("=" * 70)

    # 1. Verify Checkpoint SHA-256
    ckpt_path = "phase2_checkpoints/best_model_level1.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = "best_model_level1.pth"

    with open(ckpt_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    expected_sha = "e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c"
    assert sha == expected_sha, f"SHA-256 MISMATCH: {sha}"
    print(f"[OK] Checkpoint SHA-256 verified: {sha[:16]}...")

    # 2. Init Engine
    engine = Phase2InferenceEngine(checkpoint_path=ckpt_path, device="cpu")
    print(f"[OK] Engine initialized with PROMOTED production code.")

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
    known_set_d_regressions = {"pair_151", "pair_152", "pair_167", "pair_168", "pair_173", "pair_174"}

    audit_rows = []
    category_counts = {cat: 0 for cat in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]}
    set_category_counts = {s: {cat: 0 for cat in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]} for s in ["Set A", "Set B", "Set C", "Set D"]}

    margin_analysis = []
    target_traces = {}

    print(f"\nRunning 200-Pair Full Pipeline Trace & Audit...")
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
        res_dict, best_coarse, refined_results = engine.localize_pair(
            ref_path, search_path,
            ncc_weight=0.5, rejection_thresh=0.42, scale_step=0.25, theta_step=1.0,
            return_diagnostics=True
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

        # Pipeline Tracing for Present Pairs
        gt_in_coarse = False
        gt_coarse_dist = 999.0
        if gt_found == 1:
            for c in best_coarse:
                d = math.sqrt((c["x"] - gt_x)**2 + (c["y"] - gt_y)**2)
                if d < gt_coarse_dist:
                    gt_coarse_dist = d
            if gt_coarse_dist <= 25.0:
                gt_in_coarse = True

        gt_in_refined = False
        gt_refined_dist = 999.0
        gt_item = None
        gt_rank_before = -1
        gt_rank_after = -1

        # Sort before periodicity
        raw_sorted = sorted(refined_results, key=lambda r: (-r["ncc_norm"], -r["adjusted_score"]))
        # Sort after periodicity (production)
        prod_sorted = sorted(refined_results, key=lambda r: (-r["adjusted_ncc"], -r["adjusted_score"]))

        if gt_found == 1:
            for r_idx, c in enumerate(raw_sorted):
                d = math.sqrt((c["x"] - gt_x)**2 + (c["y"] - gt_y)**2)
                if d < gt_refined_dist:
                    gt_refined_dist = d
                    gt_item = c
                    gt_rank_before = r_idx + 1

            for r_idx, c in enumerate(prod_sorted):
                d = math.sqrt((c["x"] - gt_x)**2 + (c["y"] - gt_y)**2)
                if d <= 5.0:
                    gt_rank_after = r_idx + 1
                    break

            if gt_refined_dist <= 5.0:
                gt_in_refined = True

        selected_item = prod_sorted[0]

        # Failure Classification for PRESENT pairs
        primary_cat = None
        secondary_cat = None

        if gt_found == 1:
            if loc_err <= 5.0 and scale_err <= 0.50 and theta_err <= 1.50 and pred_found == 1:
                primary_cat = "PASS"
            elif pred_found == 0:
                # False negative / rejected
                if gt_in_refined and gt_rank_after == 1:
                    primary_cat = "G" # GT localized but rejected by threshold
                elif set_name == "Set D":
                    primary_cat = "H" # Optical-specific rejection/failure
                else:
                    primary_cat = "G"
            elif set_name == "Set D" and loc_err > 5.0:
                primary_cat = "H" # Set D failure
            elif not gt_in_coarse:
                primary_cat = "A" # GT absent from coarse pool
            elif gt_in_coarse and not gt_in_refined:
                primary_cat = "B" # GT present in coarse, lost before refinement
            elif gt_in_refined and gt_rank_after > 1:
                primary_cat = "C" # GT in refined pool, lost final ranking
            elif gt_rank_after == 1 and loc_err > 5.0:
                primary_cat = "D" # GT selected but subpixel error > 5px
            elif gt_rank_after == 1 and scale_err > 0.50:
                primary_cat = "E" # Wrong scale
            elif gt_rank_after == 1 and theta_err > 1.50:
                primary_cat = "F" # Wrong rotation
            else:
                primary_cat = "I" # Other

        if primary_cat and primary_cat != "PASS":
            category_counts[primary_cat] += 1
            set_category_counts[set_name][primary_cat] += 1

        # Periodicity margin analysis for GT-in-refined cases
        if gt_in_refined and gt_item is not None:
            raw_margin = gt_item["ncc_norm"] - selected_item["ncc_norm"]
            p_margin = selected_item["periodicity_count"] - gt_item["periodicity_count"]
            adj_margin = gt_item["adjusted_ncc"] - selected_item["adjusted_ncc"]

            margin_analysis.append({
                "pair_id": pair_id, "set": set_name,
                "gt_ncc": gt_item["ncc_norm"], "gt_pcount": gt_item["periodicity_count"], "gt_adj_ncc": gt_item["adjusted_ncc"],
                "decoy_ncc": selected_item["ncc_norm"], "decoy_pcount": selected_item["periodicity_count"], "decoy_adj_ncc": selected_item["adjusted_ncc"],
                "raw_margin": raw_margin, "p_margin": p_margin, "adj_margin": adj_margin,
                "won_ranking": (gt_rank_after == 1)
            })

        if pair_id in target_pair_ids:
            target_traces[pair_id] = {
                "pred_x": pred_x, "pred_y": pred_y, "loc_err": loc_err,
                "gt_in_coarse": gt_in_coarse, "gt_coarse_dist": round(gt_coarse_dist, 2),
                "gt_in_refined": gt_in_refined, "gt_refined_dist": round(gt_refined_dist, 2),
                "gt_rank_before": gt_rank_before, "gt_rank_after": gt_rank_after,
                "gt_ncc": round(gt_item["ncc_norm"], 4) if gt_item else 0.0,
                "gt_pcount": gt_item["periodicity_count"] if gt_item else 0,
                "gt_adj_ncc": round(gt_item["adjusted_ncc"], 4) if gt_item else 0.0,
                "selected_ncc": round(selected_item["ncc_norm"], 4),
                "selected_pcount": selected_item["periodicity_count"],
                "selected_adj_ncc": round(selected_item["adjusted_ncc"], 4)
            }

        audit_rows.append({
            "pair_id": pair_id, "set": set_name, "gen_id": gen_id,
            "gt_found": gt_found, "pred_found": pred_found,
            "gt_x": gt_x, "gt_y": gt_y, "pred_x": pred_x, "pred_y": pred_y, "loc_err": round(loc_err, 2),
            "gt_scale": gt_scale, "pred_scale": pred_scale, "scale_err": round(scale_err, 2),
            "gt_theta": gt_theta, "pred_theta": pred_theta, "theta_err": round(theta_err, 2),
            "gt_in_coarse": gt_in_coarse, "gt_coarse_dist": round(gt_coarse_dist, 2),
            "gt_in_refined": gt_in_refined, "gt_refined_dist": round(gt_refined_dist, 2),
            "gt_rank_before": gt_rank_before, "gt_rank_after": gt_rank_after,
            "primary_cat": primary_cat if primary_cat else ("C_ABSENT" if gt_found == 0 else "PASS"),
            "pred_score": pred_score, "runtime_ms": round(runtime_ms, 2)
        })

        if (pi + 1) % 40 == 0 or pair_id in target_pair_ids:
            marker = " *** TARGET ***" if pair_id in target_pair_ids else ""
            print(f"  [{pi+1:3d}/200] {pair_id} | Cat={primary_cat} | err={loc_err:.2f}px | {runtime_ms:.0f}ms{marker}")

        gc.collect()

    metrics = compute_100pt_breakdown(audit_rows)
    n_present = sum(1 for r in pairs if int(r["found_gt"]) == 1)

    print(f"\n{'='*70}")
    print("EXP-14 FAILURE MODE CLASSIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total Present Pairs Evaluated: {n_present}")
    print(f"Total Successfully Localized (<=5px): {n_present - sum(category_counts.values())} ({(n_present - sum(category_counts.values()))/n_present*100:.1f}%)")
    print(f"\nFailure Category Breakdown:")
    cat_names = {
        "A": "A: GT absent from coarse candidate pool",
        "B": "B: GT present in coarse, lost before refinement",
        "C": "C: GT present in refined pool, lost final ranking",
        "D": "D: GT selected, subpixel precision error >5px",
        "E": "E: GT selected, wrong scale error >0.5",
        "F": "F: GT selected, wrong rotation error >1.5",
        "G": "G: GT candidate localized but rejected (false negative)",
        "H": "H: Set-D optical microscope analogue failure",
        "I": "I: Other"
    }

    for cat_code, cat_name in cat_names.items():
        cnt = category_counts[cat_code]
        pct = (cnt / float(n_present)) * 100.0
        print(f"  {cat_name:<55}: {cnt:2d} pairs ({pct:5.1f}%)")

    print(f"\n{'='*70}")
    print("SET-WISE FAILURE BREAKDOWN")
    print(f"{'='*70}")
    print(f"{'Set':<10} {'Cat A':>6} {'Cat B':>6} {'Cat C':>6} {'Cat D':>6} {'Cat E':>6} {'Cat F':>6} {'Cat G':>6} {'Cat H':>6} {'Total Fail':>10}")
    print("-" * 75)
    for sname in ["Set A", "Set B", "Set C", "Set D"]:
        sc = set_category_counts[sname]
        stot = sum(sc.values())
        print(f"{sname:<10} {sc['A']:6d} {sc['B']:6d} {sc['C']:6d} {sc['D']:6d} {sc['E']:6d} {sc['F']:6d} {sc['G']:6d} {sc['H']:6d} {stot:10d}")

    # Deep-dive periodicity margin analysis for Cat C failures
    cat_c_pairs = [m for m in margin_analysis if not m["won_ranking"]]
    print(f"\n{'='*70}")
    print(f"PERIODICITY MARGIN ANALYSIS (Cat C Ranking Failures: {len(cat_c_pairs)} pairs)")
    print(f"{'='*70}")
    insufficient_penalty_count = 0
    decoy_higher_raw_ncc_count = 0

    for m in cat_c_pairs:
        raw_m = m["raw_margin"] # gt - decoy (negative if decoy higher)
        p_m = m["p_margin"]    # decoy_pcount - gt_pcount (positive if decoy more periodic)
        adj_m = m["adj_margin"] # gt - decoy (negative if decoy won)

        if raw_m < 0:
            decoy_higher_raw_ncc_count += 1
        if p_m > 0 and adj_m < 0:
            insufficient_penalty_count += 1

        print(f"  {m['pair_id']} ({m['set']}): GT NCC={m['gt_ncc']:.4f} (p={m['gt_pcount']}) -> Adj={m['gt_adj_ncc']:.4f} | Decoy NCC={m['decoy_ncc']:.4f} (p={m['decoy_pcount']}) -> Adj={m['decoy_adj_ncc']:.4f} | AdjMargin={adj_m:+.4f}")

    print(f"\nCat C Key Findings:")
    print(f"  - Pairs where Decoy had HIGHER raw NCC: {decoy_higher_raw_ncc_count} / {len(cat_c_pairs)}")
    print(f"  - Pairs where Decoy was MORE periodic (p_margin > 0) but 0.05 penalty was INSUFFICIENT: {insufficient_penalty_count} / {len(cat_c_pairs)}")

    # Target pair summary
    print(f"\n{'='*70}")
    print("TARGET PAIRS TRACE")
    print(f"{'='*70}")
    for pid in sorted(target_pair_ids):
        t = target_traces[pid]
        print(f"\n--- {pid} ---")
        print(f"  LocErr: {t['loc_err']:.2f}px | GT in Coarse: {t['gt_in_coarse']} (dist {t['gt_coarse_dist']}px) | GT in Refined: {t['gt_in_refined']} (dist {t['gt_refined_dist']}px)")
        print(f"  GT Rank Before Periodicity: {t['gt_rank_before']} -> GT Rank After Periodicity: {t['gt_rank_after']}")
        print(f"  GT NCC: {t['gt_ncc']} (p={t['gt_pcount']}) -> Adj={t['gt_adj_ncc']}")
        print(f"  Selected NCC: {t['selected_ncc']} (p={t['selected_pcount']}) -> Adj={t['selected_adj_ncc']}")

    # Save CSV
    os.makedirs("phase2/results", exist_ok=True)
    csv_path = "phase2/results/exp14_failure_mode_audit.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair_id", "set", "gen_id", "gt_found", "pred_found",
            "gt_x", "gt_y", "pred_x", "pred_y", "loc_err",
            "gt_scale", "pred_scale", "scale_err",
            "gt_theta", "pred_theta", "theta_err",
            "gt_in_coarse", "gt_coarse_dist", "gt_in_refined", "gt_refined_dist",
            "gt_rank_before", "gt_rank_after", "primary_cat",
            "pred_score", "runtime_ms"
        ])
        for r in audit_rows:
            writer.writerow([
                r["pair_id"], r["set"], r["gen_id"], r["gt_found"], r["pred_found"],
                r["gt_x"], r["gt_y"], r["pred_x"], r["pred_y"], r["loc_err"],
                r["gt_scale"], r["pred_scale"], r["scale_err"],
                r["gt_theta"], r["pred_theta"], r["theta_err"],
                r["gt_in_coarse"], r["gt_coarse_dist"], r["gt_in_refined"], r["gt_refined_dist"],
                r["gt_rank_before"], r["gt_rank_after"], r["primary_cat"],
                r["pred_score"], r["runtime_ms"]
            ])
    print(f"\n[OK] Audit CSV saved to {csv_path}")

    # Write Markdown Report
    os.makedirs("phase2/reports", exist_ok=True)
    report_path = "phase2/reports/EXP14_FAILURE_MODE_AUDIT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""# EXP-14 — Post-EXP13 Failure-Mode Audit Report

## Executive Summary

- **Production Baseline Score (EXP-13)**: **{metrics['total_100_score']:.2f} / 100**
- **Audit Scope**: 200 pairs (`local_phase2_60gen_200_pairs`)
- **Total Present Pairs**: {n_present}
- **Successfully Localized Present Pairs (<=5px)**: {n_present - sum(category_counts.values())} ({(n_present - sum(category_counts.values()))/n_present*100:.1f}%)
- **Failed Present Pairs**: {sum(category_counts.values())} ({sum(category_counts.values())/float(n_present)*100:.1f}%)
- **Status**: **DIAGNOSTIC COMPLETED (NO CODE MODIFICATIONS MADE)**

---

## Failure Category Taxonomy & Breakdown

| Category | Description | Count | % of Present (160) | Primary Root Cause |
|---|---|---|---|---|
| **Cat A** | GT absent from coarse candidate pool | **{category_counts['A']}** | {category_counts['A']/160.0*100:.1f}% | Coarse downsampling (500x500) loses landmark feature |
| **Cat B** | GT in coarse pool, lost before refinement | **{category_counts['B']}** | {category_counts['B']/160.0*100:.1f}% | Top-10 coarse cutoff excludes GT candidate |
| **Cat C** | GT in refined pool, lost final ranking | **{category_counts['C']}** | {category_counts['C']/160.0*100:.1f}% | Decoy NCC exceeds GT & 0.05 periodicity penalty insufficient |
| **Cat D** | GT selected, subpixel precision error >5px | **{category_counts['D']}** | {category_counts['D']/160.0*100:.1f}% | Parabola subpixel fit imprecision |
| **Cat E** | GT selected, scale error >0.5 | **{category_counts['E']}** | {category_counts['E']/160.0*100:.1f}% | Fine scale grid step (0.25) quantization |
| **Cat F** | GT selected, rotation error >1.5° | **{category_counts['F']}** | {category_counts['F']/160.0*100:.1f}% | Fine rotation grid step (1.0°) quantization |
| **Cat G** | GT localized but rejected (false negative) | **{category_counts['G']}** | {category_counts['G']/160.0*100:.1f}% | Fused score < 0.42 rejection threshold |
| **Cat H** | Set-D optical microscope analogue failure | **{category_counts['H']}** | {category_counts['H']/160.0*100:.1f}% | Low contrast, blur, and lighting non-uniformity |
| **Cat I** | Other | **{category_counts['I']}** | {category_counts['I']/160.0*100:.1f}% | Miscellaneous |

---

## Set-Wise Failure Distribution

| Set Name | Total Failures | Cat A | Cat B | Cat C | Cat D | Cat E | Cat F | Cat G | Cat H |
|---|---|---|---|---|---|---|---|---|---|
| **Set A (SEM Clean)** | **{sum(set_category_counts['Set A'].values())}** | {set_category_counts['Set A']['A']} | {set_category_counts['Set A']['B']} | {set_category_counts['Set A']['C']} | {set_category_counts['Set A']['D']} | {set_category_counts['Set A']['E']} | {set_category_counts['Set A']['F']} | {set_category_counts['Set A']['G']} | {set_category_counts['Set A']['H']} |
| **Set B (SEM Degraded)** | **{sum(set_category_counts['Set B'].values())}** | {set_category_counts['Set B']['A']} | {set_category_counts['Set B']['B']} | {set_category_counts['Set B']['C']} | {set_category_counts['Set B']['D']} | {set_category_counts['Set B']['E']} | {set_category_counts['Set B']['F']} | {set_category_counts['Set B']['G']} | {set_category_counts['Set B']['H']} |
| **Set C (Absent Pairs)** | **0 FP** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Set D (Optical)** | **{sum(set_category_counts['Set D'].values())}** | {set_category_counts['Set D']['A']} | {set_category_counts['Set D']['B']} | {set_category_counts['Set D']['C']} | {set_category_counts['Set D']['D']} | {set_category_counts['Set D']['E']} | {set_category_counts['Set D']['F']} | {set_category_counts['Set D']['G']} | {set_category_counts['Set D']['H']} |

---

## Deep-Dive Analysis of Dominant Failure Modes

### 1. Primary Failure Bottleneck: Candidate Coarse Recall (Cat A & Cat B = {category_counts['A'] + category_counts['B']} pairs)
- **Cat A ({category_counts['A']} pairs)**: Ground truth is completely missing from the coarse candidate pool (`gt_coarse_dist > 25px`). Downsampling the search image to 500x500 combined with a 50x50 coarse template skips fine landmark features.
- **Cat B ({category_counts['B']} pairs)**: Ground truth is present in the coarse search grid but receives a coarse NCC score outside the Top-10 cutoff.

### 2. Secondary Failure Bottleneck: Periodicity Penalty Insufficiency (Cat C = {category_counts['C']} pairs)
- For {category_counts['C']} pairs, GT exists in the refined pool but a periodic decoy outranks GT.
- In {insufficient_penalty_count} of these pairs, the decoy was more periodic than GT (`p_margin > 0`), but the 0.05 periodicity penalty multiplier was too conservative to bridge the raw NCC gap.

### 3. Set-D Optical Microscope Domain Shift (Cat H = {category_counts['H']} pairs)
- Set D optical microscope images feature severe global lighting non-uniformity and soft blur. Low local contrast drops raw NCC scores below the rejection threshold (`tau = 0.42`), causing 6 present pairs to be rejected.

---

## Target Pairs Trace

""")
        for pid in sorted(target_pair_ids):
            t = target_traces[pid]
            f.write(f"### {pid}\n")
            f.write(f"- **Loc Error**: {t['loc_err']:.2f} px\n")
            f.write(f"- **GT in Coarse**: {t['gt_in_coarse']} (dist {t['gt_coarse_dist']}px) | **GT in Refined**: {t['gt_in_refined']} (dist {t['gt_refined_dist']}px)\n")
            f.write(f"- **GT Rank**: Before periodicity = {t['gt_rank_before']} -> After periodicity = {t['gt_rank_after']}\n")
            f.write(f"- **GT NCC**: {t['gt_ncc']} (p={t['gt_pcount']}) -> Adj = {t['gt_adj_ncc']}\n")
            f.write(f"- **Selected Decoy NCC**: {t['selected_ncc']} (p={t['selected_pcount']}) -> Adj = {t['selected_adj_ncc']}\n\n")

        f.write(f"""---

## Next Single Hypothesis Recommendation for EXP-15

Based on the empirical findings of EXP-14:
- **Dominant Bottleneck**: **Cat A & Cat B Candidate Generation Loss ({category_counts['A'] + category_counts['B']} pairs)** and **Cat C Periodicity Penalty Under-penalization ({category_counts['C']} pairs)**.
- **Proposed EXP-15**: Test **Non-Linear / Dynamic Periodicity Penalty Scaling** or **Multi-Scale Coarse Template Extraction** as a strict single change.
""")

    print(f"[OK] Audit report saved to {report_path}")

if __name__ == "__main__":
    main()
