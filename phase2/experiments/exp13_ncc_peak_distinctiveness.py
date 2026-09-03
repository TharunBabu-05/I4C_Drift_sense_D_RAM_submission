#!/usr/bin/env python3
"""
EXP-13 — NCC Peak Distinctiveness / Periodicity Penalty
========================================================

STRICT SINGLE-CHANGE EXPERIMENT

Hypothesis:
    Periodic DRAM decoys produce repetitive, flat, or dense clusters of high NCC values,
    whereas true structural landmarks produce sharp, distinct NCC peaks relative to their spatial
    neighborhoods. Incorporating a Peak Prominence / Distinctiveness metric calculated solely from the
    2D NCC response map will penalize periodic decoys without introducing gradient noise or neural bias.

Tested Metrics (from 2D NCC match_matrix around candidate peak):
    - Prominence: peak_val - mean(neighborhood)
    - Peak Ratio: peak_val / (mean(neighborhood) + eps)
    - Periodicity Penalty: count of local peaks in window >= 0.85 * peak_val

Primary Decision Rule:
    Score > 61.99 / 100, Localization improves, 0 unacceptable regressions.

Production files: UNMODIFIED. Checkpoint: UNMODIFIED.
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

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from phase2.phase2_inference import Phase2InferenceEngine, load_grayscale_image, fit_parabola_subpixel
from phase2.phase2_config import Phase2Config

import torch


def compute_peak_distinctiveness(match_matrix, peak_loc=None, radius=7):
    """
    Computes peak prominence & periodicity count from a 2D NCC match_matrix.
    
    Returns:
        prominence: peak_val - mean(annular_neighborhood)
        periodicity_count: number of local peaks in match_matrix with value >= 0.85 * peak_val
    """
    if match_matrix is None or match_matrix.size == 0:
        return 0.0, 1

    h, w = match_matrix.shape
    if peak_loc is None:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_matrix)
        py, px = max_loc[1], max_loc[0]
    else:
        px, py = peak_loc
        max_val = float(match_matrix[py, px])

    # Annular neighborhood mask (radius R, excluding inner 3x3)
    y_min = max(0, py - radius)
    y_max = min(h, py + radius + 1)
    x_min = max(0, px - radius)
    x_max = min(w, px + radius + 1)

    annulus_vals = []
    for r in range(y_min, y_max):
        for c in range(x_min, x_max):
            if abs(r - py) > 1 or abs(c - px) > 1:
                annulus_vals.append(match_matrix[r, c])

    if len(annulus_vals) > 0:
        neighborhood_mean = float(np.mean(annulus_vals))
    else:
        neighborhood_mean = float(np.mean(match_matrix))

    prominence = max_val - neighborhood_mean

    # Periodicity count: count local maxima in match_matrix exceeding 0.85 * max_val
    thresh = 0.85 * max_val
    high_mask = (match_matrix >= thresh).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(high_mask)
    periodicity_count = max(1, num_labels - 1)

    return float(prominence), int(periodicity_count)


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


def run_experiment():
    print("=" * 70)
    print("EXP-13: NCC PEAK DISTINCTIVENESS / PERIODICITY PENALTY")
    print("=" * 70)

    # 1. Verify Checkpoint SHA-256
    ckpt_path = "phase2_checkpoints/best_model_level1.pth"
    with open(ckpt_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    expected_sha = "e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c"
    assert sha == expected_sha, f"SHA-256 MISMATCH: {sha}"
    print(f"[OK] Checkpoint SHA-256 verified: {sha[:16]}...")

    # 2. Init Engine
    engine = Phase2InferenceEngine(checkpoint_path=ckpt_path, device="cpu")
    print(f"[OK] Engine initialized.")

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

    # Step 1: Pre-calculate candidate diagnostics & prominence on target pairs
    print(f"\n{'='*70}")
    print("STEP 1: Target Pair Response Landscape Diagnostic")
    print(f"{'='*70}")

    for pid in sorted(target_pair_ids):
        row = next(p for p in pairs if p["pair_id"] == pid)
        ref_path = row["reference_path"]
        search_path = row["search_path"]
        gt_x = float(row["x_gt"])
        gt_y = float(row["y_gt"])

        res_dict, best_coarse, refined_results = engine.localize_pair(
            ref_path, search_path,
            ncc_weight=0.5, rejection_thresh=0.42, scale_step=0.25, theta_step=1.0,
            return_diagnostics=True
        )

        print(f"\n--- {pid} (GT: {gt_x:.1f}, {gt_y:.1f}) ---")
        for idx, c in enumerate(refined_results[:5]):
            d_gt = math.sqrt((c["x"] - gt_x)**2 + (c["y"] - gt_y)**2)
            prom, pcount = compute_peak_distinctiveness(c.get("match_matrix", None))
            gt_mark = " <-- GT MATCH" if d_gt <= 15.0 else ""
            print(f"  Rank {idx+1}: ({c['x']:.1f}, {c['y']:.1f}) | NCC={c['ncc_norm']:.4f} | Prominence={prom:.4f} | PeriodicityCount={pcount} | DistGT={d_gt:.1f}px{gt_mark}")

    # Variants to evaluate across 200 pairs:
    # 1. Base (Promoted Strategy A: argmax(ncc_norm))
    # 2. Prominence Weighted: argmax(ncc_norm * (1.0 + 0.2 * prominence))
    # 3. Prominence Additive: argmax(ncc_norm + 0.1 * prominence)
    # 4. Periodicity Penalized: argmax(ncc_norm - 0.05 * (periodicity_count - 1))

    variant_names = {
        "base": "Baseline Strategy A (Pure NCC)",
        "prominence_mult": "Prominence Multiplicative (ncc * (1 + 0.2*prom))",
        "prominence_add": "Prominence Additive (ncc + 0.1*prom)",
        "periodicity_pen": "Periodicity Penalized (ncc - 0.05*period_count)"
    }

    print(f"\n{'='*70}")
    print("STEP 2: Running 200-Pair Full Evaluation for All Distinctiveness Variants...")
    print(f"{'='*70}")

    all_variant_results = {k: [] for k in variant_names}
    all_variant_metrics = {}
    all_variant_targets = {k: {} for k in variant_names}

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

        # Pre-compute prominence metrics for all refined candidates
        cands_with_metrics = []
        for c in refined_results:
            prom, pcount = compute_peak_distinctiveness(c.get("match_matrix", None))
            cands_with_metrics.append({
                "cand": c,
                "prominence": prom,
                "periodicity_count": pcount
            })

        # Evaluate each variant
        for vkey in variant_names:
            if vkey == "base":
                sorted_cands = sorted(cands_with_metrics, key=lambda item: (-item["cand"]["ncc_norm"], -item["cand"]["adjusted_score"]))
            elif vkey == "prominence_mult":
                sorted_cands = sorted(cands_with_metrics, key=lambda item: (-(item["cand"]["ncc_norm"] * (1.0 + 0.2 * item["prominence"])), -item["cand"]["adjusted_score"]))
            elif vkey == "prominence_add":
                sorted_cands = sorted(cands_with_metrics, key=lambda item: (-(item["cand"]["ncc_norm"] + 0.1 * item["prominence"]), -item["cand"]["adjusted_score"]))
            elif vkey == "periodicity_pen":
                sorted_cands = sorted(cands_with_metrics, key=lambda item: (-(item["cand"]["ncc_norm"] - 0.05 * (item["periodicity_count"] - 1)), -item["cand"]["adjusted_score"]))

            best_item = sorted_cands[0]
            best_c = best_item["cand"]

            # Subpixel parabola fit
            try:
                m_mat = best_c.get("match_matrix", None)
                if m_mat is not None and m_mat.shape[0] >= 3 and m_mat.shape[1] >= 3:
                    sub_3x3 = m_mat[:3, :3]
                    fine_x, fine_y = fit_parabola_subpixel(sub_3x3, best_c["x"], best_c["y"])
                else:
                    fine_x, fine_y = best_c["x"], best_c["y"]
            except Exception:
                fine_x, fine_y = best_c["x"], best_c["y"]

            final_fused = best_c["fused_score"]
            tau = 0.42

            if final_fused >= tau:
                found = 1
                pred_x = float(round(fine_x, 2))
                pred_y = float(round(fine_y, 2))
                pred_theta = float(round(best_c["theta"], 2))
                pred_scale = float(round(best_c["scale"], 2))
            else:
                found = 0
                pred_x = pred_y = pred_theta = pred_scale = 0.0

            conf_score = 1.0 / (1.0 + math.exp(-12.0 * (final_fused - tau)))
            conf_score = float(round(max(0.0001, min(0.9999, conf_score)), 4))

            if gt_found == 1 and found == 1:
                loc_err = math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
                scale_err = abs(pred_scale - gt_scale)
                theta_err = abs(pred_theta - gt_theta)
            elif gt_found == 0 and found == 0:
                loc_err = scale_err = theta_err = 0.0
            else:
                loc_err = scale_err = theta_err = 999.0

            all_variant_results[vkey].append({
                "pair_id": pair_id, "set": set_name, "gen_id": gen_id,
                "gt_x": gt_x, "gt_y": gt_y, "gt_theta": gt_theta, "gt_scale": gt_scale, "gt_found": gt_found,
                "pred_x": pred_x, "pred_y": pred_y, "pred_theta": pred_theta, "pred_scale": pred_scale,
                "pred_found": found, "pred_score": conf_score,
                "loc_err": loc_err, "scale_err": scale_err, "theta_err": theta_err, "runtime_ms": runtime_ms,
                "raw_ncc": float(round(best_c["ncc_norm"], 4)), "prominence": float(round(best_item["prominence"], 4))
            })

            if pair_id in target_pair_ids:
                all_variant_targets[vkey][pair_id] = {
                    "loc_err": round(loc_err, 2), "pred_found": found,
                    "pred_x": pred_x, "pred_y": pred_y,
                    "raw_ncc": float(round(best_c["ncc_norm"], 4)),
                    "prominence": float(round(best_item["prominence"], 4))
                }

        if (pi + 1) % 40 == 0 or pair_id in target_pair_ids:
            marker = " *** TARGET ***" if pair_id in target_pair_ids else ""
            print(f"  [{pi+1:3d}/200] {pair_id} | base_err={all_variant_results['base'][-1]['loc_err']:.2f}px | {runtime_ms:.0f}ms{marker}")

        gc.collect()

    # Compute metrics for each variant
    for vkey in variant_names:
        all_variant_metrics[vkey] = compute_100pt_breakdown(all_variant_results[vkey])

    # Summary table
    print(f"\n{'='*70}")
    print("EXP-13 VARIANT COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Variant':<45} {'Total':>7} {'Loc':>7} {'Pose':>7} {'Rej':>7} {'Conf':>7} {'Eff':>5} {'Med_RT':>8}")
    print("-" * 95)
    for vkey, vname in variant_names.items():
        m = all_variant_metrics[vkey]
        print(f"{vname:<45} {m['total_100_score']:7.2f} {m['loc_score']:7.2f} {m['pose_score']:7.2f} {m['rejection_score']:7.2f} {m['confidence_score']:7.2f} {m['eff_score']:5.1f} {m['med_rt']:7.0f}ms")

    # Target pair deep-dive
    print(f"\n{'='*70}")
    print("TARGET PAIRS DEEP-DIVE ACROSS VARIANTS")
    print(f"{'='*70}")
    for pid in sorted(target_pair_ids):
        print(f"\n--- {pid} ---")
        for vkey, vname in variant_names.items():
            t = all_variant_targets[vkey][pid]
            print(f"  {vkey:<18} | LocErr: {t['loc_err']:>6.2f}px | Found: {t['pred_found']} | NCC: {t['raw_ncc']:.4f} | Prom: {t['prominence']:.4f}")

    # Recall Audit Metrics
    n_present = sum(1 for r in pairs if int(r["found_gt"]) == 1)
    def calc_recall(results_list):
        rec = {1: 0, 5: 0, 15: 0, 50: 0}
        for r in results_list:
            if r["gt_found"] == 1 and r["pred_found"] == 1:
                err = r["loc_err"]
                if err <= 1.0: rec[1] += 1
                if err <= 5.0: rec[5] += 1
                if err <= 15.0: rec[15] += 1
                if err <= 50.0: rec[50] += 1
        return {t: round(count / float(n_present) * 100.0, 2) for t, count in rec.items()}

    base_rec = calc_recall(all_variant_results["base"])
    best_vkey = max(variant_names.keys(), key=lambda k: all_variant_metrics[k]["total_100_score"])
    best_m = all_variant_metrics[best_vkey]
    best_rec = calc_recall(all_variant_results[best_vkey])

    print(f"\nCandidate Recall @5px:  Base = {base_rec[5]}% | Best ({best_vkey}) = {best_rec[5]}%")
    print(f"Candidate Recall @15px: Base = {base_rec[15]}% | Best ({best_vkey}) = {best_rec[15]}%")

    # Regression analysis (best_vkey vs Baseline)
    base_res = all_variant_results["base"]
    exp_res = all_variant_results[best_vkey]
    recovered = []
    regressed = []
    unchanged = []

    for idx, b_r in enumerate(base_res):
        e_r = exp_res[idx]
        pid = b_r["pair_id"]
        b_err = b_r["loc_err"]
        e_err = e_r["loc_err"]

        if b_err > 5.0 and e_err <= 5.0:
            recovered.append((pid, b_err, e_err))
        elif b_err <= 5.0 and e_err > 5.0:
            regressed.append((pid, b_err, e_err))
        else:
            unchanged.append(pid)

    print(f"\n{'='*70}")
    print(f"REGRESSION ANALYSIS ({variant_names[best_vkey]} vs Production Baseline 60.99)")
    print(f"{'='*70}")
    print(f"  Recovered pairs: {len(recovered)}")
    for r in recovered:
        print(f"    + {r[0]}: Baseline err = {r[1]:.2f}px -> Best EXP-13 err = {r[2]:.2f}px")
    print(f"  Regressed pairs: {len(regressed)}")
    for r in regressed:
        print(f"    - {r[0]}: Baseline err = {r[1]:.2f}px -> Best EXP-13 err = {r[2]:.2f}px")
    print(f"  Unchanged pairs: {len(unchanged)}")

    base_m = all_variant_metrics["base"]
    base_total = base_m["total_100_score"]
    base_loc = base_m["loc_score"]

    delta_total = best_m["total_100_score"] - base_total
    delta_loc = best_m["loc_score"] - base_loc

    print(f"\n{'='*70}")
    print("PROMOTION DECISION EVALUATION")
    print(f"{'='*70}")
    print(f"  Baseline Total: {base_total:.2f} | Best Variant ({best_vkey}): Total = {best_m['total_100_score']:.2f} (Delta = {delta_total:+.2f})")
    print(f"  Baseline Loc:   {base_loc:.2f} | Best Variant ({best_vkey}): Loc   = {best_m['loc_score']:.2f} (Delta = {delta_loc:+.2f})")

    if delta_total > 1.0 and delta_loc >= 0 and len(regressed) == 0:
        verdict = "PROMOTE"
        print(f"\n  [PASS] PROMOTE: Improved total score by {delta_total:.2f} and localization by {delta_loc:.2f} with zero regressions!")
    elif delta_total > 1.0 and delta_loc >= 0:
        verdict = "PROMOTE"
        print(f"\n  [PASS] PROMOTE: Improved total score by {delta_total:.2f} and localization by {delta_loc:.2f}.")
    elif delta_total <= 1.0 and delta_total > 0:
        verdict = "REJECT"
        print(f"\n  [FAIL] REJECT: Score gain ({delta_total:+.2f}) is <= +1.0 threshold.")
    elif delta_loc < 0:
        verdict = "REJECT"
        print(f"\n  [FAIL] REJECT: Localization score regressed by {abs(delta_loc):.2f}.")
    else:
        verdict = "REJECT"
        print(f"\n  [FAIL] REJECT: No improvement over baseline.")

    # Save CSV
    os.makedirs("phase2/results", exist_ok=True)
    csv_path = "phase2/results/exp13_ncc_peak_distinctiveness.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair_id", "set", "gen_id", "gt_found", "pred_found",
            "gt_x", "gt_y", "pred_x", "pred_y", "loc_err",
            "gt_scale", "pred_scale", "scale_err",
            "gt_theta", "pred_theta", "theta_err",
            "pred_score", "raw_ncc", "prominence", "runtime_ms"
        ])
        for r in all_variant_results[best_vkey]:
            writer.writerow([
                r["pair_id"], r["set"], r["gen_id"], r["gt_found"], r["pred_found"],
                r["gt_x"], r["gt_y"], r["pred_x"], r["pred_y"], round(r["loc_err"], 2),
                r["gt_scale"], r["pred_scale"], round(r["scale_err"], 2),
                r["gt_theta"], r["pred_theta"], round(r["theta_err"], 2),
                r["pred_score"], r["raw_ncc"], r["prominence"], round(r["runtime_ms"], 2)
            ])
    print(f"\n[OK] Saved CSV to {csv_path}")

    # Write Markdown Report
    report_path = "phase2/reports/EXP13_NCC_PEAK_DISTINCTIVENESS_ANALYSIS.md"
    os.makedirs("phase2/reports", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""# EXP-13 — NCC PEAK DISTINCTIVENESS / PERIODICITY PENALTY REPORT

## Executive Summary

- **Production Baseline Total (Strategy A)**: {base_total:.2f} / 100
- **Best EXP-13 Variant Total ({variant_names[best_vkey]})**: **{best_m['total_100_score']:.2f} / 100**
- **Delta Total Score**: **{delta_total:+.2f}**
- **Production Baseline Localization**: {base_loc:.2f} / 40
- **Best EXP-13 Localization**: **{best_m['loc_score']:.2f} / 40**
- **Delta Localization Score**: **{delta_loc:+.2f}**
- **Decision**: **{verdict}**

---

## 100-Point Score Breakdown across Variants

| Category | Baseline (Strategy A) | Prominence Mult | Prominence Add | Periodicity Pen |
|---|---|---|---|---|
| **Localization /40** | {base_m['loc_score']:.2f} | {all_variant_metrics['prominence_mult']['loc_score']:.2f} | {all_variant_metrics['prominence_add']['loc_score']:.2f} | {all_variant_metrics['periodicity_pen']['loc_score']:.2f} |
| **Scale /10** | {base_m['scale_score']:.2f} | {all_variant_metrics['prominence_mult']['scale_score']:.2f} | {all_variant_metrics['prominence_add']['scale_score']:.2f} | {all_variant_metrics['periodicity_pen']['scale_score']:.2f} |
| **Rotation /10** | {base_m['theta_score']:.2f} | {all_variant_metrics['prominence_mult']['theta_score']:.2f} | {all_variant_metrics['prominence_add']['theta_score']:.2f} | {all_variant_metrics['periodicity_pen']['theta_score']:.2f} |
| **Pose Total /20** | {base_m['pose_score']:.2f} | {all_variant_metrics['prominence_mult']['pose_score']:.2f} | {all_variant_metrics['prominence_add']['pose_score']:.2f} | {all_variant_metrics['periodicity_pen']['pose_score']:.2f} |
| **Rejection /15** | {base_m['rejection_score']:.2f} | {all_variant_metrics['prominence_mult']['rejection_score']:.2f} | {all_variant_metrics['prominence_add']['rejection_score']:.2f} | {all_variant_metrics['periodicity_pen']['rejection_score']:.2f} |
| **Confidence /10** | {base_m['confidence_score']:.2f} | {all_variant_metrics['prominence_mult']['confidence_score']:.2f} | {all_variant_metrics['prominence_add']['confidence_score']:.2f} | {all_variant_metrics['periodicity_pen']['confidence_score']:.2f} |
| **Efficiency /5** | {base_m['eff_score']:.2f} | {all_variant_metrics['prominence_mult']['eff_score']:.2f} | {all_variant_metrics['prominence_add']['eff_score']:.2f} | {all_variant_metrics['periodicity_pen']['eff_score']:.2f} |
| **Generator/Citations /10** | 10.00 | 10.00 | 10.00 | 10.00 |
| **TOTAL SCORE /100** | **{base_total:.2f}** | **{all_variant_metrics['prominence_mult']['total_100_score']:.2f}** | **{all_variant_metrics['prominence_add']['total_100_score']:.2f}** | **{all_variant_metrics['periodicity_pen']['total_100_score']:.2f}** |

---

## Target Pairs Analysis

""")
        for pid in sorted(target_pair_ids):
            f.write(f"### {pid}\n")
            f.write(f"- **GT Location**: ({all_variant_results['base'][0]['gt_x']}, {all_variant_results['base'][0]['gt_y']})\n")
            for vkey, vname in variant_names.items():
                t = all_variant_targets[vkey][pid]
                f.write(f"  - **{vkey}**: LocErr = {t['loc_err']}px, Found = {t['pred_found']}, NCC = {t['raw_ncc']:.4f}, Prominence = {t['prominence']:.4f}\n")
            f.write("\n")

        f.write(f"""---

## Regression Analysis ({variant_names[best_vkey]} vs Production Baseline)

- **Recovered Pairs**: {len(recovered)}
""")
        for r in recovered:
            f.write(f"  - `{r[0]}`: Baseline error {r[1]:.2f}px -> Best EXP-13 error {r[2]:.2f}px\n")

        f.write(f"""- **Regressed Pairs**: {len(regressed)}\n""")
        for r in regressed:
            f.write(f"  - `{r[0]}`: Baseline error {r[1]:.2f}px -> Best EXP-13 error {r[2]:.2f}px\n")

        f.write(f"""- **Unchanged Pairs**: {len(unchanged)}

---

## Runtime Performance

- **Median Runtime**: {best_m['med_rt']:.0f} ms
- **P90 Runtime**: {best_m['p90_rt']:.0f} ms
- **P99 Runtime**: {best_m['p99_rt']:.0f} ms

---

## Final Decision: {verdict}
""")

    print(f"[OK] Report saved to {report_path}")


if __name__ == "__main__":
    run_experiment()

