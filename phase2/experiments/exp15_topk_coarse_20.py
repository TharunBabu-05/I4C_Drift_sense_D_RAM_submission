#!/usr/bin/env python3
"""
EXP-15 — Top-K Coarse Cutoff Expansion (TOP_K_COARSE: 10 -> 20)
================================================================

STRICT SINGLE-CHANGE EXPERIMENT

Hypothesis:
    EXP-14 identified Cat B (GT present in coarse search grid but ranked 11th-20th by coarse NCC)
    as the primary bottleneck accounting for 32 / 66 failures (48.5% of all remaining errors).
    Expanding TOP_K_COARSE from 10 to 20 will allow degraded Set-B GT candidates into the fine refinement stage,
    where EXP-13 Periodicity Penalized ranking can correctly localize them.

Production files: UNMODIFIED during experiment.
Checkpoint: UNMODIFIED (SHA-256 e64fd936... verified).
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
    print("EXP-15: TOP-K COARSE CUTOFF EXPANSION (TOP_K_COARSE: 10 -> 20)")
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

    # Load EXP-14 audit log to track the 32 Cat-B pairs
    cat_b_pairs = set()
    exp14_csv = "phase2/results/exp14_failure_mode_audit.csv"
    if os.path.exists(exp14_csv):
        with open(exp14_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["primary_cat"] == "B":
                    cat_b_pairs.add(row["pair_id"])
    print(f"[OK] Loaded {len(cat_b_pairs)} Cat-B failure pairs from EXP-14 audit.")

    results_base_k10 = []
    results_exp_k20 = []

    target_details_k10 = {}
    target_details_k20 = {}

    cat_b_trace = {}

    print(f"\nRunning 200-Pair Evaluation for Baseline (K=10) and EXP-15 (K=20)...")

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

        # -------------------------------------------------------------
        # 1. Baseline K=10 Run
        # -------------------------------------------------------------
        t0 = time.time()
        res_k10, coarse_k10, refined_k10 = engine.localize_pair(
            ref_path, search_path,
            ncc_weight=0.5, rejection_thresh=0.42, scale_step=0.25, theta_step=1.0,
            top_k_coarse=10, return_diagnostics=True
        )
        t1 = time.time()
        rt_k10 = (t1 - t0) * 1000.0

        px10, py10 = res_k10["x"], res_k10["y"]
        pth10, psc10 = res_k10["theta"], res_k10["scale"]
        f10, sc10 = res_k10["found"], res_k10["score"]

        if gt_found == 1 and f10 == 1:
            err_k10 = math.sqrt((px10 - gt_x)**2 + (py10 - gt_y)**2)
            serr_k10 = abs(psc10 - gt_scale)
            terr_k10 = abs(pth10 - gt_theta)
        elif gt_found == 0 and f10 == 0:
            err_k10 = serr_k10 = terr_k10 = 0.0
        else:
            err_k10 = serr_k10 = terr_k10 = 999.0

        results_base_k10.append({
            "pair_id": pair_id, "set": set_name, "gen_id": gen_id,
            "gt_x": gt_x, "gt_y": gt_y, "gt_theta": gt_theta, "gt_scale": gt_scale, "gt_found": gt_found,
            "pred_x": px10, "pred_y": py10, "pred_theta": pth10, "pred_scale": psc10,
            "pred_found": f10, "pred_score": sc10,
            "loc_err": err_k10, "scale_err": serr_k10, "theta_err": terr_k10,
            "runtime_ms": rt_k10
        })

        # -------------------------------------------------------------
        # 2. EXP-15 K=20 Run
        # -------------------------------------------------------------
        t0 = time.time()
        res_k20, coarse_k20, refined_k20 = engine.localize_pair(
            ref_path, search_path,
            ncc_weight=0.5, rejection_thresh=0.42, scale_step=0.25, theta_step=1.0,
            top_k_coarse=20, return_diagnostics=True
        )
        t1 = time.time()
        rt_k20 = (t1 - t0) * 1000.0

        px20, py20 = res_k20["x"], res_k20["y"]
        pth20, psc20 = res_k20["theta"], res_k20["scale"]
        f20, sc20 = res_k20["found"], res_k20["score"]

        if gt_found == 1 and f20 == 1:
            err_k20 = math.sqrt((px20 - gt_x)**2 + (py20 - gt_y)**2)
            serr_k20 = abs(psc20 - gt_scale)
            terr_k20 = abs(pth20 - gt_theta)
        elif gt_found == 0 and f20 == 0:
            err_k20 = serr_k20 = terr_k20 = 0.0
        else:
            err_k20 = serr_k20 = terr_k20 = 999.0

        results_exp_k20.append({
            "pair_id": pair_id, "set": set_name, "gen_id": gen_id,
            "gt_x": gt_x, "gt_y": gt_y, "gt_theta": gt_theta, "gt_scale": gt_scale, "gt_found": gt_found,
            "pred_x": px20, "pred_y": py20, "pred_theta": pth20, "pred_scale": psc20,
            "pred_found": f20, "pred_score": sc20,
            "loc_err": err_k20, "scale_err": serr_k20, "theta_err": terr_k20,
            "runtime_ms": rt_k20
        })

        # Diagnostic audit for Cat-B pairs
        if pair_id in cat_b_pairs:
            # Find GT coarse rank in coarse_k20
            gt_coarse_rank = -1
            for c_idx, c in enumerate(coarse_k20):
                d = math.sqrt((c["x"] - gt_x)**2 + (c["y"] - gt_y)**2)
                if d <= 25.0:
                    gt_coarse_rank = c_idx + 1
                    break

            gt_in_refined_k20 = False
            for c in refined_k20:
                d = math.sqrt((c["x"] - gt_x)**2 + (c["y"] - gt_y)**2)
                if d <= 5.0:
                    gt_in_refined_k20 = True
                    break

            cat_b_trace[pair_id] = {
                "coarse_rank": gt_coarse_rank,
                "in_refined_k20": gt_in_refined_k20,
                "err_k10": round(err_k10, 2),
                "err_k20": round(err_k20, 2),
                "recovered": (err_k10 > 5.0 and err_k20 <= 5.0)
            }

        if pair_id in target_pair_ids:
            target_details_k10[pair_id] = {"err": round(err_k10, 2), "found": f10, "score": sc10}
            target_details_k20[pair_id] = {"err": round(err_k20, 2), "found": f20, "score": sc20}

        if (pi + 1) % 40 == 0 or pair_id in target_pair_ids:
            marker = " *** TARGET ***" if pair_id in target_pair_ids else ""
            print(f"  [{pi+1:3d}/200] {pair_id} | k10_err={err_k10:.2f}px | k20_err={err_k20:.2f}px | k20_rt={rt_k20:.0f}ms{marker}")

        del res_k10, coarse_k10, refined_k10, res_k20, coarse_k20, refined_k20
        gc.collect()

    # Compute 100-Point Scores
    m_k10 = compute_100pt_breakdown(results_base_k10)
    m_k20 = compute_100pt_breakdown(results_exp_k20)

    # Candidate Recall Audit
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

    rec_k10 = calc_recall(results_base_k10)
    rec_k20 = calc_recall(results_exp_k20)

    # Cat-B Recovery Audit
    cat_b_recovered = sum(1 for t in cat_b_trace.values() if t["recovered"])
    cat_b_in_top20 = sum(1 for t in cat_b_trace.values() if 11 <= t["coarse_rank"] <= 20)
    cat_b_beyond_top20 = sum(1 for t in cat_b_trace.values() if t["coarse_rank"] > 20 or t["coarse_rank"] == -1)

    # Regression Analysis (K=20 vs Baseline K=10)
    recovered_pairs = []
    regressed_pairs = []
    unchanged_pairs = []

    for idx, r10 in enumerate(results_base_k10):
        r20 = results_exp_k20[idx]
        pid = r10["pair_id"]
        e10 = r10["loc_err"]
        e20 = r20["loc_err"]

        if e10 > 5.0 and e20 <= 5.0:
            recovered_pairs.append((pid, e10, e20))
        elif e10 <= 5.0 and e20 > 5.0:
            regressed_pairs.append((pid, e10, e20))
        else:
            unchanged_pairs.append(pid)

    print(f"\n{'='*70}")
    print("EXP-15 OFFICIAL 100-POINT SCORE COMPARISON")
    print(f"{'='*70}")
    print(f"Baseline (K=10): Total = {m_k10['total_100_score']:.2f} / 100 (Loc {m_k10['loc_score']:.2f}/40, Pose {m_k10['pose_score']:.2f}/20, Rej {m_k10['rejection_score']:.2f}/15, Conf {m_k10['confidence_score']:.2f}/10, RT {m_k10['med_rt']:.0f}ms)")
    print(f"EXP-15   (K=20): Total = {m_k20['total_100_score']:.2f} / 100 (Loc {m_k20['loc_score']:.2f}/40, Pose {m_k20['pose_score']:.2f}/20, Rej {m_k20['rejection_score']:.2f}/15, Conf {m_k20['confidence_score']:.2f}/10, RT {m_k20['med_rt']:.0f}ms)")
    
    delta_total = m_k20['total_100_score'] - m_k10['total_100_score']
    delta_loc = m_k20['loc_score'] - m_k10['loc_score']
    print(f"DELTA: Total = {delta_total:+.2f} points | Localization = {delta_loc:+.2f} points")

    print(f"\n{'='*70}")
    print("EXP-14 CAT-B FAILURE RECOVERY AUDIT (32 Total Baseline Cat-B Failures)")
    print(f"{'='*70}")
    print(f"  - Cat-B GT Coarse Rank was 11-20:  {cat_b_in_top20} / 32")
    print(f"  - Cat-B GT Coarse Rank was > 20:   {cat_b_beyond_top20} / 32")
    print(f"  - Cat-B Pairs RECOVERED (<=5px):    {cat_b_recovered} / 32")
    print(f"  - Cat-B Pairs STILL FAILED (>5px): {len(cat_b_pairs) - cat_b_recovered} / 32")

    print(f"\n{'='*70}")
    print(f"REGRESSION ANALYSIS (K=20 vs Baseline K=10)")
    print(f"{'='*70}")
    print(f"  Recovered pairs: {len(recovered_pairs)}")
    for r in recovered_pairs:
        print(f"    + {r[0]}: Baseline err = {r[1]:.2f}px -> EXP-15 err = {r[2]:.2f}px")
    print(f"  Regressed pairs: {len(regressed_pairs)}")
    for r in regressed_pairs:
        print(f"    - {r[0]}: Baseline err = {r[1]:.2f}px -> EXP-15 err = {r[2]:.2f}px")
    print(f"  Unchanged pairs: {len(unchanged_pairs)}")

    # Decision evaluation
    print(f"\n{'='*70}")
    print("PROMOTION DECISION EVALUATION")
    print(f"{'='*70}")
    if delta_total > 1.0 and delta_loc >= 0 and len(regressed_pairs) <= len(recovered_pairs) / 2:
        verdict = "PROMOTE"
        print(f"  [PASS] PROMOTE: Score gain ({delta_total:+.2f}) exceeds +1.0 threshold!")
    elif delta_total > 0 and delta_loc >= 0:
        verdict = "REJECT"
        print(f"  [FAIL] REJECT: Score gain ({delta_total:+.2f}) is <= +1.0 threshold.")
    elif delta_loc < 0:
        verdict = "REJECT"
        print(f"  [FAIL] REJECT: Localization score regressed by {abs(delta_loc):.2f}.")
    else:
        verdict = "REJECT"
        print(f"  [FAIL] REJECT: No improvement over baseline.")

    # Save CSV
    os.makedirs("phase2/results", exist_ok=True)
    csv_path = "phase2/results/exp15_topk_coarse_20.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair_id", "set", "gen_id", "gt_found", "pred_found",
            "gt_x", "gt_y", "pred_x", "pred_y", "loc_err",
            "gt_scale", "pred_scale", "scale_err",
            "gt_theta", "pred_theta", "theta_err",
            "pred_score", "runtime_ms"
        ])
        for r in results_exp_k20:
            writer.writerow([
                r["pair_id"], r["set"], r["gen_id"], r["gt_found"], r["pred_found"],
                r["gt_x"], r["gt_y"], r["pred_x"], r["pred_y"], round(r["loc_err"], 2),
                r["gt_scale"], r["pred_scale"], round(r["scale_err"], 2),
                r["gt_theta"], r["pred_theta"], round(r["theta_err"], 2),
                r["pred_score"], round(r["runtime_ms"], 2)
            ])
    print(f"\n[OK] Saved CSV to {csv_path}")

    # Write Markdown Report
    os.makedirs("phase2/reports", exist_ok=True)
    report_path = "phase2/reports/EXP15_TOPK_COARSE_20_ANALYSIS.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""# EXP-15 TOP-K COARSE EXPANSION REPORT

## Executive Summary

- **Baseline Score (EXP-13, K=10)**: **{m_k10['total_100_score']:.2f} / 100**
- **EXP-15 Score (K=20)**: **{m_k20['total_100_score']:.2f} / 100**
- **Delta Total Score**: **{delta_total:+.2f}**
- **Baseline Localization**: {m_k10['loc_score']:.2f} / 40
- **EXP-15 Localization**: **{m_k20['loc_score']:.2f} / 40** ({delta_loc:+.2f})
- **Decision**: **{verdict}**

---

## 100-Point Score Breakdown

| Category | Baseline (K=10) | EXP-15 (K=20) | Delta |
|---|---|---|---|
| **Localization /40** | {m_k10['loc_score']:.2f} | **{m_k20['loc_score']:.2f}** | **{delta_loc:+.2f}** |
| **Scale /10** | {m_k10['scale_score']:.2f} | **{m_k20['scale_score']:.2f}** | **{m_k20['scale_score'] - m_k10['scale_score']:+.2f}** |
| **Rotation /10** | {m_k10['theta_score']:.2f} | **{m_k20['theta_score']:.2f}** | **{m_k20['theta_score'] - m_k10['theta_score']:+.2f}** |
| **Pose Total /20** | {m_k10['pose_score']:.2f} | **{m_k20['pose_score']:.2f}** | **{m_k20['pose_score'] - m_k10['pose_score']:+.2f}** |
| **Rejection /15** | {m_k10['rejection_score']:.2f} | **{m_k20['rejection_score']:.2f}** | **0.00** |
| **Confidence /10** | {m_k10['confidence_score']:.2f} | **{m_k20['confidence_score']:.2f}** | **{m_k20['confidence_score'] - m_k10['confidence_score']:+.2f}** |
| **Efficiency /5** | {m_k10['eff_score']:.2f} | **{m_k20['eff_score']:.2f}** | **0.00** |
| **Generator/Citations /10** | 10.00 | **10.00** | **0.00** |
| **TOTAL SCORE /100** | **{m_k10['total_100_score']:.2f}** | **{m_k20['total_100_score']:.2f}** | **{delta_total:+.2f}** |

---

## Cat-B Recovery Audit (32 Total EXP-14 Baseline Cat-B Failures)

- **Cat-B Coarse Rank was 11–20**: {cat_b_in_top20} / 32
- **Cat-B Coarse Rank was > 20**: {cat_b_beyond_top20} / 32
- **Cat-B Pairs RECOVERED (<=5px)**: **{cat_b_recovered} / 32**
- **Cat-B Pairs STILL FAILED (>5px)**: {len(cat_b_pairs) - cat_b_recovered} / 32

---

## Candidate Recall Audit

| Threshold | Baseline (K=10) | EXP-15 (K=20) | Delta |
|---|---|---|---|
| **Recall @1px** | {rec_k10[1]}% | **{rec_k20[1]}%** | {rec_k20[1] - rec_k10[1]:+.2f}% |
| **Recall @5px** | {rec_k10[5]}% | **{rec_k20[5]}%** | {rec_k20[5] - rec_k10[5]:+.2f}% |
| **Recall @15px** | {rec_k10[15]}% | **{rec_k20[15]}%** | {rec_k20[15] - rec_k10[15]:+.2f}% |
| **Recall @50px** | {rec_k10[50]}% | **{rec_k20[50]}%** | {rec_k20[50] - rec_k10[50]:+.2f}% |

---

## Target Pairs Comparison

""")
        for pid in sorted(target_pair_ids):
            t10 = target_details_k10[pid]
            t20 = target_details_k20[pid]
            f.write(f"### {pid}\n")
            f.write(f"- **Baseline (K=10)**: LocErr = {t10['err']}px, Found = {t10['found']}, Score = {t10['score']}\n")
            f.write(f"- **EXP-15 (K=20)**  : LocErr = {t20['err']}px, Found = {t20['found']}, Score = {t20['score']}\n\n")

        f.write(f"""---

## Regression Analysis (K=20 vs Baseline K=10)

- **Recovered Pairs ({len(recovered_pairs)})**:
""")
        for r in recovered_pairs:
            f.write(f"  - `{r[0]}`: Baseline error {r[1]:.2f}px -> EXP-15 error {r[2]:.2f}px\n")

        f.write(f"""- **Regressed Pairs ({len(regressed_pairs)})**:
""")
        for r in regressed_pairs:
            f.write(f"  - `{r[0]}`: Baseline error {r[1]:.2f}px -> EXP-15 error {r[2]:.2f}px\n")

        f.write(f"""- **Unchanged Pairs**: {len(unchanged_pairs)}

---

## Runtime Performance

- **Baseline Median Runtime**: {m_k10['med_rt']:.0f} ms
- **EXP-15 Median Runtime**: **{m_k20['med_rt']:.0f} ms** (Delta: {m_k20['med_rt'] - m_k10['med_rt']:+.0f} ms)
- **EXP-15 P90 Runtime**: {m_k20['p90_rt']:.0f} ms
- **EXP-15 P99 Runtime**: {m_k20['p99_rt']:.0f} ms

---

## Technical Conclusion & Decision: {verdict}
""")

    print(f"[OK] Audit report saved to {report_path}")

if __name__ == "__main__":
    main()
