#!/usr/bin/env python3
"""
EXP-16 — Sunday Checkpoint-Only Validation Script
==================================================

STRICT SINGLE-CHANGE VALIDATION — NO ALGORITHM MODIFICATION

Evaluates the Sunday fine-tuned checkpoint (`checkpoints_phase2_v2_sunday/best_model_phase2.pth`)
against the old baseline checkpoint (`phase2_checkpoints/best_model_level1.pth`) using the exact
EXP-13 Production Algorithm.

All 200 official pairs evaluated across Set A, B, C, D.
"""

import os
import sys
import json
import time
import math
import hashlib
import csv
import gc
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from phase2.phase2_inference import Phase2InferenceEngine

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
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "prec": prec, "rec": rec, "f1": f1, "auc": auc, "med_rt": med_rt,
        "p90_rt": float(np.percentile(runtimes, 90)),
        "p99_rt": float(np.percentile(runtimes, 99)),
    }

def main():
    print("=" * 70)
    print("EXP-16: FRESH SUNDAY CHECKPOINT VALIDATION")
    print("=" * 70)

    # 1. Record Checkpoint Metadata for Sunday Model
    sunday_ckpt_path = "checkpoints_phase2_v2_sunday/best_model_phase2.pth"
    assert os.path.exists(sunday_ckpt_path), f"Sunday checkpoint not found: {sunday_ckpt_path}"
    sunday_size = os.path.getsize(sunday_ckpt_path)

    with open(sunday_ckpt_path, "rb") as f:
        sunday_sha = hashlib.sha256(f.read()).hexdigest()

    # 2. Record Checkpoint Metadata for Old Baseline Model
    old_ckpt_path = "phase2_checkpoints/best_model_level1.pth"
    if not os.path.exists(old_ckpt_path):
        old_ckpt_path = "best_model_level1.pth"

    with open(old_ckpt_path, "rb") as f:
        old_sha = hashlib.sha256(f.read()).hexdigest()

    print(f"Old Checkpoint:    {old_ckpt_path} (SHA: {old_sha[:16]}...)")
    print(f"Sunday Checkpoint: {sunday_ckpt_path} (Size: {sunday_size} bytes, SHA: {sunday_sha[:16]}...)")

    # 3. Init Engines
    engine_old = Phase2InferenceEngine(checkpoint_path=old_ckpt_path, device="cpu")
    engine_sunday = Phase2InferenceEngine(checkpoint_path=sunday_ckpt_path, device="cpu")

    data_dir = "local_phase2_60gen_200_pairs"
    manifest_path = os.path.join(data_dir, "phase2_60generator_manifest.csv")
    assert os.path.exists(manifest_path), f"Manifest not found: {manifest_path}"

    pairs = []
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(row)
    print(f"[OK] Loaded {len(pairs)} pairs from manifest")

    target_pairs = {"pair_006", "pair_066", "pair_116", "pair_186", "pair_160"}

    results_old = []
    results_sunday = []

    target_details_old = {}
    target_details_sunday = {}

    print(f"\nRunning 200-Pair Evaluation for Both Checkpoints...")
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
        # Old Checkpoint Evaluation
        # -------------------------------------------------------------
        t0 = time.time()
        res_o = engine_old.localize_pair(
            ref_path, search_path,
            ncc_weight=0.5, rejection_thresh=0.42, scale_step=0.25, theta_step=1.0
        )
        t1 = time.time()
        rt_o = (t1 - t0) * 1000.0

        if gt_found == 1 and res_o["found"] == 1:
            err_o = math.sqrt((res_o["x"] - gt_x)**2 + (res_o["y"] - gt_y)**2)
            serr_o = abs(res_o["scale"] - gt_scale)
            terr_o = abs(res_o["theta"] - gt_theta)
        elif gt_found == 0 and res_o["found"] == 0:
            err_o = serr_o = terr_o = 0.0
        else:
            err_o = serr_o = terr_o = 999.0

        results_old.append({
            "pair_id": pair_id, "set": set_name, "gen_id": gen_id,
            "gt_x": gt_x, "gt_y": gt_y, "gt_theta": gt_theta, "gt_scale": gt_scale, "gt_found": gt_found,
            "pred_x": res_o["x"], "pred_y": res_o["y"], "pred_theta": res_o["theta"], "pred_scale": res_o["scale"],
            "pred_found": res_o["found"], "pred_score": res_o["score"],
            "loc_err": err_o, "scale_err": serr_o, "theta_err": terr_o,
            "raw_ncc": res_o.get("raw_ncc", 0.0), "raw_siamese": res_o.get("raw_siamese", 0.0),
            "runtime_ms": rt_o
        })

        # -------------------------------------------------------------
        # Sunday Checkpoint Evaluation
        # -------------------------------------------------------------
        t0 = time.time()
        res_s = engine_sunday.localize_pair(
            ref_path, search_path,
            ncc_weight=0.5, rejection_thresh=0.42, scale_step=0.25, theta_step=1.0
        )
        t1 = time.time()
        rt_s = (t1 - t0) * 1000.0

        if gt_found == 1 and res_s["found"] == 1:
            err_s = math.sqrt((res_s["x"] - gt_x)**2 + (res_s["y"] - gt_y)**2)
            serr_s = abs(res_s["scale"] - gt_scale)
            terr_s = abs(res_s["theta"] - gt_theta)
        elif gt_found == 0 and res_s["found"] == 0:
            err_s = serr_s = terr_s = 0.0
        else:
            err_s = serr_s = terr_s = 999.0

        results_sunday.append({
            "pair_id": pair_id, "set": set_name, "gen_id": gen_id,
            "gt_x": gt_x, "gt_y": gt_y, "gt_theta": gt_theta, "gt_scale": gt_scale, "gt_found": gt_found,
            "pred_x": res_s["x"], "pred_y": res_s["y"], "pred_theta": res_s["theta"], "pred_scale": res_s["scale"],
            "pred_found": res_s["found"], "pred_score": res_s["score"],
            "loc_err": err_s, "scale_err": serr_s, "theta_err": terr_s,
            "raw_ncc": res_s.get("raw_ncc", 0.0), "raw_siamese": res_s.get("raw_siamese", 0.0),
            "runtime_ms": rt_s
        })

        if pair_id in target_pairs:
            target_details_old[pair_id] = res_o
            target_details_old[pair_id]["loc_err"] = round(err_o, 2)
            target_details_sunday[pair_id] = res_s
            target_details_sunday[pair_id]["loc_err"] = round(err_s, 2)

        if (pi + 1) % 40 == 0 or pair_id in target_pairs:
            marker = " *** TARGET ***" if pair_id in target_pairs else ""
            print(f"  [{pi+1:3d}/200] {pair_id} | old_err={err_o:.2f}px | sunday_err={err_s:.2f}px | rt={rt_s:.0f}ms{marker}")

        gc.collect()

    # Compute 100-Point Metrics
    m_old = compute_100pt_breakdown(results_old)
    m_sun = compute_100pt_breakdown(results_sunday)

    # Set-wise breakdowns
    def compute_set_metrics(res_list):
        s_dict = {}
        for sname in ["Set A", "Set B", "Set C", "Set D"]:
            s_entries = [r for r in res_list if r["set"] == sname]
            tot = len(s_entries)
            if sname in ["Set A", "Set B", "Set D"]:
                localized = sum(1 for r in s_entries if r["gt_found"] == 1 and r["pred_found"] == 1 and r["loc_err"] <= 5.0)
                failed_loc = sum(1 for r in s_entries if r["gt_found"] == 1 and (r["pred_found"] == 0 or r["loc_err"] > 5.0))
                fn = sum(1 for r in s_entries if r["gt_found"] == 1 and r["pred_found"] == 0)
                fp = sum(1 for r in s_entries if r["gt_found"] == 0 and r["pred_found"] == 1)
                tn = sum(1 for r in s_entries if r["gt_found"] == 0 and r["pred_found"] == 0)
            else:
                localized = 0
                failed_loc = 0
                fn = sum(1 for r in s_entries if r["gt_found"] == 1 and r["pred_found"] == 0)
                fp = sum(1 for r in s_entries if r["gt_found"] == 0 and r["pred_found"] == 1)
                tn = sum(1 for r in s_entries if r["gt_found"] == 0 and r["pred_found"] == 0)
            s_dict[sname] = {
                "tot": tot, "localized": localized, "failed_loc": failed_loc,
                "tn": tn, "fp": fp, "fn": fn
            }
        return s_dict

    set_m_old = compute_set_metrics(results_old)
    set_m_sun = compute_set_metrics(results_sunday)

    # Regression analysis (Sunday vs Old)
    recovered = []
    regressed = []
    unchanged = []

    for idx, r_o in enumerate(results_old):
        r_s = results_sunday[idx]
        pid = r_o["pair_id"]
        e_o = r_o["loc_err"]
        e_s = r_s["loc_err"]

        if e_o > 5.0 and e_s <= 5.0:
            recovered.append((pid, e_o, e_s))
        elif e_o <= 5.0 and e_s > 5.0:
            regressed.append((pid, e_o, e_s))
        else:
            unchanged.append(pid)

    delta_total = m_sun["total_100_score"] - m_old["total_100_score"]
    delta_loc = m_sun["loc_score"] - m_old["loc_score"]

    print(f"\n{'='*70}")
    print("EXP-16 OFFICIAL COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"Old Baseline (best_model_level1.pth): Total = {m_old['total_100_score']:.2f} / 100 (Loc {m_old['loc_score']:.2f}, Pose {m_old['pose_score']:.2f}, Rej {m_old['rejection_score']:.2f}, Conf {m_old['confidence_score']:.2f})")
    print(f"Sunday Model (best_model_phase2.pth): Total = {m_sun['total_100_score']:.2f} / 100 (Loc {m_sun['loc_score']:.2f}, Pose {m_sun['pose_score']:.2f}, Rej {m_sun['rejection_score']:.2f}, Conf {m_sun['confidence_score']:.2f})")
    print(f"DELTA: Total = {delta_total:+.2f} | Loc = {delta_loc:+.2f} | Rejection = {m_sun['rejection_score'] - m_old['rejection_score']:+.2f} | Confidence = {m_sun['confidence_score'] - m_old['confidence_score']:+.2f}")

    if delta_total > 1.0 and delta_loc >= 0 and len(regressed) == 0:
        verdict = "PROMOTE CHECKPOINT ONLY"
        print(f"\n  [PASS] PROMOTE CHECKPOINT ONLY: Improved total score by {delta_total:+.2f} points with zero regressions!")
    elif delta_total > 1.0 and delta_loc >= 0:
        verdict = "PROMOTE CHECKPOINT ONLY"
        print(f"\n  [PASS] PROMOTE CHECKPOINT ONLY: Improved total score by {delta_total:+.2f} points.")
    else:
        verdict = "REJECT"
        print(f"\n  [FAIL] REJECT: Score gain ({delta_total:+.2f}) is <= +1.0 threshold.")

    # Save CSV
    os.makedirs("phase2/results", exist_ok=True)
    csv_path = "phase2/results/exp16_sunday_checkpoint_validation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair_id", "set", "gen_id", "gt_found", "old_pred_found", "sunday_pred_found",
            "gt_x", "gt_y", "old_pred_x", "old_pred_y", "sunday_pred_x", "sunday_pred_y",
            "old_loc_err", "sunday_loc_err", "old_score", "sunday_score", "runtime_ms"
        ])
        for idx, r_s in enumerate(results_sunday):
            r_o = results_old[idx]
            writer.writerow([
                r_s["pair_id"], r_s["set"], r_s["gen_id"], r_s["gt_found"], r_o["pred_found"], r_s["pred_found"],
                r_s["gt_x"], r_s["gt_y"], r_o["pred_x"], r_o["pred_y"], r_s["pred_x"], r_s["pred_y"],
                r_o["loc_err"], r_s["loc_err"], r_o["pred_score"], r_s["pred_score"], round(r_s["runtime_ms"], 2)
            ])
    print(f"\n[OK] Saved CSV to {csv_path}")

    # Write Markdown Report
    os.makedirs("phase2/reports", exist_ok=True)
    report_path = "phase2/reports/EXP16_SUNDAY_CHECKPOINT_VALIDATION.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""# EXP-16 SUNDAY CHECKPOINT VALIDATION REPORT

OLD EXP-13 BASELINE:

    {m_old['total_100_score']:.2f} /100

SUNDAY CHECKPOINT:

    {m_sun['total_100_score']:.2f} /100

DELTA:

    {delta_total:+.2f}

Localization:

    {m_sun['loc_score']:.2f} /40

Scale:

    {m_sun['scale_score']:.2f} /10

Rotation:

    {m_sun['theta_score']:.2f} /10

Pose:

    {m_sun['pose_score']:.2f} /20

Rejection:

    {m_sun['rejection_score']:.2f} /15

Confidence:

    {m_sun['confidence_score']:.2f} /10

Efficiency:

    {m_sun['eff_score']:.2f} /5

Generator/Citations:

    {m_sun['gen_score']:.2f} /10

Runtime:

    Median = {m_sun['med_rt']:.0f} ms
    P90    = {m_sun['p90_rt']:.0f} ms
    P99    = {m_sun['p99_rt']:.0f} ms

Recovered:

    {len(recovered)} pairs

Regressed:

    {len(regressed)} pairs

Unchanged:

    {len(unchanged)} pairs

---

## Set-Wise Breakdown

### Set A (SEM Clean - 70 pairs)
- Old Baseline: Localized = {set_m_old['Set A']['localized']}, Failed = {set_m_old['Set A']['failed_loc']}
- Sunday Model: Localized = {set_m_sun['Set A']['localized']}, Failed = {set_m_sun['Set A']['failed_loc']}

### Set B (SEM Degraded - 70 pairs)
- Old Baseline: Localized = {set_m_old['Set B']['localized']}, Failed = {set_m_old['Set B']['failed_loc']}
- Sunday Model: Localized = {set_m_sun['Set B']['localized']}, Failed = {set_m_sun['Set B']['failed_loc']}

### Set C (Absent Pairs - 40 pairs)
- Old Baseline: Correct Rejections (TN) = {set_m_old['Set C']['tn']}/40, False Positives (FP) = {set_m_old['Set C']['fp']}/40
- Sunday Model: Correct Rejections (TN) = {set_m_sun['Set C']['tn']}/40, False Positives (FP) = {set_m_sun['Set C']['fp']}/40

### Set D (Optical Analogue - 20 pairs)
- Old Baseline: Localized = {set_m_old['Set D']['localized']}, Failed = {set_m_old['Set D']['failed_loc']}
- Sunday Model: Localized = {set_m_sun['Set D']['localized']}, Failed = {set_m_sun['Set D']['failed_loc']}

---

## Target Pairs Trace

""")
        for pid in sorted(target_pairs):
            t_o = target_details_old[pid]
            t_s = target_details_sunday[pid]
            f.write(f"### {pid}\n")
            f.write(f"- **Old Baseline**: LocErr = {t_o['loc_err']}px, Found = {t_o['found']}, Score = {t_o['score']}\n")
            f.write(f"- **Sunday Model**: LocErr = {t_s['loc_err']}px, Found = {t_s['found']}, Score = {t_s['score']}\n\n")

        f.write(f"""---

## Checkpoint Metadata & Integrity

- **Checkpoint File**: `{sunday_ckpt_path}`
- **File Size**: {sunday_size} bytes
- **SHA-256 Hash**: `{sunday_sha}`

---

## DECISION

**{verdict}**
""")

    print(f"[OK] Audit report saved to {report_path}")

if __name__ == "__main__":
    main()
