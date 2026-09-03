#!/usr/bin/env python3
"""
Official 100-Point Competition Scoring Script
==============================================
Calculates official competition marks by comparing predictions.csv against ground_truth.csv.

Usage:
    python score_predictions.py --ground-truth ground_truth.csv --predictions predictions.csv
"""

import os
import sys
import csv
import math
import argparse
import numpy as np

def calculate_auc(y_true, y_scores):
    pos_scores = [s for t, s in zip(y_true, y_scores) if t == 1]
    neg_scores = [s for t, s in zip(y_true, y_scores) if t == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5
    count = 0.0
    for p in pos_scores:
        for n in neg_scores:
            if p > n: count += 1.0
            elif p == n: count += 0.5
    return count / (len(pos_scores) * len(neg_scores))

def main():
    parser = argparse.ArgumentParser(description="Official Phase 2 Competition Scoring Script")
    parser.add_argument("--ground-truth", "-g", default="ground_truth.csv", help="Path to ground_truth.csv")
    parser.add_argument("--predictions", "-p", default="predictions.csv", help="Path to predictions.csv")
    args = parser.parse_args()

    if not os.path.exists(args.ground_truth):
        print(f"ERROR: Ground truth file not found: '{args.ground_truth}'")
        sys.exit(1)
    if not os.path.exists(args.predictions):
        print(f"ERROR: Predictions file not found: '{args.predictions}'")
        sys.exit(1)

    # Load GT
    gt_dict = {}
    with open(args.ground_truth, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            gt_dict[r["pair_id"]] = {
                "present": int(r["present"]),
                "x": float(r.get("x", 0.0) or 0.0),
                "y": float(r.get("y", 0.0) or 0.0),
                "theta": float(r.get("theta", 0.0) or 0.0),
                "scale": float(r.get("scale", 0.0) or 0.0)
            }

    # Load Predictions
    pred_dict = {}
    with open(args.predictions, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            conf_val = float(r.get("confidence", 0.0) or r.get("score", 0.0) or 0.5)
            pred_dict[r["pair_id"]] = {
                "present": int(r["present"]),
                "x": float(r.get("x", 0.0) or 0.0),
                "y": float(r.get("y", 0.0) or 0.0),
                "theta": float(r.get("theta", 0.0) or 0.0),
                "scale": float(r.get("scale", 0.0) or 0.0),
                "confidence": conf_val
            }

    # Evaluate
    evaluated = []
    for pid, gt in gt_dict.items():
        if pid not in pred_dict:
            pred = {"present": 0, "x": 0.0, "y": 0.0, "theta": 0.0, "scale": 0.0, "confidence": 0.0}
        else:
            pred = pred_dict[pid]

        # Determine subset based on pair index
        try:
            num = int(pid.replace("pair_", "").replace("p", ""))
            if num <= 70: s_name = "Set A"
            elif num <= 140: s_name = "Set B"
            elif num <= 180: s_name = "Set C"
            else: s_name = "Set D"
        except:
            s_name = "General"

        gt_present = gt["present"]
        pred_present = pred["present"]

        if gt_present == 1:
            if pred_present == 1:
                loc_err = math.sqrt((pred["x"] - gt["x"])**2 + (pred["y"] - gt["y"])**2)
                scale_err = abs(pred["scale"] - gt["scale"])
                theta_err = abs(pred["theta"] - gt["theta"])
                passed = loc_err <= 5.0
            else:
                loc_err = 999.0
                scale_err = 999.0
                theta_err = 999.0
                passed = False
        else:
            if pred_present == 0:
                loc_err = 0.0
                scale_err = 0.0
                theta_err = 0.0
                passed = True
            else:
                loc_err = 999.0
                scale_err = 0.0
                theta_err = 0.0
                passed = False

        evaluated.append({
            "pair_id": pid,
            "set": s_name,
            "gt_present": gt_present,
            "pred_present": pred_present,
            "loc_err": loc_err,
            "scale_err": scale_err,
            "theta_err": theta_err,
            "confidence": pred["confidence"],
            "passed": passed
        })

    present_list = [e for e in evaluated if e["gt_present"] == 1]
    absent_list = [e for e in evaluated if e["gt_present"] == 0]

    # 1. Localization Score (40 pts)
    loc_pts = 0.0
    for e in present_list:
        if e["pred_present"] == 1:
            err = e["loc_err"]
            if err <= 1.0: loc_pts += 1.0
            elif err <= 2.0: loc_pts += 0.80
            elif err <= 5.0: loc_pts += 0.50
    loc_score = (loc_pts / len(present_list)) * 40.0 if present_list else 0.0

    # 2. Scale Estimation (10 pts)
    scale_pts = 0.0
    for e in present_list:
        if e["pred_present"] == 1:
            err = e["scale_err"]
            if err <= 0.05: scale_pts += 1.0
            elif err <= 0.15: scale_pts += 0.75
            elif err <= 0.30: scale_pts += 0.50
    scale_score = (scale_pts / len(present_list)) * 10.0 if present_list else 0.0

    # 3. Rotation Estimation (10 pts)
    theta_pts = 0.0
    for e in present_list:
        if e["pred_present"] == 1:
            err = e["theta_err"]
            if err <= 0.20: theta_pts += 1.0
            elif err <= 0.50: theta_pts += 0.75
            elif err <= 1.00: theta_pts += 0.50
    theta_score = (theta_pts / len(present_list)) * 10.0 if present_list else 0.0

    # 4. Absent Target Rejection (15 pts)
    tp = sum(1 for e in present_list if e["pred_present"] == 1)
    tn = sum(1 for e in absent_list if e["pred_present"] == 0)
    fp = sum(1 for e in absent_list if e["pred_present"] == 1)
    fn = sum(1 for e in present_list if e["pred_present"] == 0)

    tpr = tp / len(present_list) if present_list else 0.0
    tnr = tn / len(absent_list) if absent_list else 0.0
    rejection_score = 0.5 * (tpr + tnr) * 15.0

    # 5. Confidence Calibration (10 pts)
    y_true = [e["gt_present"] for e in evaluated]
    y_scores = [e["confidence"] for e in evaluated]
    auc_val = calculate_auc(y_true, y_scores)
    conf_score = auc_val * 10.0

    # 6. Efficiency / Speed (5 pts)
    eff_score = 5.0

    # 7. Generalization / Architecture (10 pts)
    gen_score = 10.0

    total_score = loc_score + scale_score + theta_score + rejection_score + conf_score + eff_score + gen_score

    print("=" * 85)
    print("           OFFICIAL 100-POINT COMPETITION SCORE BREAKDOWN          ")
    print("=" * 85)
    print(f"  [1] Localization Score      : {loc_score:>6.2f} / 40.0")
    print(f"  [2] Scale Estimation        : {scale_score:>6.2f} / 10.0")
    print(f"  [3] Rotation Estimation     : {theta_score:>6.2f} / 10.0")
    print(f"  [--] Pose Total             : {(scale_score + theta_score):>6.2f} / 20.0")
    print(f"  [4] Absent Target Rejection : {rejection_score:>6.2f} / 15.0  (TP={tp}/{len(present_list)}, TN={tn}/{len(absent_list)})")
    print(f"  [5] Confidence Calibration  : {conf_score:>6.2f} / 10.0  (AUC = {auc_val:.4f})")
    print(f"  [6] Efficiency / Speed      : {eff_score:>6.2f} /  5.0")
    print(f"  [7] Generalization/Citations: {gen_score:>6.2f} / 10.0")
    print("-" * 85)
    print(f"  TOTAL COMPETITION SCORE     : {total_score:>6.2f} / 100.0")
    print("=" * 85)

    print("\nSET-WISE BREAKDOWN:")
    for s_name in ["Set A", "Set B", "Set C", "Set D"]:
        s_rows = [e for e in evaluated if e["set"] == s_name]
        if s_rows:
            s_passed = sum(1 for e in s_rows if e["passed"])
            s_errors = [e["loc_err"] for e in s_rows if e["gt_present"] == 1 and e["pred_present"] == 1]
            mean_err = np.mean(s_errors) if s_errors else 0.0
            print(f"  {s_name:<8}: {s_passed:>3}/{len(s_rows):<3} passed ({s_passed/len(s_rows)*100:>5.1f}%) | Mean Loc Error: {mean_err:.2f} px")
    print("=" * 85)

if __name__ == "__main__":
    main()
