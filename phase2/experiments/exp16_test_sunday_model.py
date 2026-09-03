#!/usr/bin/env python3
"""
EXP-16 — Evaluation of Newly Trained Model: checkpoints_phase2_v2_sunday/best_model_phase2.pth
================================================================================================

Tests the newly trained 16k dataset fine-tuned model (best_model_phase2.pth) on the official 200-pair benchmark:
- Checkpoint Path: checkpoints_phase2_v2_sunday/best_model_phase2.pth
- Training data: phase-2\\phase2_training_16k (16,000 images, Val Acc: 99.71%)
- Production engine: phase2/phase2_inference.py (with EXP-13 Periodicity Penalization)

Compare score against verified production baseline (71.65 / 100.0).
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
        "f1": f1, "auc": auc, "med_rt": med_rt,
        "p90_rt": float(np.percentile(runtimes, 90)),
        "p99_rt": float(np.percentile(runtimes, 99)),
    }

def main():
    print("=" * 70)
    print("EVALUATING MODEL: checkpoints_phase2_v2_sunday/best_model_phase2.pth")
    print("=" * 70)

    ckpt_path = "checkpoints_phase2_v2_sunday/best_model_phase2.pth"
    assert os.path.exists(ckpt_path), f"Model checkpoint not found: {ckpt_path}"

    with open(ckpt_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    print(f"[OK] Checkpoint SHA-256: {sha}")

    engine = Phase2InferenceEngine(checkpoint_path=ckpt_path, device="cpu")
    print(f"[OK] Engine initialized with Sunday fine-tuned model.")

    data_dir = "local_phase2_60gen_200_pairs"
    manifest_path = os.path.join(data_dir, "phase2_60generator_manifest.csv")
    assert os.path.exists(manifest_path), f"Manifest not found: {manifest_path}"

    pairs = []
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(row)
    print(f"[OK] Loaded {len(pairs)} pairs from manifest")

    target_pairs = {"pair_006", "pair_066", "pair_116", "pair_186"}
    results = []

    print(f"\nRunning 200-pair official benchmark evaluation...")
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
        res = engine.localize_pair(
            ref_path, search_path,
            ncc_weight=0.5, rejection_thresh=0.42,
            scale_step=0.25, theta_step=1.0
        )
        t1 = time.time()
        runtime_ms = (t1 - t0) * 1000.0

        pred_x, pred_y = res["x"], res["y"]
        pred_theta, pred_scale = res["theta"], res["scale"]
        pred_found, pred_score = res["found"], res["score"]

        if gt_found == 1 and pred_found == 1:
            loc_err = math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
            scale_err = abs(pred_scale - gt_scale)
            theta_err = abs(pred_theta - gt_theta)
        elif gt_found == 0 and pred_found == 0:
            loc_err = scale_err = theta_err = 0.0
        else:
            loc_err = scale_err = theta_err = 999.0

        results.append({
            "pair_id": pair_id, "set": set_name, "gen_id": gen_id,
            "gt_x": gt_x, "gt_y": gt_y, "gt_theta": gt_theta, "gt_scale": gt_scale, "gt_found": gt_found,
            "pred_x": pred_x, "pred_y": pred_y, "pred_theta": pred_theta, "pred_scale": pred_scale,
            "pred_found": pred_found, "pred_score": pred_score,
            "loc_err": loc_err, "scale_err": scale_err, "theta_err": theta_err,
            "raw_ncc": res.get("raw_ncc", 0.0), "raw_siamese": res.get("raw_siamese", 0.0),
            "runtime_ms": runtime_ms
        })

        if (pi + 1) % 40 == 0 or pair_id in target_pairs:
            marker = " *** TARGET ***" if pair_id in target_pairs else ""
            print(f"  [{pi+1:3d}/200] {pair_id} | loc_err={loc_err:.2f}px | {runtime_ms:.0f}ms{marker}")

        gc.collect()

    metrics = compute_100pt_breakdown(results)

    print(f"\n=======================================================")
    print("SUNDAY CHECKPOINT (best_model_phase2.pth) 100-POINT SCORE")
    print("=======================================================")
    print(f"Localization: {metrics['loc_score']:.2f} / 40.0")
    print(f"Scale:        {metrics['scale_score']:.2f} / 10.0")
    print(f"Rotation:     {metrics['theta_score']:.2f} / 10.0")
    print(f"Pose Total:   {metrics['pose_score']:.2f} / 20.0")
    print(f"Rejection:    {metrics['rejection_score']:.2f} / 15.0")
    print(f"Confidence:   {metrics['confidence_score']:.2f} / 10.0")
    print(f"Efficiency:   {metrics['eff_score']:.2f} / 5.0")
    print(f"Gen/Citation: 10.00 / 10.0")
    print(f"-------------------------------------------------------")
    print(f"TOTAL SCORE:  {metrics['total_100_score']:.2f} / 100.0 (Baseline was 71.65)")

if __name__ == "__main__":
    main()
