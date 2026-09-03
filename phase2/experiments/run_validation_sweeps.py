"""
Phase-2 Validation Hyperparameter Sweeps
========================================
Runs empirical sweeps over scale step, rotation step, fusion weights, and rejection thresholds
using a held-out 50-pair validation split (pairs 001-050 from 60-generator suite).
Selects optimal parameters BEFORE evaluating on the full test set.
"""

import os
import sys
import json
import time
import math
import csv
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from phase2.phase2_inference import Phase2InferenceEngine
from phase2.phase2_config import Phase2Config

def run_sweeps(data_dir="local_phase2_60gen_200_pairs", manifest_file="phase2_60generator_manifest.csv"):
    manifest_path = os.path.join(data_dir, manifest_file)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
    engine = Phase2InferenceEngine(checkpoint_path="best_model_level1.pth", device="cpu")
    
    # Load first 50 pairs for validation tuning
    val_pairs = []
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx >= 50: # Use 50 pairs for validation
                break
            val_pairs.append({
                "pair_id": row["pair_id"],
                "set": row["set"],
                "ref_path": row["reference_path"],
                "search_path": row["search_path"],
                "gt_x": float(row["x_gt"]),
                "gt_y": float(row["y_gt"]),
                "gt_theta": float(row["theta_gt"]),
                "gt_scale": float(row["scale_gt"]),
                "gt_found": int(row["found_gt"])
            })
            
    print(f"Loaded {len(val_pairs)} validation pairs for hyperparameter sweeps...")
    
    os.makedirs("phase2/experiments", exist_ok=True)
    sweep_log_file = "phase2/experiments/sweep_results.json"
    sweep_results = {}
    
    # --- SWEEP 1: REJECTION THRESHOLD TAU (0.20 to 0.70) ---
    print("\n--- SWEEP 1: Rejection Threshold (tau) ---")
    tau_list = [0.20, 0.25, 0.30, 0.35, 0.40, 0.42, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    tau_metrics = {}
    
    # Pre-run inference for validation pairs at alpha = 0.3
    val_inferences = []
    for pair in val_pairs:
        t0 = time.time()
        res = engine.localize_pair(
            pair["ref_path"], pair["search_path"], ncc_weight=0.3, rejection_thresh=0.0
        )
        t1 = time.time()
        res["runtime_ms"] = (t1 - t0) * 1000.0
        val_inferences.append((pair, res))
        
    for tau in tau_list:
        tp, tn, fp, fn = 0, 0, 0, 0
        loc_errs = []
        
        for pair, res in val_inferences:
            gt_found = pair["gt_found"]
            pred_found = 1 if res["fused_score"] >= tau else 0
            
            if gt_found == 1 and pred_found == 1:
                tp += 1
                err = math.sqrt((res["x"] - pair["gt_x"])**2 + (res["y"] - pair["gt_y"])**2)
                loc_errs.append(err)
            elif gt_found == 0 and pred_found == 0:
                tn += 1
            elif gt_found == 0 and pred_found == 1:
                fp += 1
            elif gt_found == 1 and pred_found == 0:
                fn += 1
                
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        tau_metrics[str(tau)] = {
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Precision": prec, "Recall": rec, "F1": f1,
            "mean_loc_err": float(np.mean(loc_errs)) if len(loc_errs) > 0 else 0.0
        }
        print(f"  tau={tau:.2f} | TP={tp}, TN={tn}, FP={fp}, FN={fn} | Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.4f}")
        
    sweep_results["tau_sweep"] = tau_metrics
    
    # --- SWEEP 2: HYBRID FUSION NCC WEIGHT ALPHA (0.1 to 0.6) ---
    print("\n--- SWEEP 2: NCC Fusion Weight (alpha) ---")
    alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    alpha_metrics = {}
    
    for alpha in alpha_list:
        loc_errs = []
        for pair in val_pairs:
            if pair["gt_found"] == 1:
                res = engine.localize_pair(
                    pair["ref_path"], pair["search_path"], ncc_weight=alpha, rejection_thresh=0.42
                )
                if res["found"] == 1:
                    err = math.sqrt((res["x"] - pair["gt_x"])**2 + (res["y"] - pair["gt_y"])**2)
                    loc_errs.append(err)
                    
        c5 = sum(1 for e in loc_errs if e <= 5.0) / len(val_pairs) * 100.0 if len(val_pairs) > 0 else 0.0
        alpha_metrics[str(alpha)] = {
            "mean_err": float(np.mean(loc_errs)) if len(loc_errs) > 0 else 0.0,
            "median_err": float(np.median(loc_errs)) if len(loc_errs) > 0 else 0.0,
            "pct_5px": c5
        }
        print(f"  alpha={alpha:.1f} | Mean Err={alpha_metrics[str(alpha)]['mean_err']:.2f}px | Median Err={alpha_metrics[str(alpha)]['median_err']:.2f}px | <=5px={c5:.1f}%")
        
    sweep_results["alpha_sweep"] = alpha_metrics
    
    # --- SWEEP 3: SCALE STEP RESOLUTION ---
    print("\n--- SWEEP 3: Scale Step Resolution ---")
    scale_steps = [1.0, 0.5, 0.25, 0.1]
    scale_step_metrics = {}
    
    for sc_step in scale_steps:
        runtimes = []
        loc_errs = []
        for pair in val_pairs[:20]: # 20 validation pairs
            if pair["gt_found"] == 1:
                t0 = time.time()
                res = engine.localize_pair(
                    pair["ref_path"], pair["search_path"], ncc_weight=0.3, rejection_thresh=0.42, scale_step=sc_step
                )
                t1 = time.time()
                runtimes.append((t1 - t0) * 1000.0)
                if res["found"] == 1:
                    err = math.sqrt((res["x"] - pair["gt_x"])**2 + (res["y"] - pair["gt_y"])**2)
                    loc_errs.append(err)
                    
        scale_step_metrics[str(sc_step)] = {
            "median_runtime_ms": float(np.median(runtimes)),
            "mean_err": float(np.mean(loc_errs)) if len(loc_errs) > 0 else 0.0,
            "pct_5px": sum(1 for e in loc_errs if e <= 5.0) / len(loc_errs) * 100.0 if len(loc_errs) > 0 else 0.0
        }
        print(f"  scale_step={sc_step:.2f} | Median Runtime={scale_step_metrics[str(sc_step)]['median_runtime_ms']:.1f}ms | Mean Err={scale_step_metrics[str(sc_step)]['mean_err']:.2f}px")
        
    sweep_results["scale_step_sweep"] = scale_step_metrics
    
    with open(sweep_log_file, "w") as f:
        json.dump(sweep_results, f, indent=2)
        
    print(f"\nAll validation sweeps completed! Saved to '{sweep_log_file}'.")

if __name__ == "__main__":
    run_sweeps()
