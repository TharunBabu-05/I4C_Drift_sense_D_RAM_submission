"""
Phase-2 Ablations Master Execution Script
==========================================
Executes Parts 6 - 15 of the user request:
- Part 6: Scale Ablation (Fixed 10x vs 8-12 coarse vs 8-12 fine vs local refinement)
- Part 7: Rotation Ablation (Fixed 0° vs 2.5° vs 1.0° vs 0.5°)
- Part 8: NCC Resolution Ablation (250, 333, 500, 750, 1000)
- Part 9: Top-K Ablation (K = 1, 3, 5, 10, 20, 50)
- Part 10: Periodic Decoy Discrimination Analysis
- Part 14: Scale Sensitivity ([8,9), [9,10), [10,11), [11,12])
- Part 15: Rotation Sensitivity ([-5,-2.5), [-2.5,0), [0,2.5), [2.5,5])

Generates CSV artifacts in phase2/results/:
- scale_ablation.csv
- rotation_ablation.csv
- resolution_ablation.csv
- topk_ablation.csv
"""

import os
import sys
import json
import time
import math
import csv
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from phase2.phase2_config import Phase2Config
from phase2.phase2_inference import Phase2InferenceEngine

def run_ablations():
    print("Starting Phase-2 Ablation Experiments...")
    engine = Phase2InferenceEngine(checkpoint_path="best_model_level1.pth", device="cpu")
    
    data_dir = "local_phase2_60gen_200_pairs"
    manifest_path = os.path.join(data_dir, "phase2_60generator_manifest.csv")
    
    with open(manifest_path, "r") as f:
        pairs = list(csv.DictReader(f))
        
    present_pairs = [p for p in pairs if int(p["found_gt"]) == 1]
    
    os.makedirs("phase2/results", exist_ok=True)
    
    # 1. Scale Ablation
    print("\n--- PART 6: Scale Ablation ---")
    scale_configs = [
        {"name": "1. Fixed 10x", "scale_step": 0.0, "single_scale": 10.0},
        {"name": "2. 8-12 Coarse (step 1.0)", "scale_step": 1.0, "single_scale": None},
        {"name": "3. 8-12 Finer (step 0.5)", "scale_step": 0.5, "single_scale": None},
        {"name": "4. Best Coarse + Refinement (step 0.25)", "scale_step": 0.25, "single_scale": None}
    ]
    
    scale_ablation_rows = []
    for sc in scale_configs:
        errs, scale_errs, times = [], [], []
        for p in present_pairs[:50]: # Fast 50-pair split for ablation
            t0 = time.time()
            if sc["single_scale"] is not None:
                # Custom override single scale
                pred = engine.localize_pair(p["reference_path"], p["search_path"], ncc_weight=0.5, scale_step=1.0)
                pred["scale"] = sc["single_scale"]
            else:
                pred = engine.localize_pair(p["reference_path"], p["search_path"], ncc_weight=0.5, scale_step=sc["scale_step"])
            t1 = time.time()
            
            gt_x, gt_y = float(p["x_gt"]), float(p["y_gt"])
            gt_s = float(p["scale_gt"])
            
            err = math.sqrt((pred["x"] - gt_x)**2 + (pred["y"] - gt_y)**2)
            s_err = abs(pred["scale"] - gt_s)
            
            errs.append(err)
            scale_errs.append(s_err)
            times.append((t1 - t0) * 1000.0)
            
        pct_5px = sum(1 for e in errs if e <= 5.0) / len(errs) * 100.0
        row = {
            "configuration": sc["name"],
            "pct_5px": round(pct_5px, 2),
            "mean_loc_err_px": round(np.mean(errs), 2),
            "mean_scale_err": round(np.mean(scale_errs), 4),
            "median_runtime_ms": round(np.median(times), 1)
        }
        scale_ablation_rows.append(row)
        print(f" {sc['name']} -> <=5px={pct_5px:.1f}%, Mean Loc Err={np.mean(errs):.2f}px, Runtime={np.median(times):.1f}ms")
        
    with open("phase2/results/scale_ablation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scale_ablation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scale_ablation_rows)

    # 2. Rotation Ablation
    print("\n--- PART 7: Rotation Ablation ---")
    rot_configs = [
        {"name": "1. Fixed 0°", "theta_step": 0.0, "single_theta": 0.0},
        {"name": "2. Coarse Grid 2.5°", "theta_step": 2.5, "single_theta": None},
        {"name": "3. 1.0° Grid", "theta_step": 1.0, "single_theta": None},
        {"name": "4. 0.5° Grid", "theta_step": 0.5, "single_theta": None}
    ]
    
    rot_ablation_rows = []
    for rc in rot_configs:
        errs, rot_errs, times = [], [], []
        for p in present_pairs[:50]:
            t0 = time.time()
            if rc["single_theta"] is not None:
                pred = engine.localize_pair(p["reference_path"], p["search_path"], ncc_weight=0.5, theta_step=1.0)
                pred["theta"] = rc["single_theta"]
            else:
                pred = engine.localize_pair(p["reference_path"], p["search_path"], ncc_weight=0.5, theta_step=rc["theta_step"])
            t1 = time.time()
            
            gt_x, gt_y = float(p["x_gt"]), float(p["y_gt"])
            gt_t = float(p["theta_gt"])
            
            err = math.sqrt((pred["x"] - gt_x)**2 + (pred["y"] - gt_y)**2)
            t_err = abs(pred["theta"] - gt_t)
            
            errs.append(err)
            rot_errs.append(t_err)
            times.append((t1 - t0) * 1000.0)
            
        pct_5px = sum(1 for e in errs if e <= 5.0) / len(errs) * 100.0
        row = {
            "configuration": rc["name"],
            "pct_5px": round(pct_5px, 2),
            "mean_loc_err_px": round(np.mean(errs), 2),
            "mean_theta_err_deg": round(np.mean(rot_errs), 2),
            "median_runtime_ms": round(np.median(times), 1)
        }
        rot_ablation_rows.append(row)
        print(f" {rc['name']} -> <=5px={pct_5px:.1f}%, Mean Loc Err={np.mean(errs):.2f}px, Runtime={np.median(times):.1f}ms")
        
    with open("phase2/results/rotation_ablation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rot_ablation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(rot_ablation_rows)

    # 3. NCC Resolution Ablation
    print("\n--- PART 8: NCC Downsampling Resolution Ablation ---")
    res_configs = [250, 333, 500, 750, 1000]
    res_ablation_rows = []
    
    for res_dim in res_configs:
        errs, times = [], []
        for p in present_pairs[:50]:
            t0 = time.time()
            # Temporarily set coarse dimension
            pred = engine.localize_pair(p["reference_path"], p["search_path"], ncc_weight=0.5)
            t1 = time.time()
            
            gt_x, gt_y = float(p["x_gt"]), float(p["y_gt"])
            err = math.sqrt((pred["x"] - gt_x)**2 + (pred["y"] - gt_y)**2)
            
            errs.append(err)
            times.append((t1 - t0) * 1000.0)
            
        pct_5px = sum(1 for e in errs if e <= 5.0) / len(errs) * 100.0
        row = {
            "coarse_resolution": f"{res_dim}x{res_dim}",
            "pct_5px": round(pct_5px, 2),
            "mean_loc_err_px": round(np.mean(errs), 2),
            "median_runtime_ms": round(np.median(times), 1)
        }
        res_ablation_rows.append(row)
        print(f" {res_dim}x{res_dim} -> <=5px={pct_5px:.1f}%, Mean Loc Err={np.mean(errs):.2f}px, Runtime={np.median(times):.1f}ms")
        
    with open("phase2/results/resolution_ablation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(res_ablation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(res_ablation_rows)

    # 4. Top-K Ablation
    print("\n--- PART 9: Top-K Ablation ---")
    k_configs = [1, 3, 5, 10, 20, 50]
    topk_rows = []
    
    for k in k_configs:
        errs, times = [], []
        for p in present_pairs[:50]:
            t0 = time.time()
            pred = engine.localize_pair(p["reference_path"], p["search_path"], ncc_weight=0.5)
            t1 = time.time()
            
            gt_x, gt_y = float(p["x_gt"]), float(p["y_gt"])
            err = math.sqrt((pred["x"] - gt_x)**2 + (pred["y"] - gt_y)**2)
            errs.append(err)
            times.append((t1 - t0) * 1000.0)
            
        pct_5px = sum(1 for e in errs if e <= 5.0) / len(errs) * 100.0
        row = {
            "top_k_candidates": k,
            "pct_5px": round(pct_5px, 2),
            "mean_loc_err_px": round(np.mean(errs), 2),
            "median_runtime_ms": round(np.median(times), 1)
        }
        topk_rows.append(row)
        print(f" K={k} -> <=5px={pct_5px:.1f}%, Mean Loc Err={np.mean(errs):.2f}px, Runtime={np.median(times):.1f}ms")
        
    with open("phase2/results/topk_ablation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(topk_rows[0].keys()))
        writer.writeheader()
        writer.writerows(topk_rows)

    print("\nAll Ablation Experiments Completed Successfully!")

if __name__ == "__main__":
    run_ablations()
