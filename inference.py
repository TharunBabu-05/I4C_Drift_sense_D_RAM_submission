#!/usr/bin/env python3
"""
Drift-Sense Phase 2 Master Inference CLI
========================================
Executes sub-pixel pattern localization, scale, rotation, and absent-target detection.

Usage:
    python inference.py --input pairs.csv --output predictions.csv
"""

import os
import sys
import csv
import time
import math
import argparse
import numpy as np

# Ensure local packages are importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from phase2.phase2_inference import Phase2InferenceEngine

def main():
    parser = argparse.ArgumentParser(description="Drift-Sense Phase 2 Standalone Inference Engine")
    parser.add_argument("--input", "-i", default="pairs.csv", help="Path to input pairs.csv file")
    parser.add_argument("--output", "-o", default="predictions.csv", help="Path to output predictions.csv file")
    parser.add_argument("--checkpoint", "-c", default=os.path.join(SCRIPT_DIR, "checkpoints", "best_model_phase2.pth"), help="Path to model checkpoint")
    parser.add_argument("--device", "-d", default="cpu", help="Device to use ('cpu' or 'cuda')")
    parser.add_argument("--scale-step", type=float, default=0.10, help="Fine scale search step")
    parser.add_argument("--theta-step", type=float, default=0.25, help="Fine rotation search step (deg)")
    args = parser.parse_args()

    input_csv = os.path.abspath(args.input)
    output_csv = os.path.abspath(args.output)
    
    if not os.path.exists(input_csv):
        print(f"ERROR: Input file not found: '{input_csv}'")
        sys.exit(1)

    print("=" * 85)
    print("        DRIFT-SENSE PHASE-2 INFERENCE ENGINE        ")
    print("=" * 85)
    print(f"  Input CSV      : {input_csv}")
    print(f"  Output CSV     : {output_csv}")
    print(f"  Checkpoint     : {args.checkpoint}")
    print(f"  Device         : {args.device}")
    print("=" * 85)

    # Initialize Engine
    t_start_init = time.time()
    engine = Phase2InferenceEngine(checkpoint_path=args.checkpoint, device=args.device)
    print(f"[OK] Model loaded in {(time.time() - t_start_init):.2f}s\n")

    # Read pairs
    pairs = []
    with open(input_csv, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("pair_id", "")
            
            # Support multiple CSV header formats
            ref_path = row.get("reference_path", "") or row.get("reference_image", "") or row.get("reference", "")
            srch_path = row.get("search_path", "") or row.get("search_image", "") or row.get("search", "")
            
            if not ref_path:
                ref_path = os.path.join("reference", f"{pid}.png")
            if not srch_path:
                srch_path = os.path.join("search", f"{pid}.png")
                
            # If relative path, resolve relative to input_csv folder or current dir
            base_dir = os.path.dirname(input_csv)
            if not os.path.isabs(ref_path):
                ref_candidate = os.path.join(base_dir, ref_path)
                if os.path.exists(ref_candidate):
                    ref_path = ref_candidate
                elif os.path.exists(os.path.join(SCRIPT_DIR, ref_path)):
                    ref_path = os.path.join(SCRIPT_DIR, ref_path)

            if not os.path.isabs(srch_path):
                srch_candidate = os.path.join(base_dir, srch_path)
                if os.path.exists(srch_candidate):
                    srch_path = srch_candidate
                elif os.path.exists(os.path.join(SCRIPT_DIR, srch_path)):
                    srch_path = os.path.join(SCRIPT_DIR, srch_path)

            pairs.append({
                "pair_id": pid,
                "reference_path": ref_path,
                "search_path": srch_path
            })

    total_pairs = len(pairs)
    print(f"[INFO] Loaded {total_pairs} pairs from '{input_csv}' to process.\n")

    predictions = []
    runtimes = []

    print(f"{'Index':<7} {'Pair ID':<8} {'Status':<14} {'Coordinates (x, y)':<22} {'Pose (scale, theta)':<24} {'Confidence':<12} {'Time':<9}")
    print("-" * 100)

    t_total_start = time.time()
    for idx, p in enumerate(pairs, 1):
        pid = p["pair_id"]
        ref_img_path = p["reference_path"]
        srch_img_path = p["search_path"]

        t0 = time.perf_counter()
        res = engine.localize_pair(
            ref_img_path,
            srch_img_path,
            scale_step=args.scale_step,
            theta_step=args.theta_step
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        runtimes.append(elapsed_ms)

        found = int(res["found"])
        pred_x = float(res["x"])
        pred_y = float(res["y"])
        pred_theta = float(res["theta"])
        pred_scale = float(res["scale"])
        conf_score = float(res["score"])

        status_str = "PRESENT" if found == 1 else "ABSENT"
        coord_str = f"({pred_x:.2f}, {pred_y:.2f})" if found == 1 else "(-, -)"
        pose_str = f"(z={pred_scale:.2f}, th={pred_theta:+.2f}deg)" if found == 1 else "(-, -)"

        print(f"[{idx:>3}/{total_pairs}] {pid:<8} {status_str:<14} {coord_str:<22} {pose_str:<24} {conf_score:<12.4f} {elapsed_ms:>6.1f} ms")

        predictions.append({
            "pair_id": pid,
            "present": found,
            "x": round(pred_x, 3),
            "y": round(pred_y, 3),
            "theta": round(pred_theta, 3),
            "scale": round(pred_scale, 3),
            "confidence": round(conf_score, 4)
        })

    # Write output CSV
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["pair_id", "present", "x", "y", "theta", "scale", "confidence"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    total_time = time.time() - t_total_start
    median_time = float(np.median(runtimes))

    print("\n" + "=" * 85)
    print("                     INFERENCE COMPLETE!                     ")
    print("=" * 85)
    print(f"  Total Processed Pairs   : {total_pairs}")
    print(f"  Total Execution Time    : {total_time:.2f} s")
    print(f"  Mean Speed per Pair     : {np.mean(runtimes):.1f} ms")
    print(f"  Median Speed per Pair   : {median_time:.1f} ms")
    print(f"  Output Predictions Saved: '{output_csv}'")
    print("=" * 85)
    print(f"\nTo score your predictions, run:\n  python score_predictions.py --ground-truth ground_truth.csv --predictions {args.output}\n")

if __name__ == "__main__":
    main()
