import os
import sys
import json
import cv2
import math
import time
import argparse
import numpy as np
import csv

# Import the inference functions from master_inference_claude.py
import master_inference_claude

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="all_60_pairs")
    parser.add_argument("--checkpoint", type=str, default="best_model_level1.pth")
    parser.add_argument("--output_csv", type=str, default="results_manifest.csv")
    args = parser.parse_args()

    data_dir = args.data_dir
    vis_dir = os.path.join(data_dir, "visualizer")
    os.makedirs(vis_dir, exist_ok=True)
    
    pairs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d != "visualizer"])
    
    print("=========================================================")
    print(f"EVALUATING HYBRID MODEL ON: {data_dir}")
    print("Generating visualizations, computing metrics & writing CSV...")
    print("=========================================================\n")
    
    errors = []
    times = []
    hits_5px = 0
    hits_0px = 0
    
    # Open CSV writer
    csv_file = open(args.output_csv, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "pair_id", "reference_path", "search_path", 
        "gt_x", "gt_y", "pred_x", "pred_y", 
        "error_px", "inference_time_ms", 
        "architecture", "noise_type"
    ])
    
    failure_cases = []
    
    for pair in pairs:
        pair_dir = os.path.join(data_dir, pair)
        files = os.listdir(pair_dir)
        
        ref_name = next((f for f in files if f in ["reference.png", "target.png"]), None)
        search_name = next((f for f in files if f in ["search.png"]), None)
        gt_name = next((f for f in files if "ground" in f.lower() or "gt" in f.lower()), None)
        
        if not ref_name or not search_name or not gt_name:
            continue
            
        ref = os.path.join(pair_dir, ref_name)
        search = os.path.join(pair_dir, search_name)
        gt_path = os.path.join(pair_dir, gt_name)
        
        with open(gt_path, 'r') as f:
            gt = json.load(f)
            if "target_x" in gt:
                gt_x, gt_y = gt["target_x"], gt["target_y"]
            elif "target" in gt and "search_center_xy" in gt["target"]:
                gt_x, gt_y = gt["target"]["search_center_xy"]
            else:
                gt_x, gt_y = float(gt.get('center_x', 500)), float(gt.get('center_y', 500))
                
        t0 = time.time()
        # Evaluate using Claude's run_hybrid_pipeline implementation
        pred_x, pred_y = master_inference_claude.run_hybrid_pipeline(ref, search, args.checkpoint, verbose=False)
        t1 = time.time()
        
        inf_time = (t1 - t0) * 1000.0
        times.append(inf_time)
        
        err = math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        errors.append(err)
        
        if err <= 5.0:
            hits_5px += 1
        if err <= 0.5:
            hits_0px += 1
            
        # Write to CSV
        csv_writer.writerow([
            pair, ref, search, 
            int(gt_x), int(gt_y), int(pred_x), int(pred_y), 
            round(err, 2), round(inf_time, 2),
            gt.get('architecture', 'N/A'),
            gt.get('applied_noise_type', 'N/A')
        ])
        
        # Visualize
        search_img = cv2.imread(search, cv2.IMREAD_GRAYSCALE)
        search_color = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)
        
        cv2.rectangle(search_color, (int(gt_x) - 50, int(gt_y) - 50), (int(gt_x) + 50, int(gt_y) + 50), (0, 255, 0), 3)
        cv2.rectangle(search_color, (int(pred_x) - 50, int(pred_y) - 50), (int(pred_x) + 50, int(pred_y) + 50), (0, 0, 255), 3)
        
        cv2.putText(search_color, f"GT: ({int(gt_x)}, {int(gt_y)})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(search_color, f"Pred: ({int(pred_x)}, {int(pred_y)})", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(search_color, f"Error: {err:.2f} px", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        
        if err > 5.0:
            cv2.line(search_color, (int(gt_x), int(gt_y)), (int(pred_x), int(pred_y)), (0, 165, 255), 3)
            print(f" [MISSED] {pair} | Error: {err:.2f}px | GT: ({int(gt_x)}, {int(gt_y)}) | Pred: ({int(pred_x)}, {int(pred_y)})")
            failure_cases.append(pair)
        else:
            print(f" [HIT]    {pair} | Error: {err:.2f}px | Time: {inf_time:.1f}ms")
            
        out_path = os.path.join(vis_dir, f"{pair}.png")
        cv2.imwrite(out_path, search_color)
        
    csv_file.close()
    
    total = len(errors)
    if total == 0:
        return
        
    mean_err = np.mean(errors)
    mean_time = np.mean(times)
    acc_5px = (hits_5px / total) * 100.0
    acc_0px = (hits_0px / total) * 100.0
    
    print("\n=========================================================")
    print("FINAL MACHINE LEARNING METRICS")
    print("=========================================================")
    print(f"[1] INFERENCE SPEED:   {mean_time:.2f} ms per image")
    print(f"[2] MEAN ERROR:        {mean_err:.2f} pixels")
    print(f"[3] ACCURACY (<= 5px): {acc_5px:.1f}% ({hits_5px} Hits, {total-hits_5px} Misses)")
    print(f"[4] PERFECT (0px err): {acc_0px:.1f}% ({hits_0px} Perfect Matches)")
    print("=========================================================")
    print(f"CSV Manifest saved to:   {args.output_csv}")
    print(f"Visualizations saved to: {vis_dir}")
    print("\n[ROBUSTNESS ANALYSIS]")
    print(f"Total Failure Cases (>5px): {len(failure_cases)}")
    if failure_cases:
        print(f"Example Failure Case: {failure_cases[0]} (Check visualization folder to analyze)")
    print("=========================================================")

if __name__ == "__main__":
    main()
