#!/usr/bin/env python3
"""
Phase-2 Evaluation Script for Retrained Model
=============================================
Evaluates the retrained model (phase2_checkpoints/best_model_level1.pth)
on the official 200-pair Phase-2 datasets and outputs a side-by-side comparison table.
"""

import os
import sys
import json
import time
import math
import csv
import gc
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from phase2.phase2_inference import Phase2InferenceEngine
from phase2.experiments.evaluate_phase2_inference import evaluate_dataset, compute_official_metrics

def main():
    checkpoint_path = "phase2_checkpoints/best_model_level1.pth"
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Trained checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    print("=" * 70)
    print(f"EVALUATING RETRAINED PHASE-2 MODEL: {checkpoint_path}")
    print("=" * 70)

    engine = Phase2InferenceEngine(checkpoint_path=checkpoint_path, device="cpu")

    # Evaluate Dataset 2 (60-generator dataset)
    print("\n[1/2] Evaluating on 60-Generator Phase-2 Dataset (DS2)...")
    res_d2 = evaluate_dataset(engine, "local_phase2_60gen_200_pairs", "phase2_60generator_manifest.csv", "ds2_60gen_retrained")
    metrics_d2 = compute_official_metrics(res_d2)

    # Evaluate Dataset 1 (Generic dataset)
    print("\n[2/2] Evaluating on Generic Phase-2 Dataset (DS1)...")
    res_d1 = evaluate_dataset(engine, "local_phase2_200_pairs", "dataset_manifest.csv", "ds1_generic_retrained")
    metrics_d1 = compute_official_metrics(res_d1)

    # Comparison Table Output
    print("\n" + "=" * 70)
    print("PHASE-2 RETRAINED MODEL BENCHMARK RESULTS")
    print("=" * 70)
    
    print("\n--- 60-Generator Dataset (DS2) ---")
    print(f"Localization Score (/40)  : {metrics_d2['loc_score']:.2f}")
    print(f"Scale Recovery Score (/10): {metrics_d2['scale_score']:.2f}")
    print(f"Rotation Score (/10)      : {metrics_d2['theta_score']:.2f}")
    print(f"Rejection Score (/15)     : {metrics_d2['rejection_score']:.2f} (F1={metrics_d2['f1_score']:.4f})")
    print(f"Confidence Score (/10)    : {metrics_d2['confidence_score']:.2f} (AUC={metrics_d2['auc']:.4f})")
    print(f"CPU Efficiency Score (/5) : {metrics_d2['eff_score']:.2f} ({metrics_d2['median_runtime_ms']:.1f}ms)")
    print(f"TOTAL ESTIMATED SCORE (/90): {metrics_d2['total_score']:.2f} / 90.00")

    print("\n--- Generic Dataset (DS1) ---")
    print(f"Localization Score (/40)  : {metrics_d1['loc_score']:.2f}")
    print(f"Scale Recovery Score (/10): {metrics_d1['scale_score']:.2f}")
    print(f"Rotation Score (/10)      : {metrics_d1['theta_score']:.2f}")
    print(f"Rejection Score (/15)     : {metrics_d1['rejection_score']:.2f} (F1={metrics_d1['f1_score']:.4f})")
    print(f"Confidence Score (/10)    : {metrics_d1['confidence_score']:.2f} (AUC={metrics_d1['auc']:.4f})")
    print(f"CPU Efficiency Score (/5) : {metrics_d1['eff_score']:.2f} ({metrics_d1['median_runtime_ms']:.1f}ms)")
    print(f"TOTAL ESTIMATED SCORE (/90): {metrics_d1['total_score']:.2f} / 90.00")

    # Save JSON summary report
    report_dict = {
        "checkpoint": checkpoint_path,
        "ds2_60gen_metrics": metrics_d2,
        "ds1_generic_metrics": metrics_d1
    }
    
    report_path = "phase2/experiments/trained_model_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)
        
    print(f"\nDetailed metrics saved to: {report_path}")

if __name__ == "__main__":
    main()
