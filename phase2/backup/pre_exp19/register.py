"""
Semicon India Hackathon — Phase 2 Official Competition Entry Point
====================================================================
"""

import os
import sys
import argparse
import csv
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phase2.phase2_inference import Phase2InferenceEngine

def main():
    parser = argparse.ArgumentParser(description="Semicon India Hackathon Phase 2 Registration Script")
    parser.add_argument("--input", type=str, required=True, help="Path to input pairs.csv file")
    parser.add_argument("--output", type=str, required=True, help="Path to save predictions.csv output")
    args = parser.parse_args()

    input_csv = args.input
    output_csv = args.output

    if not os.path.exists(input_csv):
        print(f"Error: Input file '{input_csv}' does not exist.")
        sys.exit(1)

    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints_phase2_v2_sunday", "best_model_phase2.pth")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model_level1.pth")

    engine = Phase2InferenceEngine(checkpoint_path=checkpoint_path, device="cpu")
    predictions = []

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair_id = row["pair_id"].strip()
            ref_path = row["reference_path"].strip()
            search_path = row["search_path"].strip()

            try:
                result = engine.localize_pair(
                    ref_path=ref_path,
                    search_path=search_path,
                    ncc_weight=0.5,
                    rejection_thresh=0.42,
                    scale_step=0.25,
                    theta_step=1.0
                )
                predictions.append({
                    "pair_id": pair_id,
                    "x": f"{result['x']:.4f}",
                    "y": f"{result['y']:.4f}",
                    "theta": f"{result['theta']:.4f}",
                    "scale": f"{result['scale']:.4f}",
                    "found": result["found"],
                    "score": f"{result['score']:.4f}"
                })
            except Exception as e:
                predictions.append({
                    "pair_id": pair_id,
                    "x": "0.0000",
                    "y": "0.0000",
                    "theta": "0.0000",
                    "scale": "0.0000",
                    "found": 0,
                    "score": "0.0000"
                })

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    fieldnames = ["pair_id", "x", "y", "theta", "scale", "found", "score"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Successfully processed {len(predictions)} pairs. Output written to '{output_csv}'.")

if __name__ == "__main__":
    main()
