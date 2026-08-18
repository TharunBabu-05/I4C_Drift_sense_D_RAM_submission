import os
import sys
import json
import time
import argparse
import subprocess
import math
import numpy as np

def run_inference(script_path, ref_path, search_path, checkpoint=None):
    cmd = ["python3", script_path, "--reference", ref_path, "--search", search_path]
    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])
        
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start
    
    # Parse output "(x, y)"
    try:
        out = result.stdout.strip().split('\n')[-1]
        out = out.replace('(', '').replace(')', '').split(',')
        x = float(out[0].strip())
        y = float(out[1].strip())
        return x, y, duration
    except Exception as e:
        print(f"Error parsing output from {script_path}: {result.stdout}")
        return -1, -1, duration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to base dataset (e.g. model/data)")
    parser.add_argument("--split", type=str, default="test", help="Which split to evaluate (train/val/test)")
    parser.add_argument("--siamese_script", type=str, default="model/inference/inference_siamese.py")
    parser.add_argument("--ncc_script", type=str, default="inference.py")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tolerance", type=float, default=5.0)
    args = parser.parse_args()

    meta_path = os.path.join(args.data_dir, "dataset_manifest.json")
    if not os.path.exists(meta_path):
        print(f"Manifest not found at {meta_path}")
        sys.exit(1)

    import json
    samples = []
    with open(meta_path, 'r') as f:
        manifest = json.load(f)
        for row in manifest.get('pairs', []):
            if row['split'] == args.split:
                samples.append(row)

    print(f"Evaluating {len(samples)} pairs...")
    print(f"{'ID':<10} | {'NCC Error':<12} | {'Siam Error':<12} | {'NCC Time':<10} | {'Siam Time':<10}")
    print("-" * 65)

    ncc_errors = []
    siam_errors = []
    ncc_times = []
    siam_times = []

    for s in samples:
        pair_id = s['pair_id']
        ref = os.path.join(args.data_dir, args.split, pair_id, "reference.png")
        search = os.path.join(args.data_dir, args.split, pair_id, "search.png")
        tx, ty = float(s['center_x']), float(s['center_y'])

        # NCC
        nx, ny, nt = run_inference(args.ncc_script, ref, search)
        n_err = math.sqrt((nx - tx)**2 + (ny - ty)**2) if nx != -1 else float('inf')
        
        # Siamese
        sx, sy, st = run_inference(args.siamese_script, ref, search, args.checkpoint)
        s_err = math.sqrt((sx - tx)**2 + (sy - ty)**2) if sx != -1 else float('inf')

        ncc_errors.append(n_err)
        siam_errors.append(s_err)
        ncc_times.append(nt)
        siam_times.append(st)

        print(f"{pair_id:<6} | {n_err:<12.2f} | {s_err:<12.2f} | {nt:<10.3f} | {st:<10.3f}")

    # Summary
    ncc_acc = sum(1 for e in ncc_errors if e <= args.tolerance) / len(samples) * 100
    siam_acc = sum(1 for e in siam_errors if e <= args.tolerance) / len(samples) * 100
    
    print("=" * 65)
    print("SUMMARY")
    print(f"Tolerance: {args.tolerance} px")
    print(f"NCC Accuracy:     {ncc_acc:.1f}%  | Mean Error: {np.mean(ncc_errors):.2f}px | Avg Time: {np.mean(ncc_times):.3f}s")
    print(f"Siamese Accuracy: {siam_acc:.1f}%  | Mean Error: {np.mean(siam_errors):.2f}px | Avg Time: {np.mean(siam_times):.3f}s")

if __name__ == "__main__":
    main()
