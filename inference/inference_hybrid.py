import argparse
import sys
import time
import os
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pyramid_siamese import PyramidSiameseNetwork

def load_grayscale(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

def non_max_suppression_peaks(scores, min_distance=10, top_k=3):
    """
    Finds the Top-K peaks in a 2D score map, ensuring they are at least `min_distance` apart.
    """
    peaks = []
    # Work on a copy so we can suppress regions
    score_map = scores.copy()
    
    for _ in range(top_k):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(score_map)
        
        # max_loc is (x, y)
        peaks.append((max_loc[0], max_loc[1], max_val))
        
        # Suppress the neighborhood around this peak
        x, y = max_loc
        y0 = max(0, y - min_distance)
        y1 = min(score_map.shape[0], y + min_distance)
        x0 = max(0, x - min_distance)
        x1 = min(score_map.shape[1], x + min_distance)
        
        score_map[y0:y1, x0:x1] = -1.0 # Suppress
        
    return peaks

def localize_hybrid(model, reference_path, search_path, device, verbose=False):
    start_time = time.time()

    # 1. Load Images (OpenCV for fast NCC, PIL for PyTorch)
    ref_img_full = load_grayscale(reference_path)
    search_img = load_grayscale(search_path)
    
    # 2. Fast Coarse NCC Search
    t_ncc_start = time.time()
    # Downscale reference to 100x100 (10x magnification difference)
    ref_img_100 = cv2.resize(ref_img_full, (100, 100), interpolation=cv2.INTER_AREA)
    
    # Histogram Equalization for robustness
    ref_eq = cv2.equalizeHist(ref_img_100)
    search_eq = cv2.equalizeHist(search_img)
    
    # Fast Gaussian Blur to suppress severe high-frequency SEM shot noise
    # This ensures the true target actually makes it into the top-20 peaks!
    search_eq = cv2.GaussianBlur(search_eq, (3, 3), 1.0)
    ref_eq = cv2.GaussianBlur(ref_eq, (3, 3), 1.0)
    
    # Fast Template Matching
    ncc_result = cv2.matchTemplate(search_eq, ref_eq, cv2.TM_CCOEFF_NORMED)
    
    # Get Top-10 candidates separated by at least 20 pixels
    # Triplet models perform best when comparing a small, highly curated pool of candidates
    top_peaks = non_max_suppression_peaks(ncc_result, min_distance=10, top_k=3)
    t_ncc_end = time.time()
    if verbose: print(f"NCC Search took {(t_ncc_end - t_ncc_start)*1000:.1f}ms")

    # 3. Siamese Disambiguation
    t_cnn_start = time.time()   
    # Prepare Reference Tensor
    ref_tensor = TF.to_tensor(ref_img_100).unsqueeze(0).to(device)
    
    candidate_patches = []
    valid_peaks = []
    
    for px, py, ncc_score in top_peaks:
        # px, py are the top-left corner of the matched template
        # The center of the 100x100 template is px + 50, py + 50
        cy = py + 50
        cx = px + 50
        
        # Crop exactly 100x100 from original search image
        y0 = cy - 50
        y1 = cy + 50
        x0 = cx - 50
        x1 = cx + 50
        
        # Handle edges
        if y0 >= 0 and y1 <= search_img.shape[0] and x0 >= 0 and x1 <= search_img.shape[1]:
            crop = search_img[y0:y1, x0:x1]
            candidate_patches.append(TF.to_tensor(crop).unsqueeze(0))
            valid_peaks.append((cx, cy, ncc_score))

    if not candidate_patches:
        # Fallback to the absolute best NCC if cropping fails on all edges
        best_x, best_y, _ = top_peaks[0]
        return best_x + 50, best_y + 50

    batch = torch.cat(candidate_patches).to(device)
    
    model.eval()
    with torch.no_grad():
        ref_emb = model.encoder(ref_tensor)
        batch_emb = model.encoder(batch)
        sim_scores = model.compute_similarity(ref_emb.expand(batch.size(0), -1), batch_emb).cpu().numpy()
        
    t_cnn_end = time.time()
    if verbose: print(f"CNN Disambiguation took {(t_cnn_end - t_cnn_start)*1000:.1f}ms")

    # 4. Fusion Strategy: Multiply NCC score (structural fit) by Siamese score (semantic disambiguation)
    # Clamp negative scores to 0.001 to prevent negative multiplication
    best_fusion_score = -1.0
    best_coord = (500, 500)
    
    for i in range(len(valid_peaks)):
        cx, cy, ncc_val = valid_peaks[i]
        siam_val = max(0.001, float(sim_scores[i]))
        ncc_val_clamped = max(0.001, float(ncc_val))
        
        # Fusion: Weighted combination is usually safest
        # NCC is very sharp, Siamese is semantically robust. 
        # We heavily trust Siamese to reject periodic hard negatives.
        fusion_score = 0.3 * ncc_val_clamped + 0.7 * siam_val
        
        if verbose:
            print(f"  Cand {i+1}: ({cx}, {cy}) | NCC: {ncc_val:.3f} | Siam: {siam_val:.3f} | Fused: {fusion_score:.3f}")
            
        if fusion_score > best_fusion_score:
            best_fusion_score = fusion_score
            best_coord = (float(cx), float(cy))

    elapsed = time.time() - start_time
    if verbose:
        print(f"\nFinal Result: {best_coord}")
        print(f"Total Time: {elapsed*1000:.1f}ms")

    return best_coord[0], best_coord[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=str, required=True)
    parser.add_argument("--search", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PyramidSiameseNetwork().to(device)
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        if args.verbose: print(f"Loaded checkpoint {args.checkpoint}")

    x, y = localize_hybrid(model, args.reference, args.search, device, args.verbose)
    print(f"({x}, {y})")

if __name__ == "__main__":
    main()
