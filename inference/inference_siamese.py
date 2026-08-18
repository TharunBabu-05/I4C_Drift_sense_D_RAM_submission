import argparse
import sys
import time
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pyramid_siamese import PyramidSiameseNetwork

def load_grayscale(path):
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.float32)

def extract_patches(image_tensor, patch_size, stride):
    """
    Extracts sliding window patches from a 2D image tensor.
    image_tensor: [1, 1, H, W]
    Returns: [N, 1, patch_size, patch_size] and list of (y, x) coordinates for the top-left of each patch
    """
    B, C, H, W = image_tensor.shape
    patches = image_tensor.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    # patches is [1, 1, n_h, n_w, patch_size, patch_size]
    
    n_h = patches.shape[2]
    n_w = patches.shape[3]
    
    patches = patches.contiguous().view(-1, 1, patch_size, patch_size)
    
    coords = []
    for i in range(n_h):
        for j in range(n_w):
            coords.append((i * stride, j * stride))
            
    return patches, coords

def localize_siamese(model, reference_path, search_path, device, verbose=False):
    start_time = time.time()

    # 1. Load and normalize images
    ref_img = load_grayscale(reference_path) / 255.0
    search_img = load_grayscale(search_path) / 255.0

    # ---------------------------------------------------------
    # LEVEL 0: Coarse Search
    # ---------------------------------------------------------
    if verbose: print("Level 0: Coarse Siamese Search...")
    # Downscale Ref 20x (to 50x50), Search 2x (to 500x500)
    ref_l0 = TF.to_tensor(Image.fromarray(ref_img).resize((50, 50), Image.LANCZOS)).unsqueeze(0).to(device)
    search_l0 = TF.to_tensor(Image.fromarray(search_img).resize((500, 500), Image.LANCZOS)).unsqueeze(0).to(device)

    # Extract patches
    stride_l0 = 15 # 15px stride for speed (15 * 2 = 30px stride in full search)
    patches_l0, coords_l0 = extract_patches(search_l0, 50, stride_l0)
    
    if verbose: print(f"  Evaluating {len(patches_l0)} patches...")
    
    model.eval()
    with torch.no_grad():
        ref_emb_l0 = model.encoder(ref_l0)
        
        # Batch process patches
        batch_size = 512
        sim_scores_l0 = []
        for i in range(0, len(patches_l0), batch_size):
            batch = patches_l0[i:i+batch_size].to(device)
            batch_emb = model.encoder(batch)
            sim = model.compute_similarity(ref_emb_l0.expand(batch.size(0), -1), batch_emb)
            sim_scores_l0.append(sim.cpu())
            
        sim_scores_l0 = torch.cat(sim_scores_l0)
    
    # Get top 20 candidates
    topk_val, topk_idx = torch.topk(sim_scores_l0, min(20, len(sim_scores_l0)))
    
    candidates_l0 = []
    for val, idx in zip(topk_val, topk_idx):
        y, x = coords_l0[idx.item()]
        # Convert to center coords in full search image
        # patch size is 50, center is y+25, x+25. Scale back by 2x.
        full_cy = (y + 25) * 2
        full_cx = (x + 25) * 2
        candidates_l0.append((full_cy, full_cx, val.item()))

    # ---------------------------------------------------------
    # LEVEL 1: Nominal Verification
    # ---------------------------------------------------------
    if verbose: print("Level 1: Nominal Siamese Verification...")
    ref_l1 = TF.to_tensor(Image.fromarray(ref_img).resize((100, 100), Image.LANCZOS)).unsqueeze(0).to(device)
    search_tensor_full = TF.to_tensor(search_img).unsqueeze(0)
    
    candidates_l1_patches = []
    valid_candidates_l0 = []
    
    for cy, cx, s0 in candidates_l0:
        # Crop 100x100 window around (cy, cx) in full search
        y0 = max(0, cy - 50)
        y1 = min(1000, cy + 50)
        x0 = max(0, cx - 50)
        x1 = min(1000, cx + 50)
        
        # If crop is too small (on edges), pad it or skip
        crop = search_tensor_full[:, :, y0:y1, x0:x1]
        if crop.shape[2] == 100 and crop.shape[3] == 100:
            candidates_l1_patches.append(crop)
            valid_candidates_l0.append((cy, cx, s0))
            
    if len(candidates_l1_patches) > 0:
        batch_l1 = torch.cat(candidates_l1_patches).to(device)
        with torch.no_grad():
            ref_emb_l1 = model.encoder(ref_l1)
            batch_emb_l1 = model.encoder(batch_l1)
            sim_scores_l1 = model.compute_similarity(ref_emb_l1.expand(batch_l1.size(0), -1), batch_emb_l1).cpu().tolist()
            
        fused_candidates = []
        for i, (cy, cx, s0) in enumerate(valid_candidates_l0):
            s1 = sim_scores_l1[i]
            # Simple weighted fusion
            fused_score = 0.35 * s0 + 0.65 * s1
            fused_candidates.append((cy, cx, fused_score))
            
        fused_candidates.sort(key=lambda c: -c[2])
    else:
        fused_candidates = candidates_l0

    best_y, best_x, best_score = fused_candidates[0]

    # ---------------------------------------------------------
    # LEVEL 2: Fine Refinement
    # ---------------------------------------------------------
    if verbose: print("Level 2: Fine Sub-pixel Refinement...")
    # In a full implementation, we would extract a tight window, upscale 2x, and run through the refinement head.
    # For now, we will use the best integer coordinate from Level 1, as the refinement head requires specific training.
    result_x, result_y = best_x, best_y

    elapsed = time.time() - start_time
    if verbose:
        print(f"\nResult: ({result_x}, {result_y})")
        print(f"Time: {elapsed:.3f}s")

    return result_x, result_y

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
    else:
        if args.verbose: print("WARNING: No checkpoint loaded, using untrained weights.")

    x, y = localize_siamese(model, args.reference, args.search, device, args.verbose)
    print(f"({x}, {y})")

if __name__ == "__main__":
    main()
