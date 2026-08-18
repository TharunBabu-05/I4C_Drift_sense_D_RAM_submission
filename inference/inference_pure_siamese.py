import argparse
import sys
import time
import os
import torch
import cv2
import numpy as np
import torchvision.transforms.functional as TF

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pyramid_siamese import PyramidSiameseNetwork

def load_grayscale(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {path}")
    return img

def extract_patches(image, centers, patch_size=100):
    """ Extracts patches centered at the given (x,y) coordinates. """
    half = patch_size // 2
    patches = []
    valid_centers = []
    
    h, w = image.shape
    for cx, cy in centers:
        y0 = cy - half
        y1 = cy + half
        x0 = cx - half
        x1 = cx + half
        
        if y0 >= 0 and y1 <= h and x0 >= 0 and x1 <= w:
            crop = image[y0:y1, x0:x1]
            patches.append(TF.to_tensor(crop).unsqueeze(0))
            valid_centers.append((cx, cy))
            
    if not patches:
        return None, []
        
    return torch.cat(patches), valid_centers

def localize_pure_siamese(model, reference_path, search_path, device, verbose=False):
    start_time = time.time()

    # 1. Load Images
    ref_img_full = load_grayscale(reference_path)
    search_img = load_grayscale(search_path)
    
    ref_img_100 = cv2.resize(ref_img_full, (100, 100), interpolation=cv2.INTER_AREA)
    ref_tensor = TF.to_tensor(ref_img_100).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        ref_emb = model.encoder(ref_tensor)
        
        # --- PHASE 1: Coarse Search (Stride 20) ---
        t_coarse_start = time.time()
        coarse_centers = []
        stride = 20
        half = 50
        h, w = search_img.shape
        
        for y in range(half, h - half, stride):
            for x in range(half, w - half, stride):
                coarse_centers.append((x, y))
                
        coarse_patches, valid_coarse_centers = extract_patches(search_img, coarse_centers, 100)
        
        best_cx, best_cy = 500, 500
        
        if coarse_patches is not None:
            coarse_patches = coarse_patches.to(device)
            # Batch process in chunks to avoid blowing up memory if CPU
            batch_size = 512
            best_sim = -float('inf')
            
            for i in range(0, coarse_patches.size(0), batch_size):
                batch = coarse_patches[i:i+batch_size]
                batch_emb = model.encoder(batch)
                sims = model.compute_similarity(ref_emb.expand(batch.size(0), -1), batch_emb).cpu().numpy()
                
                max_idx = np.argmax(sims)
                if sims[max_idx] > best_sim:
                    best_sim = sims[max_idx]
                    best_cx, best_cy = valid_coarse_centers[i + max_idx]
        
        if verbose: print(f"Coarse search found ({best_cx}, {best_cy}) in {(time.time()-t_coarse_start)*1000:.1f}ms")

        # --- PHASE 2: Fine Search (Stride 1 around best coarse center) ---
        t_fine_start = time.time()
        fine_centers = []
        radius = 20
        
        for y in range(best_cy - radius, best_cy + radius + 1, 1):
            for x in range(best_cx - radius, best_cx + radius + 1, 1):
                fine_centers.append((x, y))
                
        fine_patches, valid_fine_centers = extract_patches(search_img, fine_centers, 100)
        
        if fine_patches is not None:
            fine_patches = fine_patches.to(device)
            batch_size = 512
            
            for i in range(0, fine_patches.size(0), batch_size):
                batch = fine_patches[i:i+batch_size]
                batch_emb = model.encoder(batch)
                sims = model.compute_similarity(ref_emb.expand(batch.size(0), -1), batch_emb).cpu().numpy()
                
                max_idx = np.argmax(sims)
                if sims[max_idx] > best_sim:
                    best_sim = sims[max_idx]
                    best_cx, best_cy = valid_fine_centers[i + max_idx]
                    
        if verbose: print(f"Fine search found ({best_cx}, {best_cy}) in {(time.time()-t_fine_start)*1000:.1f}ms")

    elapsed = time.time() - start_time
    if verbose:
        print(f"\nFinal Result: ({best_cx}, {best_cy})")
        print(f"Total Time: {elapsed*1000:.1f}ms")

    return float(best_cx), float(best_cy)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=str, required=True)
    parser.add_argument("--search", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--encoder", type=str, default="mobilenet")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PyramidSiameseNetwork(encoder_type=args.encoder).to(device)
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        if args.verbose: print(f"Loaded checkpoint {args.checkpoint}")

    x, y = localize_pure_siamese(model, args.reference, args.search, device, args.verbose)
    print(f"({x}, {y})")

if __name__ == "__main__":
    main()
