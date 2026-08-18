#!/usr/bin/env python3
"""
Drift-Sense: Master Inference Script
=====================================

Unifies two localization pipelines behind a single entry point:

  1. Pure NCC Pyramid pipeline  (from inference.py, v3.0)
     - 3-level coarse-to-fine Normalized Cross-Correlation pyramid
     - No model weights required
     - This is the DEFAULT pipeline.

  2. Hybrid NCC + Siamese pipeline  (from inference_hybrid.py)
     - Fast coarse NCC search -> top-K candidates
     - PyramidSiameseNetwork (DL model) used to disambiguate candidates
     - Requires a trained checkpoint (e.g. resnet_final_16k_correct_Dataset_TLM.pt)
     - Activated automatically whenever --checkpoint is supplied.

Switching logic (DEFAULT IS NOW HYBRID)
----------------------------------------
    # Runs the hybrid NCC + Siamese pipeline (inference_hybrid.py behavior)
    # Uses DEFAULT_CHECKPOINT_PATH below automatically -- no flag needed.
    python master_inference.py --reference ref.png --search search.png

    # Runs the pure NCC pyramid pipeline (inference.py behavior)
    python master_inference.py --reference ref.png --search search.png --ncc

If the default checkpoint can't be found / torch or the model module can't be
imported, the script prints a clear error and falls back to the NCC pyramid
pipeline (never fails silently) unless --strict is passed, in which case it
exits with an error instead of falling back.

You can still point at a different checkpoint explicitly with --checkpoint.

Usage:
  python master_inference.py --reference ref.png --search search.png
  python master_inference.py --reference ref.png --search search.png --verbose
  python master_inference.py --reference ref.png --search search.png --ncc
  python master_inference.py --reference ref.png --search search.png --ncc --use_edge
  python master_inference.py --reference ref.png --search search.png --ncc --use_robust
  python master_inference.py --reference ref.png --search search.png --checkpoint other_model.pt
  python master_inference.py --reference ref.png --search search.png --strict

Output:
  Prints the predicted center (x, y) of the reference pattern in the search image.
"""

# Default checkpoint used by the hybrid pipeline when --checkpoint is not
# explicitly passed. Edit this path to match where your trained model lives.
DEFAULT_CHECKPOINT_PATH = "model/_resnet_final_16k_correct_Dataset_TLM/best_model_level1.pth"

import argparse
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image


# =============================================================================
# Shared helpers
# =============================================================================

def load_grayscale_pil(path):
    """Load an image as grayscale uint8 array via PIL (used by NCC pyramid path)."""
    img = Image.open(path).convert('L')
    return np.ascontiguousarray(np.array(img, dtype=np.uint8))


def load_grayscale_cv2(path):
    """Load an image as grayscale uint8 array via OpenCV (used by hybrid path)."""
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


# =============================================================================
# ============================  NCC PYRAMID PATH  ============================
# (ported from inference.py v3.0 -- unchanged logic)
# =============================================================================

def histogram_equalize(image):
    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.equalizeHist(img_uint8)


def light_denoise(image, sigma=1.0):
    ksize = int(round(sigma * 3)) | 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def median_denoise(image, size=3):
    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.medianBlur(img_uint8, size)


def bilateral_denoise(image, d=7, sigma_color=25, sigma_space=7):
    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.bilateralFilter(img_uint8, d, sigma_color, sigma_space)


def preprocess(image, denoise_sigma=0.8):
    img = histogram_equalize(image)
    if denoise_sigma > 0:
        img = light_denoise(img, sigma=denoise_sigma)
    return img


def preprocess_with_edge(image, denoise_sigma=0.8, edge_weight=0.6):
    img = histogram_equalize(image)
    if denoise_sigma > 0:
        img = light_denoise(img, sigma=denoise_sigma)

    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = cv2.magnitude(dx, dy)
    max_v = edge_mag.max()
    if max_v > 0:
        edge_mag = (edge_mag / max_v) * 255.0

    blended = edge_weight * edge_mag + (1.0 - edge_weight) * img.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def preprocess_robust(image, median_size=3, bilateral_d=7, sigma_color=25,
                       sigma_space=7, gauss_sigma=1.2):
    img = histogram_equalize(image)
    img = median_denoise(img, size=median_size)
    img = bilateral_denoise(img, d=bilateral_d,
                             sigma_color=sigma_color, sigma_space=sigma_space)
    img = light_denoise(img, sigma=gauss_sigma)
    return img


def resize_image(image, new_size):
    pil = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode='L')
    pil = pil.resize(new_size, Image.LANCZOS)
    return np.array(pil, dtype=np.float32)


def build_pyramid_level(reference, search, ref_target_size, search_target_size=None):
    template = resize_image(reference, (ref_target_size, ref_target_size))

    if search_target_size is not None and search_target_size != search.shape[1]:
        search_resized = resize_image(search, (search_target_size, search_target_size))
        scale_factor = search.shape[1] / search_target_size
    else:
        search_resized = search
        scale_factor = 1.0

    return template, search_resized, scale_factor


def ncc_search(search_image, template, top_k=20, min_score=-1.0):
    search_f32 = search_image.astype(np.float32)
    template_f32 = template.astype(np.float32)

    res = cv2.matchTemplate(search_f32, template_f32, cv2.TM_CCOEFF_NORMED)
    th, tw = template.shape

    candidates = []
    res_copy = res.copy()

    for _ in range(top_k):
        _, max_val, _, max_loc = cv2.minMaxLoc(res_copy)
        if max_val < min_score:
            break

        x, y = max_loc
        cy = y + th // 2
        cx = x + tw // 2
        candidates.append((cy, cx, float(max_val)))

        y_lo = max(0, y - th // 2)
        y_hi = min(res_copy.shape[0], y + th // 2)
        x_lo = max(0, x - tw // 2)
        x_hi = min(res_copy.shape[1], x + tw // 2)
        res_copy[y_lo:y_hi, x_lo:x_hi] = -np.inf

    return candidates


def disambiguate(candidates, image_center=(500, 500), ncc_threshold=0.05):
    if not candidates:
        return image_center
    if len(candidates) == 1:
        return (candidates[0][0], candidates[0][1])

    best_score = candidates[0][2]
    tied = [(y, x, s) for y, x, s in candidates if best_score - s < ncc_threshold]

    if len(tied) <= 1:
        return (candidates[0][0], candidates[0][1])

    cy, cx = image_center
    best = min(tied, key=lambda c: (c[0] - cy) ** 2 + (c[1] - cx) ** 2)
    return (best[0], best[1])


def localize_ncc_pyramid(reference_path, search_path, verbose=False,
                          use_edge=False, use_robust=False):
    """Pure NCC pyramid pipeline (identical behavior to inference.py v3.0)."""
    start_time = time.time()

    if verbose:
        print("Loading images...")
    reference = load_grayscale_pil(reference_path)
    search = load_grayscale_pil(search_path)

    if verbose:
        print("Preprocessing...")
        if use_edge:
            print("  Using edge enhancement")
        if use_robust:
            print("  Using robust preprocessing")

    if use_edge:
        ref_proc = preprocess_with_edge(reference, denoise_sigma=0.5, edge_weight=0.6)
        search_proc = preprocess_with_edge(search, denoise_sigma=0.8, edge_weight=0.6)
    elif use_robust:
        ref_proc = preprocess_robust(reference, median_size=3, bilateral_d=7, gauss_sigma=0.5)
        search_proc = preprocess_robust(search, median_size=3, bilateral_d=7, gauss_sigma=0.8)
    else:
        ref_proc = preprocess(reference, denoise_sigma=0.5)
        search_proc = preprocess(search, denoise_sigma=0.8)

    # LEVEL 0 -----------------------------------------------------------
    if verbose:
        print("Level 0: Coarse NCC search (500x500 search, 50x50 template)...")

    template_l0, search_l0, scale_l0 = build_pyramid_level(
        ref_proc, search_proc, ref_target_size=50, search_target_size=500
    )
    candidates_l0 = ncc_search(search_l0, template_l0, top_k=20, min_score=-0.5)

    if verbose:
        print(f"  Found {len(candidates_l0)} coarse candidates")

    candidates_l0_full = [
        (int(round(y * scale_l0)), int(round(x * scale_l0)), s)
        for y, x, s in candidates_l0
    ]

    # LEVEL 1 -----------------------------------------------------------
    if verbose:
        print("Level 1: Nominal NCC refinement (1000x1000 search, 100x100 template)...")

    template_l1, search_l1, scale_l1 = build_pyramid_level(
        ref_proc, search_proc, ref_target_size=100, search_target_size=None
    )
    all_l1 = ncc_search(search_l1, template_l1, top_k=30, min_score=-0.5)

    def best_l1_near(cy, cx, radius=60):
        best = None
        best_score = -np.inf
        for y, x, s in all_l1:
            if abs(y - cy) <= radius and abs(x - cx) <= radius and s > best_score:
                best_score = s
                best = (y, x, s)
        return best

    fused_candidates = []
    for cy0, cx0, s0 in candidates_l0_full:
        c1 = best_l1_near(cy0, cx0, radius=70)
        if c1 is not None:
            y1, x1, s1 = c1
            fused_score = 0.35 * s0 + 0.65 * s1
            fused_candidates.append((y1, x1, fused_score))
        else:
            fused_candidates.append((cy0, cx0, s0 * 0.5))

    fused_candidates.sort(key=lambda c: -c[2])

    if verbose:
        print("  Top-5 fused candidates:")
        for i, (y, x, s) in enumerate(fused_candidates[:5]):
            print(f"    #{i + 1}: ({x}, {y}) score={s:.4f}")

    # LEVEL 2 -----------------------------------------------------------
    best_y, best_x, best_score = fused_candidates[0] if fused_candidates else (500, 500, 0)

    if verbose:
        print(f"Level 2: Fine refinement around ({best_x}, {best_y})...")

    win_half = 100
    wy0 = max(0, best_y - win_half)
    wy1 = min(search_proc.shape[0], best_y + win_half)
    wx0 = max(0, best_x - win_half)
    wx1 = min(search_proc.shape[1], best_x + win_half)

    search_window = search_proc[wy0:wy1, wx0:wx1]

    if search_window.shape[0] > 20 and search_window.shape[1] > 20:
        fine_scale = 2
        win_h, win_w = search_window.shape
        search_fine = resize_image(search_window, (win_w * fine_scale, win_h * fine_scale))

        template_l2 = resize_image(ref_proc, (200, 200))

        if (search_fine.shape[0] >= template_l2.shape[0] and
                search_fine.shape[1] >= template_l2.shape[1]):
            fine_cands = ncc_search(search_fine, template_l2, top_k=5, min_score=-0.5)
            if fine_cands:
                fy, fx, fs = fine_cands[0]
                fy_win = fy / fine_scale
                fx_win = fx / fine_scale
                fy_full = int(round(wy0 + fy_win))
                fx_full = int(round(wx0 + fx_win))
                if verbose:
                    print(f"  Fine result: ({fx_full}, {fy_full}) score={fs:.4f}")
                if fs > best_score - 0.1:
                    best_y, best_x = fy_full, fx_full

    # Disambiguation ------------------------------------------------------
    if verbose:
        print("Disambiguating...")

    image_center = (search_proc.shape[0] // 2, search_proc.shape[1] // 2)
    result_y, result_x = disambiguate(
        fused_candidates[:10], image_center=image_center, ncc_threshold=0.03
    )

    if abs(best_y - result_y) <= 15 and abs(best_x - result_x) <= 15:
        result_y, result_x = best_y, best_x

    result_x = max(0, min(search_proc.shape[1] - 1, int(round(result_x))))
    result_y = max(0, min(search_proc.shape[0] - 1, int(round(result_y))))

    elapsed = time.time() - start_time

    if verbose:
        print(f"\nResult: ({result_x}, {result_y})")
        print(f"Time: {elapsed:.3f}s")

    return (result_x, result_y)


# =============================================================================
# =========================  HYBRID NCC + SIAMESE PATH  ======================
# (ported from inference_hybrid.py -- unchanged logic, imports kept lazy so
#  torch / the model module are only required when this path is actually used)
# =============================================================================

def non_max_suppression_peaks(scores, min_distance=10, top_k=3):
    """Top-K peaks in a 2D score map, at least `min_distance` apart."""
    peaks = []
    score_map = scores.copy()

    for _ in range(top_k):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(score_map)
        peaks.append((max_loc[0], max_loc[1], max_val))

        x, y = max_loc
        y0 = max(0, y - min_distance)
        y1 = min(score_map.shape[0], y + min_distance)
        x0 = max(0, x - min_distance)
        x1 = min(score_map.shape[1], x + min_distance)

        score_map[y0:y1, x0:x1] = -1.0

    return peaks


def _import_hybrid_deps():
    """
    Lazily import torch / torchvision / PyramidSiameseNetwork.
    Kept lazy so users who never pass --checkpoint don't need torch installed.
    """
    import torch
    import torchvision.transforms.functional as TF

    # Mirrors inference_hybrid.py's sys.path handling so `models.pyramid_siamese`
    # can be found relative to this script's parent directory.
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from models.pyramid_siamese import PyramidSiameseNetwork

    return torch, TF, PyramidSiameseNetwork


def localize_hybrid(model, reference_path, search_path, device, TF, torch, verbose=False):
    """Hybrid NCC + Siamese pipeline (identical behavior to inference_hybrid.py)."""
    start_time = time.time()

    ref_img_full = load_grayscale_cv2(reference_path)
    search_img = load_grayscale_cv2(search_path)

    t_ncc_start = time.time()
    ref_img_100 = cv2.resize(ref_img_full, (100, 100), interpolation=cv2.INTER_AREA)

    ref_eq = cv2.equalizeHist(ref_img_100)
    search_eq = cv2.equalizeHist(search_img)

    search_eq = cv2.GaussianBlur(search_eq, (3, 3), 1.0)
    ref_eq = cv2.GaussianBlur(ref_eq, (3, 3), 1.0)

    ncc_result = cv2.matchTemplate(search_eq, ref_eq, cv2.TM_CCOEFF_NORMED)

    top_peaks = non_max_suppression_peaks(ncc_result, min_distance=10, top_k=3)
    t_ncc_end = time.time()
    if verbose:
        print(f"NCC Search took {(t_ncc_end - t_ncc_start) * 1000:.1f}ms")

    t_cnn_start = time.time()
    ref_tensor = TF.to_tensor(ref_img_100).unsqueeze(0).to(device)

    candidate_patches = []
    valid_peaks = []

    for px, py, ncc_score in top_peaks:
        cy = py + 50
        cx = px + 50

        y0 = cy - 50
        y1 = cy + 50
        x0 = cx - 50
        x1 = cx + 50

        if y0 >= 0 and y1 <= search_img.shape[0] and x0 >= 0 and x1 <= search_img.shape[1]:
            crop = search_img[y0:y1, x0:x1]
            candidate_patches.append(TF.to_tensor(crop).unsqueeze(0))
            valid_peaks.append((cx, cy, ncc_score))

    if not candidate_patches:
        best_x, best_y, _ = top_peaks[0]
        return best_x + 50, best_y + 50

    batch = torch.cat(candidate_patches).to(device)

    model.eval()
    with torch.no_grad():
        ref_emb = model.encoder(ref_tensor)
        batch_emb = model.encoder(batch)
        sim_scores = model.compute_similarity(
            ref_emb.expand(batch.size(0), -1), batch_emb
        ).cpu().numpy()

    t_cnn_end = time.time()
    if verbose:
        print(f"CNN Disambiguation took {(t_cnn_end - t_cnn_start) * 1000:.1f}ms")

    best_fusion_score = -1.0
    best_coord = (500, 500)

    for i in range(len(valid_peaks)):
        cx, cy, ncc_val = valid_peaks[i]
        siam_val = max(0.001, float(sim_scores[i]))
        ncc_val_clamped = max(0.001, float(ncc_val))

        fusion_score = 0.3 * ncc_val_clamped + 0.7 * siam_val

        if verbose:
            print(f"  Cand {i + 1}: ({cx}, {cy}) | NCC: {ncc_val:.3f} | "
                  f"Siam: {siam_val:.3f} | Fused: {fusion_score:.3f}")

        if fusion_score > best_fusion_score:
            best_fusion_score = fusion_score
            best_coord = (float(cx), float(cy))

    elapsed = time.time() - start_time
    if verbose:
        print(f"\nFinal Result: {best_coord}")
        print(f"Total Time: {elapsed * 1000:.1f}ms")

    return best_coord[0], best_coord[1]


def run_hybrid_pipeline(reference_path, search_path, checkpoint_path, verbose=False):
    """
    Sets up torch/model/device and runs the hybrid pipeline.
    Raises RuntimeError with a clear message if anything required is missing.
    """
    try:
        torch, TF, PyramidSiameseNetwork = _import_hybrid_deps()
    except ImportError as e:
        raise RuntimeError(
            f"Hybrid pipeline requires torch/torchvision and models.pyramid_siamese "
            f"to be importable, but import failed: {e}"
        )

    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PyramidSiameseNetwork(encoder_type="resnet").to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    if verbose:
        print(f"Loaded checkpoint {checkpoint_path}")

    return localize_hybrid(model, reference_path, search_path, device, TF, torch, verbose=verbose)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Drift-Sense: Master Inference (Hybrid Siamese default + NCC Pyramid fallback)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline selection:
  (nothing / no flag)  -> runs the hybrid NCC + Siamese pipeline (inference_hybrid.py),
                          using DEFAULT_CHECKPOINT_PATH defined at the top of this file
  --ncc            -> runs the pure NCC pyramid pipeline (inference.py), no model needed
  --checkpoint PATH      -> runs the hybrid pipeline with a specific checkpoint instead of
                          the default one

Examples:
  python master_inference.py --reference ref.png --search search.png
  python master_inference.py --reference ref.png --search search.png --ncc
  python master_inference.py --reference ref.png --search search.png --ncc --use_edge
  python master_inference.py --reference ref.png --search search.png --ncc --use_robust
  python master_inference.py --reference ref.png --search search.png --checkpoint other_model.pt
        """
    )
    parser.add_argument("--reference", type=str, required=True,
                         help="Path to the reference image")
    parser.add_argument("--search", type=str, required=True,
                         help="Path to the search image")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_PATH,
                         help="Path to a trained Siamese model checkpoint. Defaults to "
                              f"'{DEFAULT_CHECKPOINT_PATH}'. Only used when --ncc "
                              "is NOT passed.")
    parser.add_argument("--ncc", action="store_true",
                         help="Force the pure NCC pyramid pipeline (no model / checkpoint "
                              "needed), instead of the default hybrid NCC+Siamese pipeline.")
    parser.add_argument("--verbose", action="store_true",
                         help="Print detailed debug output")
    parser.add_argument("--use_edge", action="store_true",
                         help="(NCC pyramid pipeline only) Enable Sobel edge enhancement")
    parser.add_argument("--use_robust", action="store_true",
                         help="(NCC pyramid pipeline only) Enable robust preprocessing")
    parser.add_argument("--strict", action="store_true",
                         help="If the hybrid pipeline can't be run (missing checkpoint/"
                              "torch/model), exit with an error instead of silently "
                              "falling back to the NCC pyramid pipeline.")

    args = parser.parse_args()

    if not os.path.isfile(args.reference):
        print(f"ERROR: Reference image not found: {args.reference}")
        sys.exit(1)
    if not os.path.isfile(args.search):
        print(f"ERROR: Search image not found: {args.search}")
        sys.exit(1)

    if args.ncc:
        # ---- Pure NCC pyramid pipeline (explicitly requested) ----
        x, y = localize_ncc_pyramid(args.reference, args.search, verbose=args.verbose,
                                     use_edge=args.use_edge, use_robust=args.use_robust)
    else:
        # ---- Hybrid NCC + Siamese pipeline (default) ----
        try:
            x, y = run_hybrid_pipeline(args.reference, args.search, args.checkpoint,
                                        verbose=args.verbose)
        except RuntimeError as e:
            print(f"ERROR: {e}")
            if args.strict:
                sys.exit(1)
            print("Falling back to the NCC pyramid pipeline...")
            x, y = localize_ncc_pyramid(args.reference, args.search, verbose=args.verbose,
                                         use_edge=args.use_edge, use_robust=args.use_robust)

    print(f"({x}, {y})")


if __name__ == "__main__":
    main()