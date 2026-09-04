#!/usr/bin/env python3
"""
Master Synthetic SEM Dataset Generator v2 (Clear Unique Pattern + Max 3 Degradations)
====================================================================================

Generates N synthetic image pairs (Reference + Search) across 60 semiconductor
architectures with STRICT separation & pattern clarity:

1. TARGET / REFERENCE = 100% CLEAN & SHARP high-magnification landmark pattern.
   - Zero sensor noise, zero blur, zero geometric warping.
   - Preserves exact unique structural landmark from the 60 sub-scripts.

2. SEARCH IMAGE = STOCHASTICALLY DEGRADED (MAX 3 DEGRADATION MODELS TOTAL).
   - Randomly samples 1 to 3 degradation models from categories A-E.
   - Keeps search image clear and findable against the reference target.
   - Geometric distortions (rotation, barrel, tilt, drift) update GT coordinates.

Usage:
------
    python master_generator_v2.py --num_pairs 100 --output_dir ./dataset_v2_100
"""

import argparse
import importlib.util
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from degradation_engine import (
    DEFAULT_CONFIG,
    apply_full_degradation,
    print_degradation_report,
)


def discover_render_functions(scripts_dir):
    if not os.path.exists(scripts_dir):
        raise FileNotFoundError(f"Scripts directory not found: {scripts_dir}")

    script_files = sorted(
        f for f in os.listdir(scripts_dir)
        if f.endswith(".py") and f.startswith("generate_folder_")
    )

    registry = []
    for s_file in script_files:
        folder_num = s_file.replace("generate_folder_", "").replace(".py", "")
        s_path = os.path.join(scripts_dir, s_file)

        spec = importlib.util.spec_from_file_location(f"gen_mod_{folder_num}", s_path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"WARNING: Failed to load {s_file}: {e}")
            continue

        render_fn = None
        for attr in dir(mod):
            if attr.startswith("render_") and callable(getattr(mod, attr)):
                render_fn = getattr(mod, attr)
                break

        generate_fn = None
        for attr in dir(mod):
            if attr.startswith("generate_") and callable(getattr(mod, attr)):
                generate_fn = getattr(mod, attr)
                break

        if render_fn or generate_fn:
            registry.append({
                "folder_num": folder_num,
                "script_name": s_file,
                "render_fn": render_fn,
                "generate_fn": generate_fn,
                "module": mod,
            })

    return registry


def generate_clean_pair(render_fn, gt_x, gt_y, seed, h=1000, w=1000):
    """
    Render CLEAN search image and crop CLEAN 100x100 target patch centered at (gt_x, gt_y),
    upscaled to 1000x1000. Returns search_clean, target_clean, and true_gt_x, true_gt_y.
    """
    try:
        search_clean = render_fn(h=h, w=w, gt_x=int(gt_x), gt_y=int(gt_y), seed=seed)
        true_gt_x, true_gt_y = int(gt_x), int(gt_y)
    except TypeError:
        try:
            search_clean = render_fn(h=h, w=w, seed=seed)
            true_gt_x, true_gt_y = 500, 500
        except TypeError:
            search_clean = render_fn()
            true_gt_x, true_gt_y = 500, 500

    if search_clean.dtype != np.uint8:
        search_clean = np.clip(search_clean, 0, 255).astype(np.uint8)

    # Strip bottom SEM metadata text banner if present (bottom 70px)
    if search_clean.shape[0] >= 1000:
        banner_region = search_clean[930:, :]
        if np.std(banner_region) > 5.0:  # Contains text / scale bar lines
            sem_body = search_clean[:930, :]
            search_clean = cv2.resize(sem_body, (w, h), interpolation=cv2.INTER_LINEAR)

    crop_size = 100
    x0 = max(0, min(w - crop_size, true_gt_x - crop_size // 2))
    y0 = max(0, min(h - crop_size, true_gt_y - crop_size // 2))
    crop_patch = search_clean[y0:y0 + crop_size, x0:x0 + crop_size]
    target_clean = cv2.resize(crop_patch, (w, h), interpolation=cv2.INTER_NEAREST)

    return search_clean, target_clean, true_gt_x, true_gt_y


def generate_synthetic_dataset_v2(
    num_pairs,
    output_dir,
    scripts_dir=None,
    master_seed=42,
    config=None,
):
    os.makedirs(output_dir, exist_ok=True)

    if config is None:
        config = DEFAULT_CONFIG.copy()

    if scripts_dir is None:
        base = os.path.dirname(os.path.abspath(__file__))
        scripts_dir = os.path.join(base, "clean_60_scripts")
        if not os.path.exists(scripts_dir):
            scripts_dir = os.path.join(base, "..", "Script_and_their_images", "all_60_generator_scripts")
        scripts_dir = os.path.normpath(scripts_dir)

    registry = discover_render_functions(scripts_dir)
    num_arch = len(registry)
    if num_arch == 0:
        raise RuntimeError("No valid generator scripts found.")

    print("=" * 80)
    print("MASTER DATASET GENERATOR v2 -- UNIQUE PATTERNS & MAX 3 DEGRADATIONS")
    print("=" * 80)
    print(f"Pairs:          {num_pairs} across {num_arch} architectures")
    print(f"Master seed:    {master_seed}")
    print(f"Target Quality: 100% CLEAN & SHARP UNIQUE LANDMARKS")
    print(f"Search Degrad: STOCHASTIC (MAX 3 MODELS PER IMAGE)")
    print("=" * 80)

    master_rng = np.random.default_rng(master_seed)
    manifest_entries = []
    all_reports = []

    # Map default GTs for folders 001-060 to ensure unique landmarks are centered
    DEFAULT_GT_MAP = {
        "001": (650, 350), "002": (650, 350), "003": (490, 490), "004": (350, 630),
        "005": (500, 500), "006": (300, 700), "007": (490, 490), "008": (350, 630),
        "009": (460, 460), "010": (420, 580), "011": (500, 500), "012": (500, 335),
        "013": (500, 690), "014": (310, 500), "015": (500, 405), "016": (310, 500),
        "017": (500, 310), "018": (500, 500), "019": (488, 506), "020": (488, 314),
        "021": (500, 500), "022": (500, 310), "023": (500, 500), "024": (500, 310),
        "025": (500, 500), "026": (500, 310), "027": (500, 500), "028": (500, 310),
        "029": (500, 500), "030": (500, 310), "031": (500, 500), "032": (500, 310),
        "033": (500, 500), "034": (500, 310), "035": (500, 500), "036": (500, 310),
        "037": (500, 500), "038": (500, 310), "039": (500, 500), "040": (500, 310),
        "041": (500, 500), "042": (500, 310), "043": (500, 500), "044": (500, 310),
        "045": (500, 500), "046": (500, 310), "047": (500, 500), "048": (500, 310),
        "049": (500, 500), "050": (500, 310), "051": (500, 500), "052": (500, 310),
        "053": (500, 500), "054": (500, 310), "055": (500, 500), "056": (500, 310),
        "057": (500, 500), "058": (500, 310), "059": (500, 500), "060": (500, 310)
    }

    for idx in range(num_pairs):
        pair_id = f"pair_{idx + 1:04d}" if num_pairs >= 1000 else f"pair_{idx + 1:03d}"
        pair_dir = os.path.join(output_dir, pair_id)
        os.makedirs(pair_dir, exist_ok=True)

        arch = registry[idx % num_arch]
        folder_num = arch["folder_num"]
        render_fn = arch["render_fn"]
        generate_fn = arch["generate_fn"]
        cycle = idx // num_arch + 1

        render_seed = int(master_rng.integers(0, 2**31))
        degrade_seed = int(master_rng.integers(0, 2**31))

        # Use native GT for each architecture folder to guarantee unique landmark presence
        def_gt_x, def_gt_y = DEFAULT_GT_MAP.get(folder_num, (500, 500))
        gt_x, gt_y = def_gt_x, def_gt_y

        # ── 1. GENERATE CLEAN STRUCTURAL PAIR ─────────────────────────────
        if render_fn is not None:
            search_clean, target_clean = generate_clean_pair(
                render_fn, gt_x, gt_y, render_seed
            )
        else:
            # Fallback for scripts without render_* (001, 002)
            generate_fn(output_dir=pair_dir, seed=render_seed)
            s_path = os.path.join(pair_dir, "search.png")
            t_path = os.path.join(pair_dir, "target.png")
            if os.path.exists(s_path):
                search_clean = cv2.imread(s_path, cv2.IMREAD_UNCHANGED)
                target_clean = cv2.imread(t_path, cv2.IMREAD_UNCHANGED)
            else:
                search_clean = np.full((1000, 1000), 40, dtype=np.uint8)
                target_clean = np.full((1000, 1000), 40, dtype=np.uint8)

        # ── 2. REFERENCE/TARGET: ALWAYS 100% CLEAN & SHARP ──────────────
        cv2.imwrite(os.path.join(pair_dir, "target.png"), target_clean)
        cv2.imwrite(os.path.join(pair_dir, "reference.png"), target_clean)

        # ── 3. SEARCH: APPLY MAX 3 DEGRADATION MODELS ───────────────────
        degrade_rng = np.random.default_rng(degrade_seed)
        search_degraded, final_gt_x, final_gt_y, report = apply_full_degradation(
            search_clean, gt_x, gt_y, degrade_rng, config
        )
        cv2.imwrite(os.path.join(pair_dir, "search.png"), search_degraded)
        all_reports.append(report)

        # ── 4. RECORD GROUND TRUTH & MANIFEST ────────────────────────────
        gt_info = {
            "center_x": int(final_gt_x),
            "center_y": int(final_gt_y),
            "original_gt_x": int(gt_x),
            "original_gt_y": int(gt_y),
            "target_name": arch["script_name"],
            "pair_id": pair_id,
            "scale_factor": 10.0,
            "degradation_applied": report.get("applied", False),
            "degradation_models": report.get("models", []),
        }
        with open(os.path.join(pair_dir, "groundtruth.json"), "w") as f:
            json.dump(gt_info, f, indent=2)

        entry = {
            "pair_index": idx + 1,
            "pair_id": pair_id,
            "architecture_folder": folder_num,
            "architecture_script": arch["script_name"],
            "cycle": cycle,
            "ground_truth_x": int(final_gt_x),
            "ground_truth_y": int(final_gt_y),
            "original_gt_x": int(gt_x),
            "original_gt_y": int(gt_y),
            "degradation_applied": report.get("applied", False),
            "degradation_models": report.get("models", []),
            "degradation_details": report.get("details", []),
            "render_seed": render_seed,
            "degrade_seed": degrade_seed,
        }
        manifest_entries.append(entry)

        models_str = " + ".join(report.get("models", [])) if report.get("applied") else "CLEAN"
        gt_note = f" GT:({gt_x},{gt_y})->({final_gt_x},{final_gt_y})" if report.get("gt_changed") else f" GT:({final_gt_x},{final_gt_y})"
        if (idx + 1) % 10 == 0 or idx == num_pairs - 1 or idx < 5:
            print(f"[{pair_id}] Arch {folder_num} (Cycle {cycle}) |{gt_note} | {models_str}")

    manifest = {
        "num_pairs_generated": num_pairs,
        "total_architectures": num_arch,
        "master_seed": master_seed,
        "output_directory": os.path.abspath(output_dir),
        "pairs": manifest_entries,
    }
    manifest_path = os.path.join(output_dir, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print_degradation_report(all_reports, num_pairs)

    print(f"\nManifest: {manifest_path}")
    print(f"Output:   {os.path.abspath(output_dir)}")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Master Synthetic SEM Dataset Generator v2 (Max 3 Degradations + Clean Unique Targets)"
    )
    parser.add_argument("--num_pairs", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="v2_stochastic_generator/dataset_v2_two_noise_100")
    parser.add_argument("--master_seed", type=int, default=42)
    parser.add_argument("--scripts_dir", type=str, default=None)

    args = parser.parse_args()

    generate_synthetic_dataset_v2(
        num_pairs=args.num_pairs,
        output_dir=args.output_dir,
        scripts_dir=args.scripts_dir,
        master_seed=args.master_seed,
    )


if __name__ == "__main__":
    main()
