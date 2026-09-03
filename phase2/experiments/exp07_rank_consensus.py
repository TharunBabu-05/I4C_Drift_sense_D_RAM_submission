#!/usr/bin/env python3
"""
EXP-07 — Top-5 Rank Consensus Candidate Selection
===================================================

STRICT ONE-CHANGE EXPERIMENT

Hypothesis:
    Raw score fusion (α·NCC + (1-α)·Siamese) is sensitive to score magnitude
    differences between periodic replicas. Rank-based fusion normalizes away
    these magnitude effects and may select the correct candidate more robustly.

Algorithm:
    For each pair:
    1. Run unmodified baseline to get Top-5 refined_results (NCC + Siamese scores).
    2. Rank candidates 1..5 by NCC (descending) → ncc_rank
    3. Rank candidates 1..5 by Siamese (descending) → siamese_rank
    4. Convert ranks to scores: rank_score = (N - rank) / (N - 1)  where N=5
       So rank 1 → 1.0, rank 5 → 0.0
    5. Fuse: consensus_score = w_ncc * ncc_rank_score + w_sia * siamese_rank_score
    6. Pick candidate with highest consensus_score (break ties by original rank).

Variants:
    A: w_ncc=1.0,  w_sia=0.0   (Pure NCC ranking)
    B: w_ncc=0.7,  w_sia=0.3
    C: w_ncc=0.5,  w_sia=0.5   (Equal rank fusion)
    D: w_ncc=0.3,  w_sia=0.7
    E: w_ncc=0.0,  w_sia=1.0   (Pure Siamese ranking)

Production files: UNMODIFIED. Model checkpoint: UNMODIFIED.
"""

import os
import sys
import json
import time
import math
import hashlib
import csv
import gc
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from phase2.phase2_inference import Phase2InferenceEngine, fit_parabola_subpixel

# ============================================================
# SCORING FUNCTIONS (Copied from official_100pt_audit.py)
# ============================================================

def calculate_auc(y_true, y_scores):
    pos_scores = [s for t, s in zip(y_true, y_scores) if t == 1]
    neg_scores = [s for t, s in zip(y_true, y_scores) if t == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5
    count = 0.0
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                count += 1.0
            elif p == n:
                count += 0.5
    return count / (len(pos_scores) * len(neg_scores))


def compute_100pt_breakdown(results):
    """Official 100-point scoring breakdown."""
    sets_data = {"Set A": [], "Set B": [], "Set C": [], "Set D": []}
    for r in results:
        sets_data[r["set"]].append(r)

    # 1. Localization (40 pts)
    def calc_loc_credit(entries):
        present = [e for e in entries if e["gt_found"] == 1]
        n = len(present)
        if n == 0:
            return 0.0, 0, 0, 0, 0, 0
        credits = []
        c1 = c2 = c3 = c5 = 0
        for e in present:
            if e["pred_found"] == 1:
                err = e["loc_err"]
                if err <= 1.0:   credits.append(1.00); c1 += 1
                elif err <= 2.0: credits.append(0.80); c2 += 1
                elif err <= 3.0: credits.append(0.60); c3 += 1
                elif err <= 5.0: credits.append(0.40); c5 += 1
                else:            credits.append(0.00)
            else:
                credits.append(0.00)
        return np.mean(credits), c1, c2, c3, c5, n

    credit_a, c1a, c2a, c3a, c5a, na = calc_loc_credit(sets_data["Set A"])
    credit_b, c1b, c2b, c3b, c5b, nb = calc_loc_credit(sets_data["Set B"])
    loc_score = (0.45 * credit_a + 0.55 * credit_b) * 40.0

    # 2. Pose (20 pts) — only when loc_err <= 5.0
    total_present = sum(1 for r in results if r["gt_found"] == 1)
    scale_credits = []
    theta_credits = []
    for r in results:
        if r["gt_found"] == 1:
            if r["pred_found"] == 1 and r["loc_err"] <= 5.0:
                s_err = r["scale_err"]
                t_err = r["theta_err"]
                scale_credits.append(1.0 if s_err <= 0.25 else (0.5 if s_err <= 0.50 else 0.0))
                theta_credits.append(1.0 if t_err <= 0.5 else (0.5 if t_err <= 1.5 else 0.0))
            else:
                scale_credits.append(0.0)
                theta_credits.append(0.0)
    scale_score = (sum(scale_credits) / total_present) * 10.0 if total_present > 0 else 0.0
    theta_score = (sum(theta_credits) / total_present) * 10.0 if total_present > 0 else 0.0
    pose_score = scale_score + theta_score

    # 3. Rejection F1 (15 pts)
    tp = sum(1 for r in results if r["gt_found"] == 1 and r["pred_found"] == 1)
    tn = sum(1 for r in results if r["gt_found"] == 0 and r["pred_found"] == 0)
    fp = sum(1 for r in results if r["gt_found"] == 0 and r["pred_found"] == 1)
    fn = sum(1 for r in results if r["gt_found"] == 1 and r["pred_found"] == 0)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    rejection_score = f1 * 15.0

    # 4. Confidence AUC (10 pts)
    y_true = [r["gt_found"] for r in results]
    y_scores = [r["pred_score"] for r in results]
    auc = calculate_auc(y_true, y_scores)
    confidence_score = auc * 10.0

    # 5. Efficiency (5 pts)
    runtimes = [r["runtime_ms"] for r in results]
    med_rt = float(np.median(runtimes))
    eff_score = 5.0 if med_rt <= 5000.0 else (2.5 if med_rt <= 10000.0 else 0.0)

    # 6. Generator / Citations (10 pts carried forward)
    gen_score = 10.0

    total = loc_score + pose_score + rejection_score + confidence_score + eff_score + gen_score

    return {
        "total_100_score": total,
        "loc_score": loc_score,
        "scale_score": scale_score,
        "theta_score": theta_score,
        "pose_score": pose_score,
        "rejection_score": rejection_score,
        "confidence_score": confidence_score,
        "eff_score": eff_score,
        "gen_score": gen_score,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "f1": f1, "auc": auc, "med_rt": med_rt,
        "loc_credit_a": credit_a, "loc_credit_b": credit_b,
        "set_a_1px": c1a, "set_a_2px": c2a, "set_a_3px": c3a, "set_a_5px": c5a, "set_a_n": na,
        "set_b_1px": c1b, "set_b_2px": c2b, "set_b_3px": c3b, "set_b_5px": c5b, "set_b_n": nb,
    }


# ============================================================
# RANK CONSENSUS SELECTION
# ============================================================

def rank_consensus_select(top5_candidates, w_ncc, w_sia, rejection_thresh, config):
    """
    Given Top-5 refined candidates (each with ncc_norm and siamese_sim),
    re-rank using rank-based consensus and return the selected candidate + metadata.
    """
    N = len(top5_candidates)
    if N == 0:
        return None, {}

    # Sort by NCC descending → assign NCC ranks
    by_ncc = sorted(range(N), key=lambda i: -top5_candidates[i]["ncc_norm"])
    ncc_rank = [0] * N
    for rank, idx in enumerate(by_ncc):
        ncc_rank[idx] = rank  # 0-based: 0 = best

    # Sort by Siamese descending → assign Siamese ranks
    by_sia = sorted(range(N), key=lambda i: -top5_candidates[i]["siamese_sim"])
    sia_rank = [0] * N
    for rank, idx in enumerate(by_sia):
        sia_rank[idx] = rank  # 0-based: 0 = best

    # Convert to normalized rank scores: rank 0 → 1.0, rank N-1 → 0.0
    denom = max(1, N - 1)
    consensus_scores = []
    for i in range(N):
        ncc_rs = (denom - ncc_rank[i]) / denom
        sia_rs = (denom - sia_rank[i]) / denom
        cs = w_ncc * ncc_rs + w_sia * sia_rs
        consensus_scores.append(cs)

    # Select candidate with highest consensus score; break ties by original adjusted_score
    best_idx = max(range(N), key=lambda i: (consensus_scores[i], top5_candidates[i].get("adjusted_score", 0.0)))
    best = top5_candidates[best_idx]

    # Use pre-computed subpixel-refined coordinates
    fine_x = best.get("fine_x", best["x"])
    fine_y = best.get("fine_y", best["y"])

    # Use original fused_score for rejection and confidence
    final_fused = best["fused_score"]

    if final_fused >= rejection_thresh:
        found = 1
        pred_x = float(round(fine_x, 2))
        pred_y = float(round(fine_y, 2))
        pred_theta = float(round(best["theta"], 2))
        pred_scale = float(round(best["scale"], 2))
    else:
        found = 0
        pred_x = pred_y = pred_theta = pred_scale = 0.0

    conf_score = 1.0 / (1.0 + math.exp(-config.CONFIDENCE_SLOPE * (final_fused - rejection_thresh)))
    conf_score = float(round(max(0.0001, min(0.9999, conf_score)), 4))

    debug_info = {
        "selected_idx": best_idx,
        "consensus_score": consensus_scores[best_idx],
        "ncc_rank": ncc_rank[best_idx],
        "sia_rank": sia_rank[best_idx],
        "all_consensus": consensus_scores,
        "all_ncc_ranks": ncc_rank,
        "all_sia_ranks": sia_rank,
    }

    return {
        "x": pred_x, "y": pred_y, "theta": pred_theta, "scale": pred_scale,
        "found": found, "score": conf_score, "fused_score": float(round(final_fused, 4)),
        "raw_ncc": float(round(best["ncc_norm"], 4)),
        "raw_siamese": float(round(best["siamese_sim"], 4)),
    }, debug_info


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():
    print("=" * 70)
    print("EXP-07: TOP-5 RANK CONSENSUS CANDIDATE SELECTION")
    print("=" * 70)

    # 1. Verify checkpoint integrity
    ckpt_path = "phase2_checkpoints/best_model_level1.pth"
    with open(ckpt_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    expected = "e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c"
    assert sha == expected, f"SHA-256 MISMATCH: {sha}"
    print(f"[OK] Checkpoint SHA-256 verified: {sha[:16]}...")

    # 2. Initialize unmodified engine
    engine = Phase2InferenceEngine(checkpoint_path="best_model_level1.pth", device="cpu")
    config = engine.config
    print(f"[OK] Engine initialized. NCC_WEIGHT={config.NCC_WEIGHT}, TAU={config.REJECTION_THRESHOLD}")

    # 3. Load dataset manifest
    data_dir = "local_phase2_60gen_200_pairs"
    manifest_path = os.path.join(data_dir, "phase2_60generator_manifest.csv")
    assert os.path.exists(manifest_path), f"Manifest not found: {manifest_path}"

    pairs = []
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(row)
    print(f"[OK] Loaded {len(pairs)} pairs from manifest")

    # 4. Define variants
    VARIANTS = {
        "A_pure_ncc":     {"w_ncc": 1.0, "w_sia": 0.0},
        "B_ncc70_sia30":  {"w_ncc": 0.7, "w_sia": 0.3},
        "C_equal":        {"w_ncc": 0.5, "w_sia": 0.5},
        "D_ncc30_sia70":  {"w_ncc": 0.3, "w_sia": 0.7},
        "E_pure_siamese": {"w_ncc": 0.0, "w_sia": 1.0},
    }

    # 5. Run inference ONCE and cache Top-5 for all variants
    print(f"\n{'='*70}")
    print("PHASE 1: Running baseline inference to extract Top-5 candidates...")
    print(f"{'='*70}")

    cached_top5 = {}
    cached_gt = {}
    cached_runtime = {}

    target_pairs = {"pair_006", "pair_066", "pair_186", "pair_116"}

    for pi, row in enumerate(pairs):
        pair_id = row["pair_id"]
        ref_path = row["reference_path"]
        search_path = row["search_path"]
        gt_x = float(row["x_gt"])
        gt_y = float(row["y_gt"])
        gt_theta = float(row["theta_gt"])
        gt_scale = float(row["scale_gt"])
        gt_found = int(row["found_gt"])
        set_name = row["set"]
        gen_id = row.get("generator_id", "generic")

        t0 = time.time()
        res_dict, best_coarse, refined_results = engine.localize_pair(
            ref_path, search_path,
            ncc_weight=0.5, rejection_thresh=0.42,
            scale_step=0.25, theta_step=1.0,
            return_diagnostics=True
        )
        t1 = time.time()
        runtime_ms = (t1 - t0) * 1000.0

        # Cache Top-5 refined results — strip match_matrix to prevent OOM
        # Pre-compute subpixel-refined coordinates for each candidate
        top5 = []
        for cand in refined_results[:5]:
            # Compute subpixel refinement now, before discarding match_matrix
            try:
                m_mat = cand["match_matrix"]
                if m_mat.shape[0] >= 3 and m_mat.shape[1] >= 3:
                    sub_3x3 = m_mat[:3, :3]
                    fine_x, fine_y = fit_parabola_subpixel(sub_3x3, cand["x"], cand["y"])
                else:
                    fine_x, fine_y = cand["x"], cand["y"]
            except Exception:
                fine_x, fine_y = cand["x"], cand["y"]

            top5.append({
                "x": cand["x"], "y": cand["y"],
                "scale": cand["scale"], "theta": cand["theta"],
                "ncc_norm": cand["ncc_norm"], "siamese_sim": cand["siamese_sim"],
                "fused_score": cand["fused_score"],
                "adjusted_score": cand.get("adjusted_score", 0.0),
                "fine_x": fine_x, "fine_y": fine_y,
                # NO match_matrix stored
            })

        # Aggressively free all large objects from inference
        del res_dict, best_coarse, refined_results
        gc.collect()

        cached_top5[pair_id] = top5
        cached_gt[pair_id] = {
            "gt_x": gt_x, "gt_y": gt_y, "gt_theta": gt_theta,
            "gt_scale": gt_scale, "gt_found": gt_found,
            "set": set_name, "gen_id": gen_id,
        }
        cached_runtime[pair_id] = runtime_ms

        if (pi + 1) % 20 == 0 or pair_id in target_pairs:
            n_cands = len(top5)
            gt_in = any(
                math.sqrt((c["x"] - gt_x)**2 + (c["y"] - gt_y)**2) <= 15.0
                for c in top5
            ) if gt_found == 1 else False
            marker = " *** TARGET ***" if pair_id in target_pairs else ""
            print(f"  [{pi+1:3d}/{len(pairs)}] {pair_id} | {n_cands} cands | GT-in-Top5={gt_in} | {runtime_ms:.0f}ms{marker}")

    print(f"\n[OK] Cached Top-5 for all {len(pairs)} pairs")

    # 6. Evaluate each variant by re-selecting from cached Top-5
    print(f"\n{'='*70}")
    print("PHASE 2: Evaluating rank-consensus variants...")
    print(f"{'='*70}")

    all_variant_results = {}
    all_variant_metrics = {}
    all_variant_target_debug = {}

    for vname, vparams in VARIANTS.items():
        w_ncc = vparams["w_ncc"]
        w_sia = vparams["w_sia"]
        print(f"\n--- Variant {vname} (w_ncc={w_ncc}, w_sia={w_sia}) ---")

        variant_results = []
        variant_target_debug = {}

        for row in pairs:
            pair_id = row["pair_id"]
            gt = cached_gt[pair_id]
            top5 = cached_top5[pair_id]
            runtime_ms = cached_runtime[pair_id]

            if len(top5) == 0:
                pred = {
                    "x": 0.0, "y": 0.0, "theta": 0.0, "scale": 0.0,
                    "found": 0, "score": 0.0, "fused_score": 0.0,
                    "raw_ncc": 0.0, "raw_siamese": 0.0,
                }
                debug = {}
            else:
                pred, debug = rank_consensus_select(
                    top5, w_ncc, w_sia,
                    rejection_thresh=0.42, config=config
                )

            # Compute errors
            if gt["gt_found"] == 1 and pred["found"] == 1:
                loc_err = math.sqrt((pred["x"] - gt["gt_x"])**2 + (pred["y"] - gt["gt_y"])**2)
                scale_err = abs(pred["scale"] - gt["gt_scale"])
                theta_err = abs(pred["theta"] - gt["gt_theta"])
            elif gt["gt_found"] == 0 and pred["found"] == 0:
                loc_err = scale_err = theta_err = 0.0
            else:
                loc_err = scale_err = theta_err = 999.0

            # GT in Top-5?
            gt_in_top5 = False
            best_top5_dist = 999.0
            if gt["gt_found"] == 1:
                for c in top5:
                    d = math.sqrt((c["x"] - gt["gt_x"])**2 + (c["y"] - gt["gt_y"])**2)
                    if d <= 15.0:
                        gt_in_top5 = True
                    if d < best_top5_dist:
                        best_top5_dist = d

            variant_results.append({
                "pair_id": pair_id,
                "set": gt["set"],
                "gen_id": gt["gen_id"],
                "gt_x": gt["gt_x"], "gt_y": gt["gt_y"],
                "gt_theta": gt["gt_theta"], "gt_scale": gt["gt_scale"],
                "gt_found": gt["gt_found"],
                "pred_x": pred["x"], "pred_y": pred["y"],
                "pred_theta": pred["theta"], "pred_scale": pred["scale"],
                "pred_found": pred["found"], "pred_score": pred["score"],
                "loc_err": loc_err, "scale_err": scale_err, "theta_err": theta_err,
                "runtime_ms": runtime_ms,
                "gt_in_top5": gt_in_top5,
                "best_top5_dist": best_top5_dist,
            })

            # Debug for target pairs
            if pair_id in target_pairs:
                variant_target_debug[pair_id] = {
                    "loc_err": round(loc_err, 2),
                    "pred_x": pred["x"], "pred_y": pred["y"],
                    "gt_x": gt["gt_x"], "gt_y": gt["gt_y"],
                    "found": pred["found"],
                    "selected_idx": debug.get("selected_idx", -1),
                    "consensus_score": round(debug.get("consensus_score", 0.0), 4),
                    "ncc_rank": debug.get("ncc_rank", -1),
                    "sia_rank": debug.get("sia_rank", -1),
                    "gt_in_top5": gt_in_top5,
                    "best_top5_dist": round(best_top5_dist, 2),
                }

        metrics = compute_100pt_breakdown(variant_results)
        all_variant_results[vname] = variant_results
        all_variant_metrics[vname] = metrics
        all_variant_target_debug[vname] = variant_target_debug

        print(f"  TOTAL: {metrics['total_100_score']:.2f} / 100")
        print(f"  Loc: {metrics['loc_score']:.2f}/40  Pose: {metrics['pose_score']:.2f}/20  Rej: {metrics['rejection_score']:.2f}/15  Conf: {metrics['confidence_score']:.2f}/10  Eff: {metrics['eff_score']:.2f}/5  Gen: 10/10")

    # 7. Print comparison table
    print(f"\n{'='*70}")
    print("VARIANT COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Variant':<22} {'Total':>7} {'Loc':>7} {'Pose':>7} {'Rej':>7} {'Conf':>7} {'Eff':>5}")
    print("-" * 70)
    # Also show baseline (which is the production code result)
    print(f"{'BASE (production)':22} {'46.77':>7} {'9.38':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>5}")
    for vname in VARIANTS:
        m = all_variant_metrics[vname]
        print(f"{vname:22} {m['total_100_score']:7.2f} {m['loc_score']:7.2f} {m['pose_score']:7.2f} {m['rejection_score']:7.2f} {m['confidence_score']:7.2f} {m['eff_score']:5.1f}")

    # 8. Target pair deep-dive
    print(f"\n{'='*70}")
    print("TARGET PAIR DEEP-DIVE (pair_006, pair_066, pair_116, pair_186)")
    print(f"{'='*70}")
    for pid in sorted(target_pairs):
        print(f"\n--- {pid} ---")
        # Show Top-5 candidate breakdown
        top5 = cached_top5.get(pid, [])
        gt = cached_gt.get(pid, {})
        if gt.get("gt_found") == 1:
            print(f"  GT: ({gt['gt_x']:.1f}, {gt['gt_y']:.1f})")
            for ci, c in enumerate(top5):
                d = math.sqrt((c["x"] - gt["gt_x"])**2 + (c["y"] - gt["gt_y"])**2)
                gt_marker = " ← GT" if d <= 15.0 else ""
                print(f"  Cand {ci}: ({c['x']:.1f}, {c['y']:.1f}) NCC={c['ncc_norm']:.4f} Sia={c['siamese_sim']:.4f} Fused={c['fused_score']:.4f} Dist={d:.1f}px{gt_marker}")

        print(f"  {'Variant':<22} {'LocErr':>8} {'Found':>6} {'SelIdx':>7} {'Consensus':>10} {'NCC_R':>6} {'SIA_R':>6}")
        for vname in VARIANTS:
            td = all_variant_target_debug.get(vname, {}).get(pid, {})
            if td:
                print(f"  {vname:22} {td['loc_err']:8.2f} {td['found']:6d} {td['selected_idx']:7d} {td['consensus_score']:10.4f} {td['ncc_rank']:6d} {td['sia_rank']:6d}")

    # 9. Determine best variant
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    best_vname = max(all_variant_metrics, key=lambda v: all_variant_metrics[v]["total_100_score"])
    best_m = all_variant_metrics[best_vname]
    base_total = 46.77
    base_loc = 9.38
    delta_total = best_m["total_100_score"] - base_total
    delta_loc = best_m["loc_score"] - base_loc

    print(f"Best variant: {best_vname}")
    print(f"  Total: {best_m['total_100_score']:.2f} / 100 (Delta = {delta_total:+.2f} vs BASE)")
    print(f"  Loc:   {best_m['loc_score']:.2f} / 40  (Delta = {delta_loc:+.2f} vs BASE)")

    if delta_total > 0 and delta_loc >= 0:
        print(f"\n  [PASS] CANDIDATE FOR PROMOTION: Total improved by {delta_total:.2f} without localization regression.")
    elif delta_total > 0 and delta_loc < 0:
        print(f"\n  [FAIL] REJECTED: Total improved but localization REGRESSED by {abs(delta_loc):.2f}.")
    else:
        print(f"\n  [FAIL] REJECTED: No improvement over baseline.")

    # 10. Save detailed results
    os.makedirs("phase2/results", exist_ok=True)
    output_path = "phase2/results/exp07_rank_consensus_results.json"
    output_data = {
        "experiment": "EXP-07",
        "description": "Top-5 Rank Consensus Candidate Selection",
        "baseline_total": base_total,
        "baseline_loc": base_loc,
        "checkpoint_sha256": sha[:16] + "...",
        "variants": {},
    }
    for vname in VARIANTS:
        m = all_variant_metrics[vname]
        output_data["variants"][vname] = {
            "params": VARIANTS[vname],
            "total_100_score": round(m["total_100_score"], 2),
            "loc_score": round(m["loc_score"], 2),
            "pose_score": round(m["pose_score"], 2),
            "rejection_score": round(m["rejection_score"], 2),
            "confidence_score": round(m["confidence_score"], 2),
            "eff_score": round(m["eff_score"], 2),
            "f1": round(m["f1"], 4),
            "auc": round(m["auc"], 4),
            "delta_total": round(m["total_100_score"] - base_total, 2),
            "delta_loc": round(m["loc_score"] - base_loc, 2),
            "target_pairs": all_variant_target_debug.get(vname, {}),
        }
    output_data["best_variant"] = best_vname
    output_data["verdict"] = "PROMOTE" if (delta_total > 0 and delta_loc >= 0) else "REJECT"

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[OK] Results saved to: {output_path}")

    # Save per-pair CSV for best variant
    csv_path = "phase2/results/exp07_best_variant_pairs.csv"
    best_results = all_variant_results[best_vname]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair_id", "set", "gen_id", "gt_found", "pred_found",
            "gt_x", "gt_y", "pred_x", "pred_y", "loc_err",
            "gt_scale", "pred_scale", "scale_err",
            "gt_theta", "pred_theta", "theta_err",
            "pred_score", "gt_in_top5", "runtime_ms"
        ])
        for r in best_results:
            writer.writerow([
                r["pair_id"], r["set"], r["gen_id"], r["gt_found"], r["pred_found"],
                r["gt_x"], r["gt_y"], r["pred_x"], r["pred_y"], round(r["loc_err"], 2),
                r["gt_scale"], r["pred_scale"], round(r["scale_err"], 2),
                r["gt_theta"], r["pred_theta"], round(r["theta_err"], 2),
                r["pred_score"], r["gt_in_top5"], round(r["runtime_ms"], 2)
            ])
    print(f"[OK] Per-pair CSV saved to: {csv_path}")


if __name__ == "__main__":
    run_experiment()
