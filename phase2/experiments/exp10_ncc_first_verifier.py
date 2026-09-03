#!/usr/bin/env python3
"""
EXP-10 — NCC-First + Siamese Verifier / Ranking Guard
======================================================

STRICT SINGLE-CHANGE EXPERIMENT

Hypothesis:
    NCC should be the PRIMARY spatial localization signal because periodic DRAM cell arrays
    cause the Siamese network to output near-perfect scores (0.99+) for spatially incorrect decoys
    while scoring true GT patches low (0.03 - 0.50).
    Using NCC as the primary candidate ranker while retaining Siamese strictly for rejection /
    confidence / optional guard will prevent periodic decoys from overriding high-NCC GT candidates.

Strategies Tested:
    Strategy A: Pure NCC Ranking (argmax(ncc_norm))
    Strategy B: NCC-First with Siamese Guard (NCC delta < 0.01 and Siamese delta > 0.10)

Production files: UNMODIFIED. Checkpoint: UNMODIFIED.
"""

import os
import sys
import json
import time
import math
import hashlib
import csv
import gc
import cv2
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from phase2.phase2_inference import Phase2InferenceEngine, fit_parabola_subpixel
from phase2.phase2_config import Phase2Config


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
    sets_data = {"Set A": [], "Set B": [], "Set C": [], "Set D": []}
    for r in results:
        sets_data[r["set"]].append(r)

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

    credit_a, _, _, _, _, _ = calc_loc_credit(sets_data["Set A"])
    credit_b, _, _, _, _, _ = calc_loc_credit(sets_data["Set B"])
    loc_score = (0.45 * credit_a + 0.55 * credit_b) * 40.0

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

    tp = sum(1 for r in results if r["gt_found"] == 1 and r["pred_found"] == 1)
    tn = sum(1 for r in results if r["gt_found"] == 0 and r["pred_found"] == 0)
    fp = sum(1 for r in results if r["gt_found"] == 0 and r["pred_found"] == 1)
    fn = sum(1 for r in results if r["gt_found"] == 1 and r["pred_found"] == 0)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    rejection_score = f1 * 15.0

    y_true = [r["gt_found"] for r in results]
    y_scores = [r["pred_score"] for r in results]
    auc = calculate_auc(y_true, y_scores)
    confidence_score = auc * 10.0

    runtimes = [r["runtime_ms"] for r in results]
    med_rt = float(np.median(runtimes))
    eff_score = 5.0 if med_rt <= 5000.0 else (2.5 if med_rt <= 10000.0 else 0.0)
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
        "f1": f1, "auc": auc, "med_rt": med_rt,
        "p90_rt": float(np.percentile(runtimes, 90)),
        "p99_rt": float(np.percentile(runtimes, 99)),
    }


def select_candidate_exp10(refined_results, strategy="strategy_a", rejection_thresh=0.42, config=None):
    """
    Selects top candidate from refined_results using EXP-10 strategies.
    
    Strategy A: Pure NCC ranking (argmax(ncc_norm))
    Strategy B: NCC-first with Siamese Guard:
                Start with max NCC candidate #1.
                If candidate #2 has |NCC_1 - NCC_2| < 0.01 and Siamese_2 > Siamese_1 + 0.10, allow override.
    """
    if len(refined_results) == 0:
        return None

    if strategy == "strategy_a":
        # Sort primarily by ncc_norm descending
        sorted_cands = sorted(refined_results, key=lambda c: (-c["ncc_norm"], -c.get("adjusted_score", 0.0)))
        best = sorted_cands[0]
    elif strategy == "strategy_b":
        # Sort primarily by ncc_norm descending
        sorted_cands = sorted(refined_results, key=lambda c: -c["ncc_norm"])
        best = sorted_cands[0]
        if len(sorted_cands) > 1:
            cand2 = sorted_cands[1]
            if (best["ncc_norm"] - cand2["ncc_norm"]) < 0.01 and (cand2["siamese_sim"] > best["siamese_sim"] + 0.10):
                best = cand2
    else:
        # Production fused score
        sorted_cands = sorted(refined_results, key=lambda c: -c.get("adjusted_score", 0.0))
        best = sorted_cands[0]

    # Subpixel refinement using match_matrix if present
    try:
        m_mat = best.get("match_matrix", None)
        if m_mat is not None and m_mat.shape[0] >= 3 and m_mat.shape[1] >= 3:
            sub_3x3 = m_mat[:3, :3]
            fine_x, fine_y = fit_parabola_subpixel(sub_3x3, best["x"], best["y"])
        else:
            fine_x, fine_y = best["x"], best["y"]
    except Exception:
        fine_x, fine_y = best["x"], best["y"]

    final_fused = best["fused_score"]
    tau = rejection_thresh

    if final_fused >= tau:
        found = 1
        pred_x = float(round(fine_x, 2))
        pred_y = float(round(fine_y, 2))
        pred_theta = float(round(best["theta"], 2))
        pred_scale = float(round(best["scale"], 2))
    else:
        found = 0
        pred_x = pred_y = pred_theta = pred_scale = 0.0

    conf_slope = config.CONFIDENCE_SLOPE if config else 12.0
    conf_score = 1.0 / (1.0 + math.exp(-conf_slope * (final_fused - tau)))
    conf_score = float(round(max(0.0001, min(0.9999, conf_score)), 4))

    return {
        "x": pred_x, "y": pred_y, "theta": pred_theta, "scale": pred_scale,
        "found": found, "score": conf_score, "fused_score": float(round(final_fused, 4)),
        "raw_ncc": float(round(best["ncc_norm"], 4)),
        "raw_siamese": float(round(best["siamese_sim"], 4)),
        "selected_cand": best
    }


def run_experiment():
    print("=" * 70)
    print("EXP-10: NCC-FIRST + SIAMESE VERIFIER / RANKING GUARD")
    print("=" * 70)

    # 1. Verify Checkpoint SHA-256
    ckpt_path = "phase2_checkpoints/best_model_level1.pth"
    with open(ckpt_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    expected = "e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c"
    assert sha == expected, f"SHA-256 MISMATCH: {sha}"
    print(f"[OK] Checkpoint SHA-256 verified: {sha[:16]}...")

    # 2. Init Engine
    engine = Phase2InferenceEngine(checkpoint_path="best_model_level1.pth", device="cpu")
    config = engine.config
    print(f"[OK] Engine initialized")

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

    target_pair_ids = {"pair_006", "pair_066", "pair_116", "pair_186"}

    # Cache inference diagnostics for all 200 pairs once using production generator
    print(f"\n{'='*70}")
    print("PHASE 1: Running production inference to capture candidates for all 200 pairs...")
    print(f"{'='*70}")

    cached_data = {}

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

        # Strip match_matrix to avoid keeping 200 * ~80 numpy arrays in memory
        clean_refined = []
        for c in refined_results:
            try:
                m_mat = c.get("match_matrix", None)
                if m_mat is not None and m_mat.shape[0] >= 3 and m_mat.shape[1] >= 3:
                    sub_3x3 = m_mat[:3, :3]
                    fine_x, fine_y = fit_parabola_subpixel(sub_3x3, c["x"], c["y"])
                else:
                    fine_x, fine_y = c["x"], c["y"]
            except Exception:
                fine_x, fine_y = c["x"], c["y"]

            clean_refined.append({
                "x": c["x"], "y": c["y"],
                "fine_x": fine_x, "fine_y": fine_y,
                "scale": c["scale"], "theta": c["theta"],
                "fused_score": c["fused_score"],
                "adjusted_score": c.get("adjusted_score", 0.0),
                "ncc_norm": c["ncc_norm"],
                "siamese_sim": c["siamese_sim"]
            })

        del res_dict, best_coarse, refined_results

        cached_data[pair_id] = {
            "gt": {"gt_x": gt_x, "gt_y": gt_y, "gt_theta": gt_theta, "gt_scale": gt_scale, "gt_found": gt_found, "set": set_name, "gen_id": gen_id},
            "prod_res": {
                "x": float(round(clean_refined[0]["fine_x"], 2)) if clean_refined[0]["fused_score"] >= 0.42 else 0.0,
                "y": float(round(clean_refined[0]["fine_y"], 2)) if clean_refined[0]["fused_score"] >= 0.42 else 0.0,
                "theta": float(round(clean_refined[0]["theta"], 2)) if clean_refined[0]["fused_score"] >= 0.42 else 0.0,
                "scale": float(round(clean_refined[0]["scale"], 2)) if clean_refined[0]["fused_score"] >= 0.42 else 0.0,
                "found": 1 if clean_refined[0]["fused_score"] >= 0.42 else 0,
                "score": float(round(max(0.0001, min(0.9999, 1.0 / (1.0 + math.exp(-12.0 * (clean_refined[0]["fused_score"] - 0.42))))), 4)),
                "fused_score": float(round(clean_refined[0]["fused_score"], 4)),
                "raw_ncc": float(round(clean_refined[0]["ncc_norm"], 4)),
                "raw_siamese": float(round(clean_refined[0]["siamese_sim"], 4))
            },
            "refined_results": clean_refined,
            "runtime_ms": runtime_ms
        }

        if (pi + 1) % 40 == 0 or pair_id in target_pair_ids:
            marker = " *** TARGET ***" if pair_id in target_pair_ids else ""
            print(f"  [{pi+1:3d}/200] {pair_id} | {len(clean_refined)} refined cands | {runtime_ms:.0f}ms{marker}")

        gc.collect()

    print(f"[OK] Caching complete.")

    # Strategies to evaluate
    strategies = {
        "production": "Production Fused Ranking (baseline)",
        "strategy_a": "Strategy A: Pure NCC Ranking (argmax(ncc))",
        "strategy_b": "Strategy B: NCC-First + Siamese Guard (delta_ncc < 0.01)"
    }

    all_strategy_results = {}
    all_strategy_metrics = {}
    all_strategy_diagnostics = {}
    all_strategy_target_traces = {}

    for strat_key, strat_name in strategies.items():
        print(f"\n{'='*70}")
        print(f"EVALUATING: {strat_name}")
        print(f"{'='*70}")

        results = []
        target_traces = {}

        # Candidate diagnostics
        diag_counts = {
            "gt_not_generated": 0,       # GT not in refined pool at all (>15px)
            "gt_lost_ncc_rank": 0,        # GT in refined pool, but NCC rank > 1
            "gt_lost_siamese_rank": 0,    # GT in refined pool, but Siamese rank > 1
            "gt_lost_fused_rank": 0,      # GT in refined pool, but Fused rank > 1
            "gt_selected_successfully": 0 # Selected prediction loc_err <= 5.0px
        }

        for row in pairs:
            pair_id = row["pair_id"]
            item = cached_data[pair_id]
            gt = item["gt"]
            refined_cands = item["refined_results"]
            runtime_ms = item["runtime_ms"]

            if strat_key == "production":
                pred = item["prod_res"]
            else:
                pred = select_candidate_exp10(refined_cands, strategy=strat_key, rejection_thresh=0.42, config=config)

            gt_x, gt_y = gt["gt_x"], gt["gt_y"]
            gt_s, gt_t = gt["gt_scale"], gt["gt_theta"]
            gt_found = gt["gt_found"]

            if gt_found == 1 and pred["found"] == 1:
                loc_err = math.sqrt((pred["x"] - gt_x)**2 + (pred["y"] - gt_y)**2)
                scale_err = abs(pred["scale"] - gt_s)
                theta_err = abs(pred["theta"] - gt_t)
            elif gt_found == 0 and pred["found"] == 0:
                loc_err = scale_err = theta_err = 0.0
            else:
                loc_err = scale_err = theta_err = 999.0

            results.append({
                "pair_id": pair_id, "set": gt["set"], "gen_id": gt["gen_id"],
                "gt_x": gt_x, "gt_y": gt_y, "gt_theta": gt_t, "gt_scale": gt_s, "gt_found": gt_found,
                "pred_x": pred["x"], "pred_y": pred["y"], "pred_theta": pred["theta"], "pred_scale": pred["scale"],
                "pred_found": pred["found"], "pred_score": pred["score"],
                "loc_err": loc_err, "scale_err": scale_err, "theta_err": theta_err, "runtime_ms": runtime_ms
            })

            # Diagnostic checks on present pairs
            if gt_found == 1:
                # Find GT candidate in refined pool
                gt_cand_idx = -1
                gt_cand_dist = 9999.0
                for c_idx, c in enumerate(refined_cands):
                    d = math.sqrt((c["x"] - gt_x)**2 + (c["y"] - gt_y)**2)
                    if d < gt_cand_dist:
                        gt_cand_dist = d
                        if d <= 15.0:
                            gt_cand_idx = c_idx

                if gt_cand_dist > 15.0:
                    diag_counts["gt_not_generated"] += 1
                else:
                    # GT exists in refined pool! Let's check its ranks
                    # NCC rank
                    by_ncc = sorted(range(len(refined_cands)), key=lambda i: -refined_cands[i]["ncc_norm"])
                    ncc_rank = by_ncc.index(gt_cand_idx) + 1 if gt_cand_idx in by_ncc else 99

                    # Siamese rank
                    by_sia = sorted(range(len(refined_cands)), key=lambda i: -refined_cands[i]["siamese_sim"])
                    sia_rank = by_sia.index(gt_cand_idx) + 1 if gt_cand_idx in by_sia else 99

                    # Fused rank
                    by_fused = sorted(range(len(refined_cands)), key=lambda i: -refined_cands[i].get("adjusted_score", 0.0))
                    fused_rank = by_fused.index(gt_cand_idx) + 1 if gt_cand_idx in by_fused else 99

                    if ncc_rank > 1: diag_counts["gt_lost_ncc_rank"] += 1
                    if sia_rank > 1: diag_counts["gt_lost_siamese_rank"] += 1
                    if fused_rank > 1: diag_counts["gt_lost_fused_rank"] += 1

                    if loc_err <= 5.0:
                        diag_counts["gt_selected_successfully"] += 1

            if pair_id in target_pair_ids:
                # Detailed trace for target pair
                gt_cand_idx = -1
                gt_cand_dist = 9999.0
                for c_idx, c in enumerate(refined_cands):
                    d = math.sqrt((c["x"] - gt_x)**2 + (c["y"] - gt_y)**2)
                    if d < gt_cand_dist:
                        gt_cand_dist = d
                        if d <= 15.0:
                            gt_cand_idx = c_idx

                by_ncc = sorted(range(len(refined_cands)), key=lambda i: -refined_cands[i]["ncc_norm"])
                by_sia = sorted(range(len(refined_cands)), key=lambda i: -refined_cands[i]["siamese_sim"])
                by_fused = sorted(range(len(refined_cands)), key=lambda i: -refined_cands[i].get("adjusted_score", 0.0))

                ncc_rank = by_ncc.index(gt_cand_idx) + 1 if gt_cand_idx != -1 else 99
                sia_rank = by_sia.index(gt_cand_idx) + 1 if gt_cand_idx != -1 else 99
                fused_rank = by_fused.index(gt_cand_idx) + 1 if gt_cand_idx != -1 else 99

                target_traces[pair_id] = {
                    "gt_x": gt_x, "gt_y": gt_y,
                    "gt_in_pool": gt_cand_dist <= 15.0,
                    "gt_dist_in_pool": round(gt_cand_dist, 2),
                    "gt_ncc_rank": ncc_rank,
                    "gt_sia_rank": sia_rank,
                    "gt_fused_rank": fused_rank,
                    "selected_x": pred["x"], "selected_y": pred["y"],
                    "loc_err": round(loc_err, 2),
                    "pred_found": pred["found"],
                    "top3_cands": [
                        {
                            "rank": r_i + 1, "x": round(c["x"], 1), "y": round(c["y"], 1),
                            "ncc": round(c["ncc_norm"], 4), "sia": round(c["siamese_sim"], 4),
                            "fused": round(c.get("fused_score", 0.0), 4),
                            "dist_gt": round(math.sqrt((c["x"]-gt_x)**2 + (c["y"]-gt_y)**2), 1)
                        } for r_i, c in enumerate(refined_cands[:5])
                    ]
                }

        metrics = compute_100pt_breakdown(results)
        all_strategy_results[strat_key] = results
        all_strategy_metrics[strat_key] = metrics
        all_strategy_diagnostics[strat_key] = diag_counts
        all_strategy_target_traces[strat_key] = target_traces

        print(f"---> TOTAL SCORE: {metrics['total_100_score']:.2f} / 100")
        print(f"     Loc: {metrics['loc_score']:.2f}/40 | Pose: {metrics['pose_score']:.2f}/20 | Rej: {metrics['rejection_score']:.2f}/15 | Conf: {metrics['confidence_score']:.2f}/10 | Eff: {metrics['eff_score']:.2f}/5")

    # Output comparison table
    print(f"\n{'='*70}")
    print("EXP-10 STRATEGY COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Strategy':<35} {'Total':>7} {'Loc':>7} {'Pose':>7} {'Rej':>7} {'Conf':>7} {'Eff':>5} {'Med_RT':>8}")
    print("-" * 85)
    for skey, sname in strategies.items():
        m = all_strategy_metrics[skey]
        print(f"{sname:<35} {m['total_100_score']:7.2f} {m['loc_score']:7.2f} {m['pose_score']:7.2f} {m['rejection_score']:7.2f} {m['confidence_score']:7.2f} {m['eff_score']:5.1f} {m['med_rt']:7.0f}ms")

    # Target pair deep-dive
    print(f"\n{'='*70}")
    print("TARGET PAIRS DETAILED TRACES")
    print(f"{'='*70}")
    for pid in sorted(target_pair_ids):
        print(f"\n--- {pid} ---")
        for skey, sname in strategies.items():
            t = all_strategy_target_traces[skey][pid]
            in_pool = "YES" if t["gt_in_pool"] else f"NO ({t['gt_dist_in_pool']}px)"
            print(f"  {skey:<15} | GT in pool: {in_pool:<10} | NCC_Rank: {t['gt_ncc_rank']:<2} | Sia_Rank: {t['gt_sia_rank']:<2} | Fused_Rank: {t['gt_fused_rank']:<2} | LocErr: {t['loc_err']}px")

    # Detailed top-3 candidates for pair_186 under strategy_a
    print(f"\n--- Detailed Top-3 candidates for pair_186 ---")
    p186_t = all_strategy_target_traces["strategy_a"]["pair_186"]
    for c in p186_t["top3_cands"]:
        print(f"  Rank {c['rank']}: ({c['x']}, {c['y']}) | NCC={c['ncc']} | Sia={c['sia']} | Fused={c['fused']} | DistGT={c['dist_gt']}px")

    # Diagnostics summary
    print(f"\n{'='*70}")
    print("CANDIDATE RETENTION DIAGNOSTICS (Present N=160)")
    print(f"{'='*70}")
    for skey, sname in strategies.items():
        dc = all_strategy_diagnostics[skey]
        print(f"\n{sname}:")
        print(f"  GT lost because not generated (>15px): {dc['gt_not_generated']}")
        print(f"  GT lost because NCC rank > 1:         {dc['gt_lost_ncc_rank']}")
        print(f"  GT lost because Siamese rank > 1:     {dc['gt_lost_siamese_rank']}")
        print(f"  GT lost because Fused rank > 1:       {dc['gt_lost_fused_rank']}")
        print(f"  GT selected successfully (<= 5.0px):  {dc['gt_selected_successfully']}")

    # Regression Analysis (Strategy A vs Baseline)
    base_m = all_strategy_metrics["production"]
    strat_a_m = all_strategy_metrics["strategy_a"]
    strat_a_res = all_strategy_results["strategy_a"]
    prod_res = all_strategy_results["production"]

    recovered = []
    regressed = []
    unchanged = []

    for idx, p_res in enumerate(prod_res):
        a_res = strat_a_res[idx]
        pid = p_res["pair_id"]
        p_err = p_res["loc_err"]
        a_err = a_res["loc_err"]

        if p_err > 5.0 and a_err <= 5.0:
            recovered.append((pid, p_err, a_err))
        elif p_err <= 5.0 and a_err > 5.0:
            regressed.append((pid, p_err, a_err))
        else:
            unchanged.append(pid)

    print(f"\n{'='*70}")
    print("REGRESSION ANALYSIS (Strategy A vs Production Baseline)")
    print(f"{'='*70}")
    print(f"  Recovered pairs (Baseline failed >5px, Strategy A passed <=5px): {len(recovered)}")
    for r in recovered:
        print(f"    + {r[0]}: Base error = {r[1]:.2f}px -> Strat A error = {r[2]:.2f}px")
    print(f"  Regressed pairs (Baseline passed <=5px, Strategy A failed >5px): {len(regressed)}")
    for r in regressed:
        print(f"    - {r[0]}: Base error = {r[1]:.2f}px -> Strat A error = {r[2]:.2f}px")
    print(f"  Unchanged pairs: {len(unchanged)}")

    # Decision
    base_total = base_m["total_100_score"]
    base_loc = base_m["loc_score"]
    a_total = strat_a_m["total_100_score"]
    a_loc = strat_a_m["loc_score"]

    delta_total = a_total - base_total
    delta_loc = a_loc - base_loc

    print(f"\n{'='*70}")
    print("DECISION RULE EVALUATION")
    print(f"{'='*70}")
    print(f"  Baseline Total: {base_total:.2f} | Strategy A Total: {a_total:.2f} (Delta = {delta_total:+.2f})")
    print(f"  Baseline Loc:   {base_loc:.2f} | Strategy A Loc:   {a_loc:.2f} (Delta = {delta_loc:+.2f})")

    if delta_total > 0 and delta_loc >= 0 and len(recovered) > 0 and len(regressed) == 0:
        verdict = "PROMOTE"
        print(f"\n  [PASS] PROMOTE: Strategy A improved total score by {delta_total:.2f} and localization by {delta_loc:.2f} with zero regressions!")
    elif delta_total > 0 and delta_loc >= 0:
        verdict = "PROMOTE"
        print(f"\n  [PASS] PROMOTE: Strategy A improved total score by {delta_total:.2f} and localization by {delta_loc:.2f}.")
    elif delta_total > 0 and delta_loc < 0:
        verdict = "REJECT"
        print(f"\n  [FAIL] REJECT: Total score improved but localization regressed by {abs(delta_loc):.2f}.")
    else:
        verdict = "REJECT"
        print(f"\n  [FAIL] REJECT: Total score did not improve over baseline.")

    # Save Strategy A CSV results
    os.makedirs("phase2/results", exist_ok=True)
    csv_path = "phase2/results/exp10_ncc_first_verifier.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair_id", "set", "gen_id", "gt_found", "pred_found",
            "gt_x", "gt_y", "pred_x", "pred_y", "loc_err",
            "gt_scale", "pred_scale", "scale_err",
            "gt_theta", "pred_theta", "theta_err",
            "pred_score", "runtime_ms"
        ])
        for r in strat_a_res:
            writer.writerow([
                r["pair_id"], r["set"], r["gen_id"], r["gt_found"], r["pred_found"],
                r["gt_x"], r["gt_y"], r["pred_x"], r["pred_y"], round(r["loc_err"], 2),
                r["gt_scale"], r["pred_scale"], round(r["scale_err"], 2),
                r["gt_theta"], r["pred_theta"], round(r["theta_err"], 2),
                r["pred_score"], round(r["runtime_ms"], 2)
            ])
    print(f"\n[OK] Saved CSV to {csv_path}")

    # Save Markdown Analysis Report
    report_path = "phase2/reports/EXP10_NCC_FIRST_VERIFIER_ANALYSIS.md"
    os.makedirs("phase2/reports", exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"""# EXP-10 — NCC-FIRST + SIAMESE VERIFIER ANALYSIS REPORT

## Executive Summary

- **Baseline Total**: {base_total:.2f} / 100
- **Strategy A Total (Pure NCC)**: {a_total:.2f} / 100
- **Strategy B Total (NCC + Guard)**: {all_strategy_metrics['strategy_b']['total_100_score']:.2f} / 100
- **Delta Total (Strategy A)**: {delta_total:+.2f}
- **Baseline Localization**: {base_loc:.2f} / 40
- **Strategy A Localization**: {a_loc:.2f} / 40
- **Delta Localization**: {delta_loc:+.2f}
- **Decision**: **{verdict}**

---

## 100-Point Score Breakdown

| Metric | Baseline (Production) | Strategy A (Pure NCC) | Strategy B (NCC + Guard) |
|---|---|---|---|
| **Localization /40** | {base_m['loc_score']:.2f} | {strat_a_m['loc_score']:.2f} | {all_strategy_metrics['strategy_b']['loc_score']:.2f} |
| **Scale /10** | {base_m['scale_score']:.2f} | {strat_a_m['scale_score']:.2f} | {all_strategy_metrics['strategy_b']['scale_score']:.2f} |
| **Rotation /10** | {base_m['theta_score']:.2f} | {strat_a_m['theta_score']:.2f} | {all_strategy_metrics['strategy_b']['theta_score']:.2f} |
| **Pose Total /20** | {base_m['pose_score']:.2f} | {strat_a_m['pose_score']:.2f} | {all_strategy_metrics['strategy_b']['pose_score']:.2f} |
| **Rejection /15** | {base_m['rejection_score']:.2f} | {strat_a_m['rejection_score']:.2f} | {all_strategy_metrics['strategy_b']['rejection_score']:.2f} |
| **Confidence /10** | {base_m['confidence_score']:.2f} | {strat_a_m['confidence_score']:.2f} | {all_strategy_metrics['strategy_b']['confidence_score']:.2f} |
| **Efficiency /5** | {base_m['eff_score']:.2f} | {strat_a_m['eff_score']:.2f} | {all_strategy_metrics['strategy_b']['eff_score']:.2f} |
| **Generator/Citations /10** | 10.00 | 10.00 | 10.00 |
| **TOTAL SCORE /100** | **{base_total:.2f}** | **{a_total:.2f}** | **{all_strategy_metrics['strategy_b']['total_100_score']:.2f}** |

---

## Candidate Diagnostics (Present N=160)

| Category | Production Baseline | Strategy A (Pure NCC) |
|---|---|---|
| **GT lost because not generated (>15px)** | {all_strategy_diagnostics['production']['gt_not_generated']} | {all_strategy_diagnostics['strategy_a']['gt_not_generated']} |
| **GT lost because NCC rank > 1** | {all_strategy_diagnostics['production']['gt_lost_ncc_rank']} | {all_strategy_diagnostics['strategy_a']['gt_lost_ncc_rank']} |
| **GT lost because Siamese rank > 1** | {all_strategy_diagnostics['production']['gt_lost_siamese_rank']} | {all_strategy_diagnostics['strategy_a']['gt_lost_siamese_rank']} |
| **GT lost because Fused rank > 1** | {all_strategy_diagnostics['production']['gt_lost_fused_rank']} | {all_strategy_diagnostics['strategy_a']['gt_lost_fused_rank']} |
| **GT selected successfully (<= 5.0px)** | {all_strategy_diagnostics['production']['gt_selected_successfully']} | {all_strategy_diagnostics['strategy_a']['gt_selected_successfully']} |

---

## Target Case Analysis

""")
        for pid in sorted(target_pair_ids):
            t_prod = all_strategy_target_traces["production"][pid]
            t_a = all_strategy_target_traces["strategy_a"][pid]
            in_pool_str = 'YES' if t_prod['gt_in_pool'] else f"NO ({t_prod['gt_dist_in_pool']}px)"
            f.write(f"### {pid}\n")
            f.write(f"- **GT Location**: ({t_prod['gt_x']}, {t_prod['gt_y']})\n")
            f.write(f"- **GT in Candidate Pool**: {in_pool_str}\n")
            f.write(f"- **GT Ranks**: NCC Rank = {t_prod['gt_ncc_rank']}, Siamese Rank = {t_prod['gt_sia_rank']}, Fused Rank = {t_prod['gt_fused_rank']}\n")
            f.write(f"- **Production Baseline Selected**: ({t_prod['selected_x']}, {t_prod['selected_y']}) — Error: {t_prod['loc_err']}px\n")
            f.write(f"- **Strategy A Selected**: ({t_a['selected_x']}, {t_a['selected_y']}) — Error: {t_a['loc_err']}px\n\n")

        f.write(f"""---

## Regression Analysis (Strategy A vs Production Baseline)

- **Recovered Pairs**: {len(recovered)}
""")
        for r in recovered:
            f.write(f"  - `{r[0]}`: Baseline error {r[1]:.2f}px -> Strategy A error {r[2]:.2f}px\n")

        f.write(f"""- **Regressed Pairs**: {len(regressed)}\n""")
        for r in regressed:
            f.write(f"  - `{r[0]}`: Baseline error {r[1]:.2f}px -> Strategy A error {r[2]:.2f}px\n")

        f.write(f"""- **Unchanged Pairs**: {len(unchanged)}

---

## Runtime Performance

- **Median Runtime**: {strat_a_m['med_rt']:.0f} ms
- **P90 Runtime**: {strat_a_m['p90_rt']:.0f} ms
- **P99 Runtime**: {strat_a_m['p99_rt']:.0f} ms

---

## Final Decision: {verdict}
""")

    print(f"[OK] Saved Markdown report to {report_path}")


if __name__ == "__main__":
    run_experiment()
