#!/usr/bin/env python3
"""
Phase-2 EXP-05: Multi-Layer ResNet Feature Distance Ratio Analysis
===================================================================
Single-Change Experiment testing whether intermediate ResNet feature distances (Layer-1, Layer-2, Layer-3)
provide superior candidate discrimination for periodic DRAM cell replicas than the final 128-D embedding alone.

CONSTRAINTS:
- Original Checkpoint: best_model_level1.pth (READ ONLY - SHA256 UNTOUCHED)
- Production Code: 100% UNTOUCHED
- No GT information used for candidate selection or inference decisions
"""

import os
import sys
import math
import time
import hashlib
import csv
import gc
import cv2
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from phase2.phase2_inference import Phase2InferenceEngine, load_grayscale_image
from phase2.experiments.official_100pt_audit import compute_100pt_breakdown

class ResNetFeatureExtractor:
    def __init__(self, engine):
        self.engine = engine
        self.encoder = engine.model.encoder
        self.features = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook

        if hasattr(self.encoder, 'layer1'):
            self.hooks.append(self.encoder.layer1.register_forward_hook(get_hook('layer1')))
        if hasattr(self.encoder, 'layer2'):
            self.hooks.append(self.encoder.layer2.register_forward_hook(get_hook('layer2')))
        if hasattr(self.encoder, 'layer3'):
            self.hooks.append(self.encoder.layer3.register_forward_hook(get_hook('layer3')))

    def extract_patch_features(self, img_patch_u8):
        """
        Passes a 100x100 grayscale image patch through the frozen ResNet encoder
        and returns intermediate feature distance representations.
        """
        if img_patch_u8.dtype != np.uint8:
            img_patch_u8 = np.clip(img_patch_u8, 0, 255).astype(np.uint8)

        # Preprocess patch using standard Phase2 engine transform
        img_f = img_patch_u8.astype(np.float32) / 255.0
        tensor_in = torch.from_numpy(img_f).unsqueeze(0).unsqueeze(0).to(self.engine.device)

        self.features.clear()
        with torch.no_grad():
            embedding = self.encoder(tensor_in)

        # Calculate normalized feature vectors/tensors
        res = {
            "embedding": embedding.squeeze(0).cpu().numpy()
        }

        for layer_name in ["layer1", "layer2", "layer3"]:
            if layer_name in self.features:
                f_map = self.features[layer_name]
                # L2 normalize over channel dimension
                f_norm = F.normalize(f_map, p=2, dim=1).squeeze(0).cpu().numpy()
                res[layer_name] = f_norm

        return res

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

def compute_layer_distance(f_ref, f_cand):
    """
    Computes Euclidean distance between normalized reference and candidate feature representations.
    """
    if isinstance(f_ref, np.ndarray) and isinstance(f_cand, np.ndarray):
        diff = f_ref - f_cand
        return float(np.sqrt(np.sum(diff ** 2)))
    return 0.0

def run_single_pass_dataset(engine, data_dir, manifest_filename):
    manifest_path = os.path.join(data_dir, manifest_filename)
    pair_records = []
    extractor = ResNetFeatureExtractor(engine)

    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair_id = row["pair_id"]
            set_name = row["set"]
            ref_path = os.path.abspath(row["reference_path"])
            search_path = os.path.abspath(row["search_path"])

            gt_x = float(row["x_gt"])
            gt_y = float(row["y_gt"])
            gt_theta = float(row["theta_gt"])
            gt_scale = float(row["scale_gt"])
            gt_found = int(row["found_gt"])
            gen_id = row.get("generator_id", "generic")

            ref_img = load_grayscale_image(ref_path)
            search_img = load_grayscale_image(search_path)
            h_img, w_img = search_img.shape[:2]

            t0 = time.time()
            res_dict, best_coarse, refined_results = engine.localize_pair(
                ref_path, search_path, ncc_weight=0.5, rejection_thresh=0.42, scale_step=0.25, theta_step=1.0,
                return_diagnostics=True
            )
            t1 = time.time()
            runtime_ms = (t1 - t0) * 1000.0

            # Resize clean reference image to 100x100 patch for encoder
            ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
            ref_feats = extractor.extract_patch_features(ref_100)

            cand_details = []
            for r_idx, cand in enumerate(refined_results[:5]):
                cand_x, cand_y = cand["x"], cand["y"]
                cand_scale, cand_theta = cand["scale"], cand["theta"]

                # Extract candidate patch from search image
                ref_curr = ref_img
                if abs(cand_theta) > 0.01:
                    h_r, w_r = ref_curr.shape[:2]
                    M_rot = cv2.getRotationMatrix2D((w_r / 2.0, h_r / 2.0), -cand_theta, 1.0)
                    ref_curr = cv2.warpAffine(ref_curr, M_rot, (w_r, h_r), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                if abs(cand_scale - 1.0) > 0.01:
                    new_w = max(4, int(round(ref_curr.shape[1] * cand_scale)))
                    new_h = max(4, int(round(ref_curr.shape[0] * cand_scale)))
                    ref_curr = cv2.resize(ref_curr, (new_w, new_h), interpolation=cv2.INTER_AREA)

                th, tw = ref_curr.shape[:2]
                x0 = max(0, int(round(cand_x - tw / 2.0)))
                y0 = max(0, int(round(cand_y - th / 2.0)))
                x1 = min(w_img, x0 + tw)
                y1 = min(h_img, y0 + th)

                crop = search_img[y0:y1, x0:x1]
                if crop.size > 0:
                    crop_100 = cv2.resize(crop, (100, 100), interpolation=cv2.INTER_AREA)
                    cand_feats = extractor.extract_patch_features(crop_100)
                else:
                    cand_feats = ref_feats

                d_emb = compute_layer_distance(ref_feats["embedding"], cand_feats["embedding"])
                d_l1 = compute_layer_distance(ref_feats.get("layer1", 0), cand_feats.get("layer1", 0))
                d_l2 = compute_layer_distance(ref_feats.get("layer2", 0), cand_feats.get("layer2", 0))
                d_l3 = compute_layer_distance(ref_feats.get("layer3", 0), cand_feats.get("layer3", 0))

                dist_gt = math.sqrt((cand_x - gt_x)**2 + (cand_y - gt_y)**2) if gt_found == 1 else 999.0
                is_gt = (dist_gt <= 15.0)

                cand_details.append({
                    "rank_orig": r_idx + 1, "x": cand_x, "y": cand_y,
                    "scale": cand_scale, "theta": cand_theta,
                    "ncc_orig": cand.get("ncc_norm", 0.0), "siamese": cand.get("siamese_sim", 0.0),
                    "fused_orig": cand.get("fused_score", 0.0),
                    "d_emb": d_emb, "d_layer1": d_l1, "d_layer2": d_l2, "d_layer3": d_l3,
                    "is_gt": is_gt, "dist_gt": dist_gt
                })

            gt_rank_top5 = None
            for c in cand_details:
                if c["is_gt"]:
                    gt_rank_top5 = c["rank_orig"]
                    break

            pair_records.append({
                "pair_id": pair_id, "set": set_name, "gen_id": gen_id,
                "gt_x": gt_x, "gt_y": gt_y, "gt_theta": gt_theta, "gt_scale": gt_scale, "gt_found": gt_found,
                "runtime_ms": runtime_ms, "candidates": cand_details,
                "gt_in_top5": gt_rank_top5 is not None, "gt_rank_top5": gt_rank_top5
            })

            gc.collect()

    extractor.remove_hooks()
    return pair_records

def evaluate_exp05_layer_strategy(pair_records, layer_key="d_layer2", lam=0.10):
    results = []
    tau = 0.42

    gt_in_top5_count = 0
    gt_selected_count = 0

    for rec in pair_records:
        cand_list = rec["candidates"]

        # Compute normalized feature distances across top 5 candidates
        d_vals = [c[layer_key] for c in cand_list]
        max_d = max(d_vals) if len(d_vals) > 0 and max(d_vals) > 1e-5 else 1.0

        eval_cands = []
        for cand in cand_list:
            norm_d = cand[layer_key] / max_d
            score = cand["fused_orig"] - lam * norm_d
            eval_cands.append({**cand, "norm_d": norm_d, "eval_score": score})

        eval_cands.sort(key=lambda c: -c["eval_score"])
        selected = eval_cands[0]

        if rec["gt_in_top5"]:
            gt_in_top5_count += 1
            if selected["is_gt"]:
                gt_selected_count += 1

        pred_x, pred_y = selected["x"], selected["y"]
        pred_scale, pred_theta = selected["scale"], selected["theta"]
        pred_fused = selected["eval_score"]

        gt_found = rec["gt_found"]
        pred_found = 1 if pred_fused >= tau else 0
        pred_score = float(round(1.0 / (1.0 + math.exp(-6.0 * (pred_fused - tau))), 4))

        if gt_found == 1 and pred_found == 1:
            loc_err = math.sqrt((pred_x - rec["gt_x"])**2 + (pred_y - rec["gt_y"])**2)
            scale_err = abs(pred_scale - rec["gt_scale"])
            theta_err = abs(pred_theta - rec["gt_theta"])
        elif gt_found == 0 and pred_found == 0:
            loc_err = 0.0; scale_err = 0.0; theta_err = 0.0
        else:
            loc_err = 999.0; scale_err = 999.0; theta_err = 999.0

        results.append({
            "pair_id": rec["pair_id"], "set": rec["set"], "gen_id": rec["gen_id"],
            "gt_x": rec["gt_x"], "gt_y": rec["gt_y"], "gt_theta": rec["gt_theta"], "gt_scale": rec["gt_scale"],
            "gt_found": gt_found, "pred_x": pred_x, "pred_y": pred_y, "pred_theta": pred_theta, "pred_scale": pred_scale,
            "pred_found": pred_found, "pred_score": pred_score, "loc_err": loc_err, "scale_err": scale_err,
            "theta_err": theta_err, "runtime_ms": rec["runtime_ms"], "gt_in_top5": rec["gt_in_top5"],
            "eval_candidates": eval_cands
        })

    metrics = compute_100pt_breakdown(results)
    metrics["gt_in_top5_count"] = gt_in_top5_count
    metrics["gt_selected_count"] = gt_selected_count
    return metrics, results

def main():
    ckpt_path = "phase2_checkpoints/best_model_level1.pth"
    with open(ckpt_path, "rb") as f:
        sha256_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"Original Checkpoint SHA-256 Hash: {sha256_hash}")
    assert sha256_hash == "e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c", "SHA-256 Mismatch!"

    engine = Phase2InferenceEngine(checkpoint_path="best_model_level1.pth", device="cpu")

    print("\n===========================================================================")
    print("PHASE-2 EXP-05: MULTI-LAYER RESNET FEATURE DISTANCE RATIO EVALUATION")
    print("===========================================================================")

    print("Running single-pass feature extraction on DS2 (local_phase2_60gen_200_pairs)...")
    sys.stdout.flush()
    ds2_records = run_single_pass_dataset(engine, "local_phase2_60gen_200_pairs", "phase2_60generator_manifest.csv")

    ablation_modes = [
        ("BASE", "d_emb", 0.00, "Baseline_Fused"),
        ("EXP05A", "d_layer1", 0.10, "Layer1_Feature_Distance"),
        ("EXP05B", "d_layer2", 0.10, "Layer2_Feature_Distance"),
        ("EXP05C", "d_layer3", 0.10, "Layer3_Feature_Distance"),
        ("EXP05D", "d_emb", 0.10, "Final_128D_Embedding_Distance")
    ]

    target_pairs = ["pair_006", "pair_066", "pair_186", "pair_116"]
    summary_records = []

    print("\nEvaluating intermediate layer ablations...")
    for exp_id, layer_key, lam, desc in ablation_modes:
        m_ds2, res_ds2 = evaluate_exp05_layer_strategy(ds2_records, layer_key=layer_key, lam=lam)

        runtimes = [r["runtime_ms"] for r in res_ds2]
        p90_rt = float(np.percentile(runtimes, 90))
        p99_rt = float(np.percentile(runtimes, 99))

        tot_present = sum(1 for r in ds2_records if r["gt_found"] == 1)
        top5_recall = (m_ds2["gt_in_top5_count"] / tot_present) * 100.0 if tot_present > 0 else 0.0

        rec = {
            "exp_id": exp_id, "layer_key": layer_key, "lambda": lam, "strategy": desc,
            "ds2_total": round(m_ds2["total_100_score"], 2),
            "ds2_loc": round(m_ds2["loc_score"], 2),
            "ds2_scale": round(m_ds2["scale_score"], 2),
            "ds2_theta": round(m_ds2["theta_score"], 2),
            "ds2_rejection": round(m_ds2["rejection_score"], 2),
            "ds2_conf": round(m_ds2["confidence_score"], 2),
            "ds2_eff": round(m_ds2["eff_score"], 2),
            "ds2_gen": round(m_ds2["gen_score"], 2),
            "ds2_set_a_5px": round(m_ds2["pct_5_a"], 1),
            "ds2_set_b_5px": round(m_ds2["pct_5_b"], 1),
            "top5_recall_pct": round(top5_recall, 1),
            "gt_selected_count": m_ds2["gt_selected_count"],
            "med_rt": round(m_ds2["med_rt"], 1),
            "p90_rt": round(p90_rt, 1),
            "p99_rt": round(p99_rt, 1)
        }
        summary_records.append(rec)
        print(f"Strategy {exp_id} ({desc}): DS2 Total = {m_ds2['total_100_score']:.2f} / 100.0 | Loc = {m_ds2['loc_score']:.2f} / 40.0 | GT Selected = {m_ds2['gt_selected_count']}/{tot_present}")
        sys.stdout.flush()

        if exp_id == "EXP05B":
            print(f"\n===========================================================================")
            print(f"EXP-05B TARGET PERIODIC FAILURES DIAGNOSTIC TABLE (Layer 2)")
            print(f"===========================================================================")
            for r in res_ds2:
                if r["pair_id"] in target_pairs:
                    print(f"\n--- Pair: {r['pair_id']} (GT: x={r['gt_x']}, y={r['gt_y']}) ---")
                    for cand in r["eval_candidates"]:
                        gt_tag = "GT LANDMARK" if cand["is_gt"] else "DECOY"
                        print(f"  [{gt_tag}] Orig Rank {cand['rank_orig']}: (x={cand['x']:.1f}, y={cand['y']:.1f}) | FusedOrig={cand['fused_orig']:.4f} | d_layer2={cand['d_layer2']:.4f} | NormD={cand['norm_d']:.4f} | EvalScore={cand['eval_score']:.4f} | DistGT={cand['dist_gt']:.1f}px")
            sys.stdout.flush()

    os.makedirs("phase2/results", exist_ok=True)
    csv_path = "phase2/results/exp05_multilayer_feature_distance.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "exp_id", "layer_key", "lambda", "strategy", "ds2_total", "ds2_loc", "ds2_scale", "ds2_theta",
            "ds2_rejection", "ds2_conf", "ds2_eff", "ds2_gen", "ds2_set_a_5px", "ds2_set_b_5px",
            "top5_recall_pct", "gt_selected_count", "med_rt", "p90_rt", "p99_rt"
        ])
        writer.writeheader()
        writer.writerows(summary_records)
    print(f"\nSaved EXP-05 CSV artifact to: {csv_path}")

    # Re-verify SHA256 Hash
    with open(ckpt_path, "rb") as f:
        sha256_hash_after = hashlib.sha256(f.read()).hexdigest()
    print(f"Post-run Checkpoint SHA-256 Hash: {sha256_hash_after}")
    assert sha256_hash_after == "e64fd936f8692bc6789174cc532f7734b185d83962ec0b7764a3974a768b922c", "SHA-256 Altered!"

if __name__ == "__main__":
    main()
