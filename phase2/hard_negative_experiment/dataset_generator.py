#!/usr/bin/env python3
"""
Phase-2 Hard-Negative Triplet Dataset Generator (Zero Memory Path Records)
===========================================================================
Generates (ref_path, search_path, gt_x, gt_y, neg_x, neg_y) path/coordinate records.
Zero numpy image memory accumulation during dataset construction.
"""

import os
import sys
import csv
import math
from PIL import Image
import numpy as np

def build_triplet_records(manifest_paths):
    """
    Builds lightweight metadata records (ref_path, search_path, gt_x, gt_y, neg_x, neg_y)
    across present training pairs.
    """
    records = []
    periodic_offsets = [
        (15, 0), (-15, 0), (0, 15), (0, -15),
        (30, 0), (-30, 0), (0, 30), (0, -30),
        (15, 15), (-15, -15), (30, 30), (-30, -30)
    ]

    for manifest_path in manifest_paths:
        if not os.path.exists(manifest_path):
            continue
        dataset_dir = os.path.dirname(manifest_path)
        with open(manifest_path, "r") as f:
            rows = list(csv.DictReader(f))

        print(f"Extracting lightweight triplet records from {manifest_path}...")
        for idx, r in enumerate(rows):
            found_gt = int(r.get("found_gt", 1))
            if found_gt == 0:
                continue

            ref_path = os.path.abspath(r.get("reference_path", r.get("ref_path")))
            search_path = os.path.abspath(r.get("search_path"))

            if not os.path.exists(ref_path) or not os.path.exists(search_path):
                continue

            gt_x, gt_y = float(r["x_gt"]), float(r["y_gt"])

            for dx, dy in periodic_offsets:
                neg_x = gt_x + dx
                neg_y = gt_y + dy
                records.append({
                    "pair_id": r.get("pair_id", f"pair_{idx}"),
                    "ref_path": ref_path,
                    "search_path": search_path,
                    "gt_x": gt_x, "gt_y": gt_y,
                    "neg_x": neg_x, "neg_y": neg_y
                })

    print(f"Extracted a total of {len(records)} periodic hard-negative records!")
    return records

if __name__ == "__main__":
    manifests = [
        "local_phase2_60gen_200_pairs/phase2_60generator_manifest.csv",
        "local_phase2_200_pairs/dataset_manifest.csv"
    ]
    recs = build_triplet_records(manifests)
    print(f"Sample record: {recs[0]}")
