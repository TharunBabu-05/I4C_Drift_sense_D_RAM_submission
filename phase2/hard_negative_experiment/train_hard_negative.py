#!/usr/bin/env python3
"""
Phase-2 Hard-Negative Triplet Siamese Fine-Tuning Script
=========================================================
Fine-tunes the Custom 4-Layer ResNet Siamese Encoder using explicit Periodic Hard-Negative Triplet Loss:
- Base Weights: phase2_checkpoints/best_model_level1.pth (READ ONLY - SHA256 UNTOUCHED)
- Output Checkpoint: phase2_checkpoints/hard_negative/best_model_hard_negative.pth

Loss Objective:
  L_total = L_InfoNCE + lambda_triplet * L_triplet
  L_triplet = max(0, sim(anchor, negative) - sim(anchor, positive) + margin)  (margin = 0.20)
"""

import os
import sys
import time
import math
import hashlib
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.pyramid_siamese import PyramidSiameseNetwork
from phase2.hard_negative_experiment.dataset_generator import build_triplet_records

def load_crop_patch_pil(img_path, cx, cy, patch_size=100):
    """Loads image via PIL context manager and crops patch_size x patch_size patch safely."""
    with Image.open(img_path) as img_file:
        img_gray = img_file.convert('L')
        w_img, h_img = img_gray.size
        half = patch_size // 2
        x0, x1 = max(0, int(round(cx - half))), min(w_img, int(round(cx + half)))
        y0, y1 = max(0, int(round(cy - half))), min(h_img, int(round(cy + half)))
        crop = img_gray.crop((x0, y0, x1, y1))
        if crop.size != (patch_size, patch_size):
            crop = crop.resize((patch_size, patch_size), Image.LANCZOS)
        arr = np.array(crop, dtype=np.uint8)
    return np.ascontiguousarray(arr)

class HardNegativeDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        anc_patch = load_crop_patch_pil(rec["ref_path"], 50.0, 50.0, patch_size=100)
        pos_patch = load_crop_patch_pil(rec["search_path"], rec["gt_x"], rec["gt_y"], patch_size=100)
        neg_patch = load_crop_patch_pil(rec["search_path"], rec["neg_x"], rec["neg_y"], patch_size=100)

        anc_t = TF.to_tensor(anc_patch)
        pos_t = TF.to_tensor(pos_patch)
        neg_t = TF.to_tensor(neg_patch)
        return anc_t, pos_t, neg_t

class TripletInfoNCELoss(nn.Module):
    def __init__(self, margin=0.20, temperature=0.07, lambda_triplet=1.0):
        super(TripletInfoNCELoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
        self.lambda_triplet = lambda_triplet

    def forward(self, anc_emb, pos_emb, neg_emb):
        pos_sim = torch.sum(anc_emb * pos_emb, dim=1)
        neg_sim = torch.sum(anc_emb * neg_emb, dim=1)

        triplet_loss = torch.relu(neg_sim - pos_sim + self.margin)
        mean_triplet_loss = torch.mean(triplet_loss)

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1) / self.temperature
        labels = torch.zeros(anc_emb.size(0), dtype=torch.long, device=anc_emb.device)
        infonce_loss = F.cross_entropy(logits, labels)

        total_loss = infonce_loss + self.lambda_triplet * mean_triplet_loss
        return total_loss, mean_triplet_loss, infonce_loss, pos_sim, neg_sim

def train_hard_negative_model(epochs=10, batch_size=32, lr=1e-4, margin=0.20):
    ckpt_src = "phase2_checkpoints/best_model_level1.pth"
    out_dir = "phase2_checkpoints/hard_negative"
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dst = os.path.join(out_dir, "best_model_hard_negative.pth")

    with open(ckpt_src, "rb") as f:
        sha_before = hashlib.sha256(f.read()).hexdigest()
    print(f"Original Checkpoint SHA-256 Before Training: {sha_before}")

    device = torch.device("cpu")
    manifests = [
        "local_phase2_60gen_200_pairs/phase2_60generator_manifest.csv",
        "local_phase2_200_pairs/dataset_manifest.csv"
    ]
    records = build_triplet_records(manifests)
    records = records[::2] # 1,920 records for optimal CPU training speed

    dataset = HardNegativeDataset(records)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PyramidSiameseNetwork(encoder_type="resnet").to(device)
    model.load_state_dict(torch.load(ckpt_src, map_location=device))
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = TripletInfoNCELoss(margin=margin, lambda_triplet=1.0)

    print(f"\n--- Starting Hard-Negative Triplet Siamese Fine-Tuning ({epochs} Epochs, {len(records)} Triplets) ---")
    train_history = []
    start_t = time.time()

    for epoch in range(1, epochs + 1):
        total_l, total_trip, mean_pos, mean_neg = 0.0, 0.0, 0.0, 0.0
        count = 0
        for anc, pos, neg in dataloader:
            anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
            optimizer.zero_grad()

            anc_emb = model.encoder(anc)
            pos_emb = model.encoder(pos)
            neg_emb = model.encoder(neg)

            loss, t_loss, _, pos_sim, neg_sim = criterion(anc_emb, pos_emb, neg_emb)
            loss.backward()
            optimizer.step()

            bs = anc.size(0)
            total_l += loss.item() * bs
            total_trip += t_loss.item() * bs
            mean_pos += torch.sum(pos_sim).item()
            mean_neg += torch.sum(neg_sim).item()
            count += bs

        avg_l = total_l / count
        avg_pos = mean_pos / count
        avg_neg = mean_neg / count
        delta = avg_pos - avg_neg

        train_history.append({
            "epoch": epoch, "loss": round(avg_l, 4),
            "gt_pos_sim": round(avg_pos, 4), "decoy_neg_sim": round(avg_neg, 4),
            "sim_delta": round(delta, 4)
        })

        print(f"Epoch {epoch:02d}/{epochs:02d}: Loss = {avg_l:.4f} | GT Pos Sim = {avg_pos:.4f} | Decoy Neg Sim = {avg_neg:.4f} | Delta = {delta:.4f}")

    torch.save(model.state_dict(), ckpt_dst)
    print(f"\nSaved fine-tuned checkpoint to: {ckpt_dst}")

    with open(ckpt_src, "rb") as f:
        sha_after = hashlib.sha256(f.read()).hexdigest()
    print(f"Original Checkpoint SHA-256 After Training: {sha_after}")
    assert sha_before == sha_after, "CRITICAL ERROR: Original Phase-1 Checkpoint was modified!"

    os.makedirs("phase2/results", exist_ok=True)
    csv_train_path = "phase2/results/hard_negative_training_results.csv"
    with open(csv_train_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "gt_pos_sim", "decoy_neg_sim", "sim_delta"])
        writer.writeheader()
        writer.writerows(train_history)
    print(f"Saved training history CSV to: {csv_train_path}")

    return ckpt_dst, train_history

if __name__ == "__main__":
    train_hard_negative_model(epochs=10, batch_size=32, lr=1e-4, margin=0.20)
