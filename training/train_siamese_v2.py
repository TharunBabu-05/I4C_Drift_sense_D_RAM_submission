import os
import sys
import json
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pyramid_siamese import PyramidSiameseNetwork
from dataset.dataset import DriftSenseSiameseDataset
from training.losses import ContrastiveLoss, InfoNCELoss


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# SEM-style augmentation, applied on GPU/CPU tensors already in [0, 1]
# Mirrors the noise model in the README so the encoder is trained to be
# invariant to the same corruptions the classical NCC pipeline struggles with.
# --------------------------------------------------------------------------
class SEMAugment:
    def __init__(self, p=0.7, gauss_std=(0.01, 0.05), poisson_scale=(0.5, 1.5),
                 blur_kernel=3, blur_sigma=(0.1, 1.0), jitter=0.15):
        self.p = p
        self.gauss_std = gauss_std
        self.poisson_scale = poisson_scale
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.jitter = jitter

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, H, W] in [0, 1]
        if random.random() > self.p:
            return x

        out = x

        # Poisson-like shot noise (signal-dependent)
        if random.random() < 0.6:
            scale = random.uniform(*self.poisson_scale)
            noisy = torch.poisson(torch.clamp(out * 255.0 * scale, min=0)) / (255.0 * scale)
            out = torch.clamp(noisy, 0, 1)

        # Gaussian read noise (signal-independent)
        if random.random() < 0.6:
            std = random.uniform(*self.gauss_std)
            out = torch.clamp(out + torch.randn_like(out) * std, 0, 1)

        # Gaussian blur (beam defocus / PSF)
        if random.random() < 0.4:
            sigma = random.uniform(*self.blur_sigma)
            out = gaussian_blur(out, kernel_size=self.blur_kernel, sigma=sigma)

        # Brightness / contrast jitter (intensity drift, gain/offset)
        if random.random() < 0.5:
            gain = 1.0 + random.uniform(-self.jitter, self.jitter)
            bias = random.uniform(-self.jitter, self.jitter) * 0.5
            out = torch.clamp(out * gain + bias, 0, 1)

        return out


def gaussian_blur(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """Depthwise Gaussian blur, no external deps."""
    device = x.device
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, -1)
    kernel_x = g.view(1, 1, 1, kernel_size)
    kernel_y = g.view(1, 1, kernel_size, 1)
    pad = kernel_size // 2
    x = nn.functional.conv2d(x, kernel_x.expand(x.size(1), -1, -1, -1),
                              padding=(0, pad), groups=x.size(1))
    x = nn.functional.conv2d(x, kernel_y.expand(x.size(1), -1, -1, -1),
                              padding=(pad, 0), groups=x.size(1))
    return x


# --------------------------------------------------------------------------
# Retrieval-style accuracy: for each triplet, is the positive actually
# closer to the anchor than the negative? This tracks what you actually
# care about (correct localization), unlike raw loss value.
# --------------------------------------------------------------------------
@torch.no_grad()
def triplet_accuracy(ref_emb, pos_emb, neg_emb):
    d_pos = (ref_emb - pos_emb).pow(2).sum(dim=1)
    d_neg = (ref_emb - neg_emb).pow(2).sum(dim=1)
    return (d_pos < d_neg).float().mean().item()


@torch.no_grad()
def infonce_accuracy(ref_emb, pos_emb, neg_embs):
    # neg_embs: [B, N, D]
    pos_sim = torch.sum(ref_emb * pos_emb, dim=1, keepdim=True)          # [B, 1]
    neg_sim = torch.bmm(neg_embs, ref_emb.unsqueeze(2)).squeeze(2)       # [B, N]
    return (pos_sim > neg_sim.max(dim=1, keepdim=True).values).float().mean().item()


def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, criterion, optimizer, scaler, device, augment,
              train: bool, use_infonce: bool, grad_clip: float, desc: str):
    model.train() if train else model.eval()
    total_loss, total_acc, n_samples = 0.0, 0.0, 0

    pbar = tqdm(loader, desc=desc, leave=False)
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for batch in pbar:
            ref = batch["reference"].to(device, non_blocking=True)
            pos = batch["positive"].to(device, non_blocking=True)

            if train and augment is not None:
                ref = augment(ref)
                pos = augment(pos)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                ref_emb = model.encoder(ref)
                pos_emb = model.encoder(pos)

                if use_infonce:
                    negs = batch["negatives"].to(device, non_blocking=True)  # [B, N, 1, H, W]
                    B, N = negs.shape[0], negs.shape[1]
                    negs_flat = negs.view(B * N, *negs.shape[2:])
                    if train and augment is not None:
                        negs_flat = augment(negs_flat)
                    neg_embs = model.encoder(negs_flat).view(B, N, -1)
                    loss = criterion(ref_emb, pos_emb, neg_embs)
                    acc = infonce_accuracy(ref_emb.detach(), pos_emb.detach(), neg_embs.detach())
                else:
                    neg = batch["negative"].to(device, non_blocking=True)
                    if train and augment is not None:
                        neg = augment(neg)
                    neg_emb = model.encoder(neg)
                    loss = criterion(ref_emb, pos_emb, neg_emb)
                    acc = triplet_accuracy(ref_emb.detach(), pos_emb.detach(), neg_emb.detach())

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            bs = ref.size(0)
            total_loss += loss.item() * bs
            total_acc += acc * bs
            n_samples += bs
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.3f}"})

    return total_loss / n_samples, total_acc / n_samples


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # ---------------- Data ----------------
    train_dataset = DriftSenseSiameseDataset(args.data_dir, split="train", level=args.level)
    val_dataset = DriftSenseSiameseDataset(args.data_dir, split="val", level=args.level)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                               drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Detect whether the dataset gives multiple negatives (hard-negative / InfoNCE mode)
    sample_batch = next(iter(train_loader))
    use_infonce = "negatives" in sample_batch
    print(f"Loss mode: {'InfoNCE (multi-negative)' if use_infonce else 'Triplet (single negative)'}")

    # ---------------- Model ----------------
    model = PyramidSiameseNetwork(embedding_dim=args.embedding_dim,
                                   encoder_type=args.encoder).to(device)

    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed weights from {args.resume}")

    # ---------------- Loss ----------------
    if use_infonce:
        criterion = InfoNCELoss(temperature=args.temperature)
    else:
        criterion = nn.TripletMarginLoss(margin=args.margin, p=2)

    # ---------------- Optimizer / schedule ----------------
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_lr_scheduler(optimizer, args.warmup_epochs, args.epochs)
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and args.amp) else None

    augment = SEMAugment(p=args.aug_prob) if args.augment else None

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.checkpoint_dir, f"config_level{args.level}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    best_val_acc = -1.0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, augment,
            train=True, use_infonce=use_infonce, grad_clip=args.grad_clip,
            desc=f"Epoch {epoch}/{args.epochs} [Train]",
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device, augment=None,
            train=False, use_infonce=use_infonce, grad_clip=args.grad_clip,
            desc=f"Epoch {epoch}/{args.epochs} [Val]",
        )

        scheduler.step()
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d} | lr {lr_now:.2e} | "
              f"train_loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.3f} | {dt:.1f}s")

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc, "lr": lr_now})

        # Checkpoint on best *validation accuracy*, not loss — this is the
        # metric that actually predicts downstream localization performance.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            ckpt_path = os.path.join(args.checkpoint_dir, f"best_model_level{args.level}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  --> New best (val_acc={val_acc:.3f}). Saved to {ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f"  --> No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= args.patience:
            print(f"\n[!] EARLY STOPPING at epoch {epoch} (best val_acc={best_val_acc:.3f})")
            break

    with open(os.path.join(args.checkpoint_dir, f"history_level{args.level}.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Best val_acc: {best_val_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--encoder", type=str, choices=['resnet', 'mobilenet'], default='resnet')

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--margin", type=float, default=1.0, help="Triplet loss margin")
    parser.add_argument("--temperature", type=float, default=0.1, help="InfoNCE temperature")

    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--aug_prob", type=float, default=0.7)

    parser.add_argument("--amp", action="store_true", default=True,
                         help="Mixed precision (CUDA only)")
    parser.add_argument("--resume", type=str, default=None,
                         help="Path to a checkpoint to resume from")

    args = parser.parse_args()
    train(args)
