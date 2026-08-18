import os
import argparse
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pyramid_siamese import PyramidSiameseNetwork
from dataset.dataset import DriftSenseSiameseDataset
from training.losses import ContrastiveLoss

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Dataset & DataLoader
    train_dataset = DriftSenseSiameseDataset(args.data_dir, split="train", level=args.level)
    val_dataset = DriftSenseSiameseDataset(args.data_dir, split="val", level=args.level)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    model = PyramidSiameseNetwork(embedding_dim=128).to(device)
    
    # Loss & Optimizer
    criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Checkpoints dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        # tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch in pbar:
            ref = batch['reference'].to(device)
            pos = batch['positive'].to(device)
            neg = batch['negative'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: Encode all three (Anchor, Positive, Negative)
            ref_emb = model.encoder(ref)
            pos_emb = model.encoder(pos)
            neg_emb = model.encoder(neg)
            
            # Triplet Loss (pulls positive closer, pushes negative away simultaneously)
            loss = criterion(ref_emb, pos_emb, neg_emb)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * ref.size(0)
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss /= len(train_loader.dataset)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                ref = batch['reference'].to(device)
                pos = batch['positive'].to(device)
                neg = batch['negative'].to(device)
                
                ref_emb = model.encoder(ref)
                pos_emb = model.encoder(pos)
                neg_emb = model.encoder(neg)
                
                loss = criterion(ref_emb, pos_emb, neg_emb)
                val_loss += loss.item() * ref.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        scheduler.step()
        
        # Save best model and Early Stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"best_model_level{args.level}.pth"))
            print("  --> Saved new best model!")
        else:
            epochs_no_improve += 1
            print(f"  --> No improvement for {epochs_no_improve} epochs.")
            
        if epochs_no_improve >= args.patience:
            print(f"\n[!] EARLY STOPPING TRIGGERED AT EPOCH {epoch}")
            break
            
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data", help="Dataset directory")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints", help="Where to save models")
    parser.add_argument("--level", type=int, default=1, help="Pyramid level to train (0, 1, 2)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()
    
    train(args)
