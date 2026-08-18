import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

class DriftSenseSiameseDataset(Dataset):
    def __init__(self, data_dir, split="train", level=1):
        """
        Args:
            data_dir: Base data directory containing train/val/test folders and dataset_manifest.json.
            split: 'train', 'val', or 'test'
            level: 0 (Coarse), 1 (Nominal), or 2 (Fine)
        """
        self.data_dir = data_dir
        self.split = split
        self.level = level
        self.samples = []
        
        manifest_path = os.path.join(data_dir, "dataset_manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found at {manifest_path}. Make sure to generate the dataset first.")
            
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            
        for entry in manifest.get('pairs', []):
            if entry['split'] == split:
                pair_id = entry['pair_id']
                pair_dir = os.path.join(data_dir, split, pair_id)
                self.samples.append({
                    'ref_path': os.path.join(pair_dir, "reference.png"),
                    'search_path': os.path.join(pair_dir, "search.png"),
                    'target_x': int(entry['center_x']),
                    'target_y': int(entry['center_y'])
                })

    def __len__(self):
        return len(self.samples)
        
    def _get_crop(self, image, center_x, center_y, crop_size):
        """Extract a crop_size x crop_size window around (center_x, center_y)."""
        h, w = image.shape
        half = crop_size // 2
        
        # Clamp centers to avoid out-of-bounds
        cx = max(half, min(w - half - 1, center_x))
        cy = max(half, min(h - half - 1, center_y))
        
        y0, y1 = cy - half, cy + half
        x0, x1 = cx - half, cx + half
        
        return image[y0:y1, x0:x1]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        ref_img = np.array(Image.open(sample['ref_path']).convert('L'), dtype=np.float32)
        search_img = np.array(Image.open(sample['search_path']).convert('L'), dtype=np.float32)
        
        tx, ty = sample['target_x'], sample['target_y']
        
        # Standardize based on pyramid level
        if self.level == 0:
            # Reference downscaled 20x to 50x50
            # Candidate window in search is 50x50 (from 500x500 downscaled search)
            # Actually, to simulate L0, we downscale search 2x (to 500x500) and ref 20x (to 50x50)
            ref_pil = Image.fromarray(ref_img).resize((50, 50), Image.LANCZOS)
            ref_tensor = TF.to_tensor(ref_pil)
            
            search_pil = Image.fromarray(search_img).resize((500, 500), Image.LANCZOS)
            search_down = np.array(search_pil, dtype=np.float32)
            tx_down, ty_down = tx // 2, ty // 2
            
            pos_crop = self._get_crop(search_down, tx_down, ty_down, 50)
            
            # Hard negative: shifted by 1-3 cells (cell pitch is ~2.5px at 500x500 scale)
            shift_x = np.random.choice([-10, -5, 5, 10])
            shift_y = np.random.choice([-10, -5, 5, 10])
            neg_crop = self._get_crop(search_down, tx_down + shift_x, ty_down + shift_y, 50)
            
        elif self.level == 1:
            # Reference downscaled 10x to 100x100
            # Candidate window in search is 100x100
            ref_pil = Image.fromarray(ref_img).resize((100, 100), Image.LANCZOS)
            ref_tensor = TF.to_tensor(ref_pil)
            
            pos_crop = self._get_crop(search_img, tx, ty, 100)
            
            # InfoNCE: 30 Negatives (15 local hard negatives, 15 random global decoys)
            neg_tensors = []
            for i in range(30):
                if i < 15:
                    shift_x = np.random.choice([-15, -10, -5, 5, 10, 15])
                    shift_y = np.random.choice([-15, -10, -5, 5, 10, 15])
                else:
                    shift_x = np.random.randint(-400, 400)
                    shift_y = np.random.randint(-400, 400)
                    if abs(shift_x) < 20: shift_x = 25
                    if abs(shift_y) < 20: shift_y = 25
                neg_crop = self._get_crop(search_img, tx + shift_x, ty + shift_y, 100)
                n_t = TF.to_tensor(neg_crop)
                n_t = n_t / 255.0 if n_t.max() > 1.0 else n_t
                neg_tensors.append(n_t)
                
            neg_tensor = torch.stack(neg_tensors, dim=0)
            
        elif self.level == 2:
            # Reference downscaled 5x to 200x200
            # Candidate window in search is 100x100 upscaled 2x to 200x200
            ref_pil = Image.fromarray(ref_img).resize((200, 200), Image.LANCZOS)
            ref_tensor = TF.to_tensor(ref_pil)
            
            # Positive candidate
            pos_base = self._get_crop(search_img, tx, ty, 100)
            pos_pil = Image.fromarray(pos_base).resize((200, 200), Image.LANCZOS)
            pos_crop = np.array(pos_pil, dtype=np.float32)
            
            # Fine-level negative: very small shift (e.g. 1-2 pixels in original search scale)
            shift_x = np.random.choice([-2, -1, 1, 2])
            shift_y = np.random.choice([-2, -1, 1, 2])
            neg_base = self._get_crop(search_img, tx + shift_x, ty + shift_y, 100)
            neg_pil = Image.fromarray(neg_base).resize((200, 200), Image.LANCZOS)
            neg_crop = np.array(neg_pil, dtype=np.float32)

        # Normalize to [0, 1]
        pos_tensor = TF.to_tensor(pos_crop) / 255.0 if pos_crop.max() > 1.0 else TF.to_tensor(pos_crop)
        
        if self.level != 1:
            neg_tensor = TF.to_tensor(neg_crop) / 255.0 if neg_crop.max() > 1.0 else TF.to_tensor(neg_crop)
            
        ref_tensor = ref_tensor / 255.0 if ref_tensor.max() > 1.0 else ref_tensor

        # Data Augmentation: Random Flips (Consistent across the triplet)
        if self.split == 'train':
            if torch.rand(1).item() > 0.5:
                ref_tensor = TF.hflip(ref_tensor)
                pos_tensor = TF.hflip(pos_tensor)
                neg_tensor = TF.hflip(neg_tensor)
            if torch.rand(1).item() > 0.5:
                ref_tensor = TF.vflip(ref_tensor)
                pos_tensor = TF.vflip(pos_tensor)
                neg_tensor = TF.vflip(neg_tensor)

        ret = {
            'reference': ref_tensor,
            'positive': pos_tensor
        }
        if self.level == 1:
            ret['negatives'] = neg_tensor
        else:
            ret['negative'] = neg_tensor
            
        return ret