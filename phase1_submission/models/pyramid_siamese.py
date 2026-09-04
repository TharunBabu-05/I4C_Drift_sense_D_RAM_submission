import torch
import torch.nn as nn
import torch.nn.functional as F
from .siamese_encoder import SiameseEncoder, MobileNetV3SiameseEncoder

class PyramidSiameseNetwork(nn.Module):
    """
    Multi-Scale Siamese Network wrapper for Drift-Sense.
    Handles the 3-level pyramid architecture using a shared encoder.
    """
    def __init__(self, embedding_dim=128, encoder_type='mobilenet'):
        super(PyramidSiameseNetwork, self).__init__()
        # Shared weights encoder for all levels
        if encoder_type == 'mobilenet':
            self.encoder = MobileNetV3SiameseEncoder(embedding_dim=embedding_dim)
        else:
            self.encoder = SiameseEncoder(embedding_dim=embedding_dim)
        
        # Optional fusion head if we want to learn Level-0 and Level-1 weighting
        # Defaulting to fixed weighting if not explicitly trained
        self.fusion_weight_l0 = nn.Parameter(torch.tensor([0.35]))
        self.fusion_weight_l1 = nn.Parameter(torch.tensor([0.65]))

        # Head for fine Level-2 sub-pixel refinement
        # Takes concatenated embeddings or feature maps and predicts dx, dy
        self.refinement_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2) # dx, dy output
        )

    def forward(self, ref, search_candidate):
        """
        Base forward pass for similarity training.
        """
        ref_emb = self.encoder(ref)
        search_emb = self.encoder(search_candidate)
        
        # Return distance/similarity logic in training loop
        return ref_emb, search_emb

    def compute_similarity(self, ref_emb, search_emb):
        """
        Computes cosine similarity between embeddings.
        """
        # Encoder already L2 normalizes, so dot product is cosine similarity
        sim = torch.sum(ref_emb * search_emb, dim=1)
        return sim

    def refine_fine_candidate(self, ref_emb, search_emb):
        """
        Level 2 Fine Refinement. Predicts dx, dy offset.
        """
        concat_emb = torch.cat((ref_emb, search_emb), dim=1)
        offset = self.refinement_head(concat_emb)
        return offset
