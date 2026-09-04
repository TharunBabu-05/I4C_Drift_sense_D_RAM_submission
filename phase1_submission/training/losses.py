import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function for Siamese networks.
    Optimized for cosine similarity embeddings (normalized to length 1).
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, ref_emb, cand_emb, label):
        """
        Args:
            ref_emb: Reference embedding [B, D] (L2 normalized)
            cand_emb: Candidate embedding [B, D] (L2 normalized)
            label: 1 for positive pair, 0 for negative pair
        """
        # Since embeddings are L2 normalized, L2 distance is related to Cosine Similarity.
        # Euclidean distance squared
        euclidean_distance = F.pairwise_distance(ref_emb, cand_emb, keepdim=True)
        
        # Loss formula:
        # L = Y * 0.5 * D^2 + (1-Y) * 0.5 * max(0, margin - D)^2
        loss_contrastive = torch.mean(
            label * 0.5 * torch.pow(euclidean_distance, 2) +
            (1 - label) * 0.5 * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        return loss_contrastive

class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for contrastive learning with multiple negatives per batch.
    """
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, ref_emb, pos_emb, neg_embs):
        """
        Args:
            ref_emb: [B, D]
            pos_emb: [B, D]
            neg_embs: [B, N, D] where N is number of negative samples
        """
        batch_size = ref_emb.size(0)
        
        # Positive logits
        pos_sim = torch.sum(ref_emb * pos_emb, dim=1, keepdim=True) / self.temperature
        
        # Negative logits
        # ref_emb: [B, 1, D], neg_embs: [B, N, D]
        neg_sim = torch.bmm(neg_embs, ref_emb.unsqueeze(2)).squeeze(2) / self.temperature
        
        # Concat pos and neg logits [B, 1+N]
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        
        # Labels are 0 (positive is at index 0)
        labels = torch.zeros(batch_size, dtype=torch.long, device=ref_emb.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
