import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from vq_mm import VectorQuantizer

class SemanticIdUniquenessLoss(nn.Module):
    """
    Calculates the semantic ID uniqueness constraint loss, pushing the semantic IDs of different items apart.
    """
    def __init__(self, margin: float = 0.5, weight: float = 1.0):
        """
        Initializes the semantic ID uniqueness constraint loss.
        
        Args:
            margin: The minimum distance threshold between semantic IDs.
            weight: The weight of the loss.
        """
        super().__init__()
        self.margin = margin
        self.weight = weight
    
    def forward(self, sem_ids: Tensor, encoded_features: Tensor) -> Tensor:
        """
        Calculates the uniqueness loss for semantic IDs within a batch.
        
        Args:
            sem_ids: A tensor of semantic IDs with shape [batch_size, n_layers].
            encoded_features: The encoder output with shape [batch_size, embed_dim].
            
        Returns:
            The uniqueness constraint loss.
        """
        batch_size, n_layers = sem_ids.shape
        
        # If the batch size is too small, do not calculate the loss.
        if batch_size <= 1:
            return torch.tensor(0.0, device=sem_ids.device)
        
        # Find pairs with identical semantic IDs.
        # Expand to [batch_size, 1, n_layers] and [1, batch_size, n_layers].
        id1 = sem_ids.unsqueeze(1)
        id2 = sem_ids.unsqueeze(0)
        
        # Check if all layers are equal.
        id_eq = (id1 == id2).all(dim=-1)
        
        # Create a diagonal mask to exclude self-comparison.
        diag_mask = ~torch.eye(batch_size, device=sem_ids.device, dtype=torch.bool)
        
        # Find pairs of identical IDs (excluding self).
        identical_pairs_mask = id_eq & diag_mask
        
        # If there are no identical ID pairs, return zero loss.
        if not identical_pairs_mask.any():
            return torch.tensor(0.0, device=sem_ids.device)
        
        # Get the indices of the identical ID pairs.
        idx_a, idx_b = torch.where(identical_pairs_mask)
        
        # To avoid duplicate calculations, only consider pairs where i < j.
        unique_pairs_mask = idx_a < idx_b
        idx_a = idx_a[unique_pairs_mask]
        idx_b = idx_b[unique_pairs_mask]

        if len(idx_a) == 0:
            return torch.tensor(0.0, device=sem_ids.device)
            
        # Get the encoded features for these pairs.
        features_a = encoded_features[idx_a]
        features_b = encoded_features[idx_b]
        
        # Normalize features to calculate cosine similarity.
        features_a_norm = F.normalize(features_a, p=2, dim=-1)
        features_b_norm = F.normalize(features_b, p=2, dim=-1)

        # Calculate cosine similarity.
        cosine_sim = (features_a_norm * features_b_norm).sum(dim=-1)
        
        # Calculate the loss: we want to push these features apart, so we apply a penalty when the similarity is higher than the margin.
        # Loss = max(0, cosine_sim - margin).
        loss = F.relu(cosine_sim - self.margin)

        # Average the loss over all conflicting pairs and multiply by the weight.
        uniqueness_loss = self.weight * loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=sem_ids.device)

        return uniqueness_loss

class ResidualVectorQuantizer(nn.Module):

    def __init__(self, n_e_list, e_dim, sk_epsilons, align = 1,
                 kmeans_init = False, kmeans_iters = 100, sk_iters=100,):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim, mu=align,
                                                        kmeans_init = self.kmeans_init,
                                                        kmeans_iters = self.kmeans_iters,
                                                        sk_epsilon=sk_epsilon,
                                                        sk_iters=sk_iters)
                                        for n_e, sk_epsilon in zip(n_e_list,sk_epsilons) ])
        self.uniq_loss_fn = SemanticIdUniquenessLoss(margin=0.5, weight=0.1)


    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)
    def get_code(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return all_codebook
    def vq_ini(self, x):
        x_q = 0
        residual = x
        for idx, quantizer in enumerate(self.vq_layers):

            x_res = quantizer.vq_init(residual, use_sk=True)
            residual = residual - x_res
            x_q = x_q + x_res

    def get_codebook_usage(self):
        """返回每层的码本利用率列表"""
        return [vq.get_usage_ratio() for vq in self.vq_layers]

    def reset_usage_counts(self):
        """重置所有层的 usage 计数"""
        for vq in self.vq_layers:
            vq.reset_usage_count()

    def forward(self, x, labels, use_sk=True):
        all_losses = []
        all_indices = []

        x_q = 0
        residual = x

        for idx, quantizer in enumerate(self.vq_layers):
            label = labels[str(idx)]
            
            x_res, loss, indices = quantizer(residual,label, idx, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        # uniqueness_loss = torch.tensor(0.0, device=x.device)
        
        # uniqueness_loss = self.uniq_loss_fn(all_indices, x)

        return x_q, mean_losses, all_indices
