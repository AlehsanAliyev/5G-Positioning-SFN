import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyWeightedFusion(nn.Module):
    def __init__(self):
        super(UncertaintyWeightedFusion, self).__init__()

    def forward(self, coord_preds, log_vars):
        """
        coord_preds: Tensor of shape (B, N, 2) → predicted (x, y) from N base stations
        log_vars:    Tensor of shape (B, N, 1) → log(σ²) predicted from each BS

        Returns:
            fused_coords: Tensor of shape (B, 2)
        """
        # Convert log variance to inverse variance (confidence weights)
        inv_vars = torch.exp(-log_vars)  # shape (B, N, 1)

        # Weighted sum of coordinates
        weighted_coords = coord_preds * inv_vars  # shape (B, N, 2)
        sum_weighted = weighted_coords.sum(dim=1)  # (B, 2)

        # Normalize by total confidence
        weight_sums = inv_vars.sum(dim=1)  # (B, 1)
        fused_coords = sum_weighted / weight_sums  # (B, 2)

        return fused_coords


# fuser = UncertaintyWeightedFusion()
# fused = fuser(coord_preds, log_vars)  # coord_preds: (B, N, 2), log_vars: (B, N, 1)
