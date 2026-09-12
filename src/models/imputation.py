"""Sparse Sensor Imputation & Random Sensor Masking Module (Innovation #6).

Simulates real Indian ITMS coverage dropouts and missing sensor infrastructure
by randomly masking 20-40% of sensors during training and imputing via spatial neighbors.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SparseSensorImputer(nn.Module):
    """Innovation #6: Robust sensor masking and graph-guided spatial imputation."""

    def __init__(self, min_dropout: float = 0.20, max_dropout: float = 0.40):
        super().__init__()
        self.min_dropout = min_dropout
        self.max_dropout = max_dropout

    def mask_sensors(self, x: torch.Tensor, p: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mask a random fraction of node sensors across batch and time.
        Args:
            x: (B, T, N, F) or (B, T, F, N)
        Returns:
            masked_x, keep_mask
        """
        if not self.training:
            return x, torch.ones_like(x[..., :1])

        if p is None:
            # Random dropout rate between min_dropout and max_dropout (20% - 40%)
            p = float(torch.empty(1).uniform_(self.min_dropout, self.max_dropout).item())

        B = x.shape[0]
        # Detect whether nodes are at dim 2 or dim 3
        if x.shape[2] > x.shape[3]:  # (B, T, N, F)
            N = x.shape[2]
            keep_mask = (torch.rand(B, 1, N, 1, device=x.device) > p).float()
        else:  # (B, T, F, N)
            N = x.shape[3]
            keep_mask = (torch.rand(B, 1, 1, N, device=x.device) > p).float()

        return x * keep_mask, keep_mask

    def spatial_impute(self, x: torch.Tensor, adj: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        """
        Impute missing sensor values from 1-hop and 2-hop topological neighbors.
        Args:
            x: (B, T, N, F)
            adj: (N, N) normalized adjacency matrix
            keep_mask: (B, T, N, 1) boolean/float indicator where 1 = observed, 0 = dropped
        """
        # Neighbor weighted average
        # message: adj @ x
        neighbor_recon = torch.einsum('ij,btjf->btif', adj, x)
        # normalize by observed neighbor weight
        neighbor_weights = torch.einsum('ij,btjf->btif', adj, keep_mask) + 1e-6
        neighbor_imputed = neighbor_recon / neighbor_weights

        # Combine: keep observed values where mask=1, use neighbor imputation where mask=0
        return keep_mask * x + (1.0 - keep_mask) * neighbor_imputed
