"""Heterogeneous Vehicle-Type Graph Convolutional Sub-Networks (Innovation #1).

Constructs distinct vehicle-type sub-graphs (Two-Wheelers, Heavy Vehicles, Public Transport)
with independent spatial message passing layers, fused at the representation bottleneck.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VehicleSubGraphConv(nn.Module):
    """Directed message passing for a vehicle modality."""

    def __init__(self, in_features: int, hidden_dim: int, nodes: int):
        super().__init__()
        self.nodes = nodes
        self.msg_linear = nn.Linear(in_features, hidden_dim)
        self.update_linear = nn.Linear(in_features + hidden_dim, hidden_dim)
        self.modality_adj = nn.Parameter(torch.eye(nodes) * 0.5 + torch.randn(nodes, nodes) * 0.05)

    def forward(self, x: torch.Tensor, base_adj: torch.Tensor | None = None) -> torch.Tensor:
        B, T, N, F_dim = x.shape
        adj = torch.softmax(self.modality_adj, dim=-1)
        if base_adj is not None:
            adj = adj * base_adj
            adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)

        msg = self.msg_linear(x)
        agg = torch.einsum('ij,btjf->btif', adj, msg)
        combined = torch.cat([x, agg], dim=-1)
        return torch.relu(self.update_linear(combined))


class HeterogeneousVehicleGraph(nn.Module):
    """Innovation #1: Multi-graph vehicle modal decomposition and fusion network."""

    def __init__(self, nodes: int, hidden_dim: int = 64):
        super().__init__()
        self.nodes = nodes
        self.hidden_dim = hidden_dim

        # G_2W: Two-wheeler channel (2W + Auto)
        self.g_2w = VehicleSubGraphConv(in_features=2, hidden_dim=hidden_dim, nodes=nodes)
        # G_HV: Heavy vehicle channel (Car + Bus/Truck)
        self.g_hv = VehicleSubGraphConv(in_features=2, hidden_dim=hidden_dim, nodes=nodes)
        # G_PT: Public transport channel (Auto + Bus)
        self.g_pt = VehicleSubGraphConv(in_features=2, hidden_dim=hidden_dim, nodes=nodes)

        self.modality_fusion = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, vehicle_features: torch.Tensor, base_adj: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            vehicle_features: (B, T, N, 4) where indices 0..3 are [2W, Auto, Bus, Car]
        Returns:
            fused_representation: (B, T, N, hidden_dim)
        """
        x_2w = vehicle_features[..., [0, 1]]
        x_hv = vehicle_features[..., [2, 3]]
        x_pt = vehicle_features[..., [1, 2]]

        h_2w = self.g_2w(x_2w, base_adj)
        h_hv = self.g_hv(x_hv, base_adj)
        h_pt = self.g_pt(x_pt, base_adj)

        fused = torch.cat([h_2w, h_hv, h_pt], dim=-1)
        return torch.relu(self.modality_fusion(fused))
