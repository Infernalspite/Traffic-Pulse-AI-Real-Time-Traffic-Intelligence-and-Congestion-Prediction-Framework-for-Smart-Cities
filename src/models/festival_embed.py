"""Festival Calendar Embedding & Hard Event Attention Layer (Innovation #3).

Encodes festival and civic event vectors:
    [is_festival, days_to_festival, festival_type, geographic_scope]
and applies hard attention to upweight anomalous event signals during critical periods.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FestivalCalendarEmbedding(nn.Module):
    """Innovation #3: Discrete/continuous event embedding and hard attention."""

    def __init__(self, in_features: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        # Continuous / categorical projection of festival features
        # festival_indicator (is_festival), festival_intensity, days_relative_festival
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Gate layer for hard event attention
        self.event_gate = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.event_boost = nn.Parameter(torch.tensor(1.5))

    def forward(self, festival_features: torch.Tensor, traffic_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            festival_features: (B, T, N, 3) where [indicator, intensity, days_relative]
            traffic_hidden: (B, T, N, H)
        Returns:
            modulated_traffic: (B, T, N, H) with event upweighting
        """
        # Embed event signals
        h_fest = self.proj(festival_features)  # (B, T, N, H)

        # Event gate: 1 when active festival / eve / post-day, 0 during normal days
        gate = self.event_gate(festival_features)  # (B, T, N, 1)

        # Hard attention threshold: when gate > 0.5, apply strong event modulation
        hard_mask = (gate > 0.5).float()
        effective_gate = hard_mask * self.event_boost + (1.0 - hard_mask) * gate

        return traffic_hidden + effective_gate * h_fest
