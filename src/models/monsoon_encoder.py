"""Monsoon-Aware Weather Fusion via ConvLSTM & FiLM Modulation (Innovation #2).

Fuses meteorological signals (rainfall intensity mm/hr, visibility, waterlogging)
into each temporal traffic state using Feature-wise Linear Modulation:
    h_fused = (1.0 + tanh(gamma(weather))) * h_traffic + beta(weather)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """2D Spatial ConvLSTM cell for meteorological grid sequence modeling."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        if state is None:
            h_cur = torch.zeros(b, self.hidden_channels, h, w, device=x.device, dtype=x.dtype)
            c_cur = torch.zeros(b, self.hidden_channels, h, w, device=x.device, dtype=x.dtype)
        else:
            h_cur, c_cur = state

        combined = torch.cat([x, h_cur], dim=1)
        gates = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class MonsoonWeatherFusion(nn.Module):
    """Innovation #2: Spatial Weather Encoder and FiLM Modulation Layer."""

    def __init__(self, weather_features: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.weather_features = weather_features
        self.hidden_dim = hidden_dim

        # Temporal sequence projection for point weather signals
        self.weather_encoder = nn.GRU(weather_features, hidden_dim, batch_first=True)
        # FiLM parameter generator: outputs gamma and beta
        self.film_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

    def forward(self, traffic_hidden: torch.Tensor, weather_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            traffic_hidden: (B, T, N, H) traffic representation
            weather_features: (B, T, N, 3) where [rainfall_mm_hr, visibility, waterlogging]
        Returns:
            fused_traffic: (B, T, N, H) FiLM-modulated traffic representation
        """
        B, T, N, _ = weather_features.shape
        w_seq = weather_features.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        w_enc, _ = self.weather_encoder(w_seq)
        w_enc = w_enc.reshape(B, N, T, self.hidden_dim).permute(0, 2, 1, 3)  # (B, T, N, H)

        film_params = self.film_generator(w_enc)  # (B, T, N, 2*H)
        gamma, beta = film_params.chunk(2, dim=-1)

        # Feature-wise Linear Modulation
        fused = (1.0 + torch.tanh(gamma)) * traffic_hidden + beta
        return fused
