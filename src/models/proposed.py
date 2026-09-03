"""India-aware traffic model components from the project workflow.

This model keeps the implementation dependency-light while exposing the
workflow's important signals: adaptive adjacency, vehicle-type channels,
weather FiLM conditioning, festival conditioning, temporal attention, and
sensor masking during training.
"""
from __future__ import annotations

import torch
from torch import nn


class IndiaAwareTrafficModel(nn.Module):
    """Multi-signal graph-temporal forecaster for the 24-feature contract."""

    def __init__(self, features: int, nodes: int, horizon: int, hidden: int = 64,
                 sensor_dropout: float = 0.25) -> None:
        super().__init__()
        self.nodes = nodes
        self.horizon = horizon
        self.sensor_dropout = sensor_dropout
        self.input_projection = nn.Linear(features, hidden)
        self.vehicle_projection = nn.Linear(4, hidden)
        self.weather_projection = nn.Linear(3, hidden * 2)
        self.festival_projection = nn.Linear(3, hidden)
        self.temporal = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.temporal_attention = nn.MultiheadAttention(hidden * 2, num_heads=4, batch_first=True)
        self.node_embedding = nn.Parameter(torch.randn(nodes, hidden) * 0.05)
        self.adaptive_logits = nn.Parameter(torch.zeros(nodes, nodes))
        self.fusion = nn.Linear(hidden * 2 + hidden, hidden)
        self.decoder = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, nodes)

    def _mask_sensors(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.sensor_dropout <= 0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, x.shape[3], device=x.device)
        return x * (keep > self.sensor_dropout).to(x.dtype)

    def forward(self, x: torch.Tensor, teacher: torch.Tensor | None = None,
                teacher_forcing: float = 0.0) -> torch.Tensor:
        batch, steps, _, nodes = x.shape
        if nodes != self.nodes:
            raise ValueError(f"expected {self.nodes} nodes, received {nodes}")
        x = self._mask_sensors(x)
        node = torch.relu(self.input_projection(x.permute(0, 1, 3, 2)))
        vehicle = torch.relu(self.vehicle_projection(x[:, :, 4:8].permute(0, 1, 3, 2)))
        weather = self.weather_projection(x[:, :, 12:15].permute(0, 1, 3, 2))
        gamma, beta = weather.chunk(2, dim=-1)
        festival = torch.relu(self.festival_projection(x[:, :, 15:18].permute(0, 1, 3, 2)))
        node = node + vehicle + festival
        node = node * (1.0 + torch.tanh(gamma)) + beta
        node = node + self.node_embedding[None, None, :, :]

        adjacency = torch.softmax(self.adaptive_logits, dim=-1)
        node = node + torch.einsum("ij,btjf->btif", adjacency, node)
        sequence = node.permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)
        encoded, _ = self.temporal(sequence)
        attended, _ = self.temporal_attention(encoded, encoded, encoded)
        context = attended[:, -1].reshape(batch, nodes, -1)
        context = torch.cat([context, node[:, -1]], dim=-1)
        state = torch.tanh(self.fusion(context)).mean(dim=1)
        decoder_input = state
        predictions = []
        for step in range(self.horizon):
            state = self.decoder(decoder_input, state)
            predictions.append(self.head(state))
            if teacher is not None and step + 1 < self.horizon and teacher_forcing > 0:
                decoder_input = state + teacher[:, step].mean(dim=1, keepdim=False).unsqueeze(-1)
            else:
                decoder_input = state
        return torch.stack(predictions, dim=1)