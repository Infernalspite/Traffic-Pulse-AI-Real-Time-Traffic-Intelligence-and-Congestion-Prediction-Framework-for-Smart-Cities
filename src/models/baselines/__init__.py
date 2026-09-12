"""Baseline traffic forecasting models."""
from __future__ import annotations

from src.models.traffic_models import (
    AGCRN,
    ARIMABaseline,
    AdaptiveGraphTemporal,
    DCRNN,
    GraphWaveNet,
    LSTMOnly,
    Persistence,
    STGCN,
)

__all__ = [
    "Persistence",
    "ARIMABaseline",
    "LSTMOnly",
    "STGCN",
    "DCRNN",
    "GraphWaveNet",
    "AGCRN",
    "AdaptiveGraphTemporal",
]
