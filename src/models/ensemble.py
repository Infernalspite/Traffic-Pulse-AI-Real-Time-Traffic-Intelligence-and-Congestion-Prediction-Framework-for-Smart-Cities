"""Multi-Model Meta-Ensemble Forecaster for Traffic Pulse AI.

Combines all trained neural models:
1. Graph WaveNet (Spatial-Temporal Dilated Inception)
2. India-Aware Proposed (Heterogeneous Vehicle Graphs + FiLM Monsoon + Festival Gate)
3. AGCRN (Adaptive Graph Recurrent Network)
4. LSTM Baseline (Temporal Recurrent)
5. Adaptive Graph GNN (Dynamic Adjacency Graph)
6. STGCN (Spatial-Temporal Graph Convolution)
7. DCRNN (Diffusion Convolutional Recurrent Network)

Accounts for all 24 spatio-temporal, environmental, vehicular, and festival parameters.
Computes Bayesian inverse-variance consensus and epistemic prediction uncertainty intervals.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.models.proposed import IndiaAwareTrafficModel
from src.models.traffic_models import (
    AGCRN,
    AdaptiveGraphTemporal,
    DCRNN,
    GraphWaveNet,
    LSTMOnly,
    STGCN,
)

MODEL_REGISTRY = {
    "gwnet": GraphWaveNet,
    "india_aware": IndiaAwareTrafficModel,
    "agcrn": AGCRN,
    "lstm": LSTMOnly,
    "graph": AdaptiveGraphTemporal,
    "stgcn": STGCN,
    "dcrnn": DCRNN,
}

# Empirical held-out validation MAE weights (lower error = higher base weight)
DEFAULT_BASE_MAE = {
    "gwnet": 0.929,
    "agcrn": 0.951,
    "india_aware": 1.231,
    "lstm": 1.227,
    "graph": 1.310,
    "stgcn": 1.354,
    "dcrnn": 1.442,
}


class TrafficPulseEnsemble:
    """Meta-Ensemble that aggregates all individual models into a unified prediction."""

    def __init__(
        self,
        checkpoint_dir: str | Path = "models",
        features: int = 24,
        nodes: int = 20,
        horizon: int = 12,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.features = features
        self.nodes = nodes
        self.horizon = horizon
        self.models: Dict[str, nn.Module] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.base_weights: Dict[str, float] = {}

        ckpt_p = Path(checkpoint_dir)
        checkpoint_map = {
            "gwnet": ckpt_p / "retrained_gwnet_latest.pt",
            "india_aware": ckpt_p / "retrained_india_aware_latest.pt",
            "agcrn": ckpt_p / "retrained_agcrn_latest.pt",
            "lstm": ckpt_p / "retrained_lstm_latest.pt",
            "graph": ckpt_p / "retrained_graph_latest.pt",
            "stgcn": ckpt_p / "retrained_stgcn_latest.pt",
            "dcrnn": ckpt_p / "retrained_dcrnn_latest.pt",
        }

        for m_name, path in checkpoint_map.items():
            if path.exists():
                try:
                    ckpt = torch.load(path, map_location="cpu", weights_only=False)
                    cls = MODEL_REGISTRY.get(m_name, LSTMOnly)
                    model = cls(features, nodes, horizon)
                    model.load_state_dict(ckpt["state_dict"])
                    model.to(self.device)
                    model.eval()
                    self.models[m_name] = model
                    self.metadata[m_name] = ckpt
                    # Inverse-variance weight
                    mae = DEFAULT_BASE_MAE.get(m_name, 1.5)
                    self.base_weights[m_name] = 1.0 / (mae ** 2)
                except Exception as exc:
                    print(f"Warning: Failed to load {m_name} from {path}: {exc}")

        # Normalize base weights
        total_w = sum(self.base_weights.values())
        if total_w > 0:
            self.base_weights = {k: v / total_w for k, v in self.base_weights.items()}

    @property
    def loaded_models(self) -> List[str]:
        return list(self.models.keys())

    def compute_contextual_weights(
        self,
        rainfall_mm_hr: float = 0.0,
        waterlogging_depth: float = 0.0,
        festival_intensity: float = 0.0,
        two_wheeler_pct: float = 0.45,
    ) -> Dict[str, float]:
        """Dynamically adjusts model weights based on domain stress factors."""
        weights = dict(self.base_weights)

        # Under monsoon, festival, or high 2-wheeler friction, boost India-Aware model
        has_monsoon = rainfall_mm_hr > 5.0 or waterlogging_depth > 0.1
        has_festival = festival_intensity > 0.3
        has_mixed_traffic = abs(two_wheeler_pct - 0.45) > 0.15

        if "india_aware" in weights:
            boost = 1.0
            if has_monsoon:
                boost += 0.6 * min(2.0, (rainfall_mm_hr / 15.0) + waterlogging_depth)
            if has_festival:
                boost += 0.5 * festival_intensity
            if has_mixed_traffic:
                boost += 0.3
            weights["india_aware"] *= boost

        # Under heavy rain, standard spatial models without weather gates face distribution drift
        if has_monsoon and "gwnet" in weights:
            weights["gwnet"] *= 0.85

        # Re-normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def predict(
        self,
        x: torch.Tensor,
        rainfall_mm_hr: float = 0.0,
        waterlogging_depth: float = 0.0,
        festival_intensity: float = 0.0,
        two_wheeler_pct: float = 0.45,
        scale: float = 7.5,
        mean: float = 26.0,
    ) -> Dict[str, Any]:
        """
        Runs full multi-model forward passes and returns ensemble consensus and uncertainty.

        Args:
            x: Input tensor of shape (B, 12, 24, 20)
            rainfall_mm_hr: Precipitation intensity
            waterlogging_depth: Water depth index (0.0 to 1.0)
            festival_intensity: Festival demand intensity (0.0 to 1.0)
            two_wheeler_pct: Modal split share of two-wheelers (0.0 to 1.0)
            scale: Target speed unnormalization scale
            mean: Target speed unnormalization mean

        Returns:
            Dict with:
                - ensemble_speeds: ndarray of shape (12, nodes) in km/h
                - uncertainty_std: ndarray of shape (12, nodes) in km/h
                - confidence_lower: 95% lower bound
                - confidence_upper: 95% upper bound
                - model_predictions: Dict[str, ndarray (12, nodes)]
                - weights_used: Dict[str, float]
        """
        if not self.models:
            raise RuntimeError("No models loaded in ensemble.")

        x = x.to(self.device)
        weights = self.compute_contextual_weights(
            rainfall_mm_hr=rainfall_mm_hr,
            waterlogging_depth=waterlogging_depth,
            festival_intensity=festival_intensity,
            two_wheeler_pct=two_wheeler_pct,
        )

        preds_dict: Dict[str, np.ndarray] = {}
        weighted_sum = None
        variance_accum = None

        with torch.no_grad():
            for m_name, model in self.models.items():
                norm_out = model(x)[0].cpu().numpy()  # (12, nodes)
                unnorm_out = norm_out * scale + mean
                preds_dict[m_name] = unnorm_out
                w = weights[m_name]
                if weighted_sum is None:
                    weighted_sum = w * unnorm_out
                else:
                    weighted_sum += w * unnorm_out

        ensemble_mean = weighted_sum

        # Compute weighted inter-model standard deviation (epistemic uncertainty)
        var_accum = np.zeros_like(ensemble_mean)
        for m_name, pred in preds_dict.items():
            w = weights[m_name]
            var_accum += w * ((pred - ensemble_mean) ** 2)
        uncertainty_std = np.sqrt(np.maximum(1e-4, var_accum))

        conf_lower = np.maximum(2.0, ensemble_mean - 1.96 * uncertainty_std)
        conf_upper = ensemble_mean + 1.96 * uncertainty_std

        return {
            "ensemble_speeds": ensemble_mean,
            "uncertainty_std": uncertainty_std,
            "confidence_lower": conf_lower,
            "confidence_upper": conf_upper,
            "model_predictions": preds_dict,
            "weights_used": weights,
        }
