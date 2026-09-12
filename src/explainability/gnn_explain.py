"""Explainable AI (XAI) for Traffic Pulse AI (Innovation #5).

Combines gradient-based feature attribution and spatial neighbor graph attribution
to generate human-readable natural language diagnostic explanations for traffic operators.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TrafficXAIExplainer:
    """Interprets model forecasts and identifies primary causes of predicted congestion."""

    def __init__(self, model: nn.Module, feature_names: list[str], junctions: list[str]):
        self.model = model
        self.feature_names = feature_names
        self.junctions = junctions

    def explain_junction(
        self,
        traffic_window: torch.Tensor,
        junction_idx: int,
        horizon_step: int = 3,  # default 15-min
        free_flow_speed: float = 40.0,
    ) -> dict:
        """
        Calculates attribution scores and generates operational explanation string.
        Args:
            traffic_window: (1, T, F, N) or (1, T, N, F)
            junction_idx: Index of focal junction
            horizon_step: Step index in forecast (e.g., 3 = 15 min, 6 = 30 min, 12 = 60 min)
        """
        self.model.eval()
        x = traffic_window.clone().detach().requires_grad_(True)
        device = next(self.model.parameters()).device
        x = x.to(device)

        # Forward pass
        pred = self.model(x)  # (1, horizon, nodes)
        target_pred = pred[0, horizon_step - 1, junction_idx]

        # Saliency attribution
        target_pred.backward(retain_graph=False)
        grad = x.grad.abs()  # same shape as x

        # Aggregate attribution across time steps
        if grad.shape[2] == len(self.feature_names):  # (1, T, F, N)
            feat_attr = grad[0, :, :, junction_idx].mean(dim=0).cpu().numpy()
            spatial_attr = grad[0, :, 0, :].mean(dim=0).cpu().numpy()
        else:  # (1, T, N, F)
            feat_attr = grad[0, :, junction_idx, :].mean(dim=0).cpu().numpy()
            spatial_attr = grad[0, :, :, 0].mean(dim=0).cpu().numpy()

        # Normalize attributions
        if feat_attr.sum() > 0:
            feat_attr = feat_attr / feat_attr.sum()
        if spatial_attr.sum() > 0:
            spatial_attr = spatial_attr / spatial_attr.sum()

        top_feat_idx = int(np.argmax(feat_attr))
        top_feature = self.feature_names[top_feat_idx] if top_feat_idx < len(self.feature_names) else "speed"

        # Identify top neighboring junction contributing to congestion
        spatial_rank = np.argsort(spatial_attr)[::-1]
        top_neighbors = [self.junctions[i] for i in spatial_rank if i != junction_idx][:2]

        pred_speed = float(target_pred.detach().cpu().item())
        capacity_pct = int(min(100, max(5, (1.0 - pred_speed / max(free_flow_speed, 1.0)) * 100)))

        # Operational cause synthesis
        j_name = self.junctions[junction_idx]
        causes = []
        if "rain" in top_feature or "monsoon" in top_feature:
            causes.append("monsoon precipitation reducing road friction and throughput")
        elif "festival" in top_feature:
            causes.append("festival shopping rush and ceremonial gatherings")
        elif "two_wheeler" in top_feature or "auto" in top_feature:
            causes.append("heterogeneous modal friction from high two-wheeler/auto density")
        elif "incident" in top_feature or "waterlogging" in top_feature:
            causes.append("incident report and localized waterlogging/lane constriction")
        elif "hour" in top_feature or "dow" in top_feature:
            causes.append("standard peak commuter rush hour pattern")
        else:
            causes.append("cascading bottlenecks from upstream arterial approaches")

        if top_neighbors:
            causes.append(f"traffic spillover from {', '.join(top_neighbors)}")

        explanation = (
            f"{j_name} is predicted to reach {capacity_pct}% capacity in {horizon_step * 5} minutes. "
            f"Primary cause: {' + '.join(causes)}."
        )

        return {
            "junction": j_name,
            "predicted_speed_kmh": round(pred_speed, 1),
            "estimated_capacity_pct": capacity_pct,
            "horizon_minutes": horizon_step * 5,
            "top_contributing_feature": top_feature,
            "top_influencing_corridors": top_neighbors,
            "explanation": explanation,
            "feature_attributions": {self.feature_names[i]: float(feat_attr[i]) for i in range(min(len(self.feature_names), len(feat_attr)))},
        }
