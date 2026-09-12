"""Unit tests for spatio-temporal architectures and India-aware innovations."""
from __future__ import annotations

import torch
import pytest

from src.models.proposed import IndiaAwareTrafficModel
from src.models.vehicle_graph import HeterogeneousVehicleGraph
from src.models.monsoon_encoder import MonsoonWeatherFusion
from src.models.festival_embed import FestivalCalendarEmbedding
from src.models.imputation import SparseSensorImputer
from src.explainability.gnn_explain import TrafficXAIExplainer


def test_innovations_forward_shapes():
    B, T, N, H = 2, 12, 20, 64

    # Innovation #1: Vehicle graph
    v_module = HeterogeneousVehicleGraph(nodes=N, hidden_dim=H)
    v_input = torch.randn(B, T, N, 4)
    v_out = v_module(v_input)
    assert v_out.shape == (B, T, N, H)

    # Innovation #2: Monsoon weather fusion
    w_module = MonsoonWeatherFusion(weather_features=3, hidden_dim=H)
    traffic_h = torch.randn(B, T, N, H)
    w_input = torch.randn(B, T, N, 3)
    w_out = w_module(traffic_h, w_input)
    assert w_out.shape == (B, T, N, H)

    # Innovation #3: Festival calendar embedding
    f_module = FestivalCalendarEmbedding(in_features=3, hidden_dim=H)
    f_input = torch.randn(B, T, N, 3)
    f_out = f_module(f_input, traffic_h)
    assert f_out.shape == (B, T, N, H)

    # Innovation #6: Sparse sensor imputation
    imputer = SparseSensorImputer(min_dropout=0.2, max_dropout=0.4)
    imputer.train()
    x_in = torch.randn(B, T, 24, N)
    x_masked, mask = imputer.mask_sensors(x_in, p=0.3)
    assert x_masked.shape == x_in.shape


def test_india_aware_proposed_forward():
    model = IndiaAwareTrafficModel(features=24, nodes=20, horizon=12, hidden=32)
    x = torch.randn(2, 12, 24, 20)
    out = model(x)
    assert out.shape == (2, 12, 20)


def test_xai_explainer():
    model = IndiaAwareTrafficModel(features=24, nodes=20, horizon=12, hidden=32)
    feature_names = [f"feat_{i}" for i in range(24)]
    junctions = [f"Junction_{i}" for i in range(20)]
    explainer = TrafficXAIExplainer(model, feature_names, junctions)

    x = torch.randn(1, 12, 24, 20)
    res = explainer.explain_junction(x, junction_idx=0, horizon_step=3)
    assert "explanation" in res
    assert "estimated_capacity_pct" in res
    assert res["junction"] == "Junction_0"
