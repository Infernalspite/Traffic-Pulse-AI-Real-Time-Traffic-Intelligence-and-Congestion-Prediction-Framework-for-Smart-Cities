"""FastAPI service for 12-step traffic forecasts."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.traffic_models import LSTMOnly


MODEL_PATH = Path(os.environ.get("TRAFFIC_MODEL_PATH", "models/retrained_traffic_forecaster_20260903.pt"))
app = FastAPI(title="Traffic Pulse AI", version="2026.09")
_model = None
_metadata: dict[str, Any] = {}


class PredictionRequest(BaseModel):
    traffic_window: list = Field(..., description="Nested [12][24][20] normalized feature tensor")


def _load_model():
    global _model, _metadata
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model checkpoint not found: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if checkpoint["model"] != "lstm":
        raise ValueError("The API loader currently supports the selected LSTM checkpoint only")
    _model = LSTMOnly(checkpoint["features"], checkpoint["nodes"], checkpoint["horizon"])
    _model.load_state_dict(checkpoint["state_dict"])
    _model.eval()
    _metadata = checkpoint
    return _model


@app.get("/status")
def status():
    try:
        _load_model()
        return {"status": "ok", "model": str(MODEL_PATH), "junctions": _metadata["junctions"], "horizon_steps": _metadata["horizon"]}
    except Exception as exc:
        return {"status": "error", "model": str(MODEL_PATH), "detail": str(exc)}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        values = np.asarray(request.traffic_window, dtype=np.float32)
        expected = (12, int(_metadata.get("features", 24)), int(_metadata.get("nodes", 20)))
        if values.shape != expected:
            raise HTTPException(status_code=422, detail=f"traffic_window must have shape {expected}, received {values.shape}")
        with torch.no_grad():
            prediction = _load_model()(torch.from_numpy(values[None]))[0].numpy()
        scale = np.asarray(_metadata["feature_scale"], dtype=float)[0]
        mean = np.asarray(_metadata["feature_mean"], dtype=float)[0]
        return {"junctions": _metadata["junctions"], "forecast_speed": (prediction * scale + mean).round(4).tolist(), "horizon_minutes": [15, 30, 60]}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc