"""FastAPI REST service for 15, 30, and 60-minute urban traffic flow forecasts."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.proposed import IndiaAwareTrafficModel
from src.models.traffic_models import (
    AGCRN,
    AdaptiveGraphTemporal,
    DCRNN,
    GraphWaveNet,
    LSTMOnly,
    STGCN,
)

MODEL_PATH = Path(os.environ.get("TRAFFIC_MODEL_PATH", "models/retrained_india_aware_latest.pt"))
if not MODEL_PATH.exists():
    fallback = Path("models/retrained_lstm_latest.pt")
    if fallback.exists():
        MODEL_PATH = fallback
    else:
        fallback_orig = Path("models/retrained_traffic_forecaster_20260903.pt")
        if fallback_orig.exists():
            MODEL_PATH = fallback_orig

app = FastAPI(
    title="Traffic Pulse AI — Intelligence & Congestion API",
    description="REST API for multi-horizon spatio-temporal traffic forecasts across metropolitan junctions",
    version="2026.09",
)

_model = None
_metadata: dict[str, Any] = {}


class PredictionRequest(BaseModel):
    traffic_window: Optional[List[List[List[float]]]] = Field(
        None,
        description="Full [12][24][20] or [12][features][nodes] normalized feature tensor",
    )
    junction_ids: Optional[List[str]] = Field(
        None,
        description="Optional list of junction names to query",
    )
    current_speeds: Optional[Dict[str, float]] = Field(
        None,
        description="Optional map of junction -> current observed speed (km/h)",
    )
    weather: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional weather JSON: {'rainfall_mm_hr': float, 'waterlogging': bool}",
    )


def _load_model():
    global _model, _metadata
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    m_type = ckpt.get("model", "india_aware")
    feats = ckpt.get("features", 24)
    nodes = ckpt.get("nodes", 20)
    horiz = ckpt.get("horizon", 12)

    model_registry = {
        "lstm": LSTMOnly,
        "stgcn": STGCN,
        "dcrnn": DCRNN,
        "gwnet": GraphWaveNet,
        "agcrn": AGCRN,
        "graph": AdaptiveGraphTemporal,
        "india_aware": IndiaAwareTrafficModel,
    }
    cls = model_registry.get(m_type, IndiaAwareTrafficModel)
    _model = cls(feats, nodes, horiz)
    _model.load_state_dict(ckpt["state_dict"])
    _model.eval()
    _metadata = ckpt
    return _model


@app.get("/status")
def status():
    """Returns operational health, loaded checkpoint type, and active node list."""
    try:
        m = _load_model()
        return {
            "status": "healthy",
            "model_path": str(MODEL_PATH),
            "architecture": _metadata.get("model", "unknown"),
            "junction_count": _metadata.get("nodes", 20),
            "junctions": _metadata.get("junctions", []),
            "forecast_horizons": ["15min", "30min", "60min"],
            "cuda_available": torch.cuda.is_available(),
        }
    except Exception as exc:
        return {"status": "degraded", "model_path": str(MODEL_PATH), "error": str(exc)}


@app.post("/predict")
def predict(request: PredictionRequest):
    """Generates 15-, 30-, and 60-minute speed forecasts across junctions."""
    try:
        m = _load_model()
        feats = int(_metadata.get("features", 24))
        nodes = int(_metadata.get("nodes", 20))
        junction_list = _metadata.get("junctions", [])

        # 1. Direct tensor provided
        if request.traffic_window is not None:
            values = np.asarray(request.traffic_window, dtype=np.float32)
            if values.shape != (12, feats, nodes):
                raise HTTPException(
                    status_code=422,
                    detail=f"Expected traffic_window shape (12, {feats}, {nodes}), received {values.shape}",
                )
        # 2. Point readings provided: construct window from current observations
        elif request.current_speeds is not None:
            values = np.zeros((12, feats, nodes), dtype=np.float32)
            scale = float(_metadata.get("feature_scale", [1.0])[0])
            mean = float(_metadata.get("feature_mean", [0.0])[0])
            for i, name in enumerate(junction_list):
                spd = request.current_speeds.get(name, mean)
                norm_spd = (spd - mean) / (scale + 1e-8)
                values[:, 0, i] = norm_spd
            # Infill weather if provided
            if request.weather:
                rain = float(request.weather.get("rainfall_mm_hr", 0.0))
                values[:, 12, :] = rain
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either 'traffic_window' tensor or 'current_speeds' map.",
            )

        with torch.no_grad():
            inp_t = torch.from_numpy(values[None])
            pred_norm = m(inp_t)[0].numpy()  # (12, nodes)

        scale = float(_metadata.get("feature_scale", [1.0])[0])
        mean = float(_metadata.get("feature_mean", [0.0])[0])
        unnorm_pred = pred_norm * scale + mean

        results = {}
        for i, name in enumerate(junction_list):
            results[name] = {
                "15min_kmh": round(float(unnorm_pred[2, i]), 2),
                "30min_kmh": round(float(unnorm_pred[5, i]), 2),
                "60min_kmh": round(float(unnorm_pred[11, i]), 2),
            }

        # Filter by requested junction IDs if specified
        if request.junction_ids:
            results = {k: v for k, v in results.items() if k in request.junction_ids}

        return {
            "status": "success",
            "model": _metadata.get("model", "unknown"),
            "forecasts": results,
            "horizon_minutes": [15, 30, 60],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
