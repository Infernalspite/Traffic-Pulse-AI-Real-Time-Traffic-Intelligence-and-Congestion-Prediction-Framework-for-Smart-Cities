#!/usr/bin/env python3
"""Produce India-stratified, multi-horizon, and sensor-dropout evaluation artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.metrics import horizon_metrics, stratified_metrics
from src.evaluation.stratified_eval import build_india_strata, sensor_dropout_curve
from src.models.proposed import IndiaAwareTrafficModel
from src.models.traffic_models import (
    AGCRN,
    AdaptiveGraphTemporal,
    DCRNN,
    GraphWaveNet,
    LSTMOnly,
    Persistence,
    STGCN,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Optional trained PyTorch checkpoint path")
    parser.add_argument("--output", type=Path, default=Path("logs/india_stratified_evaluation.json"))
    args = parser.parse_args()

    data = np.load(args.windows, allow_pickle=True)
    x = data["X_test"]
    scale = float(data["feature_scale"][0])
    mean = float(data["feature_mean"][0])
    actual = data["y_test"] * scale + mean

    if args.checkpoint and args.checkpoint.exists():
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        m_name = ckpt.get("model", "india_aware")
        classes = {
            "lstm": LSTMOnly,
            "stgcn": STGCN,
            "dcrnn": DCRNN,
            "gwnet": GraphWaveNet,
            "agcrn": AGCRN,
            "graph": AdaptiveGraphTemporal,
            "india_aware": IndiaAwareTrafficModel,
        }
        model_cls = classes.get(m_name, IndiaAwareTrafficModel)
        model = model_cls(ckpt["features"], ckpt["nodes"], ckpt["horizon"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        def predictor(batch: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                tensor_in = torch.tensor(batch, dtype=torch.float32)
                pred_norm = model(tensor_in).cpu().numpy()
            return pred_norm * scale + mean
    else:
        m_name = "persistence"
        predictor = lambda batch: (
            Persistence(actual.shape[1]).predict(
                torch.tensor(batch, dtype=torch.float32)
            ).numpy() * scale + mean
        )

    predicted = predictor(x)
    strata = build_india_strata(len(actual))

    result = {
        "model_evaluated": m_name,
        "test_samples": len(actual),
        "horizon_metrics": horizon_metrics(actual, predicted),
        "stratified_metrics": stratified_metrics(actual, predicted, strata),
        "sensor_dropout_curve": sensor_dropout_curve(x, actual, predictor),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
