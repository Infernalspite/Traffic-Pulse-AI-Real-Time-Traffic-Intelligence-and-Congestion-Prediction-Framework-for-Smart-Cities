#!/usr/bin/env python3
"""Produce India-stratified and sensor-dropout evaluation artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.metrics import horizon_metrics, stratified_metrics
from src.evaluation.stratified_eval import build_india_strata, sensor_dropout_curve
from src.models.traffic_models import Persistence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("logs/india_stratified_evaluation.json"))
    args = parser.parse_args()
    data = np.load(args.windows, allow_pickle=True)
    x = data["X_test"]
    actual = data["y_test"] * float(data["feature_scale"][0]) + float(data["feature_mean"][0])
    predictor = lambda batch: (
        Persistence(actual.shape[1]).predict(
            __import__("torch").tensor(batch, dtype=__import__("torch").float32)
        ).numpy() * float(data["feature_scale"][0]) + float(data["feature_mean"][0])
    )
    predicted = predictor(x)
    strata = build_india_strata(len(actual))
    result = {
        "horizon_metrics": horizon_metrics(actual, predicted),
        "stratified_metrics": stratified_metrics(actual, predicted, strata),
        "sensor_dropout": sensor_dropout_curve(x, actual, predictor),
        "data_note": "Optional rainfall, waterlogging, and festival labels were not present in this export; dry/normal masks are explicit defaults.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()