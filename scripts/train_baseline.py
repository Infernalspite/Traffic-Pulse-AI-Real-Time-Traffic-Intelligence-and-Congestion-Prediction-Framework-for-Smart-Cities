#!/usr/bin/env python3
"""Reproducible chronological baseline for Chennai traffic-speed forecasting.

The script owns the data contract previously duplicated in an exploratory notebook.
It predicts the next observed speed for each junction using current information only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DEFAULT_INPUTS = (Path("data/processed/chennai_full_dataset.csv"), Path("data/raw/chennai_traffic_log.csv"))
BASE_COLUMNS = ["current_speed", "free_flow_speed", "current_travel_time", "free_flow_travel_time", "confidence", "lat", "lon"]
FEATURE_COLUMNS = BASE_COLUMNS + ["speed_lag_1", "speed_lag_2", "speed_lag_3", "speed_roll_3", "free_flow_ratio", "junction_code"]
TARGET = "target_next_speed"

def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    frame = frame.rename(columns={key: value for key, value in {"time": "timestamp", "junction_name": "junction"}.items() if key in frame})
    missing = {"timestamp", "junction", "current_speed"} - set(frame.columns)
    if missing: raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    for column in BASE_COLUMNS:
        if column not in frame: frame[column] = np.nan
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    for column in BASE_COLUMNS: frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[["timestamp", "junction", *BASE_COLUMNS]]

def load_data(inputs: Iterable[Path]) -> pd.DataFrame:
    frames = [_read_csv(path) for path in inputs if path.exists()]
    if not frames: raise FileNotFoundError("No input CSV found. Pass --input with a traffic CSV.")
    frame = pd.concat(frames, ignore_index=True).dropna(subset=["timestamp", "junction", "current_speed"])
    frame["junction"] = frame["junction"].astype(str).str.strip()
    return frame.drop_duplicates(subset=["timestamp", "junction"], keep="last").sort_values(["timestamp", "junction"]).reset_index(drop=True)

def make_supervised(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    grouped = result.groupby("junction", sort=False)["current_speed"]
    result["speed_lag_1"] = grouped.shift(1)
    result["speed_lag_2"] = grouped.shift(2)
    result["speed_lag_3"] = grouped.shift(3)
    result["speed_roll_3"] = grouped.transform(lambda values: values.shift(1).rolling(3, min_periods=1).mean())
    result["free_flow_ratio"] = result["current_speed"] / result["free_flow_speed"].replace(0, np.nan)
    result["junction_code"] = pd.Categorical(result["junction"]).codes
    result[TARGET] = grouped.shift(-1)
    return result.dropna(subset=FEATURE_COLUMNS + [TARGET]).sort_values("timestamp").reset_index(drop=True)

def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    actual_values = actual.to_numpy(dtype=float); predicted_values = np.asarray(predicted, dtype=float)
    nonzero = np.abs(actual_values) > 1e-9
    return {"MAE": round(float(mean_absolute_error(actual_values, predicted_values)), 4), "RMSE": round(float(np.sqrt(mean_squared_error(actual_values, predicted_values))), 4), "MAPE": round(float(np.mean(np.abs((actual_values[nonzero] - predicted_values[nonzero]) / actual_values[nonzero])) * 100), 4)}

def train_and_evaluate(data: pd.DataFrame, seed: int = 42) -> dict:
    if len(data) < 30: raise ValueError(f"Need at least 30 supervised rows; found {len(data)}")
    timestamps = data["timestamp"].sort_values().reset_index(drop=True)
    train_end = timestamps.iloc[int(len(timestamps) * 0.70)]; validation_end = timestamps.iloc[int(len(timestamps) * 0.80)]
    train = data[data["timestamp"] < train_end]; validation = data[(data["timestamp"] >= train_end) & (data["timestamp"] < validation_end)]; test = data[data["timestamp"] >= validation_end]
    if min(len(train), len(validation), len(test)) == 0: raise ValueError("Chronological split produced an empty partition")
    model = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1, min_samples_leaf=2); model.fit(train[FEATURE_COLUMNS], train[TARGET])
    predicted = model.predict(test[FEATURE_COLUMNS])
    return {"dataset": {"rows": int(len(data)), "junctions": int(data["junction"].nunique()), "start": data["timestamp"].min().isoformat(), "end": data["timestamp"].max().isoformat(), "train_rows": int(len(train)), "validation_rows": int(len(validation)), "test_rows": int(len(test)), "train_end": train_end.isoformat(), "validation_end": validation_end.isoformat()}, "features": FEATURE_COLUMNS, "target": "next observed current_speed per junction", "split": "chronological 70/10/20; no shuffled rows", "metrics": {"last_value": regression_metrics(test[TARGET], test["current_speed"]), "random_forest": regression_metrics(test[TARGET], predicted)}}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--input", action="append", type=Path); parser.add_argument("--output", type=Path, default=Path("logs/retrained_baseline_metrics.json")); parser.add_argument("--seed", type=int, default=42); args = parser.parse_args()
    metrics = train_and_evaluate(make_supervised(load_data(args.input or list(DEFAULT_INPUTS))), seed=args.seed); args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8"); print(json.dumps(metrics, indent=2))

if __name__ == "__main__": main()
