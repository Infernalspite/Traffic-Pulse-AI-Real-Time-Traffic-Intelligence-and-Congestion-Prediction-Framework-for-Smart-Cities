"""Train a leakage-safe traffic-speed baseline from repository and sheet exports.

Usage:
  python src/training/train_baseline.py --repo-csv data/raw/chennai_traffic_log.csv --sheet-csv exports/sheet.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = [
    "current_speed", "free_flow_speed", "current_travel_time",
    "free_flow_travel_time", "confidence", "lat", "lon",
    "speed_lag_1", "speed_lag_2", "speed_lag_3", "speed_roll_3",
    "free_flow_ratio", "junction_code",
]
REQUIRED = [
    "timestamp", "junction", "lat", "lon", "current_speed",
    "free_flow_speed", "current_travel_time", "free_flow_travel_time",
    "confidence", "road_closure",
]


def load_csv(path: Path, source: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if "time_stamp" in frame and "timestamp" not in frame:
        frame = frame.rename(columns={"time_stamp": "timestamp"})
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    frame["source"] = source
    return frame[REQUIRED + ["source"]]


def prepare(repo_csv: Path, sheet_csv: Path | None) -> pd.DataFrame:
    frames = [load_csv(repo_csv, "repository")]
    if sheet_csv:
        frames.append(load_csv(sheet_csv, "google_sheet"))
    frame = pd.concat(frames, ignore_index=True)
    frame["junction"] = frame["junction"].astype(str).str.strip()
    # The repository and Sheet exports use both space-separated and ISO timestamps.
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True, format="mixed")
    frame = frame.dropna(subset=["timestamp", "junction", "current_speed"])
    frame = frame.drop_duplicates(subset=["timestamp", "junction"], keep="last")
    return frame.sort_values(["timestamp", "junction"]).reset_index(drop=True)


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    grouped = frame.groupby("junction", group_keys=False)
    for lag in (1, 2, 3):
        frame[f"speed_lag_{lag}"] = grouped["current_speed"].shift(lag)
    frame["speed_roll_3"] = grouped["current_speed"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).mean()
    )
    frame["free_flow_ratio"] = frame["current_speed"] / frame["free_flow_speed"].replace(0, np.nan)
    frame["target_speed"] = grouped["current_speed"].shift(-1)
    frame["junction_code"] = pd.Categorical(frame["junction"]).codes
    return frame.dropna(subset=FEATURES + ["target_speed"]).copy()


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": round(float(mean_absolute_error(actual, predicted)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(actual, predicted))), 4),
        "MAPE": round(float(np.mean(np.abs((actual - predicted) / np.maximum(np.abs(actual), 1.0))) * 100), 4),
    }


def train(repo_csv: Path, sheet_csv: Path | None, model_out: Path, metrics_out: Path) -> dict:
    raw = prepare(repo_csv, sheet_csv)
    data = make_features(raw)
    cutoff = data["timestamp"].quantile(0.80)
    train = data[data["timestamp"] <= cutoff]
    test = data[data["timestamp"] > cutoff]
    model = RandomForestRegressor(
        n_estimators=120, max_depth=18, min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    )
    model.fit(train[FEATURES], train["target_speed"])
    predicted = model.predict(test[FEATURES])
    last_value = test["speed_lag_1"].to_numpy()
    result = {
        "dataset": {
            "rows": int(len(raw)), "usable_rows": int(len(data)),
            "junctions": int(raw["junction"].nunique()),
            "start": raw["timestamp"].min().isoformat(),
            "end": raw["timestamp"].max().isoformat(),
            "repository_rows": int((raw["source"] == "repository").sum()),
            "google_sheet_rows": int((raw["source"] == "google_sheet").sum()),
            "train_rows": int(len(train)), "test_rows": int(len(test)),
            "train_cutoff": cutoff.isoformat(),
        },
        "features": FEATURES,
        "target": "next observed current_speed per junction",
        "metrics": {
            "last_value": metrics(test["target_speed"], last_value),
            "random_forest_retrained": metrics(test["target_speed"], predicted),
        },
    }
    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "target": result["target"], "cutoff": cutoff.isoformat()}, model_out)
    metrics_out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-csv", type=Path, required=True)
    parser.add_argument("--sheet-csv", type=Path)
    parser.add_argument("--model-out", type=Path, default=Path("models/retrained_random_forest.joblib"))
    parser.add_argument("--metrics-out", type=Path, default=Path("logs/retrained_baseline_metrics.json"))
    args = parser.parse_args()
    print(json.dumps(train(args.repo_csv, args.sheet_csv, args.model_out, args.metrics_out), indent=2))


if __name__ == "__main__":
    main()
