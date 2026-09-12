import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "src" / "training" / "train_baseline.py"
spec = importlib.util.spec_from_file_location("train_baseline", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(timestamp, junction, speed, source_fields=True):
    return {
        "timestamp": timestamp,
        "junction": junction,
        "lat": 13.0,
        "lon": 80.2,
        "current_speed": speed,
        "free_flow_speed": 30,
        "current_travel_time": 100,
        "free_flow_travel_time": 80,
        "confidence": 1.0,
        "road_closure": False,
    }


def test_prepare_normalizes_headers_timestamps_and_deduplicates(tmp_path):
    repo = pd.DataFrame([
        row("2026-07-19 16:46:31", "Kathipara", 20),
        row("2026-07-19 16:50:00", "Kathipara", 21),
    ]).rename(columns={"timestamp": "time_stamp"})
    sheet = pd.DataFrame([
        row("2026-07-19T16:46:31+00:00", "Kathipara", 22),
        row("2026-07-19T16:55:00+00:00", "Kathipara", 23),
    ])
    repo_path = tmp_path / "repo.csv"
    sheet_path = tmp_path / "sheet.csv"
    repo.to_csv(repo_path, index=False)
    sheet.to_csv(sheet_path, index=False)

    prepared = module.prepare(repo_path, sheet_path)

    assert len(prepared) == 3
    assert prepared["timestamp"].dt.tz is not None
    assert prepared["source"].tolist().count("google_sheet") == 2
    assert prepared.loc[prepared["timestamp"].dt.hour == 16, "current_speed"].max() == 22


def test_make_features_is_causal():
    rows = [row(f"2026-07-19 16:{minute:02d}:00", "Kathipara", 20 + minute) for minute in range(6)]
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    features = module.make_features(frame)
    assert "target_speed" in features
    assert features.iloc[0]["speed_lag_1"] == 20
    assert features.iloc[0]["target_speed"] == 21
