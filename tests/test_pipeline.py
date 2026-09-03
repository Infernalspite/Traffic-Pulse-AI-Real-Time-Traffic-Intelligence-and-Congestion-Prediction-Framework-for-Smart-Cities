from pathlib import Path

import numpy as np
import pandas as pd

from src.data.pipeline import TensorConfig, make_windows, to_feature_frame


def test_sheet_style_timestamps_and_missing_features():
    frame = pd.DataFrame(
        {
            "time_stamp": ["2026-01-01 00:00:01", "2026-01-01T00:05:01.123+00:00"] * 20,
            "junction": ["J1", "J2"] * 20,
            "current_speed": np.linspace(10, 20, 40),
            "road_closure": [False] * 40,
        }
    )
    frame["timestamp"] = pd.to_datetime(frame.pop("time_stamp"), format="mixed", utc=True)
    features, names = to_feature_frame(frame)
    assert len(names) == 24
    windows = make_windows(features, TensorConfig(input_steps=1, forecast_steps=1))
    assert windows["X_train"].shape[-2:] == (24, 2)
    assert not np.isnan(windows["X_train"]).any()