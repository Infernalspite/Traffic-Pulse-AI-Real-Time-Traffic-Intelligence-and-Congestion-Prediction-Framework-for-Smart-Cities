import numpy as np

from src.evaluation.metrics import horizon_metrics


def test_horizon_metrics_uses_requested_minutes():
    actual = np.ones((2, 12, 3))
    predicted = np.zeros_like(actual)
    result = horizon_metrics(actual, predicted)
    assert set(result) == {"15min", "30min", "60min"}
    assert result["60min"]["MAE"] == 1.0