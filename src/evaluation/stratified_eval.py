"""India-specific strata and sensor-dropout evaluation helpers."""
from __future__ import annotations

from collections.abc import Callable
import numpy as np

from .metrics import horizon_metrics


def build_india_strata(
    sample_count: int,
    rainfall_mm_hr: np.ndarray | None = None,
    waterlogging: np.ndarray | None = None,
    festival_state: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build aligned masks for monsoon and festival reporting.

    Missing optional signals are represented as an explicit normal/dry mask;
    this avoids inventing weather or calendar labels.
    """
    rain = np.zeros(sample_count, dtype=float) if rainfall_mm_hr is None else np.asarray(rainfall_mm_hr, dtype=float)
    water = np.zeros(sample_count, dtype=bool) if waterlogging is None else np.asarray(waterlogging, dtype=bool)
    festival = np.zeros(sample_count, dtype=int) if festival_state is None else np.asarray(festival_state, dtype=int)
    if not (len(rain) == len(water) == len(festival) == sample_count):
        raise ValueError("stratification arrays must match the number of samples")
    return {
        "dry": (rain <= 0) & ~water,
        "light_rain": (rain > 0) & (rain < 5) & ~water,
        "heavy_rain": (rain >= 5) & ~water,
        "waterlogging": water,
        "normal_day": festival == 0,
        "festival_eve": festival == 1,
        "festival_day": festival == 2,
        "post_festival_day": festival == 3,
    }


def sensor_dropout_curve(
    x: np.ndarray,
    actual: np.ndarray,
    predictor: Callable[[np.ndarray], np.ndarray],
    availability: tuple[float, ...] = (1.0, .9, .8, .7, .6, .5, .4, .3),
    seed: int = 42,
) -> dict[str, dict]:
    """Measure error as a percentage of node sensors are retained."""
    rng = np.random.default_rng(seed)
    results = {}
    for fraction in availability:
        masked = x.copy()
        node_mask = rng.random((x.shape[0], x.shape[-1])) < fraction
        masked *= node_mask[:, None, None, :]
        prediction = predictor(masked)
        results[f"{int(fraction * 100)}%"] = horizon_metrics(actual, prediction)
    return results