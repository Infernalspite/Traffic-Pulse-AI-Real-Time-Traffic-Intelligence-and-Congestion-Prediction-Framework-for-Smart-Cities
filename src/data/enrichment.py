"""Credential-explicit weather and air-quality enrichment helpers."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import requests


def fetch_open_meteo(latitude: float, longitude: float, start_date: str, end_date: str,
                     output: Path) -> pd.DataFrame:
    """Fetch hourly historical weather without requiring an API key."""
    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": latitude, "longitude": longitude,
            "start_date": start_date, "end_date": end_date,
            "hourly": "precipitation,visibility",
            "timezone": "UTC",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {})
    if not hourly.get("time"):
        raise ValueError("Open-Meteo returned no hourly observations")
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"], utc=True),
        "rain_mm": hourly.get("precipitation"),
        "visibility": hourly.get("visibility"),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def fetch_cpcb_json(url: str, output: Path) -> dict:
    """Fetch a CPCB export endpoint supplied by the operator.

    CPCB endpoints and access rules change, so the URL is deliberately
    explicit instead of baking an undocumented endpoint into the pipeline.
    """
    response = requests.get(url, timeout=30, headers={"Accept": "application/json"})
    response.raise_for_status()
    payload = response.json()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    return payload