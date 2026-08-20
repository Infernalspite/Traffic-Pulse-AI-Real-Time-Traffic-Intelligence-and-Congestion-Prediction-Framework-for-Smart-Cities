import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

API_KEY = os.environ.get("TOMTOM_API_KEY")
API_URL = os.environ.get(
    "TOMTOM_FLOW_URL",
    "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json",
)
OUTPUT_PATH = Path(
    os.environ.get(
        "TRAFFIC_OUTPUT_PATH",
        Path(__file__).resolve().parents[2] / "data" / "raw" / "chennai_traffic_log.csv",
    )
)
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("TRAFFIC_REQUEST_TIMEOUT_SECONDS", "10"))
REQUEST_PAUSE_SECONDS = float(os.environ.get("TRAFFIC_REQUEST_PAUSE_SECONDS", "0.5"))

JUNCTIONS = {
    "Kathipara": (13.0107, 80.2016),
    "Guindy": (13.0067, 80.2206),
    "T_Nagar_Panagal": (13.0418, 80.2341),
    "Anna_Salai_Teynampet": (13.0455, 80.2500),
    "Egmore": (13.0732, 80.2609),
    "Central_Station": (13.0827, 80.2757),
    "Koyambedu": (13.0694, 80.1948),
    "Vadapalani": (13.0503, 80.2121),
    "Ashok_Nagar": (13.0385, 80.2107),
    "Adyar_Signal": (13.0064, 80.2570),
    "Velachery_Junction": (12.9791, 80.2211),
    "Tambaram": (12.9249, 80.1000),
    "Perungudi_OMR": (12.9634, 80.2431),
    "Thoraipakkam_OMR": (12.9407, 80.2350),
    "Porur": (13.0381, 80.1564),
    "Anna_Nagar_Roundtana": (13.0850, 80.2101),
    "Perambur": (13.1141, 80.2417),
    "Mylapore": (13.0339, 80.2698),
    "Chromepet": (12.9516, 80.1462),
    "Kelambakkam_OMR": (12.7925, 80.2183),
}


def fetch_traffic_data(junctions: dict[str, tuple[float, float]], api_key: str) -> pd.DataFrame:
    if not api_key:
        raise RuntimeError("TOMTOM_API_KEY is required; set it in the environment, never in source code.")

    timestamp = datetime.now(timezone.utc).isoformat()
    rows = []
    with requests.Session() as session:
        for name, (lat, lon) in junctions.items():
            try:
                response = session.get(
                    API_URL,
                    params={"point": f"{lat},{lon}", "key": api_key},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                segment = response.json().get("flowSegmentData", {})
                rows.append({
                    "timestamp": timestamp,
                    "junction": name,
                    "lat": lat,
                    "lon": lon,
                    "current_speed": segment.get("currentSpeed"),
                    "free_flow_speed": segment.get("freeFlowSpeed"),
                    "current_travel_time": segment.get("currentTravelTime"),
                    "free_flow_travel_time": segment.get("freeFlowTravelTime"),
                    "confidence": segment.get("confidence"),
                    "road_closure": segment.get("roadClosure"),
                })
            except (requests.RequestException, ValueError, KeyError) as exc:
                print(f"Failed for {name}: {exc}")
            time.sleep(REQUEST_PAUSE_SECONDS)
    return pd.DataFrame(rows)


def append_snapshot(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        print("No traffic rows collected; leaving output unchanged.")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, mode="a", header=not output_path.exists(), index=False)
    print(f"[{datetime.now(timezone.utc).isoformat()}] Saved {len(df)} rows to {output_path}.")


if __name__ == "__main__":
    append_snapshot(fetch_traffic_data(JUNCTIONS, API_KEY), OUTPUT_PATH)
