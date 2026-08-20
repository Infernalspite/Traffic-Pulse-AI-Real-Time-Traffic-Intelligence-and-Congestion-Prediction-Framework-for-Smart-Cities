import requests
import pandas as pd
from datetime import datetime
import time
import os

API_KEY = "kDTaToysSfirkXV1oMpJde88xATAIgDn"

junctions = {
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

def fetch_traffic_data(junctions, api_key):
    rows = []
    timestamp = datetime.now().isoformat()
    for name, (lat, lon) in junctions.items():
        url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
        params = {"point": f"{lat},{lon}", "key": api_key}
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()["flowSegmentData"]
            rows.append({
                "timestamp": timestamp,
                "junction": name,
                "lat": lat,
                "lon": lon,
                "current_speed": data.get("currentSpeed"),
                "free_flow_speed": data.get("freeFlowSpeed"),
                "current_travel_time": data.get("currentTravelTime"),
                "free_flow_travel_time": data.get("freeFlowTravelTime"),
                "confidence": data.get("confidence"),
                "road_closure": data.get("roadClosure"),
            })
        except Exception as e:
            print(f"Failed for {name}: {e}")
        time.sleep(0.5)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "chennai_traffic_log.csv")
    df = fetch_traffic_data(junctions, API_KEY)
    file_exists = os.path.exists(output_path)
    df.to_csv(output_path, mode="a", header=not file_exists, index=False)
    print(f"[{datetime.now()}] Saved {len(df)} rows.")
