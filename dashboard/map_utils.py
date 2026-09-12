"""Interactive Folium OpenStreetMap visualization for urban traffic corridors."""
from __future__ import annotations

import folium
from folium import plugins

# Coordinates for 20 Chennai metropolitan junctions
DEFAULT_JUNCTION_COORDS = {
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


def get_congestion_color(capacity_pct: float) -> str:
    """Return color code: green (<50%), amber (50-79%), red (>=80%)."""
    if capacity_pct >= 80:
        return "#e63946"  # Red / Severe Congestion Gridlock
    if capacity_pct >= 50:
        return "#f4a261"  # Amber / Moderate Delay
    return "#2a9d8f"      # Emerald Green / Free Flow


def build_traffic_folium_map(
    junction_predictions: dict[str, dict],
    center_lat: float = 13.025,
    center_lon: float = 80.215,
    zoom_start: int = 11,
) -> folium.Map:
    """
    Constructs Folium map with color-coded nodes, ripple alerts, and diagnosis popups.
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
    )

    for name, coords in DEFAULT_JUNCTION_COORDS.items():
        data = junction_predictions.get(
            name,
            {"speed": 28.0, "free_flow": 35.0, "capacity_pct": 20.0, "explanation": "Normal traffic flow", "alert": False},
        )
        cap = data.get("capacity_pct", 20.0)
        speed = data.get("speed", 25.0)
        ff = data.get("free_flow", 35.0)
        color = get_congestion_color(cap)
        explanation = data.get("explanation", "Normal conditions")
        is_alert = data.get("alert", cap >= 80)

        alert_badge = "<div style='background:#ffdddd; border:1px solid #ff4d4f; padding:4px 8px; border-radius:4px; color:#cf1322; font-weight:bold; margin-bottom:6px;'>🚨 ALERT: &gt;80% Capacity Threshold Exceeded</div>" if is_alert else ""

        popup_html = f"""
        <div style='width:250px; font-family:-apple-system,BlinkMacSystemFont,sans-serif; font-size:13px;'>
            <h4 style='margin:0 0 6px 0; color:#1f2937;'>{name.replace('_', ' ')}</h4>
            {alert_badge}
            <table style='width:100%; border-collapse:collapse; margin-bottom:6px;'>
                <tr><td><b>Predicted Speed:</b></td><td style='text-align:right;'>{speed:.1f} km/h</td></tr>
                <tr><td><b>Free-Flow Speed:</b></td><td style='text-align:right;'>{ff:.1f} km/h</td></tr>
                <tr><td><b>Capacity Reached:</b></td><td style='text-align:right; color:{color}; font-weight:bold;'>{cap:.0f}%</td></tr>
            </table>
            <div style='background:#f3f4f6; padding:6px; border-radius:4px; font-size:11px; color:#4b5563;'>
                <b>Diagnosis:</b> {explanation}
            </div>
        </div>
        """

        # Inner filled marker
        folium.CircleMarker(
            location=[coords[0], coords[1]],
            radius=11 if not is_alert else 16,
            color="#ffffff",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{name.replace('_', ' ')}: {cap:.0f}% capacity ({speed:.1f} km/h)",
        ).add_to(m)

        # Pulse circle for alert junctions
        if is_alert:
            folium.CircleMarker(
                location=[coords[0], coords[1]],
                radius=24,
                color=color,
                weight=1.5,
                fill=False,
                opacity=0.6,
            ).add_to(m)

    return m
