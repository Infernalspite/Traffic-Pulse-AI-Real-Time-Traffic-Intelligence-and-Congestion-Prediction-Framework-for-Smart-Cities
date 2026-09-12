"""Interactive Multilingual Traffic Intelligence & XAI Dashboard.

Features:
- Language toggle: English, Hindi, Kannada, Tamil, Marathi
- Dynamic Folium OpenStreetMap overlay with green/yellow/red congestion coding
- Real-time >80% capacity threshold congestion alert trigger
- GNN & SHAP attribution diagnosis explaining root causes of predicted bottlenecks
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch

from dashboard.map_utils import DEFAULT_JUNCTION_COORDS, build_traffic_folium_map, get_congestion_color
from src.explainability.gnn_explain import TrafficXAIExplainer
from src.models.proposed import IndiaAwareTrafficModel
from src.models.traffic_models import (
    AGCRN,
    AdaptiveGraphTemporal,
    DCRNN,
    GraphWaveNet,
    LSTMOnly,
    STGCN,
)

st.set_page_config(
    page_title="Traffic Pulse AI — Congestion Intelligence",
    page_icon="🚦",
    layout="wide",
)

# 1. Translation Manager
TRANS_DIR = Path(__file__).resolve().parent / "translations"
LANGUAGES = {
    "English": "en",
    "हिंदी (Hindi)": "hi",
    "ಕನ್ನಡ (Kannada)": "kn",
    "தமிழ் (Tamil)": "ta",
    "मराठी (Marathi)": "mr",
}

selected_lang_name = st.sidebar.selectbox("🌐 Language / भाषा / மொழி", list(LANGUAGES.keys()))
lang_code = LANGUAGES[selected_lang_name]
trans_file = TRANS_DIR / f"{lang_code}.json"
if trans_file.exists():
    t = json.loads(trans_file.read_text(encoding="utf-8"))
else:
    t = json.loads((TRANS_DIR / "en.json").read_text(encoding="utf-8"))

st.title(f"🚦 {t['title']}")
st.caption(f"{t['subtitle']} • *{t['retrained_note']}*")

# 2. Model Selection & Loader
CHECKPOINTS = {
    "India-Aware Proposed (Retrained)": "models/retrained_india_aware_latest.pt",
    "LSTM Baseline (Retrained)": "models/retrained_lstm_latest.pt",
    "Graph WaveNet (Retrained)": "models/retrained_gwnet_latest.pt",
    "AGCRN (Retrained)": "models/retrained_agcrn_latest.pt",
    "STGCN (Retrained)": "models/retrained_stgcn_latest.pt",
    "Adaptive Graph (Retrained)": "models/retrained_graph_latest.pt",
}

chosen_model_label = st.sidebar.selectbox("🧠 Select Model Checkpoint", list(CHECKPOINTS.keys()))
model_rel_path = CHECKPOINTS[chosen_model_label]
model_path = Path(model_rel_path)
if not model_path.exists():
    fallback = Path("models/retrained_traffic_forecaster_20260903.pt")
    if fallback.exists():
        model_path = fallback

horizon_min = st.sidebar.select_slider(f"⏱️ {t['forecast_horizon']}", options=[15, 30, 60], value=15)
horizon_step = horizon_min // 5  # 3, 6, 12


@st.cache_resource
def load_checkpoint(path_str: str):
    p = Path(path_str)
    if not p.exists():
        return None, None
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    m_type = ckpt.get("model", "lstm")
    feats = ckpt.get("features", 24)
    nodes = ckpt.get("nodes", 20)
    horiz = ckpt.get("horizon", 12)

    classes = {
        "lstm": LSTMOnly,
        "stgcn": STGCN,
        "dcrnn": DCRNN,
        "gwnet": GraphWaveNet,
        "agcrn": AGCRN,
        "graph": AdaptiveGraphTemporal,
        "india_aware": IndiaAwareTrafficModel,
    }
    model_cls = classes.get(m_type, LSTMOnly)
    instance = model_cls(feats, nodes, horiz)
    instance.load_state_dict(ckpt["state_dict"])
    instance.eval()
    return instance, ckpt


model_instance, meta = load_checkpoint(str(model_path))

# Status metric in sidebar
if model_instance is not None:
    st.sidebar.success(f"✅ {t['model_status']}: Ready ({meta.get('model', 'Model')})")
else:
    st.sidebar.warning(f"⚠️ {t['model_status']}: Checkpoint not found, using baseline fallback.")

# 3. Load sample validation window from retrained data for live visualization
WINDOWS_PATH = Path("data/processed/chennai_sheet_windows_retrained.npz")
if not WINDOWS_PATH.exists():
    WINDOWS_PATH = Path("data/processed/chennai_sheet_windows_20260903.npz")

if WINDOWS_PATH.exists():
    npz_data = np.load(WINDOWS_PATH, allow_pickle=True)
    junction_names = list(npz_data["junctions"])
    feature_names = list(npz_data["feature_names"])
    scale = float(npz_data["feature_scale"][0])
    mean = float(npz_data["feature_mean"][0])
    sample_window = torch.tensor(npz_data["X_test"][0:1], dtype=torch.float32)
else:
    junction_names = list(DEFAULT_JUNCTION_COORDS.keys())
    feature_names = ["speed", "volume", "rainfall_mm_hr", "festival_indicator"]
    scale, mean = 7.5, 26.0
    sample_window = torch.randn(1, 12, 24, 20)

# 4. Generate Predictions & Congestion Capacity
junction_preds = {}
alerts = []
free_flow_defaults = {
    "Kathipara": 35.0, "Guindy": 35.0, "T_Nagar_Panagal": 30.0,
    "Anna_Salai_Teynampet": 35.0, "Central_Station": 28.0, "Koyambedu": 25.0,
}

if model_instance is not None:
    with torch.no_grad():
        norm_pred = model_instance(sample_window)[0].cpu().numpy()  # (12, nodes)
    raw_speeds = norm_pred * scale + mean
else:
    raw_speeds = np.full((12, len(junction_names)), 24.0)

for idx, j_name in enumerate(junction_names):
    spd = float(raw_speeds[horizon_step - 1, idx])
    ff = free_flow_defaults.get(j_name, 35.0)
    cap = float(min(100.0, max(5.0, (1.0 - spd / ff) * 100.0)))
    is_alert = cap >= 80.0

    cause = "Regular commuter flow"
    if cap >= 80.0:
        cause = "Extreme arterial saturation + localized bottleneck"
    elif cap >= 50.0:
        cause = "Moderate peak hour slowdown"

    junction_preds[j_name] = {
        "speed": spd,
        "free_flow": ff,
        "capacity_pct": cap,
        "explanation": cause,
        "alert": is_alert,
    }
    if is_alert:
        alerts.append((j_name, cap, spd))

# 5. Alert Banner Section
st.subheader(f"🚨 {t['alerts_header']}")
if alerts:
    for alert_j, alert_cap, alert_spd in alerts:
        st.error(
            f"⚠️ **PUSH ALERT TRIGGERED:** **{alert_j.replace('_', ' ')}** is operating at **{alert_cap:.0f}% capacity** "
            f"(predicted speed: **{alert_spd:.1f} km/h** at {horizon_min} min horizon). Immediate signal green-split recommended!"
        )
else:
    st.success(f"🟢 {t['no_alerts']}")

# 6. Map & Overview Layout
col1, col2 = st.columns([7, 5])

with col1:
    st.subheader(f"🗺️ {t['map_header']} ({horizon_min} min)")
    folium_map = build_traffic_folium_map(junction_preds)
    components.html(folium_map._repr_html_(), height=500)

with col2:
    st.subheader(f"📊 {t['overview_metrics']}")
    metric_cols = st.columns(2)
    display_junctions = ["Kathipara", "Guindy", "T_Nagar_Panagal", "Anna_Salai_Teynampet"]
    for i, jn in enumerate(display_junctions):
        if jn in junction_preds:
            p = junction_preds[jn]
            with metric_cols[i % 2]:
                st.metric(
                    label=jn.replace("_", " "),
                    value=f"{p['speed']:.1f} km/h",
                    delta=f"{p['capacity_pct']:.0f}% cap",
                    delta_color="inverse" if p["capacity_pct"] >= 50 else "normal",
                )

# 7. Explainable AI (XAI) Diagnosis Panel
st.markdown("---")
st.subheader(f"🔍 {t['xai_header']}")

focal_junction = st.selectbox(f"{t['select_junction']}", junction_names, index=0)
focal_idx = junction_names.index(focal_junction)

if model_instance is not None:
    explainer = TrafficXAIExplainer(model_instance, feature_names, junction_names)
    explanation_dict = explainer.explain_junction(
        sample_window,
        junction_idx=focal_idx,
        horizon_step=horizon_step,
        free_flow_speed=free_flow_defaults.get(focal_junction, 35.0),
    )

    col_x1, col_x2 = st.columns([6, 6])
    with col_x1:
        st.info(f"📋 **Diagnostic Summary:**\n\n> {explanation_dict['explanation']}")
        st.write(f"**{t['predicted_speed']}:** {explanation_dict['predicted_speed_kmh']} km/h")
        st.write(f"**{t['capacity_reached']}:** {explanation_dict['estimated_capacity_pct']}%")
        st.write(f"**Upstream Spillover Corridors:** {', '.join(explanation_dict['top_influencing_corridors']) if explanation_dict['top_influencing_corridors'] else 'Isolated'}")

    with col_x2:
        st.write("**Feature Importance Attributions (Saliency / SHAP):**")
        attrs = explanation_dict["feature_attributions"]
        sorted_attrs = dict(sorted(attrs.items(), key=lambda item: item[1], reverse=True)[:6])
        st.bar_chart(sorted_attrs)
else:
    st.write("XAI explainer is ready once a trained PyTorch model checkpoint is loaded.")
