"""Small multilingual Streamlit dashboard for model health and forecasts."""
from __future__ import annotations

import os
from pathlib import Path
import streamlit as st

TRANSLATIONS = {
    "English": {"title": "Traffic Pulse AI", "status": "Model status", "note": "Upload a normalized 12-step traffic window to inspect a forecast."},
    "Hindi": {"title": "ट्रैफिक पल्स एआई", "status": "मॉडल स्थिति", "note": "पूर्वानुमान देखने के लिए सामान्यीकृत ट्रैफिक विंडो अपलोड करें।"},
    "Kannada": {"title": "ಟ್ರಾಫಿಕ್ ಪಲ್ಸ್ AI", "status": "ಮಾದರಿ ಸ್ಥಿತಿ", "note": "ಮುನ್ಸೂಚನೆ ನೋಡಲು ಸಾಮಾನ್ಯೀಕರಿಸಿದ ಟ್ರಾಫಿಕ್ ವಿಂಡೋ ಅಪ್ಲೋಡ್ ಮಾಡಿ."},
    "Tamil": {"title": "டிராஃபிக் பல்ஸ் AI", "status": "மாதிரி நிலை", "note": "முன்னறிவிப்பைப் பார்க்க இயல்பாக்கப்பட்ட போக்குவரத்து சாளரத்தைப் பதிவேற்றவும்."},
    "Marathi": {"title": "ट्रॅफिक पल्स AI", "status": "मॉडेल स्थिती", "note": "अंदाज पाहण्यासाठी सामान्यीकृत वाहतूक विंडो अपलोड करा."},
}

st.set_page_config(page_title="Traffic Pulse AI", layout="wide")
language = st.sidebar.selectbox("Language / भाषा", list(TRANSLATIONS))
copy = TRANSLATIONS[language]
st.title(copy["title"])
st.caption(copy["note"])
model_path = Path(os.environ.get("TRAFFIC_MODEL_PATH", "models/retrained_traffic_forecaster_20260903.pt"))
st.metric(copy["status"], "Ready" if model_path.exists() else "Missing checkpoint")
st.info("API endpoint: POST /predict. The dashboard intentionally does not invent weather or festival explanations when those signals are unavailable.")