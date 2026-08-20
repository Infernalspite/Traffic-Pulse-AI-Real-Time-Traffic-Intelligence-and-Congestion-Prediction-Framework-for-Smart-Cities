# Workflow alignment status

Audited against the attached Urban Traffic Flow Prediction — Complete Project Workflow. Status is evidence-based: a checked item has a tracked implementation or artifact; an unchecked item is not claimed as complete.

## Present in the repository

- Chennai traffic collection from TomTom for 20 configured junctions.
- Open-Meteo weather merge in the notebook.
- A 2026 Chennai festival seed CSV.
- Chennai road graph artifacts and junction-to-OSM-node mapping.
- Raw, weather-enriched, processed traffic CSVs and vehicle-count sample data.
- Exploratory notebook and generated result visualizations.
- Basic historical-average, last-value, linear-regression, random-forest, and experimental MonsoonAwareSTGNN result records.

## Not yet implemented

- METR-LA / PeMS-BAY download and reproducible loaders.
- IMD, CPCB AQI, Google Maps/TomTom continuous multi-source collection, and source attribution.
- The required 24-feature tensor, India-specific imputation, train-only Z-score normalization, 12-step windows, and 70/10/20 chronological split.
- ARIMA, LSTM-only, STGCN, DCRNN, Graph WaveNet, and AGCRN baselines with common evaluation.
- The proposed heterogeneous vehicle graph, ConvLSTM/FiLM weather fusion, festival embedding, sparse sensor masking, scheduled sampling, and adaptive adjacency.
- Transfer/meta-learning modules, full training strategy, stratified evaluation, ablations, and sensor-dropout robustness.
- FastAPI predict/status endpoints, Streamlit multilingual XAI dashboard, Folium overlays, alerts, ONNX export, PostgreSQL logging, Docker Compose, experiment tracking, CI, and tests.

## Cleanup decisions

- Removed duplicate collector checkpoint, notebook checkpoint/cache, and duplicate generated dashboard exports from the alignment branch only.
- Preserved datasets, models, the primary notebook, main reports, road graph artifacts, and the collector.
- Credentials are now read from TOMTOM_API_KEY; no secret belongs in source control.

The next implementation milestone should build the deterministic preprocessing contract and baseline evaluation before adding the proposed model.
