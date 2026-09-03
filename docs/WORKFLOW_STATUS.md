# Workflow alignment status

Audited against the attached Urban Traffic Flow Prediction — Complete Project
Workflow. A component is marked implemented only when it has runnable code or
an checked artifact in this repository.

## Implemented locally

- TomTom collection for the configured Chennai junctions with explicit
  `TOMTOM_API_KEY` handling.
- Open-Meteo historical rainfall/visibility collector and an explicit CPCB JSON
  adapter; provider URLs and credentials are never silently guessed.
- 24-feature tensor contract, mixed timestamp handling, 5-minute regularization,
  interpolation, train-only Z-score normalization, 12x12 windows, and
  chronological 70/10/20 splitting.
- Persistence and differenced ARIMA-style statistical baselines.
- LSTM, STGCN, DCRNN, Graph WaveNet, AGCRN, adaptive graph, and India-aware
  graph-temporal model implementations sharing one trainer contract.
- India-stratified metrics, sensor dropout curves from 100% to 30%, checkpoint
  metadata, ONNX export utility, and file-backed experiment registry.
- Adaptive adjacency, vehicle-channel fusion, weather FiLM conditioning,
  festival conditioning, temporal attention, and training-time sensor masking
  in `src/models/proposed.py`.
- Spectral graph alignment and 3/7/14-day few-shot slice utilities.
- FastAPI `/status` and `/predict`, multilingual Streamlit shell, Docker Compose
  services, local PostgreSQL service, tests, and GitHub Actions CI.
- Imported spreadsheet retraining artifacts and a documented recommended LSTM
  checkpoint under `docs/RETRAINING_20260903.md`.

## Explicit external prerequisites

- METR-LA and PeMS-BAY still require current download URLs and acceptance of
  their source terms; `scripts/download_benchmarks.py` downloads operator-
  supplied URLs without embedding stale links.
- IMD, CPCB, Google Maps, and continuous TomTom collection require provider
  access and credentials. The local workflow does not fabricate those records.
- OSMnx graph construction requires a live Overpass/network request and a
  selected city. Existing Chennai graph artifacts are preserved.
- MLflow/W&B, SHAP/GNNExplainer, push notifications, and translation providers
  are integration points; the core path runs offline without them.
- A research paper and demo video are deliverables outside executable source
  code and are not represented as completed claims.

## Reproducibility boundary

The spreadsheet retraining is a valid Chennai speed-forecast experiment, but
its export lacks weather, AQI, vehicle-count, and infrastructure observations.
Those 24-feature positions use documented defaults, so monsoon- and
vehicle-specific metrics remain unavailable until enriched data is supplied.
