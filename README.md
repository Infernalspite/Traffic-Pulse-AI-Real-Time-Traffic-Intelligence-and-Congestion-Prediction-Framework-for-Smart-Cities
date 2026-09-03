# Traffic Pulse AI

Traffic Pulse AI is a Chennai-focused prototype for traffic intelligence and congestion prediction. The repository currently contains data collection, weather enrichment, exploratory modeling, road-network artifacts, and experiment outputs.

## Current state

The repository now contains a runnable, dependency-light implementation of the
workflow's data contract, model comparisons, India-specific evaluation hooks,
inference API, dashboard shell, export path, transfer utilities, CI, and
container configuration. See docs/WORKFLOW_STATUS.md for the exact boundary
between implemented local components and external data collection that still
requires provider access.

## Run collection

1. Create a virtual environment and install requirements.txt.
2. Set TOMTOM_API_KEY in the environment (use .env.example as a template).
3. Run python src/data/collector.py for one snapshot, or python collect_loop.py for periodic collection.

The collector writes append-only snapshots to data/raw/chennai_traffic_log.csv. Collection interval and output path are configurable with environment variables.

## Data and notebooks

- data/raw/ contains collected traffic, weather, festival, and vehicle-count inputs.
- data/graphs/ and data/road_network/ contain Chennai graph artifacts.
- notebooks/trafficmodelv2.ipynb contains the current exploratory workflow.
- logs/final_results.json records the current prototype comparison.
- docs/RETRAINING_20260903.md records the new spreadsheet retraining run and its metrics.

## Latest retraining

The latest raw traffic export is stored at `data/raw/chennai_traffic_log_sheet_20260903.csv`.
To rebuild its chronological windows and retrain the available baselines:

```bash
python scripts/build_chennai_tensors.py \
  --input data/raw/chennai_traffic_log_sheet_20260903.csv \
  --output data/processed/chennai_sheet_windows_20260903.npz
python scripts/train_traffic_models.py \
  --windows data/processed/chennai_sheet_windows_20260903.npz \
  --model lstm \
  --checkpoint models/retrained_traffic_forecaster_20260903.pt \
  --output logs/retrained_lstm_metrics_20260903.json
```

The recommended checkpoint from this run is `models/retrained_traffic_forecaster_20260903.pt`.

## Run on a personal computer

Requirements: Python 3.10+ and Git. CPU training works; CUDA is used
automatically when available.

```bash
git clone https://github.com/Infernalspite/Traffic-Pulse-AI-Real-Time-Traffic-Intelligence-and-Congestion-Prediction-Framework-for-Smart-Cities.git
cd Traffic-Pulse-AI-Real-Time-Traffic-Intelligence-and-Congestion-Prediction-Framework-for-Smart-Cities
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-workflow.txt  # API/dashboard/graph tools
```

Build windows and reproduce the comparison:

```bash
python scripts/build_chennai_tensors.py --input data/raw/chennai_traffic_log_sheet_20260903.csv --output data/processed/chennai_sheet_windows_20260903.npz
python scripts/train_traffic_models.py --windows data/processed/chennai_sheet_windows_20260903.npz --model persistence --output logs/pc_persistence.json
python scripts/train_traffic_models.py --windows data/processed/chennai_sheet_windows_20260903.npz --model lstm --epochs 40 --batch-size 128 --checkpoint models/pc_lstm.pt --output logs/pc_lstm.json
python scripts/train_traffic_models.py --windows data/processed/chennai_sheet_windows_20260903.npz --model graph --epochs 40 --batch-size 128 --checkpoint models/pc_graph.pt --output logs/pc_graph.json
python scripts/evaluate_workflow.py --windows data/processed/chennai_sheet_windows_20260903.npz --output logs/pc_india_evaluation.json
```

Run the API and dashboard:

```bash
set TRAFFIC_MODEL_PATH=models\retrained_traffic_forecaster_20260903.pt  # Windows
# export TRAFFIC_MODEL_PATH=models/retrained_traffic_forecaster_20260903.pt # macOS/Linux
uvicorn api.app:app --reload --port 8000
# In another terminal:
streamlit run dashboard/app.py
```

Run the test suite:

```bash
PYTHONPATH=. python -m pytest -q tests  # macOS/Linux
python -m pytest -q tests              # Windows PowerShell
```

Or use Docker Compose for API, dashboard, and local PostgreSQL:

```bash
docker compose up --build
```

Optional collectors are explicit:

```bash
python scripts/fetch_open_meteo.py --latitude 13.0827 --longitude 80.2707 --start-date 2026-07-19 --end-date 2026-08-26 --output data/raw/chennai_open_meteo.csv
python scripts/download_benchmarks.py --metr-la-url <current-download-url> --pems-bay-url <current-download-url>
```

## Safety and reproducibility

Never commit API keys, generated caches, local virtual environments, or notebook checkpoints. Large binary assets use Git LFS pointers; ensure Git LFS is installed before fetching them.
