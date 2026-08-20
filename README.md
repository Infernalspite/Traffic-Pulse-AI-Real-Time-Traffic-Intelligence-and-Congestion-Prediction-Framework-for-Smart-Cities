# Traffic Pulse AI

Traffic Pulse AI is a Chennai-focused prototype for traffic intelligence and congestion prediction. The repository currently contains data collection, weather enrichment, exploratory modeling, road-network artifacts, and experiment outputs.

## Current state

This is not yet a complete implementation of the attached 12-phase research workflow. See docs/WORKFLOW_STATUS.md for the verified gap list. The research claims and metrics in generated logs should be treated as prototype results until the missing reproducible preprocessing, baselines, splits, tests, and evaluation reports are implemented.

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

## Safety and reproducibility

Never commit API keys, generated caches, local virtual environments, or notebook checkpoints. Large binary assets use Git LFS pointers; ensure Git LFS is installed before fetching them.
