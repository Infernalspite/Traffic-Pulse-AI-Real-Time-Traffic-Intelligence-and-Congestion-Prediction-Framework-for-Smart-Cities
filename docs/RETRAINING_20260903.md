# Traffic Pulse AI — Spreadsheet Retraining Run

## Data provenance

- Source: [Chennai traffic log spreadsheet](https://docs.google.com/spreadsheets/d/1_uHZd9C5s1b_6TWCy4LdhF8oLH_UBOROGmCixMjLNv8/edit)
- Export file: `data/raw/chennai_traffic_log_sheet_20260903.csv`
- Raw records: 26,718
- Junctions: 20
- Time range: 2026-07-19 16:46:31 through 2026-08-26 06:41:06 UTC
- Observed 5-minute bins: 1,314

The export contains the raw traffic fields: timestamp, junction, latitude,
longitude, current/free-flow speed, current/free-flow travel time, confidence,
and road-closure flag. It does not contain the optional weather, AQI, vehicle
count, or infrastructure fields. The preprocessing contract therefore keeps
the 24-feature tensor shape, uses the observed speed and closure fields, and
fills unavailable optional features with their documented defaults.

## Preprocessing contract

The pipeline now:

1. Accepts both `timestamp` and the sheet's `time_stamp` header.
2. Parses mixed timestamp representations safely.
3. Floors timestamp jitter into 5-minute bins.
4. Creates a complete 20-junction time grid and interpolates missing values.
5. Computes normalization statistics from the chronological training portion only.
6. Creates 12-step input and 12-step forecast windows.
7. Splits windows chronologically into 70% train, 10% validation, and 20% test.

Generated windows:

| Split | Samples | Shape |
| --- | ---: | --- |
| Train | 7,553 | `(12, 24, 20)` input / `(12, 20)` target |
| Validation | 1,083 | `(12, 24, 20)` input / `(12, 20)` target |
| Test | 2,165 | `(12, 24, 20)` input / `(12, 20)` target |

The split boundaries are 2026-08-15 00:05 UTC and 2026-08-18 18:20 UTC.

## Held-out test results

All metrics are speed forecast errors in the original speed units. The test
set is shared across models.

| Model | Horizon | MAE | RMSE | MAPE |
| --- | --- | ---: | ---: | ---: |
| Persistence | 15 min | 0.513 | 1.011 | 2.711% |
| Persistence | 30 min | 0.926 | 1.756 | 4.876% |
| Persistence | 60 min | 1.492 | 2.636 | 7.739% |
| LSTM | 15 min | 1.194 | 1.973 | 6.479% |
| LSTM | 30 min | 1.215 | 1.994 | 6.588% |
| LSTM | 60 min | 1.253 | 2.044 | 6.795% |
| Adaptive graph | 15 min | 1.490 | 2.087 | 7.801% |
| Adaptive graph | 30 min | 1.562 | 2.204 | 8.125% |
| Adaptive graph | 60 min | 1.734 | 2.459 | 8.900% |

The LSTM is the recommended learned model for this export because it has the
lowest learned-model MAE at all three horizons. The persistence baseline is
still retained as a useful sanity check, and the adaptive graph checkpoint is
available for future experiments.

## Outputs

- `data/processed/chennai_sheet_windows_20260903.npz`
- `data/processed/chennai_sheet_windows_20260903.json`
- `models/retrained_traffic_forecaster_20260903.pt` — recommended LSTM
- `models/retrained_lstm_20260903.pt` — same LSTM checkpoint
- `models/retrained_adaptive_graph_20260903.pt`
- `logs/retrained_persistence_metrics_20260903.json`
- `logs/retrained_lstm_metrics_20260903.json`
- `logs/retrained_graph_metrics_20260903.json`

The model checkpoints include the model type, tensor dimensions, junction
ordering, feature names, train-only normalization statistics, and random seed
needed for inference.

## Reproduction

From the repository root:

```bash
python scripts/build_chennai_tensors.py \
  --input data/raw/chennai_traffic_log_sheet_20260903.csv \
  --output data/processed/chennai_sheet_windows_20260903.npz

python scripts/train_traffic_models.py \
  --windows data/processed/chennai_sheet_windows_20260903.npz \
  --model persistence \
  --output logs/retrained_persistence_metrics_20260903.json

python scripts/train_traffic_models.py \
  --windows data/processed/chennai_sheet_windows_20260903.npz \
  --model lstm \
  --epochs 40 \
  --batch-size 128 \
  --checkpoint models/retrained_lstm_20260903.pt \
  --output logs/retrained_lstm_metrics_20260903.json

python scripts/train_traffic_models.py \
  --windows data/processed/chennai_sheet_windows_20260903.npz \
  --model graph \
  --epochs 40 \
  --batch-size 128 \
  --checkpoint models/retrained_adaptive_graph_20260903.pt \
  --output logs/retrained_graph_metrics_20260903.json
```

## Scope note

This is a retraining of the repository's current baseline pipeline, not a
claim that every phase in the attached research workflow is implemented. The
spreadsheet's absent weather, AQI, vehicle, and infrastructure signals should
be supplied in a future enriched export before making monsoon- or
vehicle-specific claims.