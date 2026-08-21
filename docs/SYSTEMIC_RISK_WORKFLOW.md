# Systemic-risk comparison contract

This branch changes the graded deliverable from a real-time traffic system to an offline India systemic-risk early-warning benchmark. The existing traffic implementation remains intact on main and other branches.

## Contract

- Target: high_stress_next_30d, where 1 marks the 30 trading days before a documented stress event.
- Shared input: data/systemic_risk/features.csv. Every model must read the same rows, feature columns, target, and date split.
- Split: chronological; train ends before the final held-out crisis period. Never shuffle time-series rows.
- Required metrics: Accuracy, Precision, Recall, F1, and ROC-AUC. F1 and ROC-AUC lead interpretation because the positive class is rare.
- Required comparison: Logistic Regression, Random Forest, XGBoost, LSTM, and graph-level GNN.

## Current implementation

- scripts/evaluate_systemic_models.py evaluates the shared matrix with Logistic Regression and Random Forest, and XGBoost when installed.
- LSTM and GNN are intentionally not marked complete until they consume the same labeled date range and produce comparable probabilities.
- Data ingestion is offline-first. Source adapters for yfinance, NSE/nsepy, FRED, RBI DBIE, and manually curated event evidence belong in the data preparation phase.

## Labeling rules

Initial event set: 2008 GFC, 2013 Taper Tantrum, IL&FS default (September 2018), YES Bank (March 2020), COVID crash (March 2020), and Adani/Hindenburg (January 2023). Store the event date and source in a manifest so labels are auditable; do not infer labels from model outputs.
